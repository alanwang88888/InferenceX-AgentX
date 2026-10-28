# SPDX-License-Identifier: Apache-2.0
"""B300 kernel-only arm i15-ar (item 15, kernel 1): served oneshot AR + residual + RMSNorm vs draft_ar two-shot, TP8.
WALL_BUDGET_S = 300  (8 ranks, no server, no weights; vLLM init + DSL compiles are most of it; image vLLM 49cb2bbf)
Run:   python3 -B -m perfbot_kernels.draft_ar.b300_ar --ms 1,7,14,42,56 --windows 600   (PYTHONPATH = overlay)
Output: stdout, one JSON line per (M, op), then one summary line; every line has "probe": "i15-ar".
Exit 0 = every M ran and the two ops' outputs are bit-equal on every rank; 1 = a bit mismatch; 2 = a rank failed
or the wall budget ran out before every M ran.

Ops, both at the draft seam's shapes (x [M, 7168] bf16 per rank, residual [M, 7168], gamma [7168], eps 1e-5):
- served: vllm.models.common.ops.fused_allreduce_rms_norm (flashinfer trtllm_allreduce_fusion, oneshot Lamport,
  kARResidualRMSNorm, fp32_acc, PDL), called exactly as dspark_mla.py calls it. Its workspace comes from vLLM's own
  get_fi_ar_workspace on the TP group.
- twoshot: perfbot_kernels.draft_ar.TwoShotBox (symmetric memory + multicast gather), parity = call index % 2.
Timing per (M, op), on every rank: one CUDA graph of CALLS back-to-back calls; WINDOWS replays, each between two
CUDA events, no host sync between replays. us per call = window / CALLS. The served op writes its residual into its
input, so the graph's values grow to inf over the replays; that does not change the kernel's work or its Lamport
polling (the sentinel is -0 only). The bit check runs first, eager, on fresh inputs.
Read: per M, max over ranks of the per-rank median (the collective is as slow as its slowest rank).
"""

import argparse
import json
import os
import socket
import sys
import time

import torch
import torch.multiprocessing as mp

TP, H, EPS, CALLS = 8, 7168, 1e-5, 20
WALL_BUDGET_S = 280  # inside the arm's 300 s PRE_EXEC timeout, so the summary line is always written
MARGIN_S = 25  # stop starting new M this close to the budget


def q(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f = int(k)
    return xs[f] + (xs[min(f + 1, len(xs) - 1)] - xs[f]) * (k - f)


def emit(row):
    print(json.dumps({"probe": "i15-ar", **row}), flush=True)


def windows_us(fn, windows):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(windows)]
    for a, b in ev:
        a.record()
        g.replay()
        b.record()
    torch.cuda.synchronize()
    return [a.elapsed_time(b) * 1e3 / CALLS for a, b in ev]


def inputs(m, rank, dev):
    g = torch.Generator(device="cpu").manual_seed(1000 * m + rank)
    x = torch.randn(m, H, generator=g).to(dev, torch.bfloat16)
    g.manual_seed(1000 * m + 99)  # residual and gamma: the same on every rank, as in the model
    res = torch.randn(m, H, generator=g).to(dev, torch.bfloat16)
    gamma = (1 + 0.1 * torch.randn(H, generator=g)).to(dev, torch.bfloat16)
    return x, res, gamma


def worker(rank, port, ms, windows, t0, queue):
    try:
        queue.put((rank, run_rank(rank, port, ms, windows, t0)))
    except BaseException as e:  # the parent reports it; the other ranks then time out on their collectives
        queue.put((rank, {"error": f"{type(e).__name__}: {e}"}))
        raise


def run_rank(rank, port, ms, windows, t0):
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import get_tp_group, init_distributed_environment, initialize_model_parallel

    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(world_size=TP, rank=rank, distributed_init_method=f"tcp://127.0.0.1:{port}",
                                     local_rank=rank, backend="nccl")
        initialize_model_parallel(tensor_model_parallel_size=TP)
        from vllm.model_executor.layers.fused_allreduce_gemma_rms_norm import _can_use_flashinfer
        from vllm.model_executor.layers.layernorm import RMSNorm
        from vllm.models.common.ops.fused_allreduce_rms_norm import fused_allreduce_rms_norm

        from perfbot_kernels.draft_ar import TwoShotBox

        group = get_tp_group()
        box = TwoShotBox(group.device_group, dev, rank)
        norm = RMSNorm(H, eps=EPS).to(dev, torch.bfloat16)
        out = {"rows": [], "skipped": []}
        for m in ms:
            # one decision for all ranks: rank 0's clock, broadcast
            late = torch.tensor([time.time() - t0 > WALL_BUDGET_S - MARGIN_S], device=dev)
            torch.distributed.broadcast(late, 0, group=group.device_group)
            if late.item():
                out["skipped"].append(m)
                continue
            x, res, gamma = inputs(m, rank, dev)
            with torch.no_grad():
                norm.weight.copy_(gamma)
            # bit check, eager, fresh inputs
            assert _can_use_flashinfer(x, TP)[0], "served flashinfer fused path not taken"
            n_s, r_s = fused_allreduce_rms_norm(x.clone(), res, norm)
            mine = [box.run(p, x, res, gamma, EPS) for p in (0, 1)]  # both parities
            torch.cuda.synchronize()
            same = lambda a, b: bool(torch.equal(a.view(torch.int16), b.view(torch.int16)))
            bits = {"normed_equal": all(same(n_s, n) for n, _ in mine),
                    "residual_equal": all(same(r_s, r) for _, r in mine),
                    "normed_max_abs": max(float((n_s.float() - n.float()).abs().max()) for n, _ in mine)}
            # timing
            xs = x.clone()
            normed = torch.empty_like(x)
            rout = torch.empty_like(x)

            def served():
                for _ in range(CALLS):
                    fused_allreduce_rms_norm(xs, res, norm)

            def twoshot():
                for k in range(CALLS):
                    box.launch(k % 2, x, res, gamma, EPS, normed.view(1, m, H), rout.view(1, m, H))

            for op, fn in (("served_oneshot", served), ("draft_ar_twoshot", twoshot)):
                torch.distributed.barrier(group=group.device_group)
                us = windows_us(fn, windows)
                out["rows"].append({"M": m, "op": op, "median_us": q(us, 0.5), "p10_us": q(us, 0.1),
                                    "p90_us": q(us, 0.9), **(bits if op == "draft_ar_twoshot" else {})})
        torch.distributed.barrier(group=group.device_group)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", default="1,7,14,42,56")
    ap.add_argument("--windows", type=int, default=600)
    a = ap.parse_args()
    ms = [int(v) for v in a.ms.split(",")]
    t0 = time.time()
    assert torch.cuda.device_count() >= TP, f"needs {TP} GPUs, sees {torch.cuda.device_count()}"
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, port, ms, a.windows, t0, queue)) for r in range(TP)]
    for p in procs:
        p.start()
    res = {}
    deadline = t0 + WALL_BUDGET_S
    while len(res) < TP and time.time() < deadline:
        try:
            r, v = queue.get(timeout=max(1.0, deadline - time.time()))
            res[r] = v
            if "error" in v:
                break
        except Exception:
            break
    for p in procs:
        p.join(timeout=5)
        if p.is_alive():
            p.kill()
    errors = {r: v["error"] for r, v in res.items() if "error" in v}
    if errors or len(res) < TP:
        emit({"summary": True, "ok": False, "errors": errors, "ranks_reported": sorted(res),
              "seconds": round(time.time() - t0, 1)})
        return 2
    rc, summary = 0, {}
    for i, row in enumerate(res[0]["rows"]):
        per_rank = [res[r]["rows"][i] for r in range(TP)]
        line = {"M": row["M"], "op": row["op"], "tp": TP, "calls_per_window": CALLS, "windows": a.windows,
                "median_us_max_rank": max(v["median_us"] for v in per_rank),
                "median_us_per_rank": [round(v["median_us"], 3) for v in per_rank],
                "p90_us_max_rank": max(v["p90_us"] for v in per_rank),
                "p10_us_min_rank": min(v["p10_us"] for v in per_rank)}
        if row["op"] == "draft_ar_twoshot":
            line["bit_equal_all_ranks"] = all(v["normed_equal"] and v["residual_equal"] for v in per_rank)
            line["normed_max_abs"] = max(v["normed_max_abs"] for v in per_rank)
            rc |= 0 if line["bit_equal_all_ranks"] else 1
        summary.setdefault(row["M"], {})[row["op"]] = line["median_us_max_rank"]
        emit(line)
    skipped = res[0]["skipped"]
    for m, d in summary.items():
        if len(d) == 2:
            d["twoshot_minus_served_us"] = d["draft_ar_twoshot"] - d["served_oneshot"]
    emit({"summary": True, "ok": rc == 0 and not skipped, "device": torch.cuda.get_device_name(0), "per_M": summary,
          "skipped_M": skipped, "seconds": round(time.time() - t0, 1)})
    return rc or (2 if skipped else 0)


if __name__ == "__main__":
    sys.exit(main())
