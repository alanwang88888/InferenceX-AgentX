"""Kernel-only G ladder (package docstring). Run in the server env (PERFBOT_PRE_EXEC), one GPU:

  python3 -B -m perfbot_kernels.skinny_ladder.selftest

Per distinct (N, K) of SITES and M = 8 x rows, rows 1..6, bf16 x [M, K] ~ N(0, 1), w [N, K] ~ N(0, 1 / K):
  served   the image's route: low_latency_gemm._run_plan on its own _build_plan, else F.linear (cuBLAS), as
           KimiK3LowLatencyLinearMethod.apply does.
  chunk<c> per row slice of at most c rows (c in CHUNKS), the image's plan entry for that slice size: dsv3_fused_a_gemm
           written into the output slice (enable_pdl=True, as served), torch.mm(out=slice) where the plan has none,
           the CuTe skinny GEMM + copy where the plan picks it. Only rows M > 16 differ from served.
Each arm: one eager call, a CUDA graph of one call, GATE_REPLAYS replays (each bit-equal to the eager output), then
REPLAYS timed replays, each between its own events after a 512 MB L2 flush (served layers read distinct weights);
median us. rel = rel-L2 vs the fp32 product.
Lines: `SKINNY n= k= m= backend= us= rel= replays_equal=`, `SKINNY-STEP rows= saving_ms=` (step_saving_ms: kernel
time; four of the six sites run on the aux stream, so exposure is the serve pair's to rule on),
`SKINNY-GATE pass=<bool>`: every chunk arm ran, its replays equal its eager call, rel <= REL_FACTOR x rel(served).
Exit 0 iff pass. Record skinny_ladder.json under $PERFBOT_WORKSPACE/$PERFBOT_RESULTS.
"""

import json
import os
import statistics
import sys
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

import perfbot_kernels.skinny_ladder as sl

REPLAYS, GATE_REPLAYS = 60, 2
REL_FACTOR = 1.25
FLUSH_BYTES = 512 << 20


def results_path(name) -> Path:
    root = Path(os.environ.get("PERFBOT_WORKSPACE", "/workspace"))
    out = root / os.environ.get("PERFBOT_RESULTS", "results/measure")
    out.mkdir(parents=True, exist_ok=True)
    return out / name


def routes():
    """(served(x, w), chunked(x, w, c)) bound to the image's router."""
    from vllm import _custom_ops as ops
    from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import shape_dynamic_skinny_gemm
    from vllm.models.kimi_k3.nvidia import low_latency_gemm as llg

    plans = {}

    def plan(w):
        key = tuple(w.shape)
        if key not in plans:
            spec = llg.KIMI_K3_PROJECTIONS.get(key)
            plans[key] = llg._build_plan(spec) if spec is not None else {}
        return plans[key]

    def served(x, w):
        out = llg._run_plan(plan(w), x, w)
        return out if out is not None else F.linear(x, w)

    def chunked(x, w, c):
        out = torch.empty((x.shape[0], w.shape[0]), dtype=x.dtype, device=x.device)
        for a, b in sl.chunks(x.shape[0], c):
            xs, os_ = x[a:b], out[a:b]
            backend, cfg = plan(w).get(b - a, (None, None))
            if backend == "dsv3_fused_a":
                ops.dsv3_fused_a_gemm(os_, xs, w.t(), enable_pdl=True)
            elif backend == "cute":
                os_.copy_(shape_dynamic_skinny_gemm(xs, w, cfg, None))
            else:
                torch.mm(xs, w.t(), out=os_)
        return out

    return served, chunked


def graph(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        res = fn()
    return g, res


def time_graph(g, flush):
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(REPLAYS)]
    for _ in range(3):
        g.replay()
    for a, b in ev:
        flush.zero_()
        a.record()
        g.replay()
        b.record()
    torch.cuda.synchronize()
    return round(statistics.median(a.elapsed_time(b) * 1e3 for a, b in ev), 2)


def arm(fn, ref, flush):
    eager = fn().clone()
    g, res = graph(fn)
    got = []
    for _ in range(GATE_REPLAYS):
        g.replay()
        torch.cuda.synchronize()
        got.append(res.clone())
    r = ((eager.double() - ref).norm() / ref.norm()).item()
    return {"us": time_graph(g, flush), "rel": r, "replays_equal": all(torch.equal(eager, x) for x in got)}


def main() -> int:
    torch.manual_seed(0)
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    served, chunked = routes()
    flush = torch.empty(FLUSH_BYTES, dtype=torch.uint8, device=dev)
    rows, us, ok = [], {}, True
    for n, k in sl.shapes():
        w = (torch.randn(n, k, device=dev) * k ** -0.5).to(torch.bfloat16)
        for r in sl.ROWS:
            m = r * sl.M_PER_ROW
            x = torch.randn(m, k, device=dev).to(torch.bfloat16)
            ref = (x.double() @ w.double().t())
            arms = {"served": lambda: served(x, w)}
            if m > 16:
                arms.update({f"chunk{c}": (lambda c=c: chunked(x, w, c)) for c in sl.CHUNKS})
            res = {}
            for name, fn in arms.items():
                try:
                    res[name] = arm(fn, ref, flush)
                except Exception as e:  # noqa: BLE001 - a failed arm is a gate failure, recorded
                    res[name] = {"error": f"{type(e).__name__}: {e}"[:300]}
                    traceback.print_exc()
                row = {"n": n, "k": k, "m": m, "backend": name, **res[name]}
                rows.append(row)
                print("SKINNY " + " ".join(f"{a}={b}" for a, b in row.items()), flush=True)
            us[(n, k, m)] = {b: v["us"] for b, v in res.items() if "us" in v}
            for name, v in res.items():
                if name != "served":
                    ok &= ("us" in v and v["replays_equal"] and "rel" in res.get("served", {})
                           and v["rel"] <= REL_FACTOR * res["served"]["rel"])
    steps = {r: round(sl.step_saving_ms(us, r), 3) for r in sl.ROWS}
    for r, s in steps.items():
        print(f"SKINNY-STEP rows={r} saving_ms={s}", flush=True)
    results_path("skinny_ladder.json").write_text(json.dumps({"rows": rows, "step_saving_ms": steps, "pass": ok},
                                                             indent=1))
    print(f"SKINNY-GATE pass={ok}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
