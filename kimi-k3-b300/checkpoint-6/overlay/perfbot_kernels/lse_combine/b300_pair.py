# SPDX-License-Identifier: Apache-2.0
"""B300 kernel-only arm i15-pair (item 15, kernels 2 + 3): the served draft DCP8 query chain + q-gather + LSE combine
+ W_UV vs the pair DraftQuery (Lamport publish) + DraftValue (Lamport), TP8 = DCP8.
WALL_BUDGET_S = 300  (8 ranks, no server, no weights; vLLM init + DSL compiles are most of it; image vLLM 49cb2bbf)
Run:   python3 -B -m perfbot_kernels.lse_combine.b300_pair --ms 1,7,14,42,56 --windows 600   (PYTHONPATH = overlay)
Output: stdout, one JSON line per (M, op), then one summary line; every line has "probe": "i15-pair".
Exit (also the first fields of the summary line: "rc", "rc_meaning"): 0 = every M ran and every bit check passed on
every rank; 1 = a bit check failed; 2 = a crash, a rank failed, the served path was not the direct symmetric-memory
one, or the wall budget ran out before every M ran. The pre-exec runs selftest/test_q_g2.py first (exit 3 = the q
judge itself is broken).

Served (exactly the served calls; vllm/models/kimi_k3/nvidia/mla.py):
  query chain   fused_q_kv_rmsnorm, F.linear (q_b), torch.bmm (W_UK_T), fused_mla_decode_q_concat_kv_cache_insert
                (FP8 cache, RoPE): perfbot_kernels.draft_query.selftest.gpu.Served.rank, the chain of mla.py:509-528 +
                690-706 (tokenspeed takes an FP8 query: tokenspeed_mla.py:282-288)
  q gather      MLADCPManager.query_gather (mla.py:711): DirectDCPQGatherWorkspace.gather (multimem)
  combine       MLADCPManager.combine (mla.py:719): DirectDCPA2AWorkspace.lse_reduce (4 launches)
  W_UV          the bmm of mla.py:_v_up_proj (:475)
  The manager is vLLM's own MLADCPManager, built with the arguments mla.py:379-390 passes for the draft: 8 local
  heads, query head dim 576 (FP8), output head dim 512 (BF16), padded heads None (the tokenspeed impl gets no
  q_pad_num_heads), LSE base 2, no PCP; dcp_comm_backend "a2a"; one ubatch.
Pair: perfbot_kernels.draft_query.LamportBox + perfbot_kernels.lse_combine.LamportBox on one mla.dcp.DcpBox
  (symmetric memory + multicast q mailbox, as TwoShot); parity = call index % 2.
Tokenspeed is not run: it is the same in both arms. Its outputs (rows [M, 64, 512] BF16, LSE [M, 64] log2) are seeded
inputs, the same bytes in both arms; seq_lens / query_start_loc: 7 rows per request, "empty" salt = empty shards.
Bit checks (eager, fresh inputs, both parities, every rank): FP8 cache bytes (served cache vs kernel 2's cache);
gathered query: kernel 2's G2 (draft_query/selftest/judge.py: every E4M3 byte of this rank's q mailbox rows and of
the served q-gather output inside the fp64 envelope; q is not bit-equal to served by design, DESIGN.md §7; byte and
max code distance vs served are reported); kernel 3's merged latent vs the served combine (bits); the [M, 1024] output of kernel 3 and of the served bmm inside the fp64 envelope of
lse_combine/selftest/judge.py (the owner's 8 sources are regenerated from the seeds on every rank).
Timing per (M, op), on every rank: one CUDA graph of CALLS back-to-back calls; WINDOWS replays, each between two CUDA
events, no host sync between replays. us per call = window / CALLS. A full device sync + group barrier separates
phases (two consecutive Lamport calls of one parity are safe only across that barrier).
Read: per M, max over ranks of the per-rank median.
One-GPU check (not the measurement): --world 1 builds the served manager on a 1-rank group (q gather falls back to
all_gather there; no multicast object on one GPU), runs the served query chain + that gather + the W_UV bmm (the
served combine refuses world_size 1, so it is not run), and only compiles the pair's two kernels for the real-rank
configuration (DcpBox needs multicast).
"""

import argparse
import functools
import json
import socket
import sys
import time

import torch
import torch.multiprocessing as mp

HEADS, LATENT, ENTRY, CALLS, Q = 8, 512, 576, 20, 7
WALL_BUDGET_S = 280  # inside the arm's 300 s PRE_EXEC timeout, so the summary line is always written
MARGIN_S = 25  # stop starting new M this close to the budget
RC_MEANING = {0: "ok", 1: "bit check failed", 2: "crash / rank failed / not direct path / M skipped"}
BIT_SALTS = (("randn", 0), ("empty", 1))  # (salt, parity): both parities, one salt with empty shards


def q(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f = int(k)
    return xs[f] + (xs[min(f + 1, len(xs) - 1)] - xs[f]) * (k - f)


def summary_line(rc, **row):
    emit({"summary": True, "rc": rc, "rc_meaning": RC_MEANING[rc], **row})
    return rc


def emit(row):
    print(json.dumps({"probe": "i15-pair", **row}), flush=True)


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


def fence(group):
    """Every rank's earlier kernels are done before any rank starts the next phase."""
    torch.cuda.synchronize()
    torch.distributed.barrier(group=group)
    torch.cuda.synchronize()


def served_manager(vllm_config, dev):
    """vLLM's MLADCPManager with the arguments mla.py:379-390 passes for the draft."""
    from vllm.v1.attention.ops.dcp import MLADCPManager

    vllm_config.parallel_config.dcp_comm_backend = "a2a"
    manager = MLADCPManager(vllm_config=vllm_config, device=dev, num_heads=HEADS, query_head_dim=ENTRY,
                            output_head_dim=LATENT, query_dtype=torch.float8_e4m3fn, output_dtype=torch.bfloat16,
                            padded_num_heads=None, is_lse_base_on_e=False, use_pcp=False)

    def direct(fn):
        return isinstance(fn, functools.partial) and fn.func.__name__.startswith("_direct_workspace")

    info = {"combine_direct": direct(manager.combine), "q_gather_direct": direct(manager.query_gather),
            "max_num_tokens": manager.max_num_tokens}
    return manager, info


class Inputs:
    """One M's seeded inputs of this rank (and, for the envelope, every rank's tokenspeed outputs)."""

    def __init__(self, m, rank, world, dev, kind, seed):
        from perfbot_kernels.draft_query.selftest.gpu import salt as query_salt
        from perfbot_kernels.lse_combine.selftest.gpu import empty_rows, salt

        # tokenspeed outputs of all ranks (the same seed on every rank: rank r takes row block r)
        rows, lse, seq, qsl = salt(m, seed, kind, ranks=8, device=dev)
        self.rows_all, self.lse_all = rows, lse
        self.empty_all = empty_rows(seq, qsl, m)
        r = rank if world == 8 else 0
        self.rows, self.lse, self.seq, self.qsl = rows[r].contiguous(), lse[r].contiguous(), seq[r].contiguous(), qsl
        if world == 1:  # a 1-rank DCP group combines its own 8 heads only (one source): the check runs, no envelope
            self.rows, self.lse = rows[0][:, :HEADS].contiguous(), lse[0][:, :HEADS].contiguous()
        # query chain inputs: qkv and positions replicated, slots [8, M] (token t's slot on rank (t + seed) % 8)
        self.qkv, self.pos, slots = query_salt(m, seed, "randn", device=dev)
        self.slots = slots[r : r + 1].contiguous()


def run_rank(rank, port, world, ms, windows, t0):
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.distributed.parallel_state import get_dcp_group

    vllm_config = VllmConfig()
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(world_size=world, rank=rank, distributed_init_method=f"tcp://127.0.0.1:{port}",
                                     local_rank=rank, backend="nccl")
        initialize_model_parallel(tensor_model_parallel_size=world, decode_context_model_parallel_size=world)
        import vllm._custom_ops  # noqa: F401  (registers the direct DCP ops)

        from perfbot_kernels.draft_query.selftest.gpu import BLOCKS, BS, EPS, Served, make_weights, rope_table
        from perfbot_kernels.draft_query.selftest import judge as qjudge
        from perfbot_kernels.lse_combine.selftest import judge
        from perfbot_kernels.lse_combine.selftest.gpu import make_wuv, served_wuv

        dcp = get_dcp_group()
        group = dcp.device_group
        manager, info = served_manager(vllm_config, dev)
        out = {"rows": [], "skipped": [], "served_path": info}
        if world == 8 and not (info["combine_direct"] and info["q_gather_direct"]):
            raise RuntimeError(f"served path is not the direct symmetric-memory one: {info}")
        weights, table = make_weights(dev), rope_table(dev)
        served_q = Served(weights, table)
        e = rank if world == 8 else 0
        gq, gkv, w_qb, w_uk = weights
        my_weights = (gq, gkv, w_qb[e * 1536 : (e + 1) * 1536].contiguous(), w_uk[e * HEADS : (e + 1) * HEADS])
        w_uv = make_wuv(dev, ranks=8)[e * HEADS : (e + 1) * HEADS]  # absorbed view: latent axis contiguous
        one = torch.ones(1, dtype=torch.float32, device=dev)
        if world == 8:
            from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.mla.dcp import DcpBox

            from perfbot_kernels.draft_query import LamportBox as QueryBox
            from perfbot_kernels.lse_combine import LamportBox as ValueBox

            box = DcpBox(group, dev, 8, rank, HEADS, cap=64)
            k2, k3 = QueryBox(box), ValueBox(box)

        def served_call(x, cache, lat_out=None, out_bf=None):
            mqa, q_c = served_q.rank(e, x.qkv, x.pos, cache[0], x.slots[0], one, one)
            gathered = manager.query_gather(mqa.view(torch.float8_e4m3fn))
            if world == 8:
                lat = manager.combine(x.rows, x.lse, seq_lens=x.seq, query_start_loc=x.qsl)
            else:  # direct_dcp_a2a_lse_reduce refuses world_size 1 (dcp_direct_a2a_lse_reduce.cu:239): not run
                lat = x.rows
            return gathered, lat, served_wuv(lat, w_uv, out=out_bf), q_c

        def q_g2(m, q_c, pos, mine, served):
            """Kernel 2's G2 over the 8 source head blocks: (check mine, check served, bytes diff, max code dist)."""
            mine, served = mine.reshape(m, 8 * HEADS, ENTRY), served.reshape(m, 8 * HEADS, ENTRY)
            cm, cs, nd, dmax = [], [], 0, 0
            for s in range(8):
                ref, err = qjudge.query_reference(q_c, weights[2][s * 1536 : (s + 1) * 1536],
                                                  weights[3][s * HEADS : (s + 1) * HEADS], table[pos], one)
                h = slice(s * HEADS, (s + 1) * HEADS)
                cm.append(qjudge.envelope("q", mine[:, h], ref, err))
                cs.append(qjudge.envelope("served_q", served[:, h], ref, err))
                n, d = qjudge.code_distance(mine[:, h], served[:, h])
                nd, dmax = nd + n, max(dmax, d)
            merge = lambda cl: {"name": f"M{m}/{kind}/{cl[0]['name']}", "ok": all(c["ok"] for c in cl),
                                "mismatches": sum(c["mismatches"] for c in cl),
                                "first": next((c["first"] for c in cl if c["first"]), None)}
            return merge(cm), merge(cs), nd, dmax

        for m in ms:
            late = torch.tensor([time.time() - t0 > WALL_BUDGET_S - MARGIN_S], device=dev)
            torch.distributed.broadcast(late, dcp.ranks[0], group=group)
            if late.item():
                out["skipped"].append(m)
                continue
            bits = {"cache_equal": True, "q_in_envelope": True, "served_q_in_envelope": True, "latent_equal": True,
                    "out_in_envelope": True, "served_out_in_envelope": True, "out_values_diff_vs_served": 0,
                    "q_bytes_diff_vs_served": 0, "q_max_code_distance": 0, "failed": [], "checks_run": 0}
            # ---- bit checks, eager, fresh inputs, both parities
            for kind, parity in BIT_SALTS:
                x = Inputs(m, rank, world, dev, kind, seed=1000 * m + 17 * parity)
                cache_s = torch.zeros(1, BLOCKS, BS, ENTRY, dtype=torch.uint8, device=dev)
                fence(group)
                gathered, lat_s, out_s, q_c = served_call(x, cache_s)
                fence(group)
                checks = []
                if world == 8:
                    cache_k = torch.zeros_like(cache_s)
                    out_k = torch.empty(1, m, HEADS * 128, dtype=torch.bfloat16, device=dev)
                    dump = torch.empty(1, m, HEADS, LATENT, dtype=torch.bfloat16, device=dev)
                    k2.run(x.qkv, my_weights, table, x.pos, cache_k, x.slots, one, one, EPS, parity=parity)
                    torch.cuda.synchronize()  # kernel 2 polled until all 8 head blocks landed; kernel 3 re-arms them
                    mailbox = box.q[parity, :m].view(torch.uint8)
                    checks.append(judge.compare_bytes(f"M{m}/{kind}/cache", cache_k, cache_s))
                    c_mine, c_served, nd, dmax = q_g2(m, q_c, x.pos[:m], mailbox, gathered.view(torch.uint8))
                    checks += [c_mine, c_served]
                    bits["q_bytes_diff_vs_served"] += nd
                    bits["q_max_code_distance"] = max(bits["q_max_code_distance"], dmax)
                    k3.run(x.rows[None], x.lse[None], x.seq[None], x.qsl, w_uv, out_k, parity=parity, dump=dump)
                    fence(group)
                    checks.append(judge.compare_bytes(f"M{m}/{kind}/latent", dump[0], lat_s))
                if world == 8:
                    xs, vs = judge.owner_inputs(x.rows_all, x.lse_all, x.empty_all, e)
                    _, _, ref, err = judge.reference(xs, vs, w_uv)
                    checks.append(judge.envelope(f"M{m}/{kind}/served_out", out_s, ref, err))
                    checks.append(judge.envelope(f"M{m}/{kind}/out", out_k[0], ref, err))
                    bits["out_values_diff_vs_served"] += int((out_k[0].view(torch.int16)
                                                              != out_s.view(torch.int16)).sum())
                bits["checks_run"] += len(checks)
                for c in checks:
                    key = {"cache": "cache_equal", "q": "q_in_envelope", "served_q": "served_q_in_envelope",
                           "latent": "latent_equal", "out": "out_in_envelope", "served_out": "served_out_in_envelope"}[c["name"].split("/")[-1]]
                    bits[key] &= c["ok"]
                    if not c["ok"]:
                        bits["failed"].append({k: c[k] for k in ("name", "mismatches", "first")})
            # ---- timing (the "randn" salt, fixed inputs)
            x = Inputs(m, rank, world, dev, "randn", seed=1000 * m + 5)
            cache_s = torch.zeros(1, BLOCKS, BS, ENTRY, dtype=torch.uint8, device=dev)
            out_bf = torch.empty(m, HEADS * 128, dtype=torch.bfloat16, device=dev)

            def served():
                for _ in range(CALLS):
                    served_call(x, cache_s, out_bf=out_bf)

            ops = [("served_q_chain_gather_combine_wuv", served)]
            if world == 8:
                cache_k = torch.zeros_like(cache_s)
                out_k = torch.empty(1, m, HEADS * 128, dtype=torch.bfloat16, device=dev)

                def pair():
                    for k in range(CALLS):
                        k2.run(x.qkv, my_weights, table, x.pos, cache_k, x.slots, one, one, EPS, parity=k % 2)
                        k3.run(x.rows[None], x.lse[None], x.seq[None], x.qsl, w_uv, out_k, parity=k % 2)

                ops.append(("pair_lamport_k2_k3", pair))
            else:
                compile_only(m, dev, x, my_weights, table, w_uv)
                bits["pair"] = "compile_only"
                bits["served_combine"] = "not_run_world1"
            for op, fn in ops:
                fence(group)
                us = windows_us(fn, windows)
                out["rows"].append({"M": m, "op": op, "median_us": q(us, 0.5), "p10_us": q(us, 0.1),
                                    "p90_us": q(us, 0.9), **(bits if op == ops[-1][0] else {})})
        fence(group)
    return out


def compile_only(m, dev, x, weights, table, w_uv):
    """One-GPU check: compile kernel 2 (Lamport, multimem, 1 rank, polls 8 head blocks) and kernel 3 (Lamport, one
    rank, one mailbox) for the real-rank configuration on tensors shaped like DcpBox's, without launching."""
    import perfbot_kernels.draft_query as dq
    import perfbot_kernels.lse_combine as lc

    q_box = torch.empty(2, 64, 8 * HEADS, ENTRY, dtype=torch.uint8, device=dev)
    from perfbot_kernels.draft_query.selftest.gpu import BLOCKS, BS

    cache = torch.zeros(1, BLOCKS, BS, ENTRY, dtype=torch.uint8, device=dev)
    i32, i64 = dict(dtype=torch.int32, device=dev), dict(dtype=torch.int64, device=dev)
    one = torch.ones(1, dtype=torch.float32, device=dev)
    gq, gkv, w_qb, w_uk = weights
    t2 = (x.qkv, gq, gkv, w_qb, w_uk, table, x.pos[:m], cache, x.slots[:, :m], one, one, q_box[0].view(1, -1),
          torch.zeros(1, 2, 8, **i32), torch.zeros(1, **i64), torch.zeros(1, **i32))
    dq.compiled(tuple(dq.layout(t) for t in t2), m, 1, 0, 8, 8, "", False, True)
    partial = torch.empty(1, 2, 8, 64, HEADS, LATENT, dtype=torch.bfloat16, device=dev)
    lse = torch.empty(1, 2, 8, 64, HEADS, dtype=torch.float32, device=dev)
    out = torch.empty(1, m, HEADS * 128, dtype=torch.bfloat16, device=dev)
    for flag, dump in ((False, torch.empty(1, 1, 1, 8, dtype=torch.bfloat16, device=dev)),
                       (True, torch.empty(1, m, HEADS, LATENT, dtype=torch.bfloat16, device=dev))):
        t3 = (x.rows_all[:1].contiguous(), x.lse_all[:1].contiguous(), x.seq[None], x.qsl, w_uv, out, dump, partial, lse, torch.zeros(1, 8, **i64),
              torch.zeros(1, 8, **i64), torch.zeros(1, 2, 8, **i32), torch.zeros(1, 8, **i64), torch.zeros(1, **i64),
              torch.zeros(1, **i32), q_box.view(1, 2, -1))
        lc.compiled(tuple(lc.layout(t) for t in t3), m, 1, 1, True, 4, "", flag, False)


def worker(rank, port, world, ms, windows, t0, queue):
    try:
        queue.put((rank, run_rank(rank, port, world, ms, windows, t0)))
    except BaseException as e:  # the parent reports it; the other ranks then time out on their collectives
        queue.put((rank, {"error": f"{type(e).__name__}: {e}"}))
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", default="1,7,14,42,56")
    ap.add_argument("--windows", type=int, default=600)
    ap.add_argument("--world", type=int, default=8, choices=(1, 8), help="1 = the one-GPU check, not a measurement")
    a = ap.parse_args()
    ms = [int(v) for v in a.ms.split(",")]
    assert max(ms) <= 64, "DcpBox / kernel capacity is 64 rows"
    t0 = time.time()
    world = a.world
    assert torch.cuda.device_count() >= world, f"needs {world} GPUs, sees {torch.cuda.device_count()}"
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, port, world, ms, a.windows, t0, queue)) for r in range(world)]
    for p in procs:
        p.start()
    res = {}
    deadline = t0 + WALL_BUDGET_S
    while len(res) < world and time.time() < deadline:
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
    if errors or len(res) < world:
        return summary_line(2, ok=False, errors=errors, ranks_reported=sorted(res), world=world,
                            seconds=round(time.time() - t0, 1))
    rc, summary = 0, {}
    emit({"served_path": res[0]["served_path"], "world": world})
    for i, row in enumerate(res[0]["rows"]):
        per_rank = [res[r]["rows"][i] for r in range(world)]
        line = {"M": row["M"], "op": row["op"], "world": world, "calls_per_window": CALLS, "windows": a.windows,
                "median_us_max_rank": max(v["median_us"] for v in per_rank),
                "median_us_per_rank": [round(v["median_us"], 3) for v in per_rank],
                "p90_us_max_rank": max(v["p90_us"] for v in per_rank),
                "p10_us_min_rank": min(v["p10_us"] for v in per_rank)}
        if "failed" in row:
            keys = ("cache_equal", "q_in_envelope", "served_q_in_envelope", "latent_equal", "out_in_envelope",
                    "served_out_in_envelope")
            line["checks_run_per_rank"] = [v["checks_run"] for v in per_rank]
            ran = all(v["checks_run"] == 6 * len(BIT_SALTS) for v in per_rank)  # 6 checks per salt at world 8
            line["bits_all_ranks"] = {k: all(v[k] for v in per_rank) if ran else "not_run" for k in keys}
            line["out_values_diff_vs_served_per_rank"] = [v["out_values_diff_vs_served"] for v in per_rank]
            line["q_bytes_diff_vs_served"] = per_rank[0]["q_bytes_diff_vs_served"]  # the gathered q: same on all ranks
            line["q_max_code_distance"] = max(v["q_max_code_distance"] for v in per_rank)
            line["failed_first"] = [f for v in per_rank for f in v["failed"]][:6]
            for key in ("pair", "served_combine"):
                if key in row:
                    line[key] = row[key]
            if world == 8:  # a check that did not run is a failure at world 8
                rc |= 0 if all(b is True for b in line["bits_all_ranks"].values()) else 1
        summary.setdefault(row["M"], {})[row["op"]] = line["median_us_max_rank"]
        emit(line)
    skipped = res[0]["skipped"]
    for m, d in summary.items():
        if len(d) == 2:
            d["pair_minus_served_us"] = d["pair_lamport_k2_k3"] - d["served_q_chain_gather_combine_wuv"]
    return summary_line(rc or (2 if skipped else 0), ok=rc == 0 and not skipped, world=world,
                        device=torch.cuda.get_device_name(0), per_M=summary, skipped_M=skipped,
                        seconds=round(time.time() - t0, 1))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # an uncaught error must not exit 1 (1 = a bit check failed)
        sys.exit(summary_line(2, ok=False, errors={"main": f"{type(e).__name__}: {e}"}))
