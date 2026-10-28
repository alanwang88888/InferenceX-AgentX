# SPDX-License-Identifier: Apache-2.0
"""GPU self-test of the draft AR consumer on ONE GPU (the stand-in; never more devices).

    python3 -B -m perfbot_kernels.draft_ar.selftest.gpu --gate [--out DIR]     G0 anchors, G1-G3, mutants
    python3 -B -m perfbot_kernels.draft_ar.selftest.gpu --time [--out DIR]     graph replay timing (b200_timed.sh)
    python3 -B -m perfbot_kernels.draft_ar.selftest.gpu --iket                 a few launches under run-iket

G0 anchors the judge: (a) MUFU.RCP / MUFU.RSQ are read from the hardware by a probe kernel; (b) the emulation is
bit-equal to the served flashinfer binary itself, run as N ranks on one GPU (N concurrent kernels over a
one-GPU Lamport workspace): TP8 at M <= 14 (co-residency bound) and TP2 at M 38..56 (the 14-warp block).
Exit 1 on any failed check.
"""

import argparse
import json
import os
import sys
import time

import torch

from perfbot_kernels.draft_ar import H, TP, StandInBox, served_warps_per_block
from perfbot_kernels.draft_ar.selftest import judge

EPS = 1e-5  # drafter config.json rms_norm_eps


# ------------------------------------------------------------------ MUFU probe (hardware oracle for the judge)

_PROBE = {}


def _probe_compiled():
    if "k" in _PROBE:
        return _PROBE["k"]
    import cutlass
    import cutlass.cute as cute
    from cutlass import Float32

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.primitives import asm

    @asm(Float32, pure=True)
    def rsq(a):
        return "rsqrt.approx.ftz.f32 $0,$1;", "=f,f", (Float32(a),)

    @asm(Float32, pure=True)
    def rcp(a):
        return "rcp.approx.ftz.f32 $0,$1;", "=f,f", (Float32(a),)

    @cute.kernel
    def probe_kernel(src: cute.Tensor, out_rsq: cute.Tensor, out_rcp: cute.Tensor):
        bid, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        i = bid * 256 + tid
        if i < src.shape[0]:
            v = src[i]
            out_rsq[i] = rsq(v)
            out_rcp[i] = rcp(v)

    @cute.jit
    def probe(src: cute.Tensor, out_rsq: cute.Tensor, out_rcp: cute.Tensor):
        n = src.shape[0]
        probe_kernel(src, out_rsq, out_rcp).launch(grid=((n + 255) // 256, 1, 1), block=(256, 1, 1))

    _PROBE["k"] = (probe, cute)
    return _PROBE["k"]


def mufu(values):
    """(MUFU.RSQ, MUFU.RCP) of fp32 ``values`` (cuda) read from the hardware."""
    from cutlass.cute.runtime import from_dlpack

    probe, cute = _probe_compiled()
    src = values.contiguous().float()
    a, b = torch.empty_like(src), torch.empty_like(src)
    views = [from_dlpack(t).mark_layout_dynamic() for t in (src, a, b)]
    if "c" not in _PROBE:
        _PROBE["c"] = cute.compile(probe, *views)
    _PROBE["c"](*views)
    return a, b


def hw_rsqrt(t):
    return mufu(t)[0]


# ------------------------------------------------------------------ the served op, N ranks on one GPU


class ServedStandIn:
    """flashinfer trtllm_allreduce_fusion (oneshot Lamport, kARResidualRMSNorm, fp32_acc, PDL) as N ranks on one GPU:
    N Lamport buffers + N flag blocks (the workspace layout of trtllm_create_ipc_workspace_for_all_reduce_fusion),
    one stream per rank, all N kernels resident at once."""

    def __init__(self, n, device, max_tokens=128):
        from flashinfer.comm import trtllm_ar

        self.ar, self.n, self.device = trtllm_ar, n, device
        self.comm = n * max_tokens * H * 2
        self.bufs = [torch.full((3 * self.comm // 2,), -32768, dtype=torch.int16, device=device) for _ in range(n)]
        self.flags = [torch.tensor([0, 0, 0, self.comm, 0], dtype=torch.int32, device=device) for _ in range(n)]
        self.dummy = torch.zeros(64, dtype=torch.int32, device=device)
        lam = [b.data_ptr() for b in self.bufs]
        self.ws = [torch.tensor([self.dummy.data_ptr()] * 2 * n + lam + [f.data_ptr()], dtype=torch.int64,
                                device=device) for f in self.flags]
        self.streams = [torch.cuda.Stream(device) for _ in range(n)]

    def launch(self, xs, res, gamma, eps, normed, rout):
        from flashinfer.comm import QuantizationSFLayout

        m = xs.shape[1]
        main = torch.cuda.current_stream(self.device)
        start = torch.cuda.Event()
        start.record(main)
        for r in range(self.n):
            s = self.streams[r]
            s.wait_event(start)
            with torch.cuda.stream(s):
                self.ar.trtllm_allreduce_fusion(
                    allreduce_in=xs[r], world_size=self.n, world_rank=r, token_num=m, hidden_dim=H,
                    workspace_ptrs=self.ws[r], launch_with_pdl=True, trigger_completion_at_end=True,
                    fp32_acc=True, pattern_code=self.ar.AllReduceFusionPattern.kARResidualRMSNorm,
                    use_oneshot=True, allreduce_out=None, residual_in=res, residual_out=rout[r],
                    norm_out=normed[r], quant_out=None, scale_out=None, rms_gamma=gamma, rms_eps=eps,
                    scale_factor=None, layout_code=QuantizationSFLayout.SWIZZLED_128x4, weight_bias=0.0)
            done = torch.cuda.Event()
            done.record(s)
            main.wait_event(done)

    def run(self, xs, res, gamma, eps):
        normed = torch.empty(self.n, *xs.shape[1:], dtype=torch.bfloat16, device=xs.device)
        rout = torch.empty_like(normed)
        self.launch(xs, res, gamma, eps, normed, rout)
        return normed, rout


# ------------------------------------------------------------------ salts


def salt(m, seed, kind="randn", device="cuda"):
    """(xs [8, M, H], residual, gamma, eps) of one pre-registered salt."""
    g = torch.Generator().manual_seed(seed)
    xs = torch.randn(TP, m, H, generator=g)
    res = torch.randn(m, H, generator=g)
    gamma = 1 + 0.1 * torch.randn(H, generator=g)
    eps = EPS
    if kind == "wide":  # rank order matters only when a column spans > ~2^16 (test_cpu)
        xs = xs * torch.exp2(torch.randint(-24, 25, xs.shape, generator=g).float())
    elif kind == "res0":
        res = torch.zeros_like(res)
    elif kind == "large":  # rows near the bf16 top: sumsq overflows to inf in row 0, 2^40 elsewhere
        xs = xs * 2.0 ** 40
        xs[:, 0] = xs[:, 0] * 2.0 ** 70
    elif kind == "eps":  # zero rows (rstd = rsqrt(eps)), tiny rows (eps dominates), bf16 denormals (FTZ)
        xs = xs * 2.0 ** -70
        res = res * 2.0 ** -70
        xs[:, 0], res[0] = 0.0, 0.0
        if m > 1:
            xs[:, 1] = xs[:, 1] * 2.0 ** -62  # 2^-132: bf16 subnormals
            res[1] = res[1] * 2.0 ** -62
        eps = 1e-30
    elif kind == "negzero":  # -0 partials (sentinel halves) and exact cancellations
        xs[:, :, ::3] = -0.0
        xs[1::2, :, 1::3] = -xs[0::2, :, 1::3]
    to = dict(device=device)
    return (xs.to(torch.bfloat16).to(**to), res.to(torch.bfloat16).to(**to), gamma.to(torch.bfloat16).to(**to), eps)


SALTS = [(1, "randn"), (7, "randn"), (14, "randn"), (37, "randn"), (38, "randn"), (42, "randn"), (56, "randn"),
         (14, "wide"), (42, "wide"), (14, "res0"), (14, "large"), (7, "eps"), (14, "negzero")]


# ------------------------------------------------------------------ gate


class Gate:
    def __init__(self):
        self.checks, self.rcp = [], None

    def add(self, items):
        for c in items:
            self.checks.append(c)
            if not c["ok"]:
                print("FAIL", json.dumps(c), flush=True)

    def want(self, xs, res, gamma, eps, tp=TP):
        m = xs.shape[1]
        return judge.emulate_served(xs[:tp].cpu(), res.cpu(), gamma.cpu(), eps, served_warps_per_block(m),
                                    rsqrt=lambda t: hw_rsqrt(t.cuda()).cpu(), rcp_hidden=self.rcp)


def used(box, parity, m):
    return [judge.rearmed(f"recv[{parity}][:, :{m}]", box.recv[parity][:, :m]),
            judge.rearmed(f"gather[{parity}][:, :{m}]", box.gather[parity][:, :m])]


def gate(out_dir):
    dev = torch.device("cuda")
    G = Gate()
    # G0a: hardware MUFU values
    rsq, rcp = mufu(torch.tensor([float(H)], device=dev))
    G.rcp = rcp.cpu()
    sweep = torch.exp2(torch.empty(1 << 22, device=dev).uniform_(-20, 40))
    torch_vs_mufu = int((torch.rsqrt(sweep) != hw_rsqrt(sweep)).sum())
    info = {"mufu_rcp_7168": float(G.rcp), "rn_1_over_7168": float(torch.tensor(1.0 / H, dtype=torch.float32)),
            "torch_rsqrt_vs_mufu_mismatch_of_4M": torch_vs_mufu,
            "sm_count": torch.cuda.get_device_properties(dev).multi_processor_count}
    print("G0a", json.dumps(info), flush=True)

    # G0b: emulation == served flashinfer binary (N ranks on one GPU)
    for n, ms, kinds in ((TP, (1, 7, 14), ("randn", "wide", "eps", "large", "negzero", "res0")),
                         (2, (38, 42, 56), ("randn", "wide"))):
        served = ServedStandIn(n, dev)
        for m in ms:
            for kind in kinds:
                xs, res, gamma, eps = salt(m, 1000 + m, kind)
                normed, rout = served.run(xs[:n], res, gamma, eps)
                torch.cuda.synchronize()
                wn, wr = G.want(xs, res, gamma, eps, tp=n)
                G.add(judge.judge_call(f"G0b/served_tp{n}/M{m}/{kind}", normed.cpu(), rout.cpu(), wn, wr))
        del served
    torch.cuda.synchronize()

    # G1 + G2 + G3 salts, one call each, parity alternating, re-arm checked after every call
    box = StandInBox(dev)
    parity = 0
    for i, (m, kind) in enumerate(SALTS):
        xs, res, gamma, eps = salt(m, i, kind)
        normed, rout = box.run(parity, xs, res, gamma, eps)
        torch.cuda.synchronize()
        G.add(judge.judge_call(f"G1/M{m}/{kind}", normed.cpu(), rout.cpu(), *G.want(xs, res, gamma, eps)))
        G.add(used(box, parity, m))
        parity ^= 1

    # G3 re-arm: 3 consecutive calls (parity 0, 1, 0), different data, no sync between
    outs, wants = [], []
    for k, p in enumerate((0, 1, 0)):
        xs, res, gamma, eps = salt(14, 50 + k, "wide")
        outs.append(box.run(p, xs, res, gamma, eps))
        wants.append(G.want(xs, res, gamma, eps))
    torch.cuda.synchronize()
    for k in range(3):
        G.add(judge.judge_call(f"G3/rearm/call{k}", outs[k][0].cpu(), outs[k][1].cpu(), *wants[k]))
    G.add(used(box, 0, 14) + used(box, 1, 14))

    # G3 graph capture + replay with swapped inputs (one pair: parity 0 then 1, as the draft graph alternates)
    for m in (14, 42):
        xs, res, gamma, eps = salt(m, 70, "randn")
        sx = [xs.clone(), xs.clone()]
        sr, sg = res.clone(), gamma.clone()
        on = [torch.empty(TP, m, H, dtype=torch.bfloat16, device=dev) for _ in range(2)]
        orr = [torch.empty_like(on[0]) for _ in range(2)]
        for p in range(2):  # warm (compile) outside capture
            box.launch(p, sx[p], sr, sg, eps, on[p], orr[p])
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for p in range(2):
                box.launch(p, sx[p], sr, sg, eps, on[p], orr[p])
        for rep in range(2):
            want = []
            for p in range(2):
                nx, nres, _, _ = salt(m, 80 + 2 * rep + p, "wide" if p else "randn")
                sx[p].copy_(nx)
                if p == 0:
                    sr.copy_(nres)
                want.append(G.want(sx[p], sr, sg, eps))
            graph.replay()
            torch.cuda.synchronize()
            for p in range(2):
                G.add(judge.judge_call(f"G3/graph/M{m}/rep{rep}/p{p}", on[p].cpu(), orr[p].cpu(), *want[p]))
            G.add(used(box, 0, m) + used(box, 1, m))
        del graph

    passed = judge.verdict(G.checks)

    # mutants: each must fail the judge
    mutants = {}
    for mutant, kind in (("order", "wide"), ("norearm", "randn"), ("noresidual", "randn")):
        box.rearm_all()
        xs, res, gamma, eps = salt(14, 90, kind)
        normed, rout = box.run(0, xs, res, gamma, eps, mutant=mutant)
        torch.cuda.synchronize()
        checks = judge.judge_call(f"mutant/{mutant}", normed.cpu(), rout.cpu(), *G.want(xs, res, gamma, eps))
        checks += used(box, 0, 14)
        failed = [c["name"] for c in checks if not c["ok"]]
        mutants[mutant] = {"caught": bool(failed), "failed_checks": failed[:6]}
        print("MUTANT", mutant, json.dumps(mutants[mutant]), flush=True)
    box.rearm_all()
    caught = all(v["caught"] for v in mutants.values())

    summary = {"gate": "AR same rank-order bits", "passed": passed, "mutants_caught": caught,
               "checks": len(G.checks), "failed": [c for c in G.checks if not c["ok"]][:20], "G0a": info,
               "mutants": mutants, "salts": SALTS}
    print("SUMMARY", json.dumps({k: summary[k] for k in ("passed", "mutants_caught", "checks")}), flush=True)
    if out_dir:
        with open(os.path.join(out_dir, "gate.json"), "w") as f:
            json.dump(summary, f, indent=1)
    return 0 if passed and caught else 1


# ------------------------------------------------------------------ timing


def graph_us(fn, calls, replays=60, blocks=2):
    """us per call over ``replays`` replays of a graph of ``calls`` calls; one value per block (A/A band)."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    out = []
    for _ in range(blocks):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(replays):
            g.replay()
        b.record()
        torch.cuda.synchronize()
        out.append(a.elapsed_time(b) * 1e3 / replays / calls)
    return out


def timing(out_dir):
    dev = torch.device("cuda")
    box = StandInBox(dev)
    calls = 20
    rows = []
    for m in (14, 42):
        xs, res, gamma, eps = salt(m, 7, "randn")
        n = torch.empty(TP, m, H, dtype=torch.bfloat16, device=dev)
        r = torch.empty_like(n)

        def mine():
            for k in range(calls):
                box.launch(k % 2, xs, res, gamma, eps, n, r)

        rows.append({"what": "draft_ar stand-in (8 ranks' work, 1 kernel)", "M": m, "us_per_call": graph_us(mine, calls)})
        print("TIME", json.dumps(rows[-1]), flush=True)
    served = ServedStandIn(TP, dev)
    for m in (14,):
        xs, res, gamma, eps = salt(m, 7, "randn")
        n = torch.empty(TP, m, H, dtype=torch.bfloat16, device=dev)
        r = torch.empty_like(n)

        def theirs():
            for _ in range(calls):
                served.launch(xs, res, gamma, eps, n, r)

        rows.append({"what": "served flashinfer stand-in (8 concurrent rank kernels)", "M": m,
                     "us_per_call": graph_us(theirs, calls)})
        print("TIME", json.dumps(rows[-1]), flush=True)
    if out_dir:
        with open(os.path.join(out_dir, "timing.json"), "w") as f:
            json.dump({"device": torch.cuda.get_device_name(dev), "calls_per_graph": calls, "replays": 60,
                       "rows": rows, "time": time.strftime("%FT%TZ", time.gmtime())}, f, indent=1)
    return 0


def iket_run():
    dev = torch.device("cuda")
    box = StandInBox(dev)
    for k, m in enumerate((14, 14, 42, 42)):
        xs, res, gamma, eps = salt(m, k, "randn")
        box.run(k % 2, xs, res, gamma, eps, iket=True)
    torch.cuda.synchronize()
    print("PERFBOT-DRAFT-AR-IKET ok", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--iket", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.out:
        os.makedirs(a.out, exist_ok=True)
    if a.iket:
        return iket_run()
    rc = 0
    if a.gate:
        rc |= gate(a.out)
    if a.time:
        rc |= timing(a.out)
    return rc


if __name__ == "__main__":
    sys.exit(main())
