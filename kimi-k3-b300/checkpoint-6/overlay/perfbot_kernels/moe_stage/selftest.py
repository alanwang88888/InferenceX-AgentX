"""Kernel-only gate for the decode MoE stage (r3-318). Run in the server env (serve.sh PERFBOT_PRE_EXEC).

  python3 -B -m perfbot_kernels.moe_stage.selftest           # identity, quant, routing, numerics vs fp64, timings
  python3 -B -m perfbot_kernels.moe_stage.selftest --graph   # CUDA-graph capture + replay per shape
  python3 -B -m perfbot_kernels.moe_stage.selftest --tail    # 8 ranks: stage -> overlay latent-tail collective
  python3 -B -m perfbot_kernels.moe_stage.selftest --iket    # a few launches only (run under run-iket)

Every gate mode updates one record, $W/$PERFBOT_RESULTS/kernel-gate.json, and exits 1 on any failed check.

Paired baseline: the overlay's real TrtLlmMxfp4ExpertsMonolithic.apply on K3's per-rank MoE shape (896 experts,
top-16, latent hidden 3584, I/partition 384, SITU 4/25, synthetic MXFP4 weights through vLLM's own TRTLLM weight
conversion, a packed top-K carrier, deferred finalize):
  control   = PERFBOT_R3_318_MOE_STAGE=0: vLLM's moe_kernel_quantize_input (MXFP8QuantizeLinearKernel) ->
              flashinfer trtllm_fp4_block_scale_routed_moe -> UnfinalizedMoEOutput; its top-K finalize is the
              image tail's (bf16 multiply and add per slot), emulated here bit for bit in torch
  candidate = PERFBOT_R3_318_MOE_STAGE=1: bf16 latent -> perfbot_kernels.moe_stage (permute + quant, fc1, fc2 with
              the weighted finalize into the tail's Lamport buffer; a plain buffer with the same layout here)
Checks:
  quant     the stage's MXFP8 rows and SF bytes equal the production quant of the same bf16 rows, bit for bit,
            including rows built to hit the scale-rounding edges (amax = 448 * 2^k, zero rows, tiny rows).
  routing   expert weights equal the packed bf16 bits (and the control's); expert assignment, per-expert counts and
            tiles equal those derived from the packed ids; the control's permutation is bijective with ordered
            disjoint expert blocks (its in-expert order comes from smem atomics: bitwise map equality is ill-posed).
  numerics  fp64 reference from the same quantized bytes: SiTU(gate = w1, up = w3), fc2, top-K weighted sum, without
            (A) and with (A_q, ceil-ue8m0 MXFP8) the intermediate quant. The stage output's max-abs and mean-abs
            error must be <= the control's (flashinfer + image-tail finalize) against both references.
  protocol  fc2 wrote every (token, rank) slot of the chosen generation and nothing else, no word equals the Lamport
            sentinel, and its fp32 accumulator is zero again after the call.
  tail      (--tail, 8 GPUs) the overlay collective's prefilled role reads the stage's multicast rows: its latent
            output equals AllReduce + RMSNorm of the ranks' rows, across generations and interleaved with the image
            role; per-layer graph-replay time of stage + tail vs quant + flashinfer + tail.
"""

import argparse
import hashlib
import importlib
import json
import os
import statistics
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

E, TOPK, H, I = 896, 16, 3584, 384
ALPHA, BETA = 4.0, 25.0
SHAPES = [1, 3, 8, 16, 24, 32, 40, 48, 56, 64, 128]
CHECK_SHAPES = [1, 8, 48, 64]
TIMED_SHAPES = [8, 48, 64]
MODULE = "vllm.model_executor.layers.fused_moe.experts.trtllm_mxfp4_moe"
COLLECTIVE = "vllm.models.kimi_k3.nvidia.ops.cute_dsl.latent_moe_tail.allreduce_rmsnorm_reduce_scatter_early_exit"
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
ENV = "PERFBOT_R3_318_MOE_STAGE"
TP, MAXM_TAIL, SELF_RANK = 8, 128, 3
SENTINEL = -0x80000000


def results_path(name="kernel-gate.json") -> Path:
    root = Path(os.environ.get("PERFBOT_WORKSPACE", "/workspace"))
    out = root / os.environ.get("PERFBOT_RESULTS", "results/measure")
    out.mkdir(parents=True, exist_ok=True)
    return out / name


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def deq_fp4(packed: torch.Tensor) -> torch.Tensor:
    lut = torch.tensor(E2M1 + [-v for v in E2M1], dtype=torch.float64, device=packed.device)
    return torch.stack([lut[(packed & 0xF).long()], lut[(packed >> 4).long()]], -1).reshape(
        *packed.shape[:-1], packed.shape[-1] * 2)


def deq_scale(sf_u8: torch.Tensor) -> torch.Tensor:
    return torch.pow(2.0, sf_u8.double() - 127.0).repeat_interleave(32, -1)


def sf_offset(rows, kblocks, n_kblocks):
    kt = (n_kblocks + 3) // 4
    return ((rows // 128) * (512 * kt) + (kblocks // 4) * 512 + (rows % 32) * 16 + ((rows % 128) // 32) * 4
            + kblocks % 4)


def quant_mxfp8_ref(x: torch.Tensor):
    """ceil-ue8m0(amax/448), RNE e4m3 over 32-column blocks: (fp8 values, ue8m0 bytes)."""
    xf = x.float().reshape(x.shape[0], -1, 32)
    norm = xf.abs().amax(-1) / 448.0
    bits = norm.view(torch.int32)
    e8 = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).int()
    e8 = torch.where(norm <= 0, torch.zeros_like(e8), e8).clamp(0, 254)
    inv = torch.where(e8 == 0, torch.zeros_like(norm), torch.pow(2.0, 127.0 - e8.float()))
    return (xf * inv[..., None]).to(torch.float8_e4m3fn).reshape(x.shape), e8.to(torch.uint8)


def situ(g, u, a=ALPHA, b=BETA):
    return a * torch.tanh(g / a) * torch.sigmoid(g) * (b * torch.tanh(u / b))


def graph_timed(fn, reps=200):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    for _ in range(10):
        graph.replay()
    times = []
    for _ in range(5):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(reps):
            graph.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1000.0 / reps)
    del graph
    return statistics.median(times)


def prod_quant(x):
    from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input

    return moe_kernel_quantize_input(x, None, quant_dtype="mxfp8", per_act_token_quant=False, block_shape=None,
                                     is_scale_swizzled=False, mx_alignment=256)


class FakeTail:
    """The tail's routed Lamport buffer layout [3 generations, 128 tokens, 8 ranks, H] bf16 in plain memory."""

    def __init__(self, dev):
        self.buf = torch.full((3, MAXM_TAIL, TP, H // 2), SENTINEL, dtype=torch.int32, device=dev)
        bpb = MAXM_TAIL * TP * H * 2
        self.flags = torch.tensor([0, 2, bpb, 0, 0, 0, 0, 0, 0], dtype=torch.int32, device=dev)
        self.gen = 0

    def arm(self, gen):
        self.gen = gen
        self.buf.fill_(SENTINEL)
        self.flags[0] = gen

    def rows(self, m, rank=SELF_RANK):
        return self.buf[self.gen, :m, rank].view(torch.bfloat16)


class Bench:
    """Synthetic K3 MoE layer (one rank) driven through the overlay's real apply()."""

    def __init__(self, seed=7):
        from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
            Mxfp4MoeBackend, convert_weight_to_mxfp4_moe_kernel_format)

        self.mod = importlib.import_module(MODULE)
        dev = self.dev = torch.device("cuda", torch.cuda.current_device())
        g = self.g = torch.Generator(device=dev).manual_seed(seed)
        u8 = lambda *s: torch.randint(0, 256, s, dtype=torch.uint8, device=dev, generator=g)
        self.w13 = u8(E, 2 * I, H // 2)  # logical [gate(w1); up(w3)] as loaded
        self.w2 = u8(E, H, I // 2)
        self.s13 = torch.randint(118, 124, (E, 2 * I, H // 32), dtype=torch.uint8, device=dev, generator=g)
        self.s2 = torch.randint(118, 124, (E, H, I // 32), dtype=torch.uint8, device=dev, generator=g)
        out = convert_weight_to_mxfp4_moe_kernel_format(
            Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8, None, self.w13.clone(), self.w2.clone(),
            self.s13.clone().view(torch.float8_e4m3fn), self.s2.clone().view(torch.float8_e4m3fn), None, None,
            _cache_permute_indices={})
        self.p13, self.p2, self.ps13, self.ps2 = out[0], out[1], out[2], out[3]
        mod = self.mod
        s = object.__new__(mod.TrtLlmMxfp4ExpertsMonolithic)
        s.quant_config = SimpleNamespace(a1_scale=None, a1_gscale=None, use_nvfp4_w4a4=False, quant_dtype="mxfp8",
                                         per_act_token_quant=False, block_shape=None, is_scale_swizzled=False,
                                         mx_alignment=256, w1_scale=self.ps13, w2_scale=self.ps2, w1_bias=None,
                                         w2_bias=None)
        s.moe_config = SimpleNamespace(max_num_tokens=256, dp_size=1, defer_moe_finalize_max_num_tokens=-1,
                                       use_deferred_moe_finalize=True, should_defer_moe_finalize=lambda n: True)
        s.routing_method_type = mod.RoutingMethodType.DeepSeekV3
        s.topk, s.intermediate_size_per_partition = TOPK, I
        s.hidden_dim = s.hidden_dim_unpadded = H
        s.local_num_experts, s.ep_rank = E, 0
        s.gemm1_alpha = torch.full((E,), ALPHA, dtype=torch.float32, device=dev)
        s.gemm1_beta = torch.full((E,), BETA, dtype=torch.float32, device=dev)
        s.gemm1_clamp_limit = None
        s.routing_replay_capture_fn, s._routing_replay_buffer = None, None
        self.s = s
        self.bias = torch.zeros(E, dtype=torch.float32, device=dev)

    def inputs(self, m, edges=True):
        dev, g = self.dev, self.g
        ids = torch.stack([torch.randperm(E, device=dev, generator=g)[:TOPK] for _ in range(m)]).int()
        wts = torch.rand(m, TOPK, device=dev, generator=g) + 0.1
        wts = (wts / wts.sum(-1, keepdim=True)).to(torch.bfloat16)
        packed = (ids << 16) | wts.view(torch.int16).int() & 0xFFFF
        carrier = torch.zeros(m, TOPK + E, dtype=torch.float32, device=dev)
        carrier.view(-1)[: m * TOPK].view(torch.int32).copy_(packed.reshape(-1))
        x = (torch.randn(m, H, device=dev, generator=g) * 0.5).to(torch.bfloat16)
        if edges and m >= 4:  # scale-rounding edges of the MXFP8 quant
            x[1].zero_()
            x[2] = x[2] * 1e-4
            for b in range(H // 32):
                x[2, 32 * b] = 448.0 * 2.0 ** (b % 24 - 16)  # amax / 448 an exact power of two
            x[3] = x[3] * 1e-30
            x[0, :64] = -0.0
        xq, xs = prod_quant(x)
        return SimpleNamespace(m=m, ids=ids, wts=wts, packed=packed.int(), carrier=carrier, x=x, xq=xq, xs=xs)

    def _apply(self, x, xs, carrier, stage: bool):
        os.environ[ENV] = "1" if stage else "0"
        try:
            return self.s.apply(x, self.p13, self.p2, carrier, self.mod.MoEActivation.SITU, E, None, xs, False,
                                num_expert_group=1, e_score_correction_bias=self.bias, routed_scaling_factor=1.0,
                                topk_group=1)
        finally:
            os.environ[ENV] = "1"

    def control(self, inp):
        """What the image runs: prepare()'s MXFP8 quant, then flashinfer's routed call."""
        xq, xs = prod_quant(inp.x)
        return self._apply(xq, xs, inp.carrier, False)

    def candidate(self, inp):
        """What the overlay runs with the stage on: prepare() hands apply() the bf16 latent."""
        return self._apply(inp.x, None, inp.carrier, True)


def tail_finalize(o, m):
    """The image tail's top-K finalize of an UnfinalizedMoEOutput, bit for bit: bf16(v * w), bf16 running sum."""
    local = torch.zeros(m, H, dtype=torch.bfloat16, device=o.expert_weights.device)
    for k in range(TOPK):
        idx = o.expanded_idx_to_permuted_idx[:, k].long()
        v = o.gemm2_permuted[idx.clamp(min=0)].float()
        w = o.expert_weights[:, k].float()[:, None]
        scaled = (v * w).to(torch.bfloat16)
        upd = (local.float() + scaled.float()).to(torch.bfloat16)
        local = torch.where((idx >= 0)[:, None], upd, local)
    return local


def references(b: Bench, inp):
    """fp64 finalized stage outputs A / A_q [m, H] and the fp64 activation per (token, k) [m, 16, I]."""
    m = inp.m
    xd = inp.xq.double() * deq_scale(inp.xs.view(torch.uint8).reshape(-1, H // 32)[:m])
    ref = {k: torch.zeros(m, TOPK, H, dtype=torch.float64, device=b.dev) for k in ("A", "A_q")}
    act_all = torch.zeros(m, TOPK, I, dtype=torch.float64, device=b.dev)
    for e in inp.ids.unique().tolist():
        sel = (inp.ids == e).nonzero()
        t_idx, k_idx = sel[:, 0], sel[:, 1]
        wg = deq_fp4(b.w13[e, :I]) * deq_scale(b.s13[e, :I])
        wu = deq_fp4(b.w13[e, I:]) * deq_scale(b.s13[e, I:])
        wd = deq_fp4(b.w2[e]) * deq_scale(b.s2[e])
        xe = xd[t_idx]
        act = situ(xe @ wg.T, xe @ wu.T)
        act_all[t_idx, k_idx] = act
        ref["A"][t_idx, k_idx] = act @ wd.T
        q, qs = quant_mxfp8_ref(act.float())
        ref["A_q"][t_idx, k_idx] = (q.double() * deq_scale(qs)) @ wd.T
    w = inp.wts.double()[..., None]
    return {k: (r * w).sum(1) for k, r in ref.items()}, act_all


def err(got, ref):
    d = (got.double() - ref).abs()
    return {"max_abs": d.max().item(), "mean_abs": d.mean().item()}


def check_shape(b: Bench, tail: FakeTail, m: int, gen: int) -> dict:
    import perfbot_kernels.moe_stage as ms

    inp = b.inputs(m)
    ctl = b.control(inp)
    tail.arm(gen)
    cand = b.candidate(inp)
    torch.cuda.synchronize()
    rec = {"candidate_is_stage": bool(getattr(cand, "perfbot_prefilled", False)),
           "control_is_image": type(ctl).__name__ == "UnfinalizedMoEOutput"}
    ws = ms._WS[b.dev.index]
    ids = inp.ids.long()
    uniq = ids.unique()
    counts = torch.bincount(ids.reshape(-1), minlength=E)
    roff = torch.cumsum(counts, 0) - counts
    ntile = (counts + ms.TN - 1) // ms.TN
    toff = torch.cumsum(ntile, 0) - ntile
    want_tiles = []
    for e in uniq.tolist():
        c = int(counts[e])
        for q in range((c + ms.TN - 1) // ms.TN):
            want_tiles.append([e, min(ms.TN, c - q * ms.TN), int(roff[e]) + q * ms.TN, 0])
    n_tiles = int(ws["cnt"][0])
    rec["tiles_equal"] = ws["tmeta"][: 4 * n_tiles].view(n_tiles, 4).cpu().tolist() == want_tiles
    rank = torch.zeros_like(ids)
    seen = torch.zeros(E, dtype=torch.long, device=b.dev)
    for t in range(m):
        rank[t] = seen[ids[t]]
        seen[ids[t]] += 1
    want_perm = (roff[ids] + rank).int()
    rec["perm_equal_expected"] = bool(torch.equal(cand.expanded_idx_to_permuted_idx, want_perm))
    rec["counts_equal"] = bool(torch.equal(seen, counts))
    rec["weights_equal_packed"] = bool(torch.equal(cand.expert_weights.view(torch.int16), inp.wts.view(torch.int16)))
    rec["weights_equal_control"] = bool(torch.equal(cand.expert_weights.view(torch.int16),
                                                    ctl.expert_weights.view(torch.int16)))
    rows_tok = torch.arange(m, device=b.dev).repeat_interleave(TOPK)
    rec["row_records_equal"] = bool(torch.equal(ws["rtok"][want_perm.reshape(-1).long()], rows_tok.int()))
    # quant: the stage's per-slot MXFP8 rows and SF = the production quant of the same bf16 rows, bit for bit
    slot_row = (toff[ids] * ms.TN + rank).reshape(-1)
    xperm = ws["x_perm"].view(-1, H)[slot_row]
    rec["quant_rows_equal_production"] = bool(torch.equal(
        xperm, inp.xq.view(torch.uint8).reshape(-1, H)[:m].repeat_interleave(TOPK, 0)))
    kb = torch.arange(H // 32, device=b.dev)[None, :]
    sf_rows = ((slot_row // ms.TN) * 128 + slot_row % ms.TN)[:, None]
    sfx = ws["sfx_perm"][sf_offset(sf_rows, kb, H // 32)]
    rec["quant_sf_equal_production"] = bool(torch.equal(
        sfx, inp.xs.view(torch.uint8).reshape(-1, H // 32)[:m].repeat_interleave(TOPK, 0)))
    cp = ctl.expanded_idx_to_permuted_idx.reshape(-1).long()
    flat_e = ids.reshape(-1)
    lo = torch.full((E,), 1 << 40, dtype=torch.long, device=b.dev).scatter_reduce(0, flat_e, cp, "amin")
    hi = torch.full((E,), -1, dtype=torch.long, device=b.dev).scatter_reduce(0, flat_e, cp, "amax")
    lo_u, hi_u = lo[uniq], hi[uniq]
    rec["control_perm_bijective"] = cp.unique().numel() == cp.numel()
    rec["control_blocks_ordered_disjoint"] = bool((lo_u[1:] > hi_u[:-1]).all()) if uniq.numel() > 1 else True
    rec["control_counts_fit_blocks"] = bool(((hi_u - lo_u + 1) >= counts[uniq]).all())
    # protocol: own slots written, no sentinel, other ranks untouched, accumulator and counters reset
    words = tail.buf[gen, :m]
    rec["slots_written_no_sentinel"] = bool((words[:, SELF_RANK] != SENTINEL).all())
    others = torch.cat([words[:, :SELF_RANK], words[:, SELF_RANK + 1:]], 1)
    rec["other_slots_untouched"] = bool((others == SENTINEL).all()) and bool(
        (tail.buf[[g for g in range(3) if g != gen]] == SENTINEL).all())
    rec["acc_reset"] = bool((ws["acc"] == 0).all())
    # numerics
    ref, act = references(b, inp)
    got_x = tail.rows(m)
    got_c = tail_finalize(ctl, m)
    num = {"control": {k: err(got_c, r) for k, r in ref.items()},
           "candidate": {k: err(got_x, r) for k, r in ref.items()},
           "ref_max_abs": ref["A"].abs().max().item(), "candidate_vs_control": err(got_x, got_c.double())}
    inter = ws["inter"].view(-1, I)[slot_row]
    sf = ws["sf2"][sf_offset(sf_rows, torch.arange(I // 32, device=b.dev)[None, :], I // 32)]
    inter_d = inter.view(torch.float8_e4m3fn).double() * deq_scale(sf)
    num["intermediate_vs_fp64"] = err(inter_d, act.reshape(-1, I))
    q, qs = quant_mxfp8_ref(act.reshape(-1, I).float())
    num["intermediate_bytes_equal_ref_quant_fraction"] = (inter == q.view(torch.uint8)).double().mean().item()
    num["intermediate_sf_equal_ref_quant_fraction"] = (sf == qs).double().mean().item()
    rec["numerics"] = num
    rec["error_le_control"] = all(
        num["candidate"][k][s] <= num["control"][k][s] for k in ("A", "A_q") for s in ("max_abs", "mean_abs"))
    rec["finite"] = bool(torch.isfinite(got_x.float()).all())
    rec["ok"] = all(v for k, v in rec.items() if isinstance(v, bool))
    return rec


def timings(b: Bench) -> dict:
    import perfbot_kernels.moe_stage as ms

    rec = {"unit": "us per call", "method": "CUDA-graph replay, median of 5 x 200",
           "control": "MXFP8 quant + flashinfer routed call (the image tail's finalize is not in either side)",
           "candidate": "permute+quant, fc1, fc2 with the weighted finalize store", "per_shape": {}}
    for m in TIMED_SHAPES:
        inp = b.inputs(m, edges=False)
        b.candidate(inp)
        row = {"layer_control_us": graph_timed(lambda: b.control(inp)),
               "layer_candidate_us": graph_timed(lambda: b.candidate(inp))}
        row["layer_delta_us"] = row["layer_candidate_us"] - row["layer_control_us"]
        args = (inp.x, inp.packed, b.p13, b.ps13, b.p2, b.ps2, 0, ALPHA, BETA)
        ms.run(*args)
        torch.cuda.synchronize()
        for which in ("permute", "fc1"):
            row[f"{which}_us"] = graph_timed(lambda: ms.run(*args, only=which))
        row["quant_us"] = graph_timed(lambda: prod_quant(inp.x))
        rec["per_shape"][f"M{m}"] = row
        print(f"TIMING M={m} {json.dumps(row)}", flush=True)
    rec["kernels_M48"] = kernel_profile(b, 48)
    return rec


def kernel_profile(b: Bench, m: int) -> dict:
    from torch.profiler import ProfilerActivity, profile

    inp = b.inputs(m, edges=False)
    rec = {}
    for name, fn in (("control", b.control), ("candidate", b.candidate)):
        fn(inp)
        torch.cuda.synchronize()
        reps = 20
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(reps):
                fn(inp)
            torch.cuda.synchronize()
        per = {}
        for evt in prof.events():
            if not str(evt.device_type).endswith("CUDA"):
                continue
            row = per.setdefault(evt.name[:120], {"calls_per_layer": 0.0, "us_per_layer": 0.0})
            row["calls_per_layer"] += 1.0 / reps
            row["us_per_layer"] += evt.time_range.elapsed_us() / reps
        rec[name] = per
    return rec


def graph_probe(b: Bench, tail: FakeTail) -> dict:
    """Capture the candidate layer per shape, replay twice on fresh inputs, compare to an eager candidate call.
    fc2's cross-CTA fp32 red.add order varies run to run, so stage rows match within bf16 rounding of the sums
    (rtol 2^-6); the image path (M 128) must match bit for bit."""
    shapes, ok = {}, True
    for m in SHAPES:
        inp = b.inputs(m)
        tail.arm(1)
        b.candidate(inp)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            b.candidate(inp)
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = b.candidate(inp)
        staged = bool(getattr(out, "perfbot_prefilled", False))
        rec = {"stage_path": staged == (m <= 64)}
        for rep in range(2):
            new = b.inputs(m)
            inp.x.copy_(new.x)
            inp.carrier.copy_(new.carrier)
            tail.arm(1)
            graph.replay()
            torch.cuda.synchronize()
            if staged:
                got = tail.rows(m).float().clone()
                gw = out.expert_weights.clone()
                tail.arm(1)
                want = b.candidate(inp)
                torch.cuda.synchronize()
                ref = tail.rows(m).float()
                scale = ref.abs().max().item()
                rec[f"replay{rep}_vs_eager"] = bool(torch.allclose(got, ref, rtol=2 ** -6, atol=1e-3 * scale)) \
                    and bool(torch.equal(gw.view(torch.int16), want.expert_weights.view(torch.int16)))
                rec[f"replay{rep}_max_abs_diff"] = (got - ref).abs().max().item()
            else:
                got = tail_finalize(out, m).clone()
                want = b.candidate(inp)
                rec[f"replay{rep}_vs_eager"] = bool(torch.equal(got, tail_finalize(want, m)))
        rec["ok"] = all(v for v in rec.values() if isinstance(v, bool))
        ok &= rec["ok"]
        shapes[f"M{m}"] = rec
        print(f"GRAPH M={m} {json.dumps(rec)}", flush=True)
        del graph
    return {"ok": ok, "shapes": shapes}


# ------------------------------------------------------------------ 8-rank tail test

def _rmsnorm_ref(rows_bf16, gamma, eps):
    """The collective's routed role in bf16 mode: fp32 rank sum -> bf16 -> bf16 square promoted, fp32 mean."""
    acc = rows_bf16.float().sum(1)
    xb = acc.to(torch.bfloat16)
    sq = (xb * xb).float().sum(-1, keepdim=True)
    inv = torch.rsqrt(sq / rows_bf16.shape[-1] + eps)
    return (xb.float() * inv * gamma.float()).to(torch.bfloat16)


def _tail_rank(rank: int, port: int, out_dir: str) -> None:
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=TP,
                            device_id=dev)
    rec = {"rank": rank}
    try:
        import perfbot_kernels.moe_stage as ms

        coll_mod = importlib.import_module(COLLECTIVE)
        rec["collective_file"] = coll_mod.__file__
        group = dist.group.WORLD
        try:
            import torch.distributed._symmetric_memory as symm_mem

            symm_mem.enable_symm_mem_for_group(group.group_name)
        except Exception:  # noqa: BLE001
            pass
        eps = 1e-6
        coll = coll_mod.CollectiveKernel(group=group, rank=rank, tp_size=TP, latent_dim=H, hidden_dim=2 * H,
                                         max_m=MAXM_TAIL, max_token_ctas=32, rms_eps=eps, fp32_internal=False,
                                         top_k=TOPK)
        rec["prefilled_ready"] = bool(getattr(coll, "perfbot_prefilled_ready", False))
        ms.set_target(dev, coll._routed_multicast_ptr, coll._routed_flags, rank, MAXM_TAIL)
        b = Bench(seed=100 + rank)
        g = torch.Generator(device=dev).manual_seed(1)
        gamma = (torch.rand(H, device=dev, generator=g) + 0.5).to(torch.bfloat16)
        steps, ok = [], rec["prefilled_ready"]
        plan = [(48, True), (8, True), (48, False), (64, True), (1, True), (48, False), (48, True), (24, True)]
        for i, (m, stage) in enumerate(plan):
            inp = b.inputs(m, edges=False)
            shared = (torch.randn(m, 2 * H, device=dev, generator=b.g) * 0.1).to(torch.bfloat16)
            gen = int(coll._routed_flags[0])
            out = b.candidate(inp) if stage else b.control(inp)
            latent, _ = coll(out, shared, gamma)
            latent = latent.clone()
            torch.cuda.synchronize()
            ws = coll._routed_workspace.view(torch.bfloat16).view(-1)
            bpb = int(coll._routed_flags[2]) // 2
            rows = ws[gen * bpb: gen * bpb + m * TP * H].view(m, TP, H)
            want = _rmsnorm_ref(rows, gamma, eps)
            d = (latent.float() - want.float()).abs()
            row = {"m": m, "stage": stage, "gen": gen, "next_gen": int(coll._routed_flags[0]),
                   "prefilled": bool(getattr(out, "perfbot_prefilled", False)),
                   "latent_vs_rows_max_abs": d.max().item(), "latent_max_abs": want.float().abs().max().item()}
            mine = rows[:, rank]
            own_ok = True
            if stage:
                ref, _ = references(b, inp)
                ctl_rows = tail_finalize(b.control(inp), m)
                row["own_rows_vs_fp64_A"] = err(mine, ref["A"])
                row["control_finalize_vs_fp64_A"] = err(ctl_rows, ref["A"])
                row["own_rows_vs_control_finalize"] = err(mine, ctl_rows.double())
                own_ok = all(row["own_rows_vs_fp64_A"][k] <= row["control_finalize_vs_fp64_A"][k]
                             for k in ("max_abs", "mean_abs"))
            row["ok"] = (row["prefilled"] == stage and own_ok and row["latent_vs_rows_max_abs"] <= 2 ** -6 * max(
                1.0, row["latent_max_abs"]) and row["next_gen"] == (gen + 1) % 3)
            ok &= row["ok"]
            steps.append(row)
        rec["steps"] = steps
        # per-layer time, graph replay on all ranks together (the Lamport poll keeps them in step)
        timing = {}
        for m in (8, 48, 64):
            inp = b.inputs(m, edges=False)
            shared = (torch.randn(m, 2 * H, device=dev, generator=b.g) * 0.1).to(torch.bfloat16)
            for name, fn in (("control", b.control), ("candidate", b.candidate)):
                def layer(fn=fn):
                    return coll(fn(inp), shared, gamma)
                layer()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    layer()
                torch.cuda.current_stream().wait_stream(s)
                with torch.cuda.graph(graph):
                    layer()
                times = []
                for _ in range(5):
                    dist.barrier()
                    a, z = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    a.record()
                    for _ in range(100):
                        graph.replay()
                    z.record()
                    z.synchronize()
                    times.append(a.elapsed_time(z) * 10.0)
                timing[f"M{m}_{name}_us"] = statistics.median(times)
                del graph
            timing[f"M{m}_delta_us"] = timing[f"M{m}_candidate_us"] - timing[f"M{m}_control_us"]
        rec["layer_timing_us"] = timing
        rec["ok"] = bool(ok)
    except Exception:  # noqa: BLE001
        rec["error"] = traceback.format_exc()
        rec["ok"] = False
        traceback.print_exc()
    Path(out_dir, f"tail-rank{rank}.json").write_text(json.dumps(rec, indent=1, default=str))
    print(f"TAIL rank={rank} ok={rec['ok']}", flush=True)
    try:
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001
        pass


def tail_test() -> dict:
    import torch.multiprocessing as mp

    out_dir = results_path().parent
    port = 29511
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_tail_rank, args=(r, port, str(out_dir))) for r in range(TP)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=900)
    ranks = []
    for r in range(TP):
        f = out_dir / f"tail-rank{r}.json"
        ranks.append(json.loads(f.read_text()) if f.is_file() else {"rank": r, "ok": False, "error": "no record"})
    exitcodes = [p.exitcode for p in procs]
    return {"ok": all(r.get("ok") for r in ranks) and all(c == 0 for c in exitcodes), "exitcodes": exitcodes,
            "ranks": ranks}


def bench_fc2() -> dict:
    """fc2 cost attribution (dev bench): the production kernel against compile-time variants on the workspace of a
    full stage call. variant 0 = production, 1 = red.add without the grid arrival and finalize, 2 = neither."""
    import cutlass
    import cutlass.cute as cute
    from cuda.bindings.driver import CUstream
    from cutlass.cute.runtime import make_ptr

    import perfbot_kernels.moe_stage as ms
    from perfbot_kernels.moe_stage.kernels import A_DTYPE, B_DTYPE, SF_DTYPE, MoeGemm

    b = Bench()
    tail = FakeTail(b.dev)
    ms.set_target(b.dev, tail.buf.data_ptr(), tail.flags, SELF_RANK, MAXM_TAIL)
    gm = cute.AddressSpace.gmem
    P = lambda t, dt: make_ptr(dt, t.data_ptr(), gm, assumed_align=16)
    i32, u8, f32 = cutlass.Int32, cutlass.Uint8, cutlass.Float32
    sms = torch.cuda.get_device_properties(b.dev).multi_processor_count
    ws = ms._WS.get(b.dev.index)
    rec = {}
    for m in (8, 48, 64):
        inp = b.inputs(m, edges=False)
        b.candidate(inp)
        ws = ms._WS[b.dev.index]
        torch.cuda.synchronize()
        row = {}
        for variant in (0, 1, 2):
            args = [P(b.p2, A_DTYPE), P(b.ps2, SF_DTYPE), P(ws["inter"], B_DTYPE), P(ws["sf2"], SF_DTYPE),
                    P(ws["tmeta"], i32), P(ws["cnt"], i32), P(ws["acc"], f32), P(ws["dummy"], u8),
                    P(ws["rtok"], i32), P(ws["rw"], i32), P(ws["ctr"], i32), P(tail.flags, i32),
                    cutlass.Int64(tail.buf.data_ptr()), cutlass.Int32(SELF_RANK), cutlass.Int32(m),
                    CUstream(torch.cuda.current_stream().cuda_stream)]
            fn = cute.compile(MoeGemm(False, sms, ALPHA, BETA, variant=variant), *args)
            us = graph_timed(lambda: fn(*args[:-1], CUstream(torch.cuda.current_stream().cuda_stream)))
            ws["acc"].zero_()
            row[f"v{variant}_us"] = us
        row["n_tiles"] = int(ws["cnt"][0])
        rec[f"M{m}"] = row
        print(f"BENCH-FC2 M={m} {json.dumps(row)}", flush=True)
    return rec


def iket_run() -> int:
    os.environ["PERFBOT_MOE_STAGE_IKET"] = "1"
    import perfbot_kernels.moe_stage as ms

    b = Bench()
    tail = FakeTail(b.dev)
    ms.set_target(b.dev, tail.buf.data_ptr(), tail.flags, SELF_RANK, MAXM_TAIL)
    for m in (48, 48, 48, 8, 64):
        b.candidate(b.inputs(m, edges=False))
    torch.cuda.synchronize()
    print("PERFBOT-MOE-STAGE-IKET ok launches=15 (5 layer calls x permute, fc1, fc2)", flush=True)
    return 0


def identity(mod) -> dict:
    import perfbot_kernels.moe_stage as ms

    coll = importlib.import_module(COLLECTIVE)
    return {"module_file": mod.__file__, "module_sha256": sha256(Path(mod.__file__)),
            "stage_file": ms.__file__, "stage_sha256": ms.source_sha256(),
            "collective_file": coll.__file__, "collective_sha256": sha256(Path(coll.__file__)),
            "overlay_patch_present": hasattr(mod.TrtLlmMxfp4ExpertsMonolithic, "_perfbot_moe_stage")
            and coll.__file__.startswith(str(Path(ms.__file__).parents[2]))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--tail", action="store_true")
    ap.add_argument("--iket", action="store_true")
    ap.add_argument("--bench-fc2", action="store_true")
    args = ap.parse_args()
    if args.iket:
        return iket_run()
    if args.bench_fc2:
        out = bench_fc2()
        results_path("bench-fc2.json").write_text(json.dumps(out, indent=1))
        return 0
    path = results_path()
    record = json.loads(path.read_text()) if (args.graph or args.tail) and path.is_file() else {}
    record.setdefault("item", "r3-318")
    record["device"] = torch.cuda.get_device_name()
    ok = False
    try:
        if args.tail:
            record["tail"] = tail_test()
            ok = record["tail"]["ok"]
        else:
            import perfbot_kernels.moe_stage as ms

            mod = importlib.import_module(MODULE)
            record["identity"] = identity(mod)
            b = Bench()
            tail = FakeTail(b.dev)
            ms.set_target(b.dev, tail.buf.data_ptr(), tail.flags, SELF_RANK, MAXM_TAIL)
            if args.graph:
                record["graph"] = graph_probe(b, tail)
                ok = record["graph"]["ok"]
            else:
                checks = {}
                for i, m in enumerate(CHECK_SHAPES):
                    checks[f"M{m}"] = check_shape(b, tail, m, i % 3)
                    print(f"CHECK M={m} {json.dumps(checks[f'M{m}'])}", flush=True)
                record["checks"] = checks
                record["ok_eager"] = all(c["ok"] for c in checks.values()) and record["identity"][
                    "overlay_patch_present"]
                record["timings"] = timings(b)
                ok = record["ok_eager"]
    except Exception:
        record["error"] = traceback.format_exc()
        traceback.print_exc()
    # The record verdict combines the eager checks with every mode that has run so far, in any order.
    ok = bool(ok) and record.get("ok_eager", False) and all(record[k]["ok"] for k in ("graph", "tail") if k in record)
    record["ok"] = ok
    path.write_text(json.dumps(record, indent=1, default=str))
    print(f"PERFBOT-MOE-STAGE-GATE ok={ok} record={path}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
