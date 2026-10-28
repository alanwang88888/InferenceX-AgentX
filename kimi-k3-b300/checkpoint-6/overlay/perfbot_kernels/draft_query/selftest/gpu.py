# SPDX-License-Identifier: Apache-2.0
"""GPU self-test of DraftQuery on ONE GPU (the stand-in: 8 emulated DCP ranks in one launch; never more devices).

    python3 -B -m perfbot_kernels.draft_query.selftest.gpu --gate [--out DIR]   G0 anchors, G1-G3, mutants
    python3 -B -m perfbot_kernels.draft_query.selftest.gpu --gate --lamport      the same on the Lamport publish
    python3 -B -m perfbot_kernels.draft_query.selftest.gpu --time [--out DIR]   graph replay timing (b200_timed.sh)
    python3 -B -m perfbot_kernels.draft_query.selftest.gpu --iket               a few launches with IKET ranges

Reference = the served chain itself, per emulated rank, on this GPU: fused_q_kv_rmsnorm (Triton), F.linear (q_b),
torch.bmm (W_UK), fused_mla_decode_q_concat_kv_cache_insert (fp8 cache, RoPE); the served q-gather is the
concatenation of the 8 ranks' mqa_q into head blocks 8r..8r+7 (dcp_direct_q_gather.cu:44-50; it needs a multicast
object, which one GPU cannot create: probe5). Exit 1 on any failed check.
"""

import argparse
import json
import os
import sys
import time

import torch

from perfbot_kernels.draft_query import CAP, HEADS, WORLD, StandInBox
from perfbot_kernels.draft_query.selftest import judge

EPS = 1e-5  # drafter config.json rms_norm_eps
MAX_POS = 1048576  # original 32768 x yarn factor 32
BLOCKS, BS = 64, 64


# ------------------------------------------------------------------ hardware MUFU (oracle for the judge emulation)

_PROBE = {}


def mufu(values):
    """(MUFU.RSQ, MUFU.RCP) of fp32 cuda ``values``."""
    import cutlass.cute as cute
    from cutlass import Float32
    from cutlass.cute.runtime import from_dlpack

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.primitives import asm

    if "k" not in _PROBE:
        @asm(Float32, pure=True)
        def rsq(a):
            return "rsqrt.approx.ftz.f32 $0,$1;", "=f,f", (Float32(a),)

        @asm(Float32, pure=True)
        def rcp(a):
            return "rcp.approx.ftz.f32 $0,$1;", "=f,f", (Float32(a),)

        @cute.kernel
        def probe_kernel(src: cute.Tensor, a: cute.Tensor, b: cute.Tensor):
            bid, _, _ = cute.arch.block_idx()
            tid, _, _ = cute.arch.thread_idx()
            i = bid * 256 + tid
            if i < src.shape[0]:
                a[i] = rsq(src[i])
                b[i] = rcp(src[i])

        @cute.jit
        def probe(src: cute.Tensor, a: cute.Tensor, b: cute.Tensor):
            probe_kernel(src, a, b).launch(grid=((src.shape[0] + 255) // 256, 1, 1), block=(256, 1, 1))

        _PROBE["k"] = probe
    src = values.contiguous().float()
    a, b = torch.empty_like(src), torch.empty_like(src)
    views = [from_dlpack(t).mark_layout_dynamic() for t in (src, a, b)]
    if "c" not in _PROBE:
        _PROBE["c"] = cute.compile(_PROBE["k"], *views)
    _PROBE["c"](*views)
    return a, b


def hw(kind):
    def f(t):
        return mufu(t)[0]

    return f


# ------------------------------------------------------------------ SM clock + cluster placement probe (timing aid)

_PLACE = {}


def placement(device, spins=4000000):
    """One cluster-8 launch of DraftQuery's grid shape (8, 8, 1): per CTA (smid, MHz = clock64 cycles spun /
    globaltimer ns over a ``spins``-cycle spin)."""
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int64, Uint32
    from cutlass.cute.runtime import from_dlpack

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.primitives import asm

    if "k" not in _PLACE:
        @asm(Uint32)
        def smid():
            return "mov.u32 $0, %smid;", "=r", ()

        @asm(Int64)
        def clock():
            return "mov.u64 $0, %clock64;", "=l", ()

        @asm(Int64)
        def gtimer():
            return "mov.u64 $0, %globaltimer;", "=l", ()

        @cute.kernel
        def place_kernel(out: cute.Tensor, n: cutlass.Int32):
            x, y, _ = cute.arch.block_idx()
            tid, _, _ = cute.arch.thread_idx()
            if tid == 0:
                c0, t0 = clock(), gtimer()
                c1 = clock()
                while c1 - c0 < Int64(n):
                    c1 = clock()
                t1 = gtimer()
                b = y * 8 + x
                out[b, 0] = Int64(smid())
                out[b, 1] = c1 - c0
                out[b, 2] = t1 - t0

        @cute.jit
        def place(out: cute.Tensor, n: cutlass.Int32):
            place_kernel(out, n).launch(grid=(8, 8, 1), block=(256, 1, 1), cluster=(8, 1, 1))

        _PLACE["k"] = place
    out = torch.zeros(64, 3, dtype=torch.int64, device=device)
    view = from_dlpack(out)
    if "c" not in _PLACE:
        _PLACE["c"] = cute.compile(_PLACE["k"], view, cutlass.Int32(spins))
    _PLACE["c"](view, cutlass.Int32(spins))
    torch.cuda.synchronize()
    o = out.cpu()
    mhz = (o[:, 1].double() / o[:, 2].double() * 1e3)
    sms = sorted(int(v) for v in o[:, 0])
    return {"sms": sms, "mhz_min": round(float(mhz.min()), 1), "mhz_max": round(float(mhz.max()), 1)}


# ------------------------------------------------------------------ the served chain, one emulated rank at a time


def rope_table(device):
    from vllm.model_executor.layers.rotary_embedding import get_rope

    params = {"rope_type": "deepseek_yarn", "factor": 32.0, "original_max_position_embeddings": 32768,
              "rope_theta": 50000.0, "beta_fast": 32, "beta_slow": 1, "mscale": 1.0, "mscale_all_dim": 1.0}
    try:
        rot = get_rope(64, max_position=MAX_POS, rope_parameters=params, is_neox_style=False, dtype=torch.float32)
    except Exception:  # CustomOp needs a config context in some builds
        from vllm.config import VllmConfig, set_current_vllm_config

        with set_current_vllm_config(VllmConfig()):
            rot = get_rope(64, max_position=MAX_POS, rope_parameters=params, is_neox_style=False,
                           dtype=torch.float32)
    table = rot.cos_sin_cache.to(device=device, dtype=torch.float32).contiguous()
    assert table.shape == (MAX_POS, 64), table.shape
    return table


class Served:
    """The served draft query chain of one rank (mla.py:509-528 + 662-707), PIECEWISE eager on this GPU."""

    def __init__(self, weights, cos_sin):
        from vllm.models.common.ops.fused_qk_rmsnorm import fused_q_kv_rmsnorm
        from vllm.models.kimi_k3.nvidia.ops.fused_mla_key_concat_kv_cache import (
            fused_mla_decode_q_concat_kv_cache_insert,
        )

        self.norm, self.concat = fused_q_kv_rmsnorm, fused_mla_decode_q_concat_kv_cache_insert
        self.gq, self.gkv, self.w_qb, self.w_uk = weights
        self.cos_sin = cos_sin

    def rank(self, e, qkv, positions, cache_e, slots_e, qsi, ksi):
        """(mqa_q [M, 8, 576] fp8 bytes, q_c normed [M, 1536] bf16); writes cache_e [blocks, bs, 576] uint8."""
        q_c, kv_c, k_pe = qkv.split([1536, 512, 64], dim=-1)
        q_c, kv_n = self.norm(q_c, kv_c, self.gq, self.gkv, EPS)
        q = torch.nn.functional.linear(q_c, self.w_qb[e * 1536 : (e + 1) * 1536]).view(-1, HEADS, 192)
        q_nope, q_pe = q.split([128, 64], dim=-1)
        ql = torch.bmm(q_nope.transpose(0, 1), self.w_uk[e * HEADS : (e + 1) * HEADS]).transpose(0, 1)
        mqa = self.concat(ql, q_pe, kv_n, k_pe, cache_e.view(torch.float8_e4m3fn), slots_e, q_scale_inv=qsi,
                          cache_scale_inv=ksi, positions=positions, cos_sin_cache=self.cos_sin)
        return mqa.view(torch.uint8), q_c

    def run(self, qkv, positions, cache, slots, qsi, ksi):
        outs, qcs = [], []
        for e in range(cache.shape[0]):
            mqa, q_c = self.rank(e, qkv, positions, cache[e], slots[e], qsi, ksi)
            outs.append(mqa)
            qcs.append(q_c)
        return torch.cat(outs, 1), qcs[0]  # gathered [M, 64, 576]


# ------------------------------------------------------------------ salts


def make_weights(device, seed=1):
    g = torch.Generator().manual_seed(seed)
    gq = (1 + 0.1 * torch.randn(1536, generator=g)).to(torch.bfloat16)
    gkv = (1 + 0.1 * torch.randn(512, generator=g)).to(torch.bfloat16)
    w_qb = (torch.randn(WORLD * 1536, 1536, generator=g) / 39.2).to(torch.bfloat16)
    w_uk = (torch.randn(WORLD * HEADS, 128, 512, generator=g) / 11.3).to(torch.bfloat16)
    return tuple(t.to(device).contiguous() for t in (gq, gkv, w_qb, w_uk))


def salt(m, seed, kind="randn", device="cuda"):
    """(qkv [M, 2112] bf16, positions [M] int64, slots [8, M] int64) of one pre-registered salt."""
    g = torch.Generator().manual_seed(seed)
    qkv = torch.randn(m, 2112, generator=g)
    pos = torch.randint(0, 4096, (m,), generator=g)
    if kind == "pos0":
        pos[:] = 0
    elif kind == "posmax":
        pos = MAX_POS - 1 - torch.randint(0, 64, (m,), generator=g)
    elif kind == "large":  # row 0: squares overflow fp32 (rstd 0); row 1: 2^40 scale; row 2: tiny (eps dominates)
        qkv[0] = qkv[0] * 2.0 ** 66
        if m > 1:
            qkv[1] = qkv[1] * 2.0 ** 40
        if m > 2:
            qkv[2] = qkv[2] * 2.0 ** -70
        if m > 3:
            qkv[3, 1536:] = 0.0
    # DCP interleave 1 over 8 ranks: token t's slot lives on rank (t + seed) % 8; -1 elsewhere; distinct slots
    perm = torch.randperm(BLOCKS * BS, generator=g)[:m]
    slots = torch.full((WORLD, m), -1, dtype=torch.int64)
    for t in range(m):
        slots[(t + seed) % WORLD, t] = perm[t]
    to = dict(device=device)
    return qkv.to(torch.bfloat16).to(**to), pos.to(**to), slots.to(**to)


SALTS = [(1, "randn"), (7, "randn"), (14, "randn"), (42, "randn"), (56, "randn"), (14, "pos0"), (14, "posmax"),
         (42, "posmax"), (14, "large"), (56, "large"), (64, "randn")]


# ------------------------------------------------------------------ gate


class Gate:
    def __init__(self, device):
        self.device = device
        self.checks = []
        self.rcp = {s: mufu(torch.tensor([float(s)], device=device))[1].cpu() for s in (1536, 512)}
        self.table = rope_table(device)
        self.weights = make_weights(device)
        self.served = Served(self.weights, self.table)
        self.qsi = torch.tensor([1.0], device=device)
        self.ksi = torch.tensor([1.0], device=device)
        self.stats = []

    def add(self, items):
        for c in items:
            self.checks.append(c)
            if not c["ok"]:
                print("FAIL", json.dumps(c), flush=True)

    def fresh_cache(self, seed):
        g = torch.Generator().manual_seed(1000 + seed)
        return torch.randint(0, 120, (WORLD, BLOCKS, BS, 576), generator=g, dtype=torch.uint8).to(self.device)

    def judge_call(self, tag, qkv, pos, slots, cache0, cache_mine, gathered_mine, scales=None):
        """G1 (cache bytes vs served + vs emulation), G2 (envelope + code distance) for one call."""
        qsi, ksi = scales or (self.qsi, self.ksi)
        cache_served = cache0.clone()
        mqa, q_c = self.served.run(qkv, pos, cache_served, slots, qsi, ksi)
        torch.cuda.synchronize()
        checks = [judge.compare_bytes(f"{tag}/G1/cache_vs_served", cache_mine, cache_served)]
        # G0: the judge's emulation equals the served binary (norm + RoPE + quant arithmetic)
        rows = judge.emulate_cache_rows(qkv, self.weights[1], self.table, pos, ksi, EPS,
                                        rsqrt=lambda t: mufu(t)[0], rcp=self.rcp[512].to(self.device))
        want = torch.stack([judge.insert(cache0[e], rows, slots[e]) for e in range(WORLD)])
        checks.append(judge.compare_bytes(f"{tag}/G0/emulation_vs_served", want, cache_served))
        qn_emul = judge.emulate_norm(qkv[:, :1536], self.weights[0], EPS, rsqrt=lambda t: mufu(t)[0],
                                     rcp=self.rcp[1536].to(self.device))
        checks.append(judge.compare_bytes(f"{tag}/G0/qnorm_emulation_vs_served", qn_emul.view(torch.uint8),
                                          q_c.view(torch.uint8)))
        # G2 per emulated rank: envelope of mine and of the served chain; code distance mine vs served
        m = qkv.shape[0]
        ndiff, dmax = 0, 0
        for e in range(WORLD):
            ref, err = judge.query_reference(q_c, self.weights[2][e * 1536 : (e + 1) * 1536],
                                             self.weights[3][e * HEADS : (e + 1) * HEADS], self.table[pos], qsi)
            got = gathered_mine[:, e * HEADS : (e + 1) * HEADS]
            served_e = mqa[:, e * HEADS : (e + 1) * HEADS]
            checks.append(judge.envelope(f"{tag}/G2/mine/rank{e}", got, ref, err))
            checks.append(judge.envelope(f"{tag}/G2/served/rank{e}", served_e, ref, err))
            n, d = judge.code_distance(got, served_e)
            ndiff, dmax = ndiff + n, max(dmax, d)
        self.stats.append({"tag": tag, "M": m, "bytes": m * 64 * 576, "bytes_diff_vs_served": ndiff,
                           "max_code_distance": dmax})
        return checks


def protocol(box, tag, calls):
    """Epoch / parity state after ``calls`` calls on a fresh box: epoch[e] == calls, completion 0, and on every
    replica signal[E & 1][s] == E (E = calls) and signal[(E - 1) & 1][s] == E - 1."""
    out = []
    E = calls
    ep = box.epoch.cpu().tolist()
    out.append({"name": f"{tag}/G3/epoch", "ok": ep == [E] * len(ep), "mismatches": sum(x != E for x in ep),
                "first": ep})
    co = box.completion.cpu().tolist()
    out.append({"name": f"{tag}/G3/completion", "ok": co == [0] * len(co), "mismatches": sum(x != 0 for x in co),
                "first": co})
    sig = box.signal.cpu()
    want = torch.zeros_like(sig)
    want[:, E & 1, :] = E
    if E >= 2:
        want[:, (E - 1) & 1, :] = E - 1
    bad = int((sig != want).sum())
    out.append({"name": f"{tag}/G3/signal_parity", "ok": bad == 0, "mismatches": bad,
                "first": sig[0].tolist() if bad else None})
    return out


def lamport_protocol(box, tag, parity, m, other_armed=True):
    """Lamport mailbox state after a call on ``parity`` (no reader ran): rows < m of every replica hold no 0xFF byte
    (every byte published; E4M3 0xFF is never produced), rows >= m are still the sentinel, and the other parity is
    still armed (``other_armed``)."""
    q = box.mailbox[:, :, : box.cap * WORLD * HEADS * 576].view(box.copies, 2, box.cap, WORLD * HEADS * 576)
    pub = q[:, parity, :m]
    out = [{"name": f"{tag}/G3/lamport_rows_published", "ok": bool((pub != 0xFF).all()),
            "mismatches": int((pub == 0xFF).sum()), "first": None},
           {"name": f"{tag}/G3/lamport_rows_beyond_m_armed", "ok": bool((q[:, parity, m:] == 0xFF).all()),
            "mismatches": int((q[:, parity, m:] != 0xFF).sum()), "first": None}]
    if other_armed:
        o = q[:, 1 - parity]
        out.append({"name": f"{tag}/G3/lamport_other_parity_armed", "ok": bool((o == 0xFF).all()),
                    "mismatches": int((o != 0xFF).sum()), "first": None})
    return out


def lamport_box(dev, ranks=WORLD, copies=WORLD, waiters=None):
    mailbox = torch.full((copies, 2, CAP * WORLD * HEADS * 576), 0xFF, dtype=torch.uint8, device=dev)
    return StandInBox(dev, ranks=ranks, copies=copies, waiters=waiters, mailbox=mailbox)


WAITED_MS = -1.0  # as lse_combine: ~0 (+-few us, cross-stream event skew) if it waited, ~-20 if not


def late_writer(G, box, m, seed, mutant="", parity=0, sleep_cycles=40_000_000):
    """Lamport poll check on one rank (rank 0 publishes head block 0 of replica 0 and polls all 8 blocks): the other
    ranks' blocks 1..7 land from a second stream after ~20 ms. Returns (ms from the late copy's end to the kernel's
    end: > WAITED_MS when the kernel waited, gathered bytes == the copied bytes)."""
    qkv, pos, slots = salt(m, seed, "randn")
    cache = torch.zeros(1, BLOCKS, BS, 576, dtype=torch.uint8, device=G.device)
    s1 = slots[:1].clone()
    late = torch.randint(0, 0x7F, (m, (WORLD - 1) * HEADS, 576), dtype=torch.uint8, device=G.device)
    view = box.mailbox[0, parity, : m * WORLD * HEADS * 576].view(m, WORLD * HEADS, 576)
    main, side = torch.cuda.current_stream(), torch.cuda.Stream()
    ev_c, ev_k = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    view[:, HEADS:].copy_(late)  # warm call (compile) with blocks 1..7 already landed, then re-arm (the reader's job)
    run_mine(box, G, qkv, pos, cache, s1, mutant=mutant, parity=parity)
    torch.cuda.synchronize()
    box.mailbox[:, parity].fill_(0xFF)
    torch.cuda.synchronize()
    side.wait_stream(main)
    with torch.cuda.stream(side):
        torch.cuda._sleep(sleep_cycles)
        view[:, HEADS:].copy_(late)
        ev_c.record(side)
    run_mine(box, G, qkv, pos, cache, s1, mutant=mutant, parity=parity)
    ev_k.record(main)
    torch.cuda.synchronize()
    return ev_c.elapsed_time(ev_k), bool((view[:, HEADS:] == late).all())


def run_mine(box, G, qkv, pos, cache, slots, mutant="", parity=0):
    box.run(qkv, G.weights, G.table, pos, cache, slots, G.qsi, G.ksi, EPS, mutant=mutant, parity=parity)


def gate(out_dir, lamport=False):
    """Epoch form (default) or the Lamport publish (``lamport``): the same salts, G1 / G2 judge and mutants; the
    protocol checks follow the form (epoch / completion / signal, or the mailbox state + the late-writer poll)."""
    dev = torch.device("cuda")
    G = Gate(dev)
    sweep = torch.exp2(torch.empty(1 << 22, device=dev).uniform_(-20, 40))
    info = {"mufu_rcp_1536": float(G.rcp[1536]), "mufu_rcp_512": float(G.rcp[512]),
            "torch_rsqrt_vs_mufu_mismatch_of_4M": int((torch.rsqrt(sweep) != mufu(sweep)[0]).sum()),
            "sm_count": torch.cuda.get_device_properties(dev).multi_processor_count, "lamport": lamport}
    print("G0a", json.dumps(info), flush=True)
    new_box = (lambda: lamport_box(dev)) if lamport else (lambda: StandInBox(dev))

    def proto(box, tag, calls, m, parity=0, other_armed=True):
        if lamport:
            return lamport_protocol(box, tag, parity, m, other_armed)
        return protocol(box, tag, calls)

    # G1 + G2 + G3 salts: one fresh box per salt (epoch 1 after the call; Lamport parity 0)
    for i, (m, kind) in enumerate(SALTS):
        qkv, pos, slots = salt(m, i, kind)
        cache0 = G.fresh_cache(i)
        box = new_box()
        cache = cache0.clone()
        run_mine(box, G, qkv, pos, cache, slots)
        torch.cuda.synchronize()
        tag = f"M{m}/{kind}"
        G.add(G.judge_call(tag, qkv, pos, slots, cache0, cache, box.gathered(0, m)))
        for c in range(1, WORLD):  # every rank's replica holds the same gathered query
            G.add([judge.compare_bytes(f"{tag}/G2/replica{c}==0", box.gathered(c, m), box.gathered(0, m))])
        G.add(proto(box, tag, 1, m))
        print("SALT", tag, json.dumps(G.stats[-1]), flush=True)

    # G3 scales != 1 (q 1 / 0.75, k 1 / 1.5)
    qkv, pos, slots = salt(14, 31, "randn")
    sc = (torch.tensor([1 / 0.75], device=dev), torch.tensor([1 / 1.5], device=dev))
    cache0 = G.fresh_cache(31)
    box, cache = new_box(), cache0.clone()
    box.run(qkv, G.weights, G.table, pos, cache, slots, sc[0], sc[1], EPS)
    torch.cuda.synchronize()
    G.add(G.judge_call("M14/scales", qkv, pos, slots, cache0, cache, box.gathered(0, 14), scales=sc))

    # G3 epoch / parity: 3 consecutive calls, different data, no sync between (Lamport: parities 0, 1, 0; the
    # harness re-arms parity 0 before the third call, as the reader would)
    box = new_box()
    res = []
    for k in range(3):
        qkv, pos, slots = salt(14, 40 + k, "randn")
        cache0 = G.fresh_cache(40 + k)
        cache = cache0.clone()
        if lamport and k == 2:
            box.mailbox[:, 0].fill_(0xFF)
        run_mine(box, G, qkv, pos, cache, slots, parity=k % 2)
        res.append((qkv, pos, slots, cache0, cache, box.gathered(0, 14, k % 2).clone()))
    torch.cuda.synchronize()
    G.add(proto(box, "consecutive/after3", 3, 14, 0, other_armed=False))
    for k, (qkv, pos, slots, cache0, cache, gq) in enumerate(res[-1:]):
        G.add(G.judge_call("consecutive/call2", qkv, pos, slots, cache0, cache, gq))

    # G3 graph capture + replay with swapped inputs (Lamport: parity 0, re-armed before each replay)
    for m in (14, 42):
        box = new_box()
        qkv, pos, slots = salt(m, 60, "randn")
        sq, sp, ss = qkv.clone(), pos.clone(), slots.clone()
        cache = G.fresh_cache(60)
        run_mine(box, G, sq, sp, cache, ss)  # compile + warm outside capture (epoch 1)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run_mine(box, G, sq, sp, cache, ss)
        for rep in range(2):
            nq, np_, ns = salt(m, 70 + rep, "posmax" if rep else "randn")
            sq.copy_(nq), sp.copy_(np_), ss.copy_(ns)
            cache0 = cache.clone()
            if lamport:
                box.mailbox.fill_(0xFF)
            graph.replay()
            torch.cuda.synchronize()
            G.add(G.judge_call(f"graph/M{m}/rep{rep}", sq, sp, ss, cache0, cache, box.gathered(0, m)))
            G.add(proto(box, f"graph/M{m}/rep{rep}", 2 + rep, m))
        del graph

    # Lamport poll (one rank, the real per-rank grid): the kernel must end after the late blocks land
    late = {}
    if lamport:
        for m in (14, 42):
            box = lamport_box(dev, ranks=1, copies=1, waiters=WORLD)
            ms, ok = late_writer(G, box, m, 80 + m)
            late[f"M{m}"] = {"kernel_end_minus_late_copy_ms": round(ms, 3), "bytes_ok": ok}
            G.add([{"name": f"late/M{m}/G3/poll_waits", "ok": ms > WAITED_MS and ok, "mismatches": int(not (ms > WAITED_MS and ok)),
                    "first": late[f"M{m}"]}])
            print("LATE", m, json.dumps(late[f"M{m}"]), flush=True)

    passed = judge.verdict(G.checks)

    # mutants: each must fail the judge
    mutants = {}
    for mutant in ("ropehalf", "nopoll" if lamport else "noepoch", "slot"):
        if mutant == "nopoll":
            box = lamport_box(dev, ranks=1, copies=1, waiters=WORLD)
            ms, ok = late_writer(G, box, 14, 94, mutant="nopoll")
            checks = [{"name": "mutant/nopoll/G3/poll_waits", "ok": ms > WAITED_MS and ok, "mismatches": 1, "first": ms}]
            late["mutant_nopoll"] = {"kernel_end_minus_late_copy_ms": round(ms, 3), "bytes_ok": ok}
        else:
            qkv, pos, slots = salt(14, 90, "randn")
            cache0 = G.fresh_cache(90)
            box, cache = new_box(), cache0.clone()
            run_mine(box, G, qkv, pos, cache, slots, mutant=mutant)
            torch.cuda.synchronize()
            checks = G.judge_call(f"mutant/{mutant}", qkv, pos, slots, cache0, cache, box.gathered(0, 14))
            checks += proto(box, f"mutant/{mutant}", 1, 14)
        failed = [c["name"] for c in checks if not c["ok"]]
        mutants[mutant] = {"caught": bool(failed), "failed_checks": failed[:6]}
        print("MUTANT", mutant, json.dumps(mutants[mutant]), flush=True)
    caught = all(v["caught"] for v in mutants.values())

    summary = {"gate": "G1 cache bit-equal, G2 fp64 envelope, G3 salts / graph / " + ("Lamport mailbox + late "
               "writer" if lamport else "epoch"), "passed": passed,
               "mutants_caught": caught, "checks": len(G.checks),
               "failed": [c for c in G.checks if not c["ok"]][:20], "G0a": info, "mutants": mutants,
               "salts": SALTS, "late": late, "G2_vs_served": G.stats}
    print("SUMMARY", json.dumps({k: summary[k] for k in ("passed", "mutants_caught", "checks")}), flush=True)
    if out_dir:
        with open(os.path.join(out_dir, "gate-lamport.json" if lamport else "gate.json"), "w") as f:
            json.dump(summary, f, indent=1)
    return 0 if passed and caught else 1


# ------------------------------------------------------------------ timing


def graph_us(fn, calls, replays=50, blocks=2):
    """us per call over ``replays`` replays of a graph of ``calls`` calls; one value per block (A/A band)."""
    for _ in range(2):
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
        out.append(round(a.elapsed_time(b) * 1e3 / replays / calls, 3))
    return out


def timing(out_dir):
    dev = torch.device("cuda")
    G = Gate.__new__(Gate)
    G.device, G.table, G.weights = dev, rope_table(dev), make_weights(dev)
    G.qsi = G.ksi = torch.tensor([1.0], device=dev)
    served = Served(G.weights, G.table)
    calls, rows = 20, []
    for m in (14, 42):
        qkv, pos, slots = salt(m, 7, "randn")
        cache = torch.zeros(1, BLOCKS, BS, 576, dtype=torch.uint8, device=dev)
        s1 = slots[:1].clone()
        s1[0] = torch.arange(m, device=dev)  # every row inserts (one rank's worst case)
        for copies in (1, 8):
            box = StandInBox(dev, ranks=1, copies=copies, waiters=1)

            def mine():
                for _ in range(calls):
                    box.run(qkv, G.weights, G.table, pos, cache, s1, G.qsi, G.ksi, EPS)

            what = f"DraftQuery one rank (64 CTAs), publish to {copies} local replica(s), waits on its own signal"
            rows.append({"what": what, "M": m, "us_per_call": graph_us(mine, calls)})
            print("TIME", json.dumps(rows[-1]), flush=True)

        def theirs():
            for _ in range(calls):
                served.rank(0, qkv, pos, cache[0], s1[0], G.qsi, G.ksi)

        rows.append({"what": "served chain one rank: Triton norms + q_b linear + W_UK bmm + QConcatKVCache "
                             "(4 launches; epoch + q_gather need multicast)", "M": m,
                     "us_per_call": graph_us(theirs, calls)})
        print("TIME", json.dumps(rows[-1]), flush=True)
    if out_dir:
        with open(os.path.join(out_dir, "timing.json"), "w") as f:
            json.dump({"device": torch.cuda.get_device_name(dev), "calls_per_graph": calls, "replays": 50,
                       "rows": rows, "time": time.strftime("%FT%TZ", time.gmtime())}, f, indent=1)
    return 0


def ablate(out_dir):
    """Timing-only cuts of DraftQuery (wrong bytes; never a gate): what each phase costs un-instrumented."""
    from perfbot_kernels.draft_query.kernel import ABLATIONS

    dev = torch.device("cuda")
    table, weights = rope_table(dev), make_weights(dev)
    one = torch.tensor([1.0], device=dev)
    served = Served(weights, table)
    calls, rows = 20, []
    rows.append({"what": "placement+clock before", **placement(dev)})
    print("PLACE", json.dumps(rows[-1]), flush=True)
    for m in (14, 42):
        qkv, pos, slots = salt(m, 7, "randn")
        cache = torch.zeros(1, BLOCKS, BS, 576, dtype=torch.uint8, device=dev)
        s1 = slots[:1].clone()
        s1[0] = torch.arange(m, device=dev)
        for variant in ("",) + ABLATIONS:
            box = StandInBox(dev, ranks=1, copies=1, waiters=1)

            def mine():
                for _ in range(calls):
                    box.run(qkv, weights, table, pos, cache, s1, one, one, EPS, mutant=variant)

            rows.append({"what": variant or "full", "M": m, "us_per_call": graph_us(mine, calls)})
            print("ABL", json.dumps(rows[-1]), flush=True)

        def theirs():
            for _ in range(calls):
                served.rank(0, qkv, pos, cache[0], s1[0], one, one)

        rows.append({"what": "served4", "M": m, "us_per_call": graph_us(theirs, calls)})
        print("ABL", json.dumps(rows[-1]), flush=True)
    rows.append({"what": "placement+clock after", **placement(dev)})
    print("PLACE", json.dumps(rows[-1]), flush=True)
    if out_dir:
        with open(os.path.join(out_dir, "ablate.json"), "w") as f:
            json.dump({"device": torch.cuda.get_device_name(dev), "rows": rows}, f, indent=1)
    return 0


def bimodal(out_dir):
    """One process's M42 read for the bimodality question: SM clock + placement, then full (release chain) and
    x_v1fence (v1's per-thread fence.sys) A/A (2 blocks each)."""
    dev = torch.device("cuda")
    table, weights = rope_table(dev), make_weights(dev)
    one = torch.tensor([1.0], device=dev)
    row = {"pid": os.getpid(), "before": placement(dev)}
    qkv, pos, slots = salt(42, 7, "randn")
    cache = torch.zeros(1, BLOCKS, BS, 576, dtype=torch.uint8, device=dev)
    s1 = slots[:1].clone()
    s1[0] = torch.arange(42, device=dev)
    for variant in ("", "x_v1fence"):
        box = StandInBox(dev, ranks=1, copies=1, waiters=1)

        def mine():
            for _ in range(20):
                box.run(qkv, weights, table, pos, cache, s1, one, one, EPS, mutant=variant)

        row[variant or "full"] = graph_us(mine, 20)
    row["after"] = placement(dev)
    print("BIMODAL", json.dumps(row), flush=True)
    return 0


def iket_run():
    """IKET ranges in the timing setting: a graph of 20 back-to-back calls (PDL overlap as in --time), M 14 and 42."""
    dev = torch.device("cuda")
    table, weights = rope_table(dev), make_weights(dev)
    one = torch.tensor([1.0], device=dev)
    for m in (14, 42):
        qkv, pos, slots = salt(m, 3, "randn")
        cache = torch.zeros(1, BLOCKS, BS, 576, dtype=torch.uint8, device=dev)
        s1 = slots[:1].clone()
        s1[0] = torch.arange(m, device=dev)
        box = StandInBox(dev, ranks=1, copies=1, waiters=1)

        def mine():
            for _ in range(20):
                box.run(qkv, weights, table, pos, cache, s1, one, one, EPS, iket=True)

        graph_us(mine, 20, replays=1, blocks=1)
    torch.cuda.synchronize()
    print("PERFBOT-DRAFT-QUERY-IKET ok", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--lamport", action="store_true")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--iket", action="store_true")
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--bimodal", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.out:
        os.makedirs(a.out, exist_ok=True)
    if a.iket:
        return iket_run()
    if a.ablate:
        return ablate(a.out)
    if a.bimodal:
        return bimodal(a.out)
    rc = 0
    if a.gate:
        rc |= gate(a.out, a.lamport)
    if a.time:
        rc |= timing(a.out)
    return rc


if __name__ == "__main__":
    sys.exit(main())
