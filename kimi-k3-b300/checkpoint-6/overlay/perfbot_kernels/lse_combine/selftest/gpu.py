# SPDX-License-Identifier: Apache-2.0
"""GPU self-test of DraftValue on ONE GPU (the stand-in: 8 emulated DCP ranks in one launch; never more devices).

    python3 -B -m perfbot_kernels.lse_combine.selftest.gpu --gate [--out DIR]   G0-G3, late writers, mutants
    python3 -B -m perfbot_kernels.lse_combine.selftest.gpu --time [--out DIR]   graph replay timing (b200_timed.sh)
    python3 -B -m perfbot_kernels.lse_combine.selftest.gpu --iket               a few launches with IKET ranges

Reference = the served op itself on this GPU: torch.ops._C.direct_dcp_a2a_lse_reduce (increment_epoch, dispatch,
signal, wait_lse_combine) run as rank 0 with peer tables shifted so destination d lands in source slot d of one local
workspace; owner o's inputs are arranged so that slot d holds source rank d's head block o (empty shards: LSE -inf,
which the served dispatch also sends). W_UV = torch.bmm as mla.py:475-481. Exit 1 on any failed check.
"""

import argparse
import json
import os
import sys
import time

import torch

from perfbot_kernels.lse_combine import CAP, HEADS, LATENT, WORLD, StandInBox
from perfbot_kernels.lse_combine.selftest import judge

Q = 7  # draft rows per request (1 + 6 drafted tokens)
SLEEP = 40_000_000  # ~20 ms of GPU spin on the late-writer stream
WAITED_MS = -1.0  # reader end - late copy end: ~0 (+-few us, cross-stream event skew) if it waited, ~-20 if not

# ------------------------------------------------------------------ hardware EX2 (oracle for the judge emulation)

_PROBE = {}


def ex2_hw(values):
    """ex2.approx.f32 (non-ftz, as exp2f without fast math) of fp32 cuda ``values``."""
    import cutlass.cute as cute
    from cutlass import Float32
    from cutlass.cute.runtime import from_dlpack

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.primitives import asm

    if "k" not in _PROBE:
        @asm(Float32, pure=True)
        def ex2(a):
            return "ex2.approx.f32 $0,$1;", "=f,f", (Float32(a),)

        @cute.kernel
        def probe_kernel(src: cute.Tensor, dst: cute.Tensor):
            bid, _, _ = cute.arch.block_idx()
            tid, _, _ = cute.arch.thread_idx()
            i = bid * 256 + tid
            if i < src.shape[0]:
                dst[i] = ex2(src[i])

        @cute.jit
        def probe(src: cute.Tensor, dst: cute.Tensor):
            probe_kernel(src, dst).launch(grid=((src.shape[0] + 255) // 256, 1, 1), block=(256, 1, 1))

        _PROBE["k"] = probe
    shape = values.shape
    src = values.contiguous().float().view(-1)
    if src.numel() == 0:
        return values.float()
    dst = torch.empty_like(src)
    views = [from_dlpack(t).mark_layout_dynamic() for t in (src, dst)]
    if "c" not in _PROBE:
        _PROBE["c"] = cute.compile(_PROBE["k"], *views)
    _PROBE["c"](*views)
    return dst.view(shape)


# ------------------------------------------------------------------ the served op, one owner at a time


class Served:
    """The served dcp_direct_a2a_lse_reduce as rank 0 on one local workspace (peer tables shifted per source)."""

    def __init__(self, device, cap=CAP):
        import vllm._custom_ops  # noqa: F401  (registers torch.ops._C.direct_dcp_a2a_lse_reduce)

        self.cap = cap
        self.recv = torch.zeros(2, WORLD, cap, HEADS, LATENT, dtype=torch.bfloat16, device=device)
        self.rlse = torch.zeros(2, WORLD, cap, HEADS, dtype=torch.float32, device=device)
        self.sig = torch.zeros(2, WORLD, dtype=torch.int32, device=device)
        self.epoch = torch.zeros(1, dtype=torch.int64, device=device)
        i64 = dict(dtype=torch.int64, device=device)
        self.peers = torch.tensor([self.recv.data_ptr() + d * cap * HEADS * LATENT * 2 for d in range(WORLD)], **i64)
        self.lpeers = torch.tensor([self.rlse.data_ptr() + d * cap * HEADS * 4 for d in range(WORLD)], **i64)
        self.speers = torch.tensor([self.sig.data_ptr() + d * 4 for d in range(WORLD)], **i64)

    def op(self, partial, plse, seq_lens=None, qsl=None, out=None):
        m = partial.shape[0]
        out = torch.empty(m, HEADS, LATENT, dtype=torch.bfloat16, device=partial.device) if out is None else out
        torch.ops._C.direct_dcp_a2a_lse_reduce(partial, plse, seq_lens, qsl, self.peers, self.lpeers, self.speers,
                                               self.recv, self.rlse, self.sig, self.epoch, out, WORLD, 0, self.cap,
                                               False)
        return out

    def owner(self, x, v):
        """Served combine of owner inputs x [8, M, 8, 512], v [8, M, 8] -> [M, 8, 512]."""
        m = x.shape[1]
        partial = x.permute(1, 0, 2, 3).reshape(m, WORLD * HEADS, LATENT).contiguous()
        plse = v.permute(1, 0, 2).reshape(m, WORLD * HEADS).contiguous()
        return self.op(partial, plse)


def served_wuv(latent, w_uv, out=None):
    """mla.py _v_up_proj: torch.bmm into the transposed [M, 8 x 128] output."""
    m = latent.shape[0]
    out = torch.empty(m, HEADS * 128, dtype=torch.bfloat16, device=latent.device) if out is None else out
    torch.bmm(latent.view(m, HEADS, LATENT).transpose(0, 1), w_uv, out=out.view(m, HEADS, 128).transpose(0, 1))
    return out


# ------------------------------------------------------------------ salts


def make_wuv(device, ranks=WORLD, seed=1):
    """W_UV [ranks x 8, 512, 128] as the served absorbed view (latent axis contiguous, mla.py:443-452)."""
    g = torch.Generator().manual_seed(seed)
    kvb = (torch.randn(ranks * HEADS * 256, LATENT, generator=g) / 22.6).to(torch.bfloat16).to(device)
    return kvb.view(ranks * HEADS, 256, LATENT)[:, 128:, :].transpose(1, 2)


def salt(m, seed, kind="randn", ranks=WORLD, device="cuda"):
    """(rows [E, M, 64, 512] bf16, lse [E, M, 64] fp32, seq_lens [E, S] int32, qsl [S + 1] int32)."""
    g = torch.Generator().manual_seed(seed)
    rows = torch.randn(ranks, m, 64, LATENT, generator=g)
    lse = torch.randn(ranks, m, 64, generator=g) * 4 + 10
    s = -(-m // Q)
    qsl = torch.tensor([min(Q * i, m) for i in range(s + 1)], dtype=torch.int32)
    seq = torch.randint(1, 4096, (ranks, s), generator=g, dtype=torch.int32)
    req = torch.arange(m) // Q
    if kind == "empty":  # ~30% (rank, request) shards empty: garbage rows / LSE there
        hole = torch.rand(ranks, s, generator=g) < 0.3
        seq[hole] = 0
        dead = hole[:, req]
        rows[dead] = float("nan")
        lse[dead] = float("nan")
    elif kind == "neginf":  # masked rows: LSE -inf with NaN payload; token 0 head 3 -inf on every rank
        hole = torch.rand(ranks, m, 64, generator=g) < 0.2
        lse[hole] = -float("inf")
        rows[hole] = float("nan")
        lse[:, 0, 3] = -float("inf")
    elif kind == "naninf":
        hole = torch.rand(ranks, m, 64, generator=g)
        lse[hole < 0.05] = float("nan")
        lse[(hole >= 0.05) & (hole < 0.1)] = float("inf")
        rows[hole < 0.1] = float("nan")
    elif kind == "large":  # LSE spread +-300 (EX2 underflow / the -126 path), rows 2^90 and 2^-100
        lse = torch.randn(ranks, m, 64, generator=g) * 120
        rows[:, ::2] *= 2.0 ** 90
        rows[:, 1::3] *= 2.0 ** -100
    elif kind == "cancel":  # rank 1 = -rank 0 at 2^20, equal LSE over ranks (rank-order salt, judge.cancel_rows)
        rows = judge.cancel_rows(rows.to(torch.bfloat16).transpose(0, 1)).transpose(0, 1).float()
        lse[:] = lse[0]
    elif kind == "cancel_solo":  # the same for the solo source map (source s = head block s of rank 0)
        rows[:, :, 0:8] *= 2.0 ** 20
        rows[:, :, 8:16] = -rows[:, :, 0:8].to(torch.bfloat16).float()
        for b in range(1, 8):
            lse[:, :, 8 * b : 8 * b + 8] = lse[:, :, 0:8]
    to = dict(device=device)
    return rows.to(torch.bfloat16).to(**to), lse.to(**to), seq.to(**to), qsl.to(**to)


def empty_rows(seq, qsl, m):
    req = torch.searchsorted(qsl[1:].cpu(), torch.arange(m), right=True).to(seq.device)
    return seq[:, req] == 0  # [E, M]


SALTS = [(1, "randn"), (7, "randn"), (14, "randn"), (42, "randn"), (56, "randn"), (64, "randn"), (14, "empty"),
         (42, "empty"), (14, "neginf"), (42, "naninf"), (14, "large"), (64, "large"), (14, "cancel"), (42, "cancel")]


# ------------------------------------------------------------------ gate


class Gate:
    def __init__(self, device):
        self.device = device
        self.checks, self.stats = [], []
        self.served = Served(device)
        self.w_uv = make_wuv(device)

    def add(self, items):
        for c in items:
            self.checks.append(c)
            if not c["ok"]:
                print("FAIL", json.dumps(c), flush=True)

    def judge_call(self, tag, rows, lse, seq, qsl, out, dump, solo=False, owners=None):
        """G0 (emulation == served), G1 (latent == served bits), G2 (envelope; served too) for every owner."""
        e_count, m = rows.shape[:2]
        empty = empty_rows(seq, qsl, m)
        checks = []
        ndiff = 0
        for o in range(e_count) if owners is None else owners:
            x, v = judge.owner_inputs(rows, lse, empty, o, solo)
            lat_served = self.served.owner(x, v)
            emu = judge.emulate_combine(x, v, ex2_hw)
            checks.append(judge.compare_bytes(f"{tag}/G0/emulation_vs_served/o{o}", emu, lat_served))
            checks.append(judge.compare_bytes(f"{tag}/G1/latent_vs_served/o{o}", dump[o], lat_served))
            w = self.w_uv[o * HEADS : (o + 1) * HEADS]
            lat, elat, ref, err = judge.reference(x, v, w)
            checks.append(judge.envelope(f"{tag}/G2/mine/o{o}", out[o], ref, err))
            out_served = served_wuv(lat_served, w)
            checks.append(judge.envelope(f"{tag}/G2/served/o{o}", out_served, ref, err))
            ndiff += int((out[o].view(torch.int16) != out_served.view(torch.int16)).sum())
        self.stats.append({"tag": tag, "M": m, "outputs": m * 1024 * (len(owners) if owners else e_count),
                           "out_values_diff_vs_served_bmm": ndiff})
        return checks


def run(box, G, data, out, dump, parity=0, mutant="", w_uv=None):
    rows, lse, seq, qsl = data
    box.run(rows, lse, seq, qsl, G.w_uv if w_uv is None else w_uv, out, parity=parity, dump=dump, mutant=mutant)


def buffers(dev, e, m):
    return (torch.full((e, m, HEADS * 128), float("nan"), dtype=torch.bfloat16, device=dev),
            torch.full((e, m, HEADS, LATENT), float("nan"), dtype=torch.bfloat16, device=dev))


def mailbox_state(box, tag, m, parity, calls=1):
    """Epoch form: epoch == calls, completion 0, signal[E & 1][*] == E on every replica (E = calls) and E - 1 on the
    other parity. Lamport form: payload / LSE mailboxes all sentinel again, q rows < m of ``parity`` re-armed."""
    out = []
    if not box.lamport:
        E = calls
        ep, co, sig = box.epoch.cpu(), box.completion.cpu(), box.signal.cpu()
        want = torch.zeros_like(sig)
        want[:, E & 1, :] = E
        if E >= 2:
            want[:, (E - 1) & 1, :] = E - 1
        out.append({"name": f"{tag}/G3/epoch", "ok": bool((ep == E).all()), "mismatches": int((ep != E).sum()),
                    "first": ep.tolist()})
        out.append({"name": f"{tag}/G3/completion", "ok": bool((co == 0).all()), "mismatches": int((co != 0).sum()),
                    "first": co.tolist()})
        out.append({"name": f"{tag}/G3/signal_parity", "ok": bool((sig == want).all()),
                    "mismatches": int((sig != want).sum()), "first": sig[0].tolist()})
        return out
    pay = box.box.view(torch.int32) != -0x80000000
    lse = box.lbox.view(torch.int32) != -1
    q = box.qbox[:, parity, : m * WORLD * HEADS * 576] != 0xFF
    for name, bad in (("payload_rearmed", pay), ("lse_rearmed", lse), ("q_rows_rearmed", q)):
        out.append({"name": f"{tag}/G3/{name}", "ok": not bool(bad.any()), "mismatches": int(bad.sum()),
                    "first": None})
    return out


def dirty_q(box, m, parity):
    """Kernel 2's published bytes stand-in: rows < m of ``parity`` hold non-sentinel bytes before kernel 3 runs."""
    box.qbox[:, parity, : m * WORLD * HEADS * 576] = 0x11


def gate_variant(G, lamport):
    dev = G.device
    tagv = "lamport" if lamport else "epoch"
    # salts: two calls per fresh box (both parities), 8 emulated ranks in one launch
    for i, (m, kind) in enumerate(SALTS):
        box = StandInBox(dev, lamport=lamport)
        for call in range(2):
            data = salt(m, 100 * i + call, kind)
            out, dump = buffers(dev, WORLD, m)
            parity = call % 2
            if lamport:
                dirty_q(box, m, parity)
            run(box, G, data, out, dump, parity)
            torch.cuda.synchronize()
            tag = f"{tagv}/M{m}/{kind}/call{call}"
            G.add(G.judge_call(tag, *data, out, dump))
            G.add(mailbox_state(box, tag, m, parity, call + 1))
            print("SALT", tag, json.dumps(G.stats[-1]), flush=True)
    # solo source map (the timing configuration): one rank, its 8 pushes on its own 8 source slots
    for i, (m, kind) in enumerate([(14, "randn"), (42, "empty"), (64, "cancel_solo"), (14, "neginf")]):
        box = StandInBox(dev, ranks=1, lamport=lamport, solo=True)
        for call in range(2):
            data = salt(m, 500 + 10 * i + call, kind, ranks=1)
            out, dump = buffers(dev, 1, m)
            if lamport:
                dirty_q(box, m, call % 2)
            run(box, G, data, out, dump, call % 2)
            torch.cuda.synchronize()
            tag = f"{tagv}/solo/M{m}/{kind}/call{call}"
            G.add(G.judge_call(tag, *data, out, dump, solo=True))
            G.add(mailbox_state(box, tag, m, call % 2, call + 1))
    # graph capture + replay with swapped inputs (Lamport: parity 0 captured; the call re-arms what it reads)
    for m in (14, 42):
        box = StandInBox(dev, lamport=lamport)
        data = salt(m, 700 + m, "randn")
        live = tuple(t.clone() for t in data)
        out, dump = buffers(dev, WORLD, m)
        run(box, G, live, out, dump)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(box, G, live, out, dump)
        for rep in range(2):
            new = salt(m, 710 + m + rep, "empty" if rep else "randn")
            for dst, src in zip(live, new):
                dst.copy_(src)
            if lamport:
                dirty_q(box, m, 0)
            graph.replay()
            torch.cuda.synchronize()
            tag = f"{tagv}/graph/M{m}/rep{rep}"
            G.add(G.judge_call(tag, *live, out, dump))
            G.add(mailbox_state(box, tag, m, 0, 2 + rep))  # eager call 1; capture runs nothing
        del graph


# ------------------------------------------------------------------ late writers (Lamport): the reader must wait


def others_fill(box, data, m, parity):
    """Sources 1..7 of owner 0 in replica 0, as the other ranks' Lamport stores would leave them."""
    rows, lse, seq, qsl = data
    empty = empty_rows(seq, qsl, m)
    for s in range(1, WORLD):
        box.box[0, parity, s, :m] = judge.lamport_payload(rows[s][:, 0:HEADS], empty[s])
        box.lbox[0, parity, s, :m] = judge.lamport_lse(lse[s][:, 0:HEADS], empty[s])


def late_k3(G, m, seed, first_mutant=""):
    """Calls A (parity 0, ``first_mutant``), B (parity 1), C (parity 0, new data, sources 1..7 land ~20 ms late from a
    second stream). Returns (ms from the late copy's end to kernel C's end, checks on C's result)."""
    dev = G.device
    box = StandInBox(dev, ranks=1, copies=WORLD, lamport=True)
    out, dump = buffers(dev, 1, m)
    for call, parity in ((0, 0), (1, 1)):
        data = salt(m, seed + call, "randn")
        others_fill(box, data, m, parity)
        run(box, G, tuple(t[:1] for t in data[:3]) + (data[3],), out, dump, parity,
            mutant=first_mutant if call == 0 else "")
        torch.cuda.synchronize()
    data = salt(m, seed + 2, "randn")
    staged = StandInBox.__new__(StandInBox)  # staging copy of the late sources, filled before the race
    staged.box = torch.empty_like(box.box[:1])
    staged.lbox = torch.empty_like(box.lbox[:1])
    others_fill(staged, data, m, 0)
    main, side = torch.cuda.current_stream(), torch.cuda.Stream()
    ev_c, ev_k = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    side.wait_stream(main)
    with torch.cuda.stream(side):
        torch.cuda._sleep(SLEEP)
        box.box[0, 0, 1:, :m].copy_(staged.box[0, 0, 1:, :m])
        box.lbox[0, 0, 1:, :m].copy_(staged.lbox[0, 0, 1:, :m])
        ev_c.record(side)
    run(box, G, tuple(t[:1] for t in data[:3]) + (data[3],), out, dump, 0)
    ev_k.record(main)
    torch.cuda.synchronize()
    ms = ev_c.elapsed_time(ev_k)
    tag = f"late_k3/M{m}" + (f"/after_{first_mutant}" if first_mutant else "")
    checks = [{"name": f"{tag}/G3/reader_waits", "ok": ms > WAITED_MS, "mismatches": int(ms <= WAITED_MS), "first": round(ms, 3)}]
    full = (data[0], data[1], data[2], data[3])
    # owner 0's sources in full form: rank s's head block 0 (rank 0 = the launched rank)
    checks += G.judge_call(tag, *full, torch.cat([out, out.new_zeros(WORLD - 1, *out.shape[1:])]),
                           torch.cat([dump, dump.new_zeros(WORLD - 1, *dump.shape[1:])]), owners=[0])
    return ms, checks


def late_pair(G, m, seed, first_mutant=""):
    """Kernel 2 (Lamport, one rank, polls all 8 head blocks of replica 0) + kernel 3 on the same q mailbox. Calls A
    (parity 0; kernel 3 with ``first_mutant``), B (parity 1), then kernel 2 alone on parity 0 with head blocks 1..7
    landing ~20 ms late. Returns ms from the late copy's end to kernel 2's end (> 0: kernel 2 waited)."""
    from perfbot_kernels.draft_query import StandInBox as QueryBox
    from perfbot_kernels.draft_query.selftest.gpu import BLOCKS, BS, EPS, make_weights, rope_table
    from perfbot_kernels.draft_query.selftest.gpu import salt as query_salt

    dev = G.device
    if "q" not in _PROBE:
        _PROBE["q"] = (make_weights(dev), rope_table(dev))
    weights, table = _PROBE["q"]
    one = torch.tensor([1.0], device=dev)
    k3 = StandInBox(dev, ranks=1, copies=WORLD, lamport=True)
    k2 = QueryBox(dev, ranks=1, copies=1, waiters=WORLD, mailbox=k3.qbox[0:1])
    qkv, pos, slots = query_salt(m, seed, "randn")
    cache = torch.zeros(1, BLOCKS, BS, 576, dtype=torch.uint8, device=dev)
    others = torch.randint(0, 0x7F, (m, (WORLD - 1) * HEADS, 576), dtype=torch.uint8, device=dev)
    out, dump = buffers(dev, 1, m)

    def q_rows(parity):
        return k3.qbox[0, parity, : m * WORLD * HEADS * 576].view(m, WORLD * HEADS, 576)

    for call, parity in ((0, 0), (1, 1)):
        q_rows(parity)[:, HEADS:] = others
        k2.run(qkv, weights, table, pos, cache, slots[:1], one, one, EPS, parity=parity)
        data = salt(m, seed + call, "randn")
        others_fill(k3, data, m, parity)
        run(k3, G, tuple(t[:1] for t in data[:3]) + (data[3],), out, dump, parity,
            mutant=first_mutant if call == 0 else "")
        torch.cuda.synchronize()
    main, side = torch.cuda.current_stream(), torch.cuda.Stream()
    ev_c, ev_k = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    side.wait_stream(main)
    with torch.cuda.stream(side):
        torch.cuda._sleep(SLEEP)
        q_rows(0)[:, HEADS:].copy_(others)
        ev_c.record(side)
    k2.run(qkv, weights, table, pos, cache, slots[:1], one, one, EPS, parity=0)
    ev_k.record(main)
    torch.cuda.synchronize()
    return ev_c.elapsed_time(ev_k)


# ------------------------------------------------------------------ gate driver


def gate(out_dir):
    dev = torch.device("cuda")
    G = Gate(dev)
    sweep = torch.empty(1 << 22, device=dev).uniform_(-200, 60)
    info = {"torch_exp2_vs_ex2_mismatch_of_4M": int((torch.exp2(sweep) != ex2_hw(sweep)).sum()),
            "sm_count": torch.cuda.get_device_properties(dev).multi_processor_count}
    print("G0a", json.dumps(info), flush=True)
    for lamport in (False, True):
        gate_variant(G, lamport)
    late = {}
    for m in (14, 42):
        ms, checks = late_k3(G, m, 900 + m)
        G.add(checks)
        late[f"k3/M{m}"] = round(ms, 3)
        ms = late_pair(G, m, 950 + m)
        late[f"pair/M{m}"] = round(ms, 3)
        G.add([{"name": f"late_pair/M{m}/G3/kernel2_waits", "ok": ms > WAITED_MS, "mismatches": int(ms <= WAITED_MS),
                "first": round(ms, 3)}])
    print("LATE", json.dumps(late), flush=True)
    passed = judge.verdict(G.checks)

    # mutants: each must fail the judge
    mutants = {}

    def record(name, checks, extra=None):
        failed = [c["name"] for c in checks if not c["ok"]]
        mutants[name] = {"caught": bool(failed), "failed_checks": failed[:6], **(extra or {})}
        print("MUTANT", name, json.dumps(mutants[name]), flush=True)

    for lamport in (False, True):
        tagv = "lamport" if lamport else "epoch"
        for mutant, (m, kind) in (("rankorder", (14, "cancel")), ("nomax", (14, "large")),
                                  ("skipzero", (14, "neginf"))):
            box = StandInBox(dev, lamport=lamport)
            data = salt(m, 990, kind)
            out, dump = buffers(dev, WORLD, m)
            run(box, G, data, out, dump, 0, mutant=mutant)
            torch.cuda.synchronize()
            record(f"{tagv}/{mutant}", G.judge_call(f"mutant/{tagv}/{mutant}", *data, out, dump))
    box = StandInBox(dev)
    data = salt(14, 991, "randn")
    out, dump = buffers(dev, WORLD, 14)
    run(box, G, data, out, dump, 0, mutant="noepoch")
    torch.cuda.synchronize()
    record("epoch/noepoch", G.judge_call("mutant/noepoch", *data, out, dump) + mailbox_state(box, "mutant/noepoch",
                                                                                              14, 0, 1))
    ms, checks = late_k3(G, 14, 1200, first_mutant="norearm")
    record("lamport/norearm (second call of parity 0)", checks, {"kernel_end_minus_late_copy_ms": round(ms, 3)})
    ms = late_pair(G, 14, 1300, first_mutant="noqrearm")
    record("pair/noqrearm (kernel 2's second call of parity 0)",
           [{"name": "mutant/noqrearm/G3/kernel2_waits", "ok": ms > WAITED_MS, "mismatches": 1, "first": round(ms, 3)}],
           {"kernel_end_minus_late_copy_ms": round(ms, 3)})
    caught = all(v["caught"] for v in mutants.values())

    summary = {"gate": "G0 emulation == served, G1 latent bit-equal to served, G2 fp64 envelope, G3 salts / parities "
                       "/ graph / mailbox state / late writers", "passed": passed, "mutants_caught": caught,
               "checks": len(G.checks), "failed": [c for c in G.checks if not c["ok"]][:20], "G0a": info,
               "late_ms": late, "mutants": mutants, "salts": SALTS, "G2_vs_served": G.stats}
    print("SUMMARY", json.dumps({k: summary[k] for k in ("passed", "mutants_caught", "checks")}), flush=True)
    if out_dir:
        with open(os.path.join(out_dir, "gate.json"), "w") as f:
            json.dump(summary, f, indent=1)
    return 0 if passed and caught else 1


# ------------------------------------------------------------------ timing


def graph_us(fn, calls, replays=50, blocks=2):
    from perfbot_kernels.draft_query.selftest.gpu import graph_us as query_graph_us

    return query_graph_us(fn, calls, replays, blocks)


def timing(out_dir, only=None):
    """One emulated rank (solo source map: one rank's store and read volume, all local) vs the served launches it
    replaces on this GPU. Pairs: kernel 2 + kernel 3 (parity alternating per call, as per layer)."""
    from perfbot_kernels.draft_query import StandInBox as QueryBox
    from perfbot_kernels.draft_query.selftest.gpu import BLOCKS, BS, EPS, Served as QueryServed, make_weights
    from perfbot_kernels.draft_query.selftest.gpu import rope_table
    from perfbot_kernels.draft_query.selftest.gpu import salt as query_salt

    dev = torch.device("cuda")
    served = Served(dev)
    w_uv = make_wuv(dev, ranks=1)
    qweights, table = make_weights(dev), rope_table(dev)
    qserved = QueryServed(qweights, table)
    one = torch.tensor([1.0], device=dev)
    calls, rows = 20, []

    def add(what, m, fn):
        if only and not any(k in what for k in only):
            return
        rows.append({"what": what, "M": m, "us_per_call": graph_us(fn, calls)})
        print("TIME", json.dumps(rows[-1]), flush=True)

    for m in (14, 42):
        data = salt(m, 7, "randn", ranks=1)
        rows_, lse_, seq_, qsl_ = data
        out = torch.empty(1, m, HEADS * 128, dtype=torch.bfloat16, device=dev)
        lat = torch.empty(m, HEADS, LATENT, dtype=torch.bfloat16, device=dev)
        sout = torch.empty(m, HEADS * 128, dtype=torch.bfloat16, device=dev)
        qkv, pos, slots = query_salt(m, 7, "randn")
        cache = torch.zeros(1, BLOCKS, BS, 576, dtype=torch.uint8, device=dev)
        s1 = slots[:1].clone()
        s1[0] = torch.arange(m, device=dev)
        boxes = {lam: StandInBox(dev, ranks=1, lamport=lam, solo=True) for lam in (False, True)}

        def served_tail():
            served.op(rows_[0], lse_[0], seq_[0], qsl_, out=lat)
            served_wuv(lat, w_uv, out=sout)

        def k3(lam):
            def fn():
                for i in range(calls):
                    boxes[lam].run(rows_, lse_, seq_, qsl_, w_uv, out, parity=i % 2)
            return fn

        def theirs_tail():
            for _ in range(calls):
                served_tail()

        add("served tail: increment_epoch + dispatch + signal + wait_lse_combine + W_UV bmm (5 launches)", m,
            theirs_tail)
        add("DraftValue epoch (solo rank, 128 CTAs)", m, k3(False))
        add("DraftValue Lamport (solo rank, 128 CTAs)", m, k3(True))

        def theirs_pair():
            for _ in range(calls):
                qserved.rank(0, qkv, pos, cache[0], s1[0], one, one)
                served_tail()

        add("served pair: q chain (4 launches) + tail (5 launches); q_gather + its epoch excluded (multicast)", m,
            theirs_pair)
        qb_epoch = QueryBox(dev, ranks=1, copies=1, waiters=1)
        qb_own = QueryBox(dev, ranks=1, copies=1, waiters=1, mailbox=boxes[True].qbox[0:1])
        qb_all = QueryBox(dev, ranks=1, copies=WORLD, waiters=WORLD, mailbox=boxes[True].qbox[0:1], shift=True)

        def pair(qb, lam):
            def fn():
                for i in range(calls):
                    qb.run(qkv, qweights, table, pos, cache, s1, one, one, EPS, parity=i % 2)
                    boxes[lam].run(rows_, lse_, seq_, qsl_, w_uv, out, parity=i % 2)
            return fn

        add("pair epoch: DraftQuery (1 replica, own signal) + DraftValue epoch", m, pair(qb_epoch, False))
        add("pair Lamport: DraftQuery (1 replica, polls own head block) + DraftValue Lamport", m, pair(qb_own, True))
        add("pair Lamport: DraftQuery (8 local head-block stores, polls all 8) + DraftValue Lamport", m,
            pair(qb_all, True))
        add("DraftQuery Lamport alone (1 replica, polls own head block; harness never re-arms: poll passes)", m,
            lambda: [qb_own.run(qkv, qweights, table, pos, cache, s1, one, one, EPS, parity=i % 2)
                     for i in range(calls)])
    if out_dir:
        with open(os.path.join(out_dir, "timing.json" if not only else "timing-" + "-".join(only) + ".json"),
                  "w") as f:
            json.dump({"device": torch.cuda.get_device_name(dev), "calls_per_graph": calls, "replays": 50,
                       "rows": rows, "time": time.strftime("%FT%TZ", time.gmtime())}, f, indent=1)
    return 0


def ablate(out_dir):
    """Timing-only cuts of DraftValue (wrong bytes; never a gate)."""
    from perfbot_kernels.lse_combine.kernel import ABLATIONS

    dev = torch.device("cuda")
    w_uv = make_wuv(dev, ranks=1)
    rows = []
    for m in (14, 42):
        data = salt(m, 7, "randn", ranks=1)
        out = torch.empty(1, m, HEADS * 128, dtype=torch.bfloat16, device=dev)
        for lam in (False, True):
            for variant in ("",) + ABLATIONS:
                if lam and variant == "x_nopush":  # Lamport readers would poll mailboxes nobody fills (hang)
                    continue
                box = StandInBox(dev, ranks=1, lamport=lam, solo=True)

                def fn():
                    for i in range(20):
                        box.run(*data, w_uv, out, parity=i % 2, mutant=variant)

                rows.append({"what": ("lamport " if lam else "epoch ") + (variant or "full"), "M": m,
                             "us_per_call": graph_us(fn, 20)})
                print("ABL", json.dumps(rows[-1]), flush=True)
    if out_dir:
        with open(os.path.join(out_dir, "ablate.json"), "w") as f:
            json.dump({"device": torch.cuda.get_device_name(dev), "rows": rows}, f, indent=1)
    return 0


def iket_run():
    """IKET ranges in the timing setting: graphs of 20 back-to-back calls, M 14 and 42, both forms."""
    dev = torch.device("cuda")
    w_uv = make_wuv(dev, ranks=1)
    for m in (14, 42):
        data = salt(m, 3, "randn", ranks=1)
        out = torch.empty(1, m, HEADS * 128, dtype=torch.bfloat16, device=dev)
        for lam in (False, True):
            box = StandInBox(dev, ranks=1, lamport=lam, solo=True)

            def fn():
                for i in range(20):
                    box.run(*data, w_uv, out, parity=i % 2, iket=True)

            graph_us(fn, 20, replays=1, blocks=1)
    torch.cuda.synchronize()
    print("PERFBOT-LSE-COMBINE-IKET ok", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--iket", action="store_true")
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.out:
        os.makedirs(a.out, exist_ok=True)
    if a.iket:
        return iket_run()
    if a.ablate:
        return ablate(a.out)
    rc = 0
    if a.gate:
        rc |= gate(a.out)
    if a.time:
        rc |= timing(a.out, [s for s in a.only.split(",") if s] or None)
    return rc


if __name__ == "__main__":
    sys.exit(main())
