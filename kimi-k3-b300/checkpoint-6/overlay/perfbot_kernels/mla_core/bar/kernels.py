"""P-08 kernel registry. Every entry has dcp.stock_decode's signature
(query, cache, block_table, lengths, max_seq, scales, out, lse, min_split, pdl, cp, totals, pack) and writes out / lse.

inc          the image's dcp.stock_decode (served seam, truth 2).
cand:<dir>   P-08/kernel/<dir>/perfbot_kernels/mla_decode (card §5 package shape); one candidate dir per process.
var:<v>      evidence-only incumbent variants (kernel/variants, v in full | qk_only | pv_only); wrong outputs by design,
             never judged, never a candidate.
mut_*        deliberately wrong kernels; the judge must FAIL each (teeth). Mutants wrap the incumbent.
emu          fp32 split-KV emulator (get_k_tile_count partition, per-split softmax, exp2 LSE merge): a correct kernel
             with another rounding schedule = a positive control that must PASS.
emu_*        the emulator with one split-merge bug (review finding 4): each must FAIL.
ctl_otherfloor  R1 positive control: the incumbent at the other split floor (planner <-> ALT[B]).
"""
from __future__ import annotations

import importlib
import math
import os
import sys

import torch

# The one table of other split floors per judged B (planner split: B1 12, B2 6, B3 4, B5 2; R11a). judge.py imports it.
ALT = {1: 24, 2: 12, 3: 8, 4: 6, 5: 4}
HERE = os.path.dirname(os.path.abspath(__file__))
KDIR = os.path.join(os.path.dirname(HERE), "kernel")
PINNED = dict(has_local_visible_lens=False, is_var_split_kv=False, skip_correction_threshold=8.0, window_left=-1,
              partial_fp16=False)


def inc():
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.mla.dcp import stock_decode
    return stock_decode


def cand_module(d: str):
    p = os.path.join(KDIR, d)
    if p not in sys.path:
        sys.path.insert(0, p)
    m = importlib.import_module("perfbot_kernels.mla_decode")
    assert os.path.dirname(os.path.dirname(os.path.dirname(m.__file__))) == p, f"stale import {m.__file__}"
    # card §3 exception of record: only the arm P-08/kernel/var-split may set is_var_split_kv=True
    var = os.path.realpath(p) == os.path.realpath(os.path.join(KDIR, "var-split"))
    pin = dict(PINNED, is_var_split_kv=True) if var else PINNED
    # item 03 ruling (perf-bot-8e 2026-10-19): the fp16-partials arm is a numerics-class lever judged by the
    # 1.02x fp64 envelope; only kernel/mla_core_fp16 may set partial_fp16=True
    if os.path.basename(os.path.realpath(p)) == "mla_core_fp16":
        pin = dict(pin, partial_fp16=True)
    got = {k: m.COMPILE_CONTRACT.get(k) for k in pin}
    assert got == pin, f"compile contract {got} != pinned {pin} (card §3)"
    return m


# (q, cache, bt, lengths, max_seq, scales, out, lse, min_split, pdl, cp, totals, pack)
def _mut(edit_in=None, edit_out=None):
    def wrap(base):
        def f(*a):
            a = list(a)
            if edit_in:
                edit_in(a)
            base(*a)
            if edit_out:
                edit_out(a[6], a[7])
        return f
    return wrap


def _droplast(a):  # drop the last visible page of every request
    a[3] = torch.clamp(a[3] - 64, min=0)
    a[11] = torch.clamp(a[11] - 64 * a[10][0], min=0)


def _skippage(a):  # read page 1 in place of page 2
    bt = a[2].clone()
    bt[:, 2] = bt[:, 1]
    a[2] = bt


def _nocausal(a):  # every q row sees every local key
    a[11] = a[11] + 7 * a[10][0]


def _qtok_off1(a):  # row i sees row i+1's bound
    a[11] = a[11] + 1


def _lowprec(o, l):
    o.copy_(o.to(torch.float8_e4m3fn).to(o.dtype))


def _headswap(o, l):  # wrong (row, head) map for one head pair of q row 3
    t = o[:, 3, 0].clone()
    o[:, 3, 0] = o[:, 3, 95]
    o[:, 3, 95] = t


def _lse_bf16(o, l):  # LSE precision dropped (the combine would hide it in the merged output)
    l.copy_(l.to(torch.bfloat16).to(l.dtype))


def _lse_base_e(o, l):  # LSE written in base e
    l.mul_(0.6931471805599453)


_STALE: dict = {}


def _stale(o, l):  # caches the first output per shape
    k = tuple(o.shape)
    if k not in _STALE:
        _STALE[k] = (o.clone(), l.clone())
    else:
        o.copy_(_STALE[k][0])
        l.copy_(_STALE[k][1])


MUTANTS = {"mut_droplast": _mut(_droplast), "mut_skippage": _mut(_skippage), "mut_nocausal": _mut(_nocausal),
           "mut_qtok_off1": _mut(_qtok_off1), "mut_lowprec": _mut(None, _lowprec),
           "mut_headswap": _mut(None, _headswap), "mut_lse_bf16": _mut(None, _lse_bf16),
           "mut_lse_base_e": _mut(None, _lse_base_e), "mut_stale": _mut(None, _stale)}


def ctl_otherfloor():
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.mla.dcp import fill_splits
    base = inc()

    def f(*a):
        a = list(a)
        B = a[0].shape[0]
        served = fill_splits(B * a[0].shape[1], a[0].device)
        a[8] = ALT[B] if a[8] == served else served
        base(*a)
    return f


def _call_at(base, a, split):
    """Run the incumbent on args a at uniform `split` into fresh NaN-filled out / lse. The seam takes a split FLOOR,
    so floor = split; the caller checks that the planner really gives `split` at this batch."""
    o = torch.full_like(a[6], float("nan"))
    l = torch.full_like(a[7], float("nan"))
    base(a[0], a[1], a[2], a[3], a[4], a[5], o, l, split, a[9], a[10], a[11], a[12])
    return o, l


def split_reachable(B, split, Q, dev) -> bool:
    """The planner gives exactly `split` at batch B with floor `split` (v5.2 [u] check)."""
    import p08
    return p08.split_for(B, split, Q, dev)[0] == split


def composed(splits):
    """v5.2 envelope member (coordinator ruling b, revised): the incumbent at per-request splits, built request by
    request. Request b's rows come from a uniform-split incumbent call at split splits[b] (vector 4,2,2,2,2: B5 floor
    4 -> 4 for request 0, the ALT[5] control's split; the served split 2 for requests 1-4). Every call is at the case's own batch (no padding);
    an unreachable split stops the judge."""
    base = inc()

    def f(*a):
        B, Q = a[0].shape[0], a[0].shape[1]
        assert len(splits) == B, f"composed: {len(splits)} splits for batch {B}"
        for s in sorted(set(splits)):
            if not split_reachable(B, s, Q, a[0].device):
                raise SystemExit(f"v5.2 [u] composed-member check FAILED: split {s} not reachable at B{B}")
            o, l = _call_at(base, a, s)
            for b in (i for i, x in enumerate(splits) if x == s):
                a[6][b].copy_(o[b])
                a[7][b].copy_(l[b])
    return f


LOG2E = 1.4426950408889634


def emulator(flags=frozenset(), splits=None):
    """fp32 emulation of split-KV MLA decode + merge over the served arguments. splits: per-request split counts
    (is_var_split_kv; the kernel partitions request b by block_split_kvs[b], mla_decode_fp8.py:2110-2125); None = the
    served uniform split. flags (one bug each):
    drop_mid     the middle split is never merged            base_e    merge weights exp(lse_j - M), not exp2
    empty_lse0   an empty split contributes lse 0, o 0       row_swap  B >= 2: request 0 merges request 1's split 0
    overlap      split j > 0 starts one 128-row tile early   stale     split 0 partials from the previous call
    thresh8      splits more than 8 bits below the max are skipped (a skip-correction bug)"""
    import p08
    assert not (flags and splits), "bug flags are defined for the served uniform split only"
    prev: dict = {}

    def f(q, cache, bt, lengths, max_seq, scales, out, lse, min_split, pdl, cp, totals, pack):
        softmax, k_scale = scales
        B, Q = q.shape[0], q.shape[1]
        split = p08.split_for(B, min_split, Q, q.device)[0]
        spb = list(splits) if splits else [split] * B
        kv = cache.view(torch.float8_e4m3fn)
        world, rank = cp[0], cp[1]
        parts = []  # per request: list of (o_j [Q,H,512], lse_j [Q,H]) in bits
        for b in range(B):
            L = int(lengths[b])
            if L == 0:
                parts.append(None)
                continue
            npg = -(-L // 64)
            k = kv[bt[b, :npg].long()].reshape(-1, 576)[:L].float()
            s = torch.einsum("qhd,kd->qhk", q[b].float(), k) * (softmax * LOG2E)  # bits; v = k_scale * latent
            T = int(totals[b])
            pos = torch.arange(L, device=q.device)
            bounds = torch.tensor([min(p08.local_bound(T - (Q - 1) + i, rank, world), L) for i in range(Q)],
                                  device=q.device)
            s = s.masked_fill(pos[None, None, :] >= bounds[:, None, None], -math.inf)
            rngs = p08.split_ranges(L, spb[b])
            pj = []
            for j, (lo, hi) in enumerate(rngs):
                if "overlap" in flags and j > 0 and hi > lo:
                    lo = max(0, lo - 128)
                sj = s[..., lo:hi]
                if hi <= lo or not torch.isfinite(sj).any():
                    if "empty_lse0" in flags:
                        pj.append((torch.zeros(Q, s.shape[1], 512, device=q.device),
                                   torch.zeros(Q, s.shape[1], device=q.device)))
                    else:
                        pj.append(None)
                    continue
                m = sj.amax(-1, keepdim=True)
                m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
                pr = torch.exp2(sj - m)
                z = pr.sum(-1)
                o = torch.einsum("qhk,kd->qhd", pr, k[lo:hi, :512] * k_scale) / z.clamp_min(1e-30)[..., None]
                l = torch.where(z > 0, m.squeeze(-1) + torch.log2(z), torch.full_like(z, -math.inf))
                pj.append((o, l))
            parts.append(pj)
        key = tuple(q.shape)
        if "stale" in flags:
            old = prev.get(key)
            prev[key] = [p[0] if p else None for p in parts]
            if old is not None:
                for b in range(B):
                    if parts[b] and old[b] is not None and parts[b][0] is not None:
                        parts[b][0] = old[b]
        if "row_swap" in flags and B >= 2 and parts[0] and parts[1]:  # v5: all B >= 2; B2 unchanged
            parts[0][0] = parts[1][0]
        for b in range(B):
            if not parts[b]:
                out[b].zero_()
                lse[b].fill_(-math.inf)
                continue
            pj = [p for j, p in enumerate(parts[b]) if p is not None and not ("drop_mid" in flags and j == spb[b] // 2)]
            if not pj:
                out[b].zero_()
                lse[b].fill_(-math.inf)
                continue
            Ls = torch.stack([p[1] for p in pj])
            M = Ls.amax(0)
            Mf = torch.where(torch.isfinite(M), M, torch.zeros_like(M))
            w = torch.exp(Ls - Mf) if "base_e" in flags else torch.exp2(Ls - Mf)
            if "thresh8" in flags:
                w = torch.where(Ls < Mf - 8, torch.zeros_like(w), w)
            W = w.sum(0)
            o = (w[..., None] * torch.stack([p[0] for p in pj])).sum(0) / W.clamp_min(1e-30)[..., None]
            out[b].copy_(o.to(out.dtype))
            lse[b].copy_(torch.where(W > 0, Mf + torch.log2(W), torch.full_like(W, -math.inf)))
    return f


EMU_FLAGS = {f"emu_{x}": frozenset([x]) for x in
             ("drop_mid", "base_e", "empty_lse0", "row_swap", "overlap", "stale", "thresh8")}


def get(name: str):  # noqa: C901
    if name == "emu":
        return emulator()
    if name in EMU_FLAGS:
        return emulator(EMU_FLAGS[name])
    if name == "ctl_otherfloor":
        return ctl_otherfloor()
    if name == "inc":
        return inc()
    if name == "pkg:mla_core":  # the bundle's own package (B300 bar arm, bar/run.py)
        from perfbot_kernels import mla_core
        got = {k: mla_core.COMPILE_CONTRACT.get(k) for k in PINNED}
        assert got == PINNED, f"compile contract {got} != pinned {PINNED} (card §3)"
        return mla_core.stock_decode
    if name.startswith("cand:"):
        return cand_module(name[5:]).stock_decode
    if name.startswith("var:"):
        os.environ["P08_VARIANT"] = name[4:]
        p = os.path.join(KDIR, "variants")
        if p not in sys.path:
            sys.path.insert(0, p)
        return importlib.import_module("seam_var").stock_decode
    if name in MUTANTS:
        return MUTANTS[name](inc())
    raise KeyError(name)
