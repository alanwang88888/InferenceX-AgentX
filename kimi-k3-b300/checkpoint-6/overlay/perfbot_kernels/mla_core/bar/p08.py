"""P-08 judge core: tokenspeed MLA decode (absorbed MQA, fp8 latent 512+64, page 64) under DCP8 at the served seam.

One case schema (Case) feeds every check: the fp64 truth, the incumbent (the image's own
`specialized_decode/cutedsl/mla/dcp.py:stock_decode`, i.e. the served call with the served compile contract), the
candidate, and the timing blocks. Case builder, fp64 truth and DCP bound are adapted from lane K
(k3-lane-k-p01 docs/expedition/problems/P-01/judge/p01judge.py, read off tokenspeed_mla 2.0.3, not assumed):
- query [B, q, H, 576] fp8, kv_cache [pages, 64, 576] fp8, block_table [B, max_pages] int32,
  lengths [B] = LOCAL rows on this rank, totals [B] = GLOBAL context (same on every rank).
- served scales (prepared.py:156-158): softmax_scale = scale * q_scale * k_scale, output_scale = k_scale.
- query token i sees global keys [0, total - (q-1) + i); local bound = _get_dcp_local_bound (interleave 1:
  (g - r + W - 1) // W).
- LSE written by the kernel is base 2 (dcp.py docstring "LSE (log2)"; DcpValue merges with exp2); checked by ready.py.
Served call (prepared.py:203-226, B requests x q 8 rows): batch = B*q, min_split = fill_splits(batch) = 148 // (2 batch),
max_seq = local_max(max_model_len 1048576, 8) = 131072, pack=True, cp=(8, rank, 1), totals = global lengths, pdl=True.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Optional

import torch

LATENT, ROPE, D = 512, 64, 576
PAGE = 64
H = 96
CP_WORLD = 8
SCALE = 192 ** -0.5  # qk_nope 128 + rope 64 (attention.scale)
FP8 = torch.float8_e4m3fn
LN2 = math.log(2.0)
MAX_MODEL_LEN = 1 << 20  # R-UP-2b/results/b0-spec/models-after-load.json: max_model_len 1048576
QLEN = 8


def local_bound(g: int, rank: int, world: int = CP_WORLD, inter: int = 1) -> int:
    """Number of this rank's keys among global positions [0, g). Mirror of _get_dcp_local_bound."""
    if g <= 0:
        return 0
    if world == 1:
        return g
    if inter == 1:
        return (g - rank + world - 1) // world
    cyc = world * inter
    full, off = divmod(g, cyc)
    return full * inter + min(max(off - rank * inter, 0), inter)


def served_max_seq() -> int:
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.mla.dcp import local_max
    return local_max(MAX_MODEL_LEN, CP_WORLD)


@dataclasses.dataclass
class Case:
    name: str
    q: torch.Tensor             # [B, q, H, 576] fp8
    kv: torch.Tensor            # [pages, 64, 576] fp8
    block_table: torch.Tensor   # [B, max_pages] int32
    lengths: torch.Tensor       # [B] int32 local rows
    totals: torch.Tensor        # [B] int32 global ctx
    q_scale: float
    k_scale: float
    rank: int
    min_split: Optional[int] = None  # None: the served fill_splits(B*q); an int forces the planner floor (salt)
    world: int = CP_WORLD

    @property
    def B(self):
        return self.q.shape[0]

    @property
    def qlen(self):
        return self.q.shape[1]

    @property
    def softmax_scale(self):
        return SCALE * self.q_scale * self.k_scale

    def split_floor(self):
        if self.min_split is not None:
            return self.min_split
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.mla.dcp import fill_splits
        return fill_splits(self.B * self.qlen, self.q.device)


def make_case(name: str, seed: int, ctx: list, rank: int = 0, qlen: int = QLEN, stress: tuple = (),
              min_split: Optional[int] = None, device="cuda", extra_pages: int = 3, q_seed: Optional[int] = None,
              split: Optional[int] = None) -> Case:
    """ctx = GLOBAL context per request (includes the q new tokens); ctx 0 = an empty padded request (pad_requests:
    length 0, total 0, block_table row of zeros). Pages are a random permutation of a pool (non-monotonic page ids).
    Values: fp8 spanning the e4m3 exponent range (log-uniform magnitudes 2^-6 .. 2^8 x 0.02, random sign), q_scale and
    k_scale uniform in [0.03, 0.3]. q_seed: draw q (and the scales) from their own seed, so the 8 ranks of a combine case
    share one query (each rank's KV shard is its own draw: together they form one global KV).

    stress: 'needle'  one planted key per page group (pairs of pages sharing a 64-dim rope signature, own latent):
                      first and last visible page, page boundary 63|64, a seeded middle; one (q row, head) aims at each
                      group, its truth is the mean of two V rows; a dropped / wrong / repeated page moves it by O(1).
            'anti'    q row i (heads 1, 4, 7, ..) aimed at the local key at its own causal bound (visible only to later
                      rows): a leaky causal mask gives an O(1) error (covers the q-row-7 causal needle).
            'qdom'    q rows with one dominant head (x16 magnitude on head 0 of every row).
            'dc'      non-zero-mean latent (+1.0) with 8 outlier channels x20.
            'graded'  (review finding 3) rows whose softmax mass is spread over the splits: for 128 (q row, head)
                      pairs, one planted key per split (`split` = the served split count) at a score drawn in 0-12
                      bits over the flat background; half the pairs also get an in-split max jump (a key at 2 bits in
                      the split's first tile, then one at 2 + 7.9 or 2 + 8.1 bits in a later tile: across
                      skip_correction_threshold 8.0). A wrong merge weighting (base, split -> row map, dropped or
                      doubled split) moves these rows by O(1e-1); needle and flat rows hide it.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    gd = torch.Generator(device=device).manual_seed(seed)
    gq = torch.Generator(device="cpu").manual_seed(seed if q_seed is None else q_seed)
    gqd = torch.Generator(device=device).manual_seed(seed if q_seed is None else q_seed)
    B = len(ctx)
    loc = [local_bound(c, rank) for c in ctx]
    npg = [max(1, -(-l // PAGE)) for l in loc]
    max_pages = max(npg)
    total = sum(npg) + extra_pages
    perm = torch.randperm(total, generator=g)
    bt = torch.zeros(B, max_pages, dtype=torch.int32)
    o = 0
    for b in range(B):
        if ctx[b] == 0:
            continue  # padded request: page 0 row of zeros
        bt[b, :npg[b]] = perm[o:o + npg[b]].to(torch.int32)
        bt[b, npg[b]:] = perm[-1].to(torch.int32)  # unused tail entries point at real pages (never read)
        o += npg[b]

    def draw(gen, *shape):
        mag = torch.exp2(torch.empty(*shape, device=device).uniform_(-6.0, 8.0, generator=gen))
        sgn = torch.where(torch.rand(*shape, device=device, generator=gen) < 0.5, -1.0, 1.0)
        return mag * sgn * 0.02

    kvf = draw(gd, total, PAGE, D)
    if "dc" in stress:
        kvf[..., :LATENT] += 1.0
        kvf[..., :8] *= 20.0
    qf = draw(gqd, B, qlen, H, D)
    q_scale = float(torch.empty(1).uniform_(0.03, 0.3, generator=gq))
    k_scale = float(torch.empty(1).uniform_(0.03, 0.3, generator=gq))
    if "qdom" in stress:
        qf[:, :, 0] *= 16.0
    kvf = kvf.clamp(-448, 448)
    s_eff = SCALE * q_scale * k_scale
    btl = bt.tolist()

    def key_at(b, pos):
        pg, off = divmod(pos, PAGE)
        return btl[b][pg], off

    taken: dict = {}
    if "anti" in stress:
        for b in range(B):
            for i in range(qlen):
                bound = local_bound(ctx[b] - (qlen - 1) + i, rank)
                if ctx[b] == 0 or bound >= loc[b]:
                    continue
                p, off = key_at(b, bound)
                key = kvf[p, off].to(FP8).float()
                sign = torch.where(key < 0, -1.0, 1.0)
                hs = list(range(1, H, 3))
                qf[b, i, hs] = sign * (12.0 / (s_eff * key.abs().sum() + 1e-30))
                taken.setdefault(b, set()).update((i, h) for h in hs)
    if "needle" in stress:
        for b in range(B):
            if ctx[b] == 0:
                continue
            vmin = min(local_bound(ctx[b] - (qlen - 1) + i, rank) for i in range(qlen))
            if vmin < 2:
                continue
            npv = -(-vmin // PAGE)
            pos = [p * PAGE + int(torch.randint(min(PAGE, vmin - p * PAGE), (1,), generator=g)) for p in range(npv)]
            pos[0] = 0
            if npv > 1:
                pos[1] = PAGE
            pos[-1] = vmin - 1
            if npv > 2 and pos[-2] == pos[-1]:
                pos[-2] = (npv - 2) * PAGE
            groups = [pos[k:k + 2] for k in range(0, len(pos), 2)]
            free = [(i, h) for i in range(qlen) for h in range(H) if (i, h) not in taken.get(b, set()) and h != 0]
            if len(groups) > len(free):  # keep both ends + a seeded random middle (every split gets needles)
                mid = torch.randperm(len(groups) - 2, generator=g)[: len(free) - 2].sort().values + 1
                groups = [groups[0]] + [groups[k] for k in mid.tolist()] + [groups[-1]]
            pf = torch.randperm(len(free), generator=g).tolist()
            sigs = torch.where(torch.rand(len(groups), ROPE, generator=g) < 0.5, -16.0, 16.0).to(device)
            pidx, oidx, qi, qh = [], [], [], []
            for gi, grp in enumerate(groups):
                for pp in grp:
                    p, off = key_at(b, pp)
                    pidx.append(p); oidx.append(off)
                i, h = free[pf[gi]]
                qi.append(i); qh.append(h)
            rep = torch.tensor([len(grp) for grp in groups], device=device)
            kvf[torch.tensor(pidx, device=device), torch.tensor(oidx, device=device), LATENT:] = \
                sigs.repeat_interleave(rep, dim=0)
            qi, qh = torch.tensor(qi, device=device), torch.tensor(qh, device=device)
            qf[b, qi, qh] = 0.0
            qf[b, qi, qh, LATENT:] = torch.sign(sigs) * (16.0 / (s_eff * 16.0 * ROPE))
    if "graded" in stress:
        assert split, "graded needs the served split count"
        a = 12.0 * LN2 / (s_eff * ROPE * 16.0)  # a key of rope magnitude 16 scores 12 bits
        for b in range(B):
            if ctx[b] == 0:
                continue
            vmin = min(local_bound(ctx[b] - (qlen - 1) + i, rank) for i in range(qlen))
            rngs = [(lo, min(hi, vmin)) for lo, hi in split_ranges(loc[b], split) if min(hi, vmin) - lo >= 2]
            pairs = [(i, h) for i in range(qlen) for h in range(2, H, 6) if (i, h) not in taken.get(b, set())]
            pidx, oidx, vals, qi, qh, qv = [], [], [], [], [], []
            for n, (i, h) in enumerate(pairs):
                sig = torch.where(torch.rand(ROPE, generator=g) < 0.5, -1.0, 1.0)
                qi.append(i); qh.append(h); qv.append(sig * a)
                bits = torch.empty(len(rngs)).uniform_(0.0, 12.0, generator=g).tolist()
                jump = int(torch.randint(len(rngs), (1,), generator=g)) if n % 2 else -1
                for j, (lo, hi) in enumerate(rngs):
                    if j == jump and hi - lo > 128:
                        p0 = lo + int(torch.randint(min(128, hi - lo), (1,), generator=g))
                        p1 = lo + 128 + int(torch.randint(hi - lo - 128, (1,), generator=g))
                        for pos, bt_ in ((p0, 2.0), (p1, 2.0 + (7.9 if n % 4 == 1 else 8.1))):
                            pp, off = key_at(b, pos)
                            pidx.append(pp); oidx.append(off); vals.append(sig * 16.0 * bt_ / 12.0)
                    else:
                        pos = lo + int(torch.randint(hi - lo, (1,), generator=g))
                        pp, off = key_at(b, pos)
                        pidx.append(pp); oidx.append(off); vals.append(sig * 16.0 * bits[j] / 12.0)
                taken.setdefault(b, set()).add((i, h))
            if pidx:
                kvf[torch.tensor(pidx, device=device), torch.tensor(oidx, device=device), LATENT:] = \
                    torch.stack(vals).to(device)
                qi_, qh_ = torch.tensor(qi, device=device), torch.tensor(qh, device=device)
                qf[b, qi_, qh_] = 0.0
                qf[b, qi_, qh_, LATENT:] = torch.stack(qv).to(device)
    q8 = qf.clamp(-448, 448).to(FP8)
    kv8 = kvf.clamp(-448, 448).to(FP8)
    return Case(name, q8, kv8, bt.to(device), torch.tensor(loc, dtype=torch.int32, device=device),
                torch.tensor(ctx, dtype=torch.int32, device=device), q_scale, k_scale, rank, min_split)


def split_for(batch: int, floor: Optional[int], q_len: int = QLEN, device="cuda") -> tuple:
    """(split, workspace bytes) the served stock_decode picks (dcp.py:305-319); floor None = fill_splits(batch * q)."""
    from tokenspeed_mla import mla_decode as md
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.mla.dcp import fill_splits
    dev = torch.device(device)
    if floor is None:
        floor = fill_splits(batch * q_len, dev)
    arch = torch.cuda.get_device_capability(dev)
    tiler, _ = md.select_mla_decode_tilers(H, q_len, is_fp8=True, compute_capability=arch)
    rows = (md.compute_q_tile_layout(H, q_len, 128)[1], 128)
    sms = md.get_num_sm(dev)
    return md._get_split_kv_and_workspace_size(batch, *rows, 512, sms, served_max_seq(), FP8, tiler, floor)


def split_of(c: Case) -> tuple:
    return split_for(c.B, c.min_split, c.qlen, c.q.device)


def split_ranges(length: int, split: int, tile: int = 128) -> list:
    """Local-row ranges of each split of one request (get_k_tile_count, mla_decode_fp8.py:2113-2125, window_left -1)."""
    total = -(-length // tile)
    per = -(-total // split) if split else 0
    out = []
    for j in range(split):
        lo = j * per
        n = max(0, min(total, lo + per) - lo)
        out.append((lo * tile, min(length, (lo + n) * tile)) if n else (length, length))
    return out


def run_incumbent(c: Case, out: Optional[torch.Tensor] = None, lse: Optional[torch.Tensor] = None):
    """The served seam, verbatim: dcp.stock_decode (attention + reducer) as prepared.stock_core calls it under DCP."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.mla.dcp import stock_decode
    if out is None:
        out = torch.empty(c.B, c.qlen, H, LATENT, dtype=torch.bfloat16, device=c.q.device)
    if lse is None:
        lse = torch.empty(c.B, c.qlen, H, dtype=torch.float32, device=c.q.device)
    stock_decode(c.q, c.kv.view(torch.uint8), c.block_table, c.lengths, served_max_seq(),
                 (c.softmax_scale, c.k_scale), out, lse, c.split_floor(), True, (c.world, c.rank, 1), c.totals, True)
    split, ws = split_of(c)
    return out, lse, {"split": split, "ws_bytes": ws}


def reference(c: Case):
    """fp64 truth over the dequantized fp8 inputs: out [B,q,H,512], lse2 [B,q,H] (base 2; -inf on rows with no key),
    pmax [B,q,H] = the largest softmax weight of the row (peaked vs flat rows)."""
    B, Q = c.B, c.qlen
    dev = c.q.device
    out = torch.zeros(B, Q, H, LATENT, dtype=torch.float64, device=dev)
    lse = torch.full((B, Q, H), -math.inf, dtype=torch.float64, device=dev)
    pmax = torch.zeros(B, Q, H, dtype=torch.float64, device=dev)
    loc, tot = c.lengths.tolist(), c.totals.tolist()
    for b in range(B):
        L = loc[b]
        if L == 0:
            continue
        npg = -(-L // PAGE)
        k = c.kv[c.block_table[b, :npg].long()].reshape(-1, D)[:L].to(torch.float64) * c.k_scale
        qd = c.q[b].to(torch.float64) * c.q_scale
        s = torch.einsum("qhd,kd->qhk", qd, k) * SCALE
        for i in range(Q):
            bound = min(local_bound(tot[b] - (Q - 1) + i, c.rank, c.world), L)
            if bound <= 0:
                continue
            si = s[i, :, :bound]
            m = si.max(-1, keepdim=True).values
            p = torch.exp(si - m)
            z = p.sum(-1, keepdim=True)
            out[b, i] = (p / z) @ k[:bound, :LATENT]
            lse[b, i] = (m.squeeze(-1) + torch.log(z.squeeze(-1))) / LN2
            pmax[b, i] = 1.0 / z.squeeze(-1)
    return out, lse, pmax


PEAK = 0.4  # page-group needle rows have pmax just under 0.5 (lane K p01judge.py:258)


@dataclasses.dataclass
class Err:
    rel_l2: float       # live rows: ||o - ref|| / ||ref|| (reported; needle rows dominate the norm)
    rel_peak: float     # rows with pmax >= PEAK: ||o - ref|| / ||ref||          (judged, 1.02x)
    err_flat: float     # rows with pmax < PEAK: ||o - ref|| / (sqrt(n) rms(V))  (judged, 1.02x; flat rows carry every split)
    lse_max_abs: float  # live rows, base 2                                     (judged, max(1.02x, 2^-16))
    n_peak: int
    n_flat: int
    n_empty: int
    empty_ok: bool      # rows with no visible key: finite out, and lse <= live min - 64, -inf or NaN (-> EMPTY_LSE)
    finite: bool

    def as_dict(self):
        return dataclasses.asdict(self)


def errors(o, lse, ref, c: Case) -> Err:
    ref_o, ref_lse, pmax = ref
    o = o.to(torch.float64)
    lse = lse.to(torch.float64)
    live = torch.isfinite(ref_lse)
    peak = live & (pmax >= PEAK)
    flat = live & ~peak
    d = o - ref_o

    def rel(m):
        return (d[m].norm() / ref_o[m].norm().clamp_min(1e-300)).item() if m.any() else 0.0

    v_rms = (c.kv[..., :LATENT].to(torch.float64) * c.k_scale).pow(2).mean().sqrt().item()
    n_fl = int(flat.sum().item()) * LATENT
    err_flat = d[flat].norm().item() / (math.sqrt(n_fl) * v_rms) if n_fl else 0.0
    le = (lse[live] - ref_lse[live]).abs()
    dead = ~live
    floor = (ref_lse[live].min().item() if live.any() else 0.0) - 64.0
    # DcpValue (dcp.py:150-157) maps NaN / <= EMPTY_LSE / >= -EMPTY_LSE and every row of a length-0 request to
    # EMPTY_LSE; a finite lse far below the live rows has merge weight ~0. Output must be finite either way.
    lse_dead = lse[dead]
    empty_ok = bool(torch.isfinite(o[dead]).all().item()) and bool(
        ((lse_dead <= floor) | torch.isnan(lse_dead) | torch.isinf(lse_dead)).all().item())
    finite = bool(torch.isfinite(o[live]).all().item()) and bool(torch.isfinite(lse[live]).all().item())
    return Err(rel(live), rel(peak), err_flat, le.max().item() if le.numel() else 0.0, int(peak.sum()),
               int(flat.sum()), int(dead.sum()), empty_ok, finite)


def run(fn, c: Case, out: Optional[torch.Tensor] = None, lse: Optional[torch.Tensor] = None):
    """Call a kernel with the served seam's arguments (prepared.stock_core under DCP, q 8). Fresh out / lse are
    NaN-filled, so a row the kernel never writes fails the judge."""
    if out is None:
        out = torch.full((c.B, c.qlen, H, LATENT), float("nan"), dtype=torch.bfloat16, device=c.q.device)
    if lse is None:
        lse = torch.full((c.B, c.qlen, H), float("nan"), dtype=torch.float32, device=c.q.device)
    fn(c.q, c.kv.view(torch.uint8), c.block_table, c.lengths, served_max_seq(), (c.softmax_scale, c.k_scale), out,
       lse, c.split_floor(), True, (c.world, c.rank, 1), c.totals, True)
    return out, lse


EMPTY_LSE = -3.0e38  # dcp.py:51


def combine(outs: list, lses: list, lens: list):
    """Torch emulation of DcpValue's merge (dcp.py:150-157, 180-227) over the 8 ranks' rows: an invalid / empty source
    (local length 0, NaN, <= EMPTY_LSE, >= -EMPTY_LSE) gets EMPTY_LSE; weight exp2(lse - max) in fp32, normalized;
    merged in fp32 from the bf16 rows, one bf16 rounding. Returns merged [B, q, H, 512] bf16."""
    L = torch.stack([l.float() for l in lses])                       # [R, B, q, H]
    n = torch.stack([ln.to(L.device) for ln in lens])               # [R, B]
    bad = torch.isnan(L) | (L <= EMPTY_LSE) | (L >= -EMPTY_LSE) | (n <= 0)[:, :, None, None]
    L = torch.where(bad, torch.full_like(L, EMPTY_LSE), L)
    top = L.max(0).values
    w = torch.where(L > EMPTY_LSE, torch.exp2(L - top), torch.zeros_like(L))
    tot = w.sum(0)
    sc = torch.where(tot > 0, w / tot, torch.zeros_like(w))
    rows = torch.stack([torch.where((ln.to(o.device) > 0)[:, None, None, None], o.float(), torch.zeros_like(o.float()))
                        for o, ln in zip(outs, lens)])
    return (sc[..., None] * rows).sum(0).to(torch.bfloat16)


def combine_truth(refs: list):
    """Exact fp64 merge of the per-rank fp64 truths (out, lse2): the global attention over the union of the shards."""
    L = torch.stack([r[1] for r in refs])
    top = L.max(0).values
    w = torch.where(torch.isfinite(L), torch.exp2(L - torch.where(torch.isfinite(top), top, torch.zeros_like(top))),
                    torch.zeros_like(L))
    z = w.sum(0)
    o = (w[..., None] * torch.stack([r[0] for r in refs])).sum(0) / z.clamp_min(1e-300)[..., None]
    return o, torch.where(z > 0, torch.log2(z) + top, torch.full_like(z, -math.inf))
