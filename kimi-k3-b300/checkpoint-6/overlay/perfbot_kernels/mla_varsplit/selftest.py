"""Kernel-only self-test of P-14 lever L6 (per-request MLA split-KV vector) on the SERVED call path, one GPU.

  cd <worktree> && PYTHONPATH=overlay python3 -B -m perfbot_kernels.mla_varsplit.selftest     (contract image)

Call: the overlay dcp.stock_decode (vLLM 49cb2bbf cutedsl/mla/dcp.py:281-366 plus `splits`), with the arguments
prepared.stock_core passes under DCP8 (prepared.py:205-226): query [B, 8, 96, 576] FP8 (DCP x 12 local heads),
paged FP8 cache (page 64), lengths = this rank's shard, totals = global lengths, pack=True, cp=(8, r, 1), pdl,
max_seq = local_max(max_model_len, 8), caller-owned out [B, 8, 96, 512] BF16 / lse [B, 8, 96] FP32 (NaN-filled).
Arms per case and rank r:  uni = splits None (the served uniform split, min_split = fill_splits(8B));
                           sgr = uniform at the grid split S_grid = 13 - B (control: same grid, no vector);
                           var = splits = var_splits(rank r's local lengths), an Int32[B] device tensor (L6).
Truth: fp64 causal MLA over the GLOBAL KV (the 8 shards put back in global order) from the dequantized FP8 q / KV;
per-rank fp64 truths over each shard; their exact LSE merge must equal the global truth (shardmap).
Inputs (adapted from P-08 judge v5.2 p08.py:86-234, lane-n-p08 582bb15): FP8 values log-uniform 2^-6..2^8 x 0.02,
random sign; q_scale, k_scale U(0.03, 0.3); shuffled page pool; one q shared by the 8 ranks, one KV draw per rank.
Planted rows: needles (one key per rank, 16 nats) at the first and last visible key and on both sides of every split
boundary of uni, sgr and var; graded rows (one key per S_grid split at 0-12 bits, half with a 2 -> 2 + 7.9 / 8.1 bit
in-split jump across skip_correction_threshold 8.0). Salts: odd lengths, a 9-token request, pad rows (judge.py:79-87).
One JSON line per check (ok null = evidence only):
  overlay   the imported dcp.stock_decode is the overlay copy and takes `splits`
  plan      B3-5: planner split == S_grid at this max_seq (no cap by K tiles or 32); workspace, reducer, grid
  vector    each rank's vector: the expected one (named cases), every split >= 1 and <= S_grid, sum <= the rule's
            units; the device kernel (kernel.produce, the served producer) writes exactly var_splits
  kernel    the device kernel vs var_splits on 6000 random B3 / B4 / B5 local length sets (zeros, ties, 1-2^24)
  shardmap  LSE merge of the 8 per-rank fp64 truths == the global fp64 truth (1e-9)
  rank      per rank r: var <= 1.02 x env + 1e-6 on rel_l2 (live rows) and err_flat (rows with pmax < 0.4, in V rms
            units), LSE max abs (log2) <= max(1.02 x env, 2^-16), env = max(uni, sgr) (P-08 envelope rule,
            judge.py:9-17); live rows finite; rows with no visible key merge-equivalent (p08.py:346-352); rows of
            length-0 requests bit-equal to uni (the served behaviour)
  combined  the 8 ranks merged as DcpValue does (dcp.py:150-227, emulated as p08.py:373-387) vs the global truth,
            the same rule on rel_l2 / err_flat; evidence: rel-L2 per (request, q row), var / env max and where
  graph     capture one var call; rewrite the splits buffer in place to another valid vector; replay: bit-equal to
            an eager call at the new vector, and not bit-equal to the replay at the old one
  order     01d (letter O): arm ord = var through core.py's split kernel (unit table + empty-unit exit), reading the
            table the device kernel writes after the vector: bit-equal to var on every case and rank; the device
            table == unit_table on the 6000 kernel sets; graph-ord: rewrite vector + table in place 3 times,
            each replay == eager; salt: the ord call while N single-CTA spin kernels hold N SMs (N in SALT_SMS), bit-equal to var
  time      evidence, rank 0: median of N graph replays with an L2 flush before each, uni / sgr / var / ord (us)
The per-(request, q row) ratio is reported, not judged: with three arms of equal noise P(var > max(uni, sgr)) = 1/3
per row, so a per-row 1.02 bar fails a correct kernel; P-08 judges per metric over the case for the same reason.
Env: MLA_VARSPLIT_MAX_MODEL_LEN (1048576: P-08 p08.py:31, R-UP-2b models-after-load.json [u here]),
MLA_VARSPLIT_RANKS (0,1,...,7; a subset skips `combined` and exits 2), MLA_VARSPLIT_ITERS (50).
Runtime [e]: 15-45 min, mostly 40 kernel compiles (cp_rank is a compile-time constant: 8 ranks x uni / sgr x2 / var
x2 reducer shapes); the fp64 truths take ~1-3 min on B300.
Exit 0 PASS, 1 FAIL, 2 incomplete (rank subset).
"""

import inspect
import json
import math
import os
import statistics
import sys
import time
import traceback

from unittest import mock

import torch

import perfbot_kernels.mla_varsplit as vs
from perfbot_kernels.mla_varsplit import DOMAIN, GATED, rule, s_grid, served, unit_table, var_splits
from perfbot_kernels.mla_varsplit.kernel import produce

W, H, Q, D, LAT, ROPE, PAGE, TILE = 8, 96, 8, 576, 512, 64, 64, 128
INTER = 1  # served cp_kv_cache_interleave_size (P-08 p08.py:14)
SCALE = 192 ** -0.5  # qk_nope 128 + rope 64 (attention.scale; p08.py:28)
FP8, F64, LN2 = torch.float8_e4m3fn, torch.float64, math.log(2.0)
EMPTY_LSE = -3.0e38  # dcp.py:51
TOL, FLOOR, LSE_FLOOR, PEAK, GRADED = 1.02, 1e-6, 2.0 ** -16, 0.4, 32
MAX_MODEL_LEN = int(os.environ.get("MLA_VARSPLIT_MAX_MODEL_LEN", 1 << 20))
RANKS = [int(x) for x in os.environ.get("MLA_VARSPLIT_RANKS", ",".join(map(str, range(W)))).split(",")]
ITERS = int(os.environ.get("MLA_VARSPLIT_ITERS", 50))
ARMS = ("uni", "sgr", "var")
K = 1000
CASES = [  # (name, global ctx per request incl. the 8 new tokens, the vector expected on every rank or None)
    ("b5", [345 * K] + [88 * K] * 4, (4, 2, 2, 2, 2)),
    ("b4", [386 * K] + [88 * K] * 3, (6, 3, 3, 3)),  # two-wave rule (item-01b): 15 units, cap 6
    ("b4-equal", [200 * K] * 4, (3, 3, 3, 3)),  # gate: one wave (run3 two-wave 1.40x)
    ("b4-pads", [386 * K, 0, 0, 0], (6, 1, 1, 1)),
    ("b3", [334 * K, 88 * K, 87 * K], (8, 4, 4)),  # two-wave rule (01c): 16 units, cap 8
    ("b5-equal", [200 * K] * 5, (3, 3, 2, 2, 2)),
    ("b5-pads", [345 * K, 0, 0, 0, 0], (8, 1, 1, 1, 1)),
    ("b5-salt", [345_017, 88_003, 9, 0, 87_995], None),  # unequal shards, rows with no key on some ranks, a pad
    ("b3-item10", [334 * K, 115 * K, 60 * K], (8, 4, 4)),  # item 10 DESIGN.md section 10 band (appended: seeds kept)
    ("b3-equal", [200 * K] * 3, (4, 4, 4)),  # gate: one wave
]
# Seed sweep (01c): case idx -> seed 1400 + 17 x (idx + SEED_SHIFT); CASE_FILTER runs only the named cases.
SEED_SHIFT = int(os.environ.get("MLA_VARSPLIT_SEED_SHIFT", 0))
CASE_FILTER = [c for c in os.environ.get("MLA_VARSPLIT_CASES", "").split(",") if c]
GRAPH_CASE = "b5"
ORD = os.environ.get("MLA_VARSPLIT_ORDER", "1") == "1"  # the 01d arm
SALT_SMS = (0, 8, 37, 74)  # SMs held by spin kernels during the salted ord calls
# Direction reading (coordinator 2026-10-19): a two-wave vector at B3 / B4, the long request at twice the served split
# (B4 15 units = 180 CTAs, B3 16 units = 192 CTAs), judged at the same envelope; arm "two", evidence for item-01b.
TWO: dict = {}  # B3 (8, 4, 4) is now the var arm (01c)
OKS: list = []


def emit(check, ok, **kw):
    OKS.append(ok)
    print(json.dumps(dict(check=check, ok=ok, **kw)), flush=True)


def local_bound(g, r):
    """This rank's keys among global positions [0, g) (_get_dcp_local_bound; p08.py:35-45)."""
    if g <= 0:
        return 0
    full, off = divmod(g, W * INTER)
    return full * INTER + min(max(off - r * INTER, 0), INTER)


def shard_positions(n, r, device):
    """Global positions of rank r's local keys 0..n-1 (the inverse of local_bound's interleave)."""
    l = torch.arange(n, device=device)
    return (l // INTER) * W * INTER + r * INTER + l % INTER


def ranges(n, split):
    """Local key ranges of the splits of one request (get_k_tile_count, mla_decode_fp8.py:2109-2126)."""
    total = -(-n // TILE)
    per = -(-total // split)
    out = []
    for j in range(split):
        lo = j * per
        cnt = max(0, min(total, lo + per) - lo)
        out.append((lo * TILE, min(n, (lo + cnt) * TILE)) if cnt else (n, n))
    return out


def draw(gen, *shape, device):  # p08.py:128-131
    mag = torch.exp2(torch.empty(*shape, device=device).uniform_(-6.0, 8.0, generator=gen))
    return mag * torch.where(torch.rand(*shape, device=device, generator=gen) < 0.5, -1.0, 1.0) * 0.02


class Case:
    """One q (shared by the 8 ranks) and 8 KV shards (shuffled page pools), needles and graded rows planted."""

    def __init__(self, idx, name, ctx, uni_split, dev):
        self.name, self.ctx, self.B, self.dev = name, ctx, len(ctx), dev
        self.sgrid = s_grid(self.B)
        g = torch.Generator().manual_seed(1400 + 17 * idx)
        self.qs, self.ks = (float(torch.empty(1).uniform_(0.03, 0.3, generator=g)) for _ in range(2))
        s_eff = SCALE * self.qs * self.ks
        c8 = float(torch.tensor(1.0 / (s_eff * ROPE)).to(FP8).float())  # q rope magnitude of a planted row
        self.lens = [[local_bound(t, r) for t in ctx] for r in range(W)]
        self.vec = [var_splits(self.lens[r]) for r in range(W)]
        vmin = [[min(local_bound(t - (Q - 1) + i, r) for i in range(Q)) for t in ctx] for r in range(W)]

        def cuts(b, r):  # first / last visible key, both sides of every split boundary of uni, sgr and var
            n, v, out = self.lens[r][b], vmin[r][b], {0, vmin[r][b] - 1}
            for s in (uni_split, self.sgrid, self.vec[r][b]):
                out |= {p for lo, _ in ranges(n, s)[1:] for p in (lo - 1, lo)}
            return sorted(p for p in out if 0 <= p < v)

        def rope_key(sig, nats):  # a key scoring `nats` against its row: 16s then one residual dim
            mag = torch.zeros(ROPE, device=dev)
            total = nats / (s_eff * c8)
            n = min(ROPE - 1, int(total // 16))
            mag[:n], mag[n] = 16.0, total - 16.0 * n
            return sig * mag

        qf = draw(torch.Generator(device=dev).manual_seed(1400 + 17 * idx), self.B, Q, H, D, device=dev)
        groups = []  # per request: needle count, [(i, h)], signs [n, 64]
        for b in range(self.B):
            nn = max((len(cuts(b, r)) for r in range(W) if vmin[r][b] >= 2), default=0)
            rows = [divmod(x, H) for x in torch.randperm(Q * H, generator=g)[: nn + GRADED].tolist()]
            sig = torch.where(torch.rand(len(rows), ROPE, generator=g) < 0.5, -1.0, 1.0).to(dev)
            for k, (i, h) in enumerate(rows):
                qf[b, i, h, :LAT], qf[b, i, h, LAT:] = 0.0, sig[k] * c8
            groups.append((nn, rows, sig))
        self.q = qf.clamp(-448, 448).to(FP8)
        self.totals = torch.tensor(ctx, dtype=torch.int32, device=dev)
        self.shards = []  # per rank: (kv uint8 [pages, 64, 576], block table int32, local lengths int32)
        for r in range(W):
            npg = [max(1, -(-n // PAGE)) for n in self.lens[r]]
            perm = torch.randperm(sum(npg) + 3, generator=g)
            bt, o = torch.zeros(self.B, max(npg), dtype=torch.int32), 0
            for b, t in enumerate(ctx):
                if t:  # a pad request keeps a row of page 0 (prepared.pad_requests)
                    bt[b, : npg[b]], bt[b, npg[b]:] = perm[o: o + npg[b]], perm[-1]
                    o += npg[b]
            kvf = draw(torch.Generator(device=dev).manual_seed(31 * idx + r), len(perm), PAGE, D, device=dev)
            pid, off, val = [], [], []
            for b in range(self.B):
                n, v, (nn, rows, sig) = self.lens[r][b], vmin[r][b], groups[b]
                if v < 2:
                    continue
                taken = set()

                def put(p, vec):
                    taken.add(p)
                    pg, o_ = divmod(p, PAGE)
                    pid.append(int(bt[b, pg]))
                    off.append(o_)
                    val.append(vec)

                pos = cuts(b, r)
                pos += [p for p in torch.randperm(v, generator=g)[: 2 * nn].tolist() if p not in pos]
                for k, p in enumerate(pos[:nn]):
                    put(p, sig[k] * 16.0)
                rng = [(lo, min(hi, v)) for lo, hi in ranges(n, self.sgrid) if min(hi, v) - lo >= 2]
                for k in range(nn, len(rows)):  # graded rows (p08.py:198-230)
                    bits = torch.empty(len(rng)).uniform_(0.0, 12.0, generator=g).tolist()
                    jump = int(torch.randint(len(rng), (1,), generator=g)) if k % 2 and rng else -1
                    for j, (lo, hi) in enumerate(rng):
                        pick = lambda m: lo + int(torch.randint(m, (1,), generator=g))  # noqa: E731
                        pts = [(pick(hi - lo), bits[j])]
                        if j == jump and hi - lo > TILE:
                            hop = 7.9 if k % 4 == 1 else 8.1
                            pts = [(pick(TILE), 2.0), (pick(hi - lo - TILE) + TILE, 2.0 + hop)]
                        for p, bt_ in pts:
                            if p not in taken:
                                put(p, rope_key(sig[k], bt_ * LN2))
            if pid:
                kvf[torch.tensor(pid, device=dev), torch.tensor(off, device=dev), LAT:] = torch.stack(val)
            kv = kvf.clamp(-448, 448).to(FP8).view(torch.uint8)
            self.shards.append((kv, bt.to(dev), torch.tensor(self.lens[r], dtype=torch.int32, device=dev)))

    def keys(self, r, b):
        """Rank r's local keys of request b, dequantized to fp64 [n, 576]."""
        kv, bt, _ = self.shards[r]
        n = self.lens[r][b]
        rows = kv.view(FP8)[bt[b, : -(-n // PAGE)].long()].reshape(-1, D)[:n]
        return rows.float().to(F64) * self.ks

    def truth(self, r=None):
        """fp64 (out [B, Q, H, 512], lse log2 [B, Q, H], pmax): rank r's shard, or the global KV (r None)."""
        B, dev = self.B, self.dev
        o = torch.zeros(B, Q, H, LAT, dtype=F64, device=dev)
        lse = torch.full((B, Q, H), -math.inf, dtype=F64, device=dev)
        pmax = torch.zeros(B, Q, H, dtype=F64, device=dev)
        for b, t in enumerate(self.ctx):
            if r is None:
                kd = torch.empty(t, D, dtype=F64, device=dev)
                for s in range(W):
                    kd[shard_positions(self.lens[s][b], s, dev)] = self.keys(s, b)
                vis = [t - (Q - 1) + i for i in range(Q)]
            else:
                kd = self.keys(r, b)
                vis = [min(local_bound(t - (Q - 1) + i, r), self.lens[r][b]) for i in range(Q)]
            qd = self.q[b].float().to(F64) * self.qs
            for i, v in enumerate(vis):
                if v <= 0:
                    continue
                s_ = (qd[i] @ kd[:v].T) * SCALE
                m = s_.amax(-1, keepdim=True)
                p = torch.exp(s_ - m)
                z = p.sum(-1, keepdim=True)
                o[b, i] = (p / z) @ kd[:v, :LAT]
                lse[b, i] = (m + torch.log(z)).squeeze(-1) / LN2
                pmax[b, i] = 1.0 / z.squeeze(-1)
            del kd
        return o, lse, pmax

    def vrms(self):  # judge.py _vrms
        return math.sqrt(statistics.mean(
            ((kv.view(FP8)[..., :LAT].float().to(F64) * self.ks) ** 2).mean().item() for kv, _, _ in self.shards))


def merge_truth(refs):
    """Exact fp64 LSE merge of per-rank truths (p08.py:390-398)."""
    L = torch.stack([x[1] for x in refs])
    top = L.max(0).values
    top0 = torch.where(torch.isfinite(top), top, torch.zeros_like(top))
    w = torch.where(torch.isfinite(L), torch.exp2(L - top0), torch.zeros_like(L))
    z = w.sum(0)
    o = (w[..., None] * torch.stack([x[0] for x in refs])).sum(0) / z.clamp_min(1e-300)[..., None]
    return o, torch.where(z > 0, torch.log2(z) + top0, torch.full_like(z, -math.inf))


def dcp_merge(outs, lses, lens):
    """DcpValue's merge of the 8 ranks' rows (dcp.py:150-157, 180-227; emulated as p08.py:373-387)."""
    L = torch.stack([x.float() for x in lses])
    n = torch.stack(lens)
    bad = torch.isnan(L) | (L <= EMPTY_LSE) | (L >= -EMPTY_LSE) | (n <= 0)[:, :, None, None]
    L = torch.where(bad, torch.full_like(L, EMPTY_LSE), L)
    w = torch.where(L > EMPTY_LSE, torch.exp2(L - L.max(0).values), torch.zeros_like(L))
    tot = w.sum(0)
    sc = torch.where(tot > 0, w / tot, torch.zeros_like(w))
    rows = torch.stack([torch.where((ln > 0)[:, None, None, None], o.float(), torch.zeros_like(o.float()))
                        for o, ln in zip(outs, lens)])
    return (sc[..., None] * rows).sum(0).to(torch.bfloat16)


def errors(o, lse, ref, vrms):
    """The metrics of one arm on one view: rel_l2 (live rows), err_flat (live rows with pmax < 0.4, V rms units),
    lse_max_abs (log2), rel-L2 per (request, q row), finite live rows, merge-equivalent dead rows."""
    ro, rl, pm = ref
    live = torch.isfinite(rl)
    flat = live & (pm < PEAK)
    o = o.to(F64)
    sq, rsq = (o - ro).pow(2).sum(-1), ro.pow(2).sum(-1)  # [B, Q, H]
    nfl = int(flat.sum()) * LAT
    e = dict(rel_l2=(sq[live].sum() / rsq[live].sum().clamp_min(1e-300)).sqrt().item() if live.any() else 0.0,
             err_flat=sq[flat].sum().sqrt().item() / (math.sqrt(nfl) * vrms) if nfl else 0.0,
             row=((sq * live).sum(-1) / (rsq * live).sum(-1).clamp_min(1e-300)).sqrt(),
             finite=bool(torch.isfinite(o[live]).all()), empty_ok=True)
    if lse is not None:
        l = lse.to(F64)
        e["lse_max_abs"] = (l[live] - rl[live]).abs().max().item() if live.any() else 0.0
        e["finite"] &= bool(torch.isfinite(l[live]).all())
        dead, floor = ~live, (rl[live].min().item() if live.any() else 0.0) - 64.0
        e["empty_ok"] = bool(torch.isfinite(o[dead]).all()) and bool(
            ((l[dead] <= floor) | torch.isnan(l[dead]) | torch.isinf(l[dead])).all())
    return e


def judge(check, errs, live_rows, extra_ok=True, **where):
    """var <= 1.02 x max(uni, sgr) (+1e-6) per metric, LSE floor 2^-16; the per-row ratio is evidence."""
    v = errs["var"]
    ms = [m for m in ("rel_l2", "err_flat", "lse_max_abs") if m in v]
    env = {m: max(errs["uni"][m], errs["sgr"][m]) for m in ms}
    bound = {m: max(TOL * env[m], LSE_FLOOR) if m == "lse_max_abs" else TOL * env[m] + FLOOR for m in ms}
    ratio = torch.where(live_rows, v["row"] / torch.maximum(errs["uni"]["row"], errs["sgr"]["row"]).clamp_min(FLOOR),
                        torch.zeros_like(v["row"]))
    worst = divmod(int(ratio.argmax()), Q)
    ok = all(v[m] <= bound[m] for m in ms) and v["finite"] and v["empty_ok"] and extra_ok
    evidence = check.endswith("-two")  # the item-01b direction arm: its envelope result is printed, not judged
    emit(check, None if evidence else ok, envelope_ok=ok, **where, **{f"{a}_{m}": errs[a][m] for a in ARMS for m in ms}, bound=bound,
         finite=v["finite"], empty_ok=v["empty_ok"], row_ratio_max=ratio.max().item(), worst_req_row=worst,
         var_row_rel_l2_max=v["row"].max().item())
    return ok


def bits_equal(a, b):
    return all(torch.equal(x.view(torch.int16 if x.element_size() == 2 else torch.int32),
                           y.view(torch.int16 if y.element_size() == 2 else torch.int32)) for x, y in zip(a, b))


def capture(fn):
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def main():  # noqa: C901
    dev = torch.device("cuda", torch.cuda.current_device())
    from tokenspeed_mla import mla_decode as md
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.mla import dcp

    ok = "splits" in inspect.signature(dcp.stock_decode).parameters
    emit("overlay", ok, file=dcp.__file__, device=torch.cuda.get_device_name(dev), ranks=RANKS)
    if not ok:
        return 1
    stock, max_seq = dcp.stock_decode, dcp.local_max(MAX_MODEL_LEN, W)
    arch, sms = torch.cuda.get_device_capability(dev), md.get_num_sm(dev)
    tiler, _ = md.select_mla_decode_tilers(H, Q, is_fp8=True, compute_capability=arch)
    tiles = md.compute_q_tile_layout(H, Q, 128)[1]
    uni_split = {}
    for B in (3, 4, 5):  # the scalar split the overlay asserts (mla_varsplit.compiler) must be S_grid exactly
        split_of = lambda floor: md._get_split_kv_and_workspace_size(  # noqa: E731
            B, tiles, 128, 512, sms, max_seq, FP8, tiler, floor)
        (su, _), (sv, ws) = split_of(dcp.fill_splits(B * Q, dev)), split_of(s_grid(B))
        uni_split[B] = su
        emit("plan", sv == s_grid(B) and ws < 1 << 31, B=B, in_domain=(B, Q) in DOMAIN, sms=sms, max_model_len=MAX_MODEL_LEN,
             max_seq=max_seq, tiler=list(tiler), served_split=su, s_grid=s_grid(B), split=sv, ws_bytes=ws,
             reducer_max_splits=md._get_reducer_max_splits(sv), reducer_d_tiles=md._get_reducer_d_tiles(
                 B, Q, H, sms, sv), grid=[2, tiles * B, sv], ctas=2 * tiles * B * sv)
    gen, bad = torch.Generator().manual_seed(14), []
    for B in (3, 4, 5):  # the device greedy (and 01d table) == the host reference (zeros, ties, short and long shards)
        out = torch.zeros(B, dtype=torch.int32, device=dev)
        tab = torch.zeros(B * (1 + s_grid(B)), dtype=torch.int32, device=dev)
        for i in range(2000):
            top = 1 << int(torch.randint(1, 25, (1,), generator=gen))
            lens = torch.randint(0, top, (B,), generator=gen)
            lens = torch.where(torch.rand(B, generator=gen) < 0.2, 0, lens)
            if i % 7 == 0:
                lens[1:] = lens[0]  # ties
            ref = var_splits(lens.tolist())
            buf = tab if i % 2 else out
            produce(lens.to(torch.int32).to(dev), buf, rule(B), served(B), B in GATED)
            got, want = buf.tolist(), list(ref) + (unit_table(lens.tolist(), ref) if i % 2 else [])
            if got != want:
                bad.append((lens.tolist(), got, want))
    emit("kernel", not bad, sets=3 * 2000, table_sets=3 * 1000, mismatches=len(bad), first=bad[:3])
    sets = [[43_125] + [11_000] * 4, [25_000] * 5, [43_125, 0, 0, 0, 0], [0] * 5]
    lens_buf, out = torch.tensor(sets[0], dtype=torch.int32, device=dev), torch.zeros(5, dtype=torch.int32, device=dev)
    graph = capture(lambda: produce(lens_buf, out, rule(5), served(5), 5 in GATED))  # the step's lengths change in place
    got = []
    for lens in sets:
        lens_buf.copy_(torch.tensor(lens, dtype=torch.int32))
        graph.replay()
        torch.cuda.synchronize()
        got.append((tuple(out.tolist()), var_splits(lens)))
    emit("kernel-graph", all(x == y for x, y in got), replays=got)
    del graph
    flush = torch.empty(max(2 * getattr(torch.cuda.get_device_properties(dev), "L2_cache_size", 0), 256 << 20),
                        dtype=torch.uint8, device=dev)
    for idx, (name, ctx, want) in enumerate(CASES):
        if CASE_FILTER and name not in CASE_FILTER:
            continue
        t0 = time.time()
        c = Case(idx + SEED_SHIFT, name, ctx, uni_split[len(ctx)], dev)
        B, served_floor = c.B, dcp.fill_splits(len(ctx) * Q, dev)
        bufs, obufs = {}, {}
        for r in range(W):  # the var arm reads the vector the served producer (the device kernel) writes
            v, bufs[r] = c.vec[r], torch.zeros(B, dtype=torch.int32, device=dev)
            produce(c.shards[r][2], bufs[r], rule(B), served(B), B in GATED)
            obufs[r] = torch.zeros(B * (1 + c.sgrid), dtype=torch.int32, device=dev)  # 01d: vector + unit table
            produce(c.shards[r][2], obufs[r], rule(B), served(B), B in GATED)
            dv = tuple(bufs[r].tolist())
            ok = (want is None or v == want) and dv == v and min(v) >= 1 and max(v) <= c.sgrid
            ok &= obufs[r].tolist() == list(v) + unit_table(c.lens[r], v)
            emit("vector", ok and sum(v) <= rule(B)[0], case=name, rank=r, lens=c.lens[r], vector=v,
                 device_vector=dv, want=want, s_grid=c.sgrid)
        arms = ARMS + (("two",) if name in TWO else ()) + (("ord",) if ORD else ())
        two = torch.tensor(TWO.get(name, c.vec[0]), dtype=torch.int32, device=dev)

        def call(arm, r, out, lse, splits=None):
            kv, bt, lens = c.shards[r]
            floor = c.sgrid if arm == "sgr" else served_floor
            if arm == "var":
                splits = bufs[r] if splits is None else splits
            if arm == "two":
                splits = two
            if arm == "ord":
                splits = (obufs[r] if splits is None else splits)[:B]
            with mock.patch.object(vs, "ORDER", arm == "ord"):
                stock(c.q, kv, bt, lens, max_seq, (SCALE * c.qs * c.ks, c.ks), out, lse, min_split=floor, pdl=True,
                      cp=(W, r, INTER), totals=c.totals, pack=True, splits=splits)

        def salted(r, n):
            """The ord call while n single-CTA spin kernels (side streams) hold n SMs."""
            o_, side = fresh(), [torch.cuda.Stream(dev) for _ in range(n)]
            for st in side:
                st.wait_stream(torch.cuda.current_stream(dev))
                with torch.cuda.stream(st):
                    torch.cuda._sleep(2_000_000)
            call("ord", r, *o_)
            for st in side:
                torch.cuda.current_stream(dev).wait_stream(st)
            return o_

        def fresh():
            return (torch.full((B, Q, H, LAT), float("nan"), dtype=torch.bfloat16, device=dev),
                    torch.full((B, Q, H), float("nan"), dtype=torch.float32, device=dev))

        glob, vr = c.truth(), c.vrms()
        loc = [c.truth(r) for r in range(W)]
        mo, ml = merge_truth(loc)
        live = torch.isfinite(glob[1])
        dev_o = ((mo - glob[0])[live].abs().max() / glob[0][live].abs().max()).item() if live.any() else 0.0
        dev_l = (ml - glob[1])[live].abs().max().item() if live.any() else 0.0
        emit("shardmap", dev_o < 1e-9 and dev_l < 1e-9 and torch.equal(torch.isfinite(ml), live), case=name,
             o_rel_dev=dev_o, lse_dev=dev_l, ref_s=round(time.time() - t0, 1))
        outs = {}
        for r in RANKS:
            t1 = time.time()
            outs[r] = {a: fresh() for a in arms}
            for a in arms:
                call(a, r, *outs[r][a])
            torch.cuda.synchronize()
            errs = {a: errors(*outs[r][a], loc[r], vr) for a in arms}
            if "two" in arms:
                judge("rank-two", dict(errs, var=errs["two"]), torch.isfinite(loc[r][1]).any(-1), case=name, rank=r,
                      vector=TWO[name])
            empty = [b for b in range(B) if c.lens[r][b] == 0]
            same = all(bits_equal([outs[r]["var"][0][b], outs[r]["var"][1][b]],
                                  [outs[r]["uni"][0][b], outs[r]["uni"][1][b]]) for b in empty)
            lv = torch.isfinite(loc[r][1])
            gap = (outs[r]["var"][1].double() - outs[r]["uni"][1].double())[lv].abs().max().item() if lv.any() else 0
            errs.pop("ord", None)
            judge("rank", errs, lv.any(-1), extra_ok=same, case=name, rank=r, vector=c.vec[r],
                  lse_var_minus_uni_max=gap, empty_requests=empty, empty_bit_equal=same,
                  call_s=round(time.time() - t1, 1))
            if ORD:
                salt = {n: bits_equal(salted(r, n), outs[r]["var"]) for n in SALT_SMS}
                torch.cuda.synchronize()
                emit("order", bits_equal(outs[r]["ord"], outs[r]["var"]) and all(salt.values()), case=name, rank=r,
                     vector=c.vec[r], table=obufs[r][B:].tolist(), ord_eq_var=bits_equal(outs[r]["ord"], outs[r]["var"]),
                     salt_eq_var=salt)
        if len(RANKS) == W:
            lens = [c.shards[r][2] for r in range(W)]
            merged = {a: dcp_merge([outs[r][a][0] for r in range(W)], [outs[r][a][1] for r in range(W)], lens)
                      for a in arms}
            errs = {a: errors(merged[a], None, glob, vr) for a in arms if a != "ord"}
            judge("combined", errs, live.any(-1), case=name, vectors=c.vec)
            if "two" in arms:
                judge("combined-two", dict(errs, var=errs["two"]), live.any(-1), case=name, vector=TWO[name])
        if name == GRAPH_CASE:  # the per-step write of the splits buffer reaches a captured graph
            r = RANKS[0]
            va = c.vec[r]
            vb = va[1:] + va[:1]
            buf = torch.tensor(va, dtype=torch.int32, device=dev)
            g_out, e_a, e_b = fresh(), fresh(), fresh()
            call("var", r, *e_a, splits=buf)
            graph = capture(lambda: call("var", r, *g_out, splits=buf))
            graph.replay()
            torch.cuda.synchronize()
            replay_a = [x.clone() for x in g_out]
            buf.copy_(torch.tensor(vb, dtype=torch.int32))
            graph.replay()
            call("var", r, *e_b, splits=buf)
            torch.cuda.synchronize()
            ok_a, ok_b, moved = bits_equal(replay_a, e_a), bits_equal(g_out, e_b), not bits_equal(g_out, replay_a)
            emit("graph", ok_a and ok_b and moved and va != vb, case=name, rank=r, old=va, new=vb,
                 replay_eq_eager_old=ok_a, replay_eq_eager_new=ok_b, output_moved=moved)
            del graph
        if name == GRAPH_CASE and ORD:  # the per-step write of vector + table reaches a captured ord graph
            r = RANKS[0]
            la = c.lens[r]
            steps = [la] + [la[k:] + la[:k] for k in (1, 2)] + [la[::-1]]  # the lengths move: vector and table move
            rows = [list(var_splits(x)) + unit_table(x, var_splits(x)) for x in steps]
            buf = torch.tensor(rows[0], dtype=torch.int32, device=dev)
            g_out = fresh()
            graph = capture(lambda: call("ord", r, *g_out, splits=buf))
            res = []
            for row in rows:
                buf.copy_(torch.tensor(row, dtype=torch.int32))
                graph.replay()
                e_ = fresh()
                call("ord", r, *e_, splits=buf)
                torch.cuda.synchronize()
                res.append(bits_equal(g_out, e_))
            emit("graph-ord", all(res) and len({tuple(x) for x in rows}) > 2, case=name, rank=r, rows=rows,
                 replay_eq_eager=res)
            del graph
        us, r = {}, RANKS[0]
        for a in arms if ITERS else ():  # ITERS 0: correctness only (timing runs take the node's timing lock)
            o_ = fresh()
            graph = capture(lambda a=a, o_=o_: call(a, r, *o_))
            ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                  for _ in range(ITERS + 10)]
            for e0, e1 in ev:
                flush.zero_()
                e0.record()
                graph.replay()
                e1.record()
            torch.cuda.synchronize()
            us[a] = statistics.median(e0.elapsed_time(e1) * 1e3 for e0, e1 in ev[10:])
            del graph
        if ITERS:
            if "two" in us:
                emit("time-two", None, case=name, rank=r, vector=TWO[name], two_over_uni=us["two"] / us["uni"])
            emit("time", None, case=name, rank=r, vector=c.vec[r], us=us, var_over_uni=us["var"] / us["uni"],
                 var_over_sgr=us["var"] / us["sgr"], ord_over_var=us["ord"] / us["var"] if "ord" in us else None,
                 iters=ITERS, case_s=round(time.time() - t0, 1))
        del c, outs, glob, loc
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    t_start = time.time()
    try:
        rc = main()
    except Exception:  # noqa: BLE001
        emit("error", False, trace=traceback.format_exc()[-3000:])
        rc = 1
    judged = [x for x in OKS if x is not None]
    passed = rc == 0 and judged and all(judged)
    incomplete = passed and len(RANKS) != W
    word = "INCOMPLETE" if incomplete else "PASS" if passed else "FAIL"
    print(f"PERFBOT-MLA-VARSPLIT {word} checks={len(judged)} failed={judged.count(False)} "
          f"seconds={round(time.time() - t_start)}", flush=True)
    sys.exit(2 if incomplete else 0 if passed else 1)
