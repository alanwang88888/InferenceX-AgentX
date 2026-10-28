"""Split-KV choice of the tokenspeed MLA decode kernel (expedition line D, item D-split).

CPU-importable (torch only, no CUDA): the case grid, the workspace formula, the wave cost model (model_split), and
split_kv, the served split (SERVED_SPLIT table, else the planner; install() behind PERFBOT_EXP_SPLITKV=1). The GPU driver is sweep.py.

Where the split is chosen (image tokenspeed_mla, mla_decode.py): tokenspeed_mla_decode calls the module global
_get_split_kv_and_workspace_size (:503) -> get_split_kv_simplified on the M128 path (:82; fp16 file :1542):
split = min(32, max(1, num_sm // B // (2 S))), blind to the context length. The value reaches the kernel only as
the runtime scalar Int32(split_kv) (:651); the compiled kernel is keyed on is_workspace_size_zero (split 1 or not),
not on the split. So the planner function is the one hook point: hook() wraps it.

Grid (mla_helpers.py:248-251, non-persistent since is_var_seq=True): (2, B * S, split) CTAs, a 2-CTA cluster per
(request, query token) (mla_decode.py:182: M128 -> cluster (2, 1, 1)); S = q_len / fold, fold = 1 at 96 heads
(get_mla_decode_fold_sq_factor: 128 // 96 = 1). A CTA walks ceil(tiles / split) k tiles of 128 keys
(mla_decode_fp8.py:1845), so the effective split is ceil(tiles / ceil(tiles / split)) (:1701-1702).
Cost model: time ~ waves(split) x (ceil(tiles / split) + K0), waves = ceil(CTAs / num_sm); K0 = per-CTA fixed cost
(prologue, partial write, reduction share) in k tiles, fitted by the sweep (fit()).
"""

import functools
import math
import statistics

from perfbot_kernels.mla_absorb.reftest import ts_plan  # host mirror of the library planner

W, HL, L, R = 8, 12, 512, 64
H = W * HL
NUM_SM = 148
CC = (10, 3)  # sm_103 (B300): M128 for every shape (the M64 tiler is sm_100 only)
CLUSTER = 2  # CTAs per (request, query token) tile on the M128 path
TILE_N = 128  # keys per k tile (select_mla_decode_tilers: qk tiler (128, 128))
MAX_SPLIT = 32  # the planner's cap on M128; the reduction kernel holds MAX_SPLITS = 256 (mla_helpers.py:331)
K0_TILES = 2  # per-CTA fixed cost in k tiles; R-split (B1 split 9: 36.6 us, B5 split 1: 293 us) gives T / 63, ~1.2 at T 78
IMAGE_Q = 8  # vllm tokenspeed_mla._TOKENSPEED_MAX_Q_LEN: the served workspace covers q_len <= 8
CASE_Q, CASE_B, CASE_CTX = [1, 8, 15], [1, 2, 4, 5, 6, 8], [30000, 100000, 300000]
FORCED = [1, 2, 4, 8, 16, 32]
GATE_RATIO = 1.02
COMP_SPLITS = [2, 3, 4, 6, 8]  # forced arms of the composition sweep (GUARDRAILS "Split-kv selection, amended")
COMP_TOL, COMP_KEEP = 0.01, 10  # served compositions: same request set within 1 % per ctx; <= 10 kept per k
SERVED_SPLIT = {(8, 3): 8, (8, 4): 8, (8, 5): 8, (8, 6): 8}  # (q_len, B) -> split; D-split-02 picks, see split_kv
MAX_FORCED_SPLIT = 8  # the served table forces no split above the composition sweep's candidate edge (COMP_SPLITS)
FORCE_BOUND = 768  # B * q_len * split of any forced entry: D-split-02 isolation crashed at 1536 (B 6, s 32), passed 768


def check_served(table) -> dict:
    """Import-time contract of a served table: every entry forces 2 <= split <= MAX_FORCED_SPLIT with
    B * q_len * split <= FORCE_BOUND; raises ValueError otherwise."""
    bad = {k: v for k, v in table.items() if not (2 <= v <= MAX_FORCED_SPLIT and k[1] * k[0] * v <= FORCE_BOUND)}
    if bad:
        raise ValueError(f"served split table out of bounds (split 2..{MAX_FORCED_SPLIT}, B*q*s <= {FORCE_BOUND}): {bad}")
    return table


check_served(SERVED_SPLIT)


def workspace_bytes(B, q_len, split, heads=H, kv_lora=L) -> int:
    """get_workspace_size (fp16 :3555 / fp8 :4184): fp32 partial O and LSE per split; 0 at split 1."""
    return 0 if split == 1 else B * heads * q_len * split * (kv_lora + 1) * 4


def image_workspace(num_sm=NUM_SM, heads=H, kv_lora=L) -> int:
    """The served buffer (vllm tokenspeed_mla._get_workspace): num_sm * heads * 8 * (L + 1) * 4 bytes (233 MB)."""
    return num_sm * heads * IMAGE_Q * (kv_lora + 1) * 4


def tiles(ctx, q_len, cp=W) -> int:
    """k tiles of the largest shard (rank 0): ceil(ceil((ctx + q_len) / cp) / 128)."""
    return -(-(-(-(ctx + q_len) // cp)) // TILE_N)


def split_eff(t, s) -> int:
    return -(-t // -(-t // s))


def ctas(B, S, s) -> int:
    return CLUSTER * B * S * s


def cost(B, S, t, s, num_sm=NUM_SM, k0=K0_TILES) -> int:
    """Wave model of one call in k-tile units."""
    return -(-ctas(B, S, split_eff(t, s)) // num_sm) * (-(-t // s) + k0)


@functools.lru_cache(maxsize=None)
def split_kv(B, q_len, ctx, num_sm=NUM_SM, cp=W, heads=H, cc=CC) -> int:
    """The served split: SERVED_SPLIT[(q_len, B)] (checked by check_served) when present, else the planner's.
    SERVED_SPLIT: the per-B argmin of the composition sweep (D-split-02, sweep.py --compositions, splitcomp.json
    `picks`) as a literal; gate_cases covers every (q_len, B) it moves. Read through lru_cache: edit the literal."""
    if (q_len, B) in SERVED_SPLIT:
        return check_served({(q_len, B): SERVED_SPLIT[(q_len, B)]})[(q_len, B)]
    return ts_plan(B, q_len, heads, L, num_sm, cc, 1)[2]


@functools.lru_cache(maxsize=None)
def model_split(B, q_len, ctx, num_sm=NUM_SM, cp=W, heads=H, cc=CC) -> int:
    """Wave-model split (N-01 draft; measured, not served): the planner's split when it is >= 2, else argmin over s
    of cost (smallest s on ties), s capped by the planner's 32, the tile count and the served workspace
    (B * heads_eff * S * s <= num_sm * heads * IMAGE_Q)."""
    _, fold, planner, _ = ts_plan(B, q_len, heads, L, num_sm, cc, 1)
    if planner >= 2:
        return planner
    S, t = q_len // fold, tiles(ctx, q_len, cp)
    cap = max(1, min(MAX_SPLIT, t, num_sm * IMAGE_Q // (B * fold * S)))
    return min(range(1, cap + 1), key=lambda s: (cost(B, S, t, s, num_sm), s))


def hook(orig, choose, seen=None, heads=None):
    """Wrap mla_decode._get_split_kv_and_workspace_size(B, q_len_eff, H_eff, kv_lora, num_sm, max_seq_len, dtype,
    tiler): split = choose(B, q_len_eff, max_seq_len, num_sm, planner_split), workspace recomputed when it moves.
    `seen` (a list) gets every split handed to the kernel. heads: choose only when H_eff == heads (else the planner)."""
    def planner(B, q, h, kv, sms, msl, dtype, tiler):
        split, ws = orig(B, q, h, kv, sms, msl, dtype, tiler)
        s = split if heads is not None and h != heads else choose(B, q, msl, sms, split)
        if s != split:
            split, ws = s, workspace_bytes(B, q, s, h, kv)
        if seen is not None:
            seen.append(split)
        return split, ws
    return planner


ENV = "PERFBOT_EXP_SPLITKV"  # "1": serve split_kv (SERVED_SPLIT); default "0" until the D6 pair + eval land it
_installed = []


def install(tmd=None, log=print) -> bool:
    """Served path, behind PERFBOT_EXP_SPLITKV=1 (overlay mla.py calls it once per worker; a no-op otherwise):
    forward_mqa (image tokenspeed_mla.py:297) -> tokenspeed_mla_decode -> mla_decode.py:503 looks the planner up as
    a module global, so replacing it once routes every decode call, CUDA-graph capture included. max_seq_len there is
    the local length (reftest R7 passes ceil(G / 8)), hence cp=1 (keys = msl + q_len); split_kv ignores it anyway.
    Logs `PERFBOT-SPLITKV` once at install and once per (q_len, B) whose split it moves. tmd: the module (tests)."""
    import os

    if os.environ.get(ENV, "0") != "1" or _installed:
        return False
    if tmd is None:
        from tokenspeed_mla import mla_decode as tmd
    check_served(SERVED_SPLIT)
    moved = set()

    def choose(B, q, msl, sms, p):
        s = split_kv(B, q, msl, sms, cp=1)
        if s != p and (q, B) not in moved:
            moved.add((q, B))
            log(f"PERFBOT-SPLITKV forced q_len={q} B={B} split={s} planner={p}", flush=True)
        return s

    tmd._get_split_kv_and_workspace_size = hook(tmd._get_split_kv_and_workspace_size, choose, heads=H)  # the gated
    # shape: the target's 96 gathered heads; the draft's 64 keep the planner
    _installed.append(tmd)
    log(f"PERFBOT-SPLITKV installed table={sorted(SERVED_SPLIT.items())}", flush=True)
    return True


# ------------------------------------------------------------------------------------------------ case bookkeeping
def cases():
    """(q_len, B, ctx) in run order: q, then ctx (one cache pool per ctx), then B."""
    return [(q, b, c) for q in CASE_Q for c in CASE_CTX for b in CASE_B]


def rank_cases(rank, world=W):
    """Sweep share of one GPU: each q_len gets a contiguous rank group (one compiled kernel pair per process),
    round-robin inside the group."""
    out = []
    for j, q in enumerate(CASE_Q):
        lo, hi = world * j // len(CASE_Q), world * (j + 1) // len(CASE_Q)
        if lo <= rank < hi:
            out += [c for c in cases() if c[0] == q][rank - lo :: hi - lo]
    return out


def gate_cases(num_sm=NUM_SM, cc=CC):
    """Cases where split_kv moves the split (elsewhere it hands the kernel the planner's own value): the grid's, plus
    every SERVED_SPLIT key at every CASE_CTX (B 3 is not on the grid). Re-checks the table's bounds."""
    check_served(SERVED_SPLIT)
    grid = cases() + [(q, b, c) for q, b in SERVED_SPLIT for c in CASE_CTX]
    return sorted({(q, b, c) for q, b, c in grid if split_kv(b, q, c, num_sm, cc=cc) != ts_plan(b, q, H, L, num_sm, cc, 1)[2]})


def spread(T, n):
    """<= n indices spread evenly over range(T), first and last included."""
    return sorted({round(i * (T - 1) / max(1, min(n, T) - 1)) for i in range(min(n, T))})


def sample_rows(B, q_len, n):
    """<= n (request, token) pairs spread evenly over the B * q_len query rows."""
    return [(i // q_len, i % q_len) for i in spread(B * q_len, n)]


def dedupe_compositions(comps, tol=COMP_TOL, keep=COMP_KEEP):
    """Served decode steps {k: [[us, [ctx ...]] or [us, [ctx ...], steps], ...]} (attn_ctx.py --dump: measured us of
    the step's split-kv call, the global context of each request) -> the same shape, one entry per distinct request
    set: a step joins the first cluster whose first step's sorted ctxs agree within tol (relative) element-wise; an
    entry is [median us, the first step's sorted ctxs, steps]; <= keep entries per k, spread evenly over the clusters
    sorted by (longest ctx, total ctx). Idempotent (cluster heads are pairwise more than tol apart)."""
    out = {}
    for k in sorted(comps, key=int):
        cl = []
        for st in comps[k]:
            cs = sorted(st[1])
            c = next((c for c in cl if len(c[1]) == len(cs) and all(
                abs(a - b) <= tol * max(a, b) for a, b in zip(c[1], cs))), None)
            if c is None:
                cl.append(c := [[], cs, 0])
            c[0].append(st[0])
            c[2] += st[2] if len(st) > 2 else 1
        cl.sort(key=lambda c: (c[1][-1], sum(c[1])))
        out[str(k)] = [[round(statistics.median(cl[i][0]), 2), cl[i][1], cl[i][2]] for i in spread(len(cl), keep)]
    return out


def arms(B, q_len, ctx, num_sm=NUM_SM, cc=CC):
    """Distinct splits to run, planner first: planner, FORCED up to the tile count, the wave model's, the served."""
    planner = ts_plan(B, q_len, H, L, num_sm, cc, 1)[2]
    t = tiles(ctx, q_len)
    return list(dict.fromkeys([planner] + [s for s in FORCED if s <= t] + [model_split(B, q_len, ctx, num_sm, cc=cc),
                                                                            split_kv(B, q_len, ctx, num_sm, cc=cc)]))


def fit(points):
    """Least squares us = a * waves * per_cta + b * waves + c over points [(waves, per_cta, us)]:
    {a (us per tile), k0 = b / a (tiles), c (us), r2}."""
    import torch

    x = torch.tensor([[w * p, w, 1.0] for w, p, _ in points], dtype=torch.float64)
    y = torch.tensor([u for _, _, u in points], dtype=torch.float64)
    a, b, c = torch.linalg.lstsq(x, y[:, None]).solution[:, 0].tolist()
    res = y - x @ torch.tensor([a, b, c], dtype=torch.float64)
    r2 = 1 - (res ** 2).sum().item() / max(((y - y.mean()) ** 2).sum().item(), 1e-300)
    return {"a_us_per_tile": a, "k0_tiles": b / a if a else None, "c_us": c, "r2": r2, "n": len(points)}
