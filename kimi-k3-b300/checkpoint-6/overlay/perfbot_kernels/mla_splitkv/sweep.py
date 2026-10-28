"""Kernel-only split-KV sweep and gate of tokenspeed_mla_decode under DCP 8 (expedition line D, item D-split).
Run in the server env (PERFBOT_PRE_EXEC):

  python3 -B -m perfbot_kernels.mla_splitkv.sweep [--deadline-s S]           # sweep: a measurement, exit 0
  python3 -B -m perfbot_kernels.mla_splitkv.sweep --gate [--deadline-s S]    # non-inferiority gate of split_kv
  python3 -B -m perfbot_kernels.mla_splitkv.sweep --isolate 8,6,100000 [--isolate q,B,ctx ...] [--isolate-s S] \
      [--isolate-splits 1,4,8] [--compositions [PATH]] [--deadline-s S]      # crash isolation, then composition sweep

SWEEP: one process per GPU, rank-local: the kernel reads only its own shard, so timing it needs no peer and no
collective. GPU r is DCP rank r (cp_world 8, cp_rank r, causal_seqs G); mla_splitkv.rank_cases shares the 54 cases
(q_len x B x ctx) out, one q_len per process. Per case: B requests of ctx cached + q_len new keys (G = ctx + q_len,
all in the cache), fp8 q [B, q_len, 96, 576] and cache, page 32, the arguments forward_mqa passes (seq_lens = local
keys, max_seq_len = ceil(G / 8), return_lse, causal; pool / inputs take one ctx per request). Per split in
mla_splitkv.arms (the planner's, FORCED up to
the tile count, the formula's), forced through mla_splitkv.hook on the library planner:
  us    one call (lse memset, causal-bound subtract, split kernel, reduction kernel) as a CUDA graph; REPLAYS
        replays, each between its own events after an L2 flush (the served layers read distinct caches); median.
  rel   rel-L2 vs fp64 (mla_mixbench.ref_rows on this rank's shard, the kernel's local causal bound) on <= REF_ROWS
        sampled (request, token) rows, all 96 heads; max_abs and bit-equality (o and lse) vs the planner's output.
  used  the split the hook handed to the kernel; CTAs, waves, workspace bytes vs the image's 233 MB buffer.
Records: splitsweep-rank<r>.jsonl (one line per case), merged into $PERFBOT_WORKSPACE/$PERFBOT_RESULTS/splitsweep.json
with the cost-model fit (mla_splitkv.fit). Table on stdout; last line `SPLITSWEEP DONE` (FAIL: no case ran).

GATE (--gate): 8 ranks (NCCL + symmetric memory), every case of mla_splitkv.gate_cases (the formula moves the
split): decode + the DCP combine (DirectDCPA2AWorkspace.lse_reduce with the served seq_lens / query_start_loc, as
overlay mla.py:715-724) captured as one CUDA graph per split, GATE_REPLAYS replays (each bit-equal to the eager call); this
rank's 12 heads vs fp64 over the global latent. PASS iff on every case and rank rel(formula) <= 1.02 x rel(planner),
the hook handed the kernel the formula's split and the planner's own, and the replays agreed.
`SPLITSWEEP-GATE pass=<bool>`; exit 0 iff pass. Record splitsweep-gate.json.

ISOLATE (--isolate q,B,ctx, repeatable; N-01 rank 3 died with an illegal memory access on (8, 6, 100000) at an
unrecorded split): one fresh interpreter per (case, split in mla_splitkv.arms, DCP rank shard 0..7, layout), on GPU
rank % gpus, CUDA_LAUNCH_BLOCKING=1: eager call, graph capture, replays, rel_ref. Layouts: n01 (N-01's pool: rows of
ceil(ceil(G / 8) / 32) pages, so the kernel's per-k-tile block-table loads run past the row, bt_overrun), and where
that over-read happens also poison (n01 with 2^30 in the allocation past the last row) and pad (whole k tiles per
row). `SPLITSWEEP-ISOLATE q= B= ctx= rank= split= layout= ok= ...` per job; unsafe = splits that crashed without an
over-read. Record splitsweep-isolate.json; `SPLITSWEEP-ISOLATE DONE` iff every job ran (FAIL: budget hit).
COMPOSITIONS (--compositions [PATH], default the package's compositions_rsplit.json, mla_splitkv.dedupe_compositions
of R-split's served steps): q_len 8, one process per GPU (GPU r = rank r shard), per composition the B = k requests'
own ctxs and splits {planner} + COMP_SPLITS minus this run's isolate `unsafe`, timed and checked as SWEEP.
`SPLITSWEEP-COMP k= idx= ...` per (composition, split), `SPLITSWEEP-COMP-PICK k= ...` per k (argmin over splits of the
median us over the k's compositions); record splitcomp.json; last line `SPLITSWEEP-COMP DONE` iff every k has a pick.
"""

import argparse
import contextlib
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

import perfbot_kernels.mla_splitkv as sk
from perfbot_kernels.mla_absorb import pshift_kernel, pshift_parts
from perfbot_kernels.mla_absorb.reftest import short_error, ts_plan
from perfbot_kernels.mla_splitkv import H, HL, L, R, W

FP8 = torch.float8_e4m3fn
PAGE = 32  # the served kernel block size for TOKENSPEED_MLA
SCALE = (128 + R) ** -0.5
REPLAYS, GATE_REPLAYS, REF_ROWS = 60, 3, 16
FLUSH_BYTES = 512 << 20
PORT = 29591
B_MAX, Q_MAX = max(sk.CASE_B), max(sk.CASE_Q)
POISON = 1 << 30
JOB_TIMEOUT_S = 150
COMP_Q = sk.IMAGE_Q
PKG_COMPS = Path(__file__).with_name("compositions_rsplit.json")
MODULE = "perfbot_kernels.mla_splitkv.sweep"


def results_path(name) -> Path:
    root = Path(os.environ.get("PERFBOT_WORKSPACE", "/workspace"))
    out = root / os.environ.get("PERFBOT_RESULTS", "results/measure")
    out.mkdir(parents=True, exist_ok=True)
    return out / name


def rel(x, ref):
    return ((x.double() - ref).norm() / ref.norm().clamp_min(1e-300)).item()


def reference(keys, q, limits):
    """fp64 latent attention of rows q [r, h, L + R] over keys [K, L + R], row i seeing key indices <= limits[i]:
    mla_mixbench.ref_rows with identity W_UK / W_UV (the kernel's output is the latent). Returns [r, h, L]."""
    from perfbot_kernels.mla_mixbench import ref_rows

    eye = torch.eye(L, dtype=torch.float64, device=q.device).expand(q.shape[1], L, L)
    return ref_rows(keys, keys[:0], q[..., :L], q[..., L:], eye, eye, limits - keys.shape[0], SCALE)


class Kernel:
    """tokenspeed_mla_decode with the split forced through mla_splitkv.hook on the library planner."""

    def __init__(self, dev):
        import cutlass
        from tokenspeed_mla import get_num_sm, tokenspeed_mla_decode
        from tokenspeed_mla import mla_decode as tmd
        from tokenspeed_mla.mla_decode_fp16 import BlackwellMultiHeadLatentAttentionForwardFP16 as K16

        self.decode, self.dev, self.ws = tokenspeed_mla_decode, dev, None
        self.num_sm, self.cc = get_num_sm(dev), tuple(torch.cuda.get_device_capability(dev))
        self.force, self.seen, self.phase = None, [], "setup"
        tmd._get_split_kv_and_workspace_size = sk.hook(
            tmd._get_split_kv_and_workspace_size, lambda B, q, msl, sms, p: self.force or p, self.seen)
        self.ws_lib = lambda B, q, s: K16.get_workspace_size(H, q, L, B, s, cutlass.Float32)

    def __call__(self, inp, out):
        return self.decode(
            query=inp.q, kv_cache=inp.cache, workspace_buffer=self.ws, kv_lora_rank=L, qk_rope_head_dim=R,
            block_tables=inp.bt, seq_lens=inp.seq, max_seq_len=inp.msl, softmax_scale=SCALE, output_scale=1.0, out=out,
            enable_pdl=False, return_lse=True, causal_mask=True, causal_seqs=inp.causal, cp_world=W, cp_rank=inp.rank)

    def alloc(self, todo):
        """One workspace for every split this process will force (and at least the image's)."""
        need = [sk.image_workspace(self.num_sm)] + [sk.workspace_bytes(B, q, s) for q, B, _, ss in todo for s in ss]
        self.ws = torch.empty(max(need), dtype=torch.int8, device=self.dev)


def pool_pages(ctx, pad, qmax=Q_MAX):
    """Pages of one request's row: its largest shard (ceil((ctx + qmax) / 8) keys), whole k tiles if pad."""
    per = sk.TILE_N // PAGE if pad else 1
    return -(-(-(-(ctx + qmax) // W)) // (PAGE * per)) * per


def bt_overrun(ctx, q_len, rank, pad=False) -> bool:
    """Does the kernel read block-table entries past the row? Per k tile it loads TILE_N / PAGE entries, unguarded
    (mla_decode_fp8.py load_k :2115-2126, load_v :2250-2255: mPT[k_index * page_per_tile + i]), over this rank's
    ceil(local / 128) tiles; the row holds pool_pages entries."""
    return pool_pages(ctx, pad, max(Q_MAX, q_len)) * PAGE < -(-(-(-(ctx + q_len - rank) // W)) // sk.TILE_N) * sk.TILE_N


def pool(dev, rank, ctxs, keep, pad=True, qmax=Q_MAX):
    """Request b: ctxs[b] cached + qmax (default Q_MAX; the isolate's q_len when larger) global positions (seeded by (ctx, b): identical on every rank), pages
    [off[b], off[b] + pages[b]) holding this rank's strided shard (global p = local c * 8 + rank). pad=False is N-01's
    layout (bt_overrun). keep: also the global latents (the gate's reference)."""
    pages = [pool_pages(c, pad, qmax) for c in ctxs]
    off = [sum(pages[:b]) for b in range(len(ctxs))]
    cache = torch.zeros(sum(pages) * PAGE, L + R, dtype=FP8, device=dev)
    lats = []
    for b, ctx in enumerate(ctxs):
        G = ctx + qmax
        g = torch.Generator(device=dev).manual_seed(7000 + 31 * b + ctx)
        lat = torch.empty(G, L + R, dtype=FP8, device=dev)
        for a in range(0, G, 1 << 16):
            e = min(G, a + (1 << 16))
            lat[a:e] = torch.randn(e - a, L + R, device=dev, generator=g).to(FP8)
        shard = lat[rank::W]
        cache[off[b] * PAGE : off[b] * PAGE + shard.shape[0]] = shard
        if keep:
            lats.append(lat)
    return types.SimpleNamespace(ctxs=list(ctxs), pages=pages, off=off, cache=cache.view(-1, PAGE, L + R), lats=lats)


def inputs(dev, rank, pl, q_len, B, poison=False):
    """Requests 0..B-1 of pl as forward_mqa passes them under DCP: seq_lens = local keys ceil((G_b - rank) / 8),
    causal_seqs = global G_b = ctx_b + q_len, max_seq_len = max ceil(G_b / 8); block-table rows padded with page 0 to
    the widest. poison: POISON in 64 int32 allocated past the last row. ctx / G / local: the longest request's."""
    ctxs = pl.ctxs[:B]
    Gs = [c + q_len for c in ctxs]
    locs = [-(-(G - rank) // W) for G in Gs]
    width = max(pl.pages[:B])
    bt = torch.zeros(B * width + (64 if poison else 0), dtype=torch.int32)
    bt[B * width :] = POISON
    for b in range(B):
        bt[b * width : b * width + pl.pages[b]] = torch.arange(pl.off[b], pl.off[b] + pl.pages[b])
    g = torch.Generator(device=dev).manual_seed(900 + q_len + max(ctxs))
    q = torch.randn(B_MAX, q_len, H, L + R, device=dev, generator=g).to(FP8)[:B]  # rows independent of B
    i32 = dict(dtype=torch.int32, device=dev)
    return types.SimpleNamespace(
        q=q, cache=pl.cache, bt=bt.to(dev)[: B * width].view(B, width), seq=torch.tensor(locs, **i32),
        msl=max(-(-G // W) for G in Gs), causal=torch.tensor(Gs, **i32), rank=rank, B=B, q_len=q_len, ctxs=ctxs,
        Gs=Gs, locs=locs, ctx=max(ctxs), G=max(Gs), local=max(locs))


def graph(fn, sync):
    """CUDA graph of fn after a side-stream warm-up: (graph, fn's return value inside the graph)."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    sync()
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
    us = [a.elapsed_time(b) * 1e3 for a, b in ev]
    return round(statistics.median(us), 2), [round(x, 2) for x in us]


def pick(o, rows):
    return o[torch.tensor([b for b, _ in rows]), torch.tensor([s for _, s in rows])]


# ------------------------------------------------------------------------------------------------------ sweep
def ref_local(pl, inp, rows, memo):
    """fp64 reference of `rows` on this rank's shard (memo keyed (q_len, b, s); q rows do not depend on B)."""
    need = {}
    for b, s in rows:
        if (inp.q_len, b, s) not in memo:
            need.setdefault(b, []).append(s)
    flat = pl.cache.view(-1, L + R)
    for b, ss in need.items():
        keys = flat[pl.off[b] * PAGE : pl.off[b] * PAGE + inp.locs[b]]
        lim = (inp.ctxs[b] + torch.tensor(ss, device=keys.device) - inp.rank) // W  # c * 8 + rank <= ctx_b + s
        ref = reference(keys, inp.q[b, ss], lim)
        memo.update({(inp.q_len, b, s): ref[i] for i, s in enumerate(ss)})
    return torch.stack([memo[(inp.q_len, b, s)] for b, s in rows])


def run_arm(k, inp, force, flush):
    k.force = force
    k.seen.clear()
    out = torch.empty(inp.B, inp.q_len, H, L, dtype=torch.bfloat16, device=k.dev)
    t = time.time()
    k.phase = "eager"
    o, lse = k(inp, out)
    torch.cuda.synchronize()
    res = {"used": k.seen[-1], "first_call_s": round(time.time() - t, 2)}
    eager = (o.clone(), lse.clone())
    k.phase = "graph"
    g, (o, lse) = graph(lambda: k(inp, out), torch.cuda.synchronize)
    k.phase = "replay"
    res["us"], res["reps_us"] = time_graph(g, flush)
    k.phase = "done"
    res["replay_equal_eager"] = torch.equal(o, eager[0]) and torch.equal(lse, eager[1])
    o, lse = o.clone(), lse.clone()
    del g
    return res, o, lse


def sweep_case(k, pl, inp, splits, flush, memo):
    B, q, ctx = inp.B, inp.q_len, inp.ctx
    _, fold, planner, _ = ts_plan(B, q, H, L, k.num_sm, k.cc, inp.msl)
    S, t = q // fold, -(-inp.local // sk.TILE_N)
    ts = [-(-x // sk.TILE_N) for x in inp.locs]
    rows = sk.sample_rows(B, q, REF_ROWS)
    t0 = time.time()
    ref = ref_local(pl, inp, rows, memo)
    rec = {"q_len": q, "B": B, "ctx": ctx, "rank": inp.rank, "G": inp.G, "local": inp.local, "tiles": t, "fold": fold,
           "num_sm": k.num_sm, "cc": k.cc, "planner": planner, "formula": sk.split_kv(B, q, ctx, k.num_sm, cc=k.cc),
           "image_ws": sk.image_workspace(k.num_sm), "ref_rows": rows, "ref_s": round(time.time() - t0, 2), "arms": []}
    base = None
    for s in splits:
        e = sk.split_eff(t, s)
        n = sk.CLUSTER * S * sum(sk.split_eff(x, s) for x in ts)  # CTAs with k tiles (uniform: sk.ctas(B, S, e))
        arm = {"split": s, "is_planner": s == planner, "is_formula": s == rec["formula"], "split_eff": e, "ctas": n,
               "waves": -(-n // k.num_sm), "per_cta_tiles": -(-t // s), "model": sk.cost(B, S, t, s, k.num_sm),
               "ws_bytes": sk.workspace_bytes(B, q, s), "ws_lib_ok": sk.workspace_bytes(B, q, s) == k.ws_lib(B, q, s)}
        arm["ws_fits_image"] = arm["ws_bytes"] <= rec["image_ws"]
        try:
            res, o, lse = run_arm(k, inp, None if s == planner else s, flush)
            arm.update(res)
            arm["rel_ref"] = rel(pick(o, rows), ref)
            if s == planner:
                base = (o, lse)
            if base is not None:
                arm["max_abs_vs_planner"] = (o.float() - base[0].float()).abs().max().item()
                arm["bit_equal_vs_planner"] = torch.equal(o, base[0]) and torch.equal(lse, base[1])
        except Exception as ex:  # noqa: BLE001
            arm["error"] = short_error(ex)
            torch.cuda.synchronize()  # a sticky CUDA error raises here and ends the worker
        rec["arms"].append(arm)
    return rec


def _run_worker(rank, out_dir, name, body):
    """One GPU (rank r = DCP shard r): body(kernel, flush buffer, write) appends records to <name>-rank<r>.jsonl;
    an exception (a sticky CUDA error ends the worker) is recorded."""
    torch.cuda.set_device(rank)
    f = Path(out_dir, f"{name}-rank{rank}.jsonl").open("w")

    def write(rec):
        f.write(json.dumps(rec, default=str) + "\n")
        f.flush()

    try:
        with torch.inference_mode():
            k = Kernel(torch.device("cuda", rank))
            body(k, torch.empty(FLUSH_BYTES, dtype=torch.int8, device=k.dev), write)
    except Exception:  # noqa: BLE001
        write({"rank": rank, "error": traceback.format_exc()[-2000:]})
        traceback.print_exc()
    f.close()


def read_jsonl(out_dir, name, world):
    recs = []
    for r in range(world):
        p = Path(out_dir, f"{name}-rank{r}.jsonl")
        recs += [json.loads(x) for x in (p.read_text().splitlines() if p.is_file() else []) if x.strip()]
    return recs


def _sweep_worker(rank, out_dir, t_end, world):
    def body(k, flush, write):
        todo = [(q, B, c, sk.arms(B, q, c, k.num_sm, k.cc)) for q, B, c in sk.rank_cases(rank, world)]
        k.alloc(todo)
        pl, memo = None, {}
        for q, B, ctx, splits in todo:
            if time.time() > t_end:
                write({"stopped_at": [q, B, ctx], "rank": rank})
                break
            if pl is None or pl.ctxs[0] != ctx:
                pl, memo = None, {}
                torch.cuda.empty_cache()
                pl = pool(k.dev, rank, [ctx] * B_MAX, False)
            write(sweep_case(k, pl, inputs(k.dev, rank, pl, q, B), splits, flush, memo))

    _run_worker(rank, out_dir, "splitsweep", body)


def report_sweep(path, world, t0, exitcodes) -> int:
    recs = read_jsonl(path.parent, "splitsweep", world)
    rows = sorted((x for x in recs if "arms" in x), key=lambda x: (x["q_len"], x["ctx"], x["B"]))
    hdr = (f"{'q':>3} {'B':>2} {'ctx':>6} {'tiles':>5} {'plan':>4} {'form':>4} {'best':>4} {'us_plan':>8} {'us_form':>8} "
           f"{'us_best':>8} {'x_form':>6} {'x_best':>6} {'rel_plan':>9} {'rel_form':>9} {'beq':>4} {'ws_form_MB':>10} "
           f"{'fits':>4} {'model_pick':>10}")
    print("SPLITSWEEP-TABLE " + hdr, flush=True)
    for x in rows:
        ok = {a["split"]: a for a in x["arms"] if a.get("us") is not None}
        pa, fa = ok.get(x["planner"], {}), ok.get(x["formula"], {})
        best = min(ok, key=lambda s: ok[s]["us"]) if ok else None
        x["best"] = best
        x["model_pick"] = min(ok, key=lambda s: (ok[s]["model"], s)) if ok else None
        sp = lambda a, b: f"{a / b:.3f}" if a and b else "NA"  # noqa: E731
        fmt = lambda v, f: "NA" if v is None else format(v, f)  # noqa: E731
        print("SPLITSWEEP-TABLE " + (
            f"{x['q_len']:>3} {x['B']:>2} {x['ctx']:>6} {x['tiles']:>5} {x['planner']:>4} {x['formula']:>4} "
            f"{fmt(best, '>4')} {fmt(pa.get('us'), '>8.1f')} {fmt(fa.get('us'), '>8.1f')} "
            f"{fmt(best and ok[best]['us'], '>8.1f')} {sp(pa.get('us'), fa.get('us')):>6} "
            f"{sp(pa.get('us'), best and ok[best]['us']):>6} {fmt(pa.get('rel_ref'), '>9.3e')} "
            f"{fmt(fa.get('rel_ref'), '>9.3e')} {str(fa.get('bit_equal_vs_planner'))[:4]:>4} "
            f"{fa.get('ws_bytes', 0) / 1e6:>10.1f} {str(fa.get('ws_fits_image'))[:4]:>4} {fmt(x['model_pick'], '>10')}"),
            flush=True)
        for a in x["arms"]:
            print(f"SPLITSWEEP-ARM q={x['q_len']} B={x['B']} ctx={x['ctx']} split={a['split']} used={a.get('used')} "
                  f"ctas={a['ctas']} waves={a['waves']} us={a.get('us')} rel_ref={a.get('rel_ref')} "
                  f"max_abs_vs_planner={a.get('max_abs_vs_planner')} bit_equal={a.get('bit_equal_vs_planner')} "
                  f"ws_bytes={a['ws_bytes']} ws_lib_ok={a['ws_lib_ok']}" + (f" error={a['error']}" if "error" in a else ""),
                  flush=True)
    fits = {}
    for q in sk.CASE_Q:
        pts = [(a["waves"], a["per_cta_tiles"], a["us"]) for x in rows if x["q_len"] == q for a in x["arms"]
               if a.get("us") is not None]
        if len(pts) >= 4:
            fits[q] = sk.fit(pts)
            print(f"SPLITSWEEP-FIT q={q} {json.dumps(fits[q])}", flush=True)
    forced_bad = [(x["q_len"], x["B"], x["ctx"], a["split"], a.get("used")) for x in rows for a in x["arms"]
                  if "used" in a and a["used"] != a["split"]]
    errors = [x for x in recs if "error" in x] + [
        {"case": (x["q_len"], x["B"], x["ctx"]), "split": a["split"], "error": a["error"]} for x in rows for a in x["arms"]
        if "error" in a]
    record = {"item": "expedition-D-split-01", "mode": "sweep", "seconds": round(time.time() - t0, 1),
              "exitcodes": exitcodes, "world": world, "cases": rows, "fit": fits, "forced_not_used": forced_bad,
              "errors": errors, "stopped": [x for x in recs if "stopped_at" in x], "expected_cases": len(sk.cases())}
    path.write_text(json.dumps(record, indent=1, default=str))
    print(f"SPLITSWEEP-SUMMARY cases={len(rows)}/{len(sk.cases())} errors={len(errors)} "
          f"forced_not_used={len(forced_bad)} seconds={record['seconds']} record={path}", flush=True)
    print("SPLITSWEEP DONE" if rows else "SPLITSWEEP FAIL", flush=True)
    return 0 if rows else 1


# ------------------------------------------------------------------------------------------------------- gate
def _gate_rank(rank, out_dir, t_end, world):
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{PORT}", rank=rank, world_size=W, device_id=dev)
    rec = {"rank": rank, "cases": []}
    path = Path(out_dir, f"splitsweep-gate-rank{rank}.json")
    save = lambda: path.write_text(json.dumps(rec, indent=1, default=str))  # noqa: E731

    def barrier():
        torch.cuda.synchronize()
        dist.barrier(device_ids=[rank])

    def any_flag(flag):
        t = torch.tensor([int(flag)], dtype=torch.int32, device=dev)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return bool(t.item())

    try:
        with torch.inference_mode():
            group = dist.group.WORLD
            try:
                import torch.distributed._symmetric_memory as symm_mem

                symm_mem.enable_symm_mem_for_group(group.group_name)
            except Exception:  # noqa: BLE001
                pass
            import vllm._C_stable_libtorch  # noqa: F401  (registers torch.ops._C.direct_dcp_*)
            from vllm.v1.attention.ops import dcp

            aws = dcp.DirectDCPA2AWorkspace(group, dev, B_MAX * Q_MAX, HL, L, torch.bfloat16, 1)
            k = Kernel(dev)
            rec["num_sm"], rec["cc"] = k.num_sm, k.cc
            todo = sorted(sk.gate_cases(k.num_sm, k.cc), key=lambda c: (c[2], c[0], c[1]))
            k.alloc([(q, B, c, [sk.split_kv(B, q, c, k.num_sm, cc=k.cc)]) for q, B, c in todo])
            mine = slice(rank * HL, (rank + 1) * HL)
            pl = None
            for q, B, ctx in todo:
                if any_flag(time.time() > t_end):
                    rec["stopped_at"] = [q, B, ctx]
                    break
                if pl is None or pl.ctxs[0] != ctx:
                    pl = None
                    torch.cuda.empty_cache()
                    pl = pool(dev, rank, [ctx] * B_MAX, True)
                inp = inputs(dev, rank, pl, q, B)
                qsl = torch.arange(B + 1, dtype=torch.int32, device=dev) * q
                planner, formula = ts_plan(B, q, H, L, k.num_sm, k.cc, inp.msl)[2], sk.split_kv(B, q, ctx, k.num_sm, cc=k.cc)
                row = {"q_len": q, "B": B, "ctx": ctx, "rank": rank, "planner": planner, "formula": formula}
                outs = {}
                for name, force in (("planner", None), ("formula", formula)):
                    k.force = force
                    k.seen.clear()
                    out = torch.empty(B, q, H, L, dtype=torch.bfloat16, device=dev)

                    def fn():
                        o, lse = k(inp, out)
                        return aws.lse_reduce(o.view(B * q, H, L), lse.view(B * q, H), False, inp.seq, qsl)

                    try:
                        eager = fn().clone()
                        barrier()
                        row[f"{name}_used"] = k.seen[-1]
                        g, res = graph(fn, barrier)
                        got = []
                        for _ in range(GATE_REPLAYS):
                            barrier()
                            g.replay()
                            barrier()
                            got.append(res.clone())
                        row[f"{name}_replays_equal"] = all(torch.equal(eager, x) for x in got)  # and vs eager
                        outs[name] = got[-1]
                        del g
                    except Exception as ex:  # noqa: BLE001
                        row[f"{name}_error"] = short_error(ex)
                rows = sk.sample_rows(B, q, REF_ROWS)
                ref = torch.cat([reference(pl.lats[b][: inp.Gs[b]], inp.q[b, [s]][:, mine],
                                           torch.tensor([inp.ctxs[b] + s], device=dev)) for b, s in rows])
                for name, o in outs.items():
                    row[f"rel_{name}"] = rel(o.view(B, q, HL, L)[torch.tensor([b for b, _ in rows]),
                                                                   torch.tensor([s for _, s in rows])], ref)
                row["ratio"] = (row["rel_formula"] / max(row["rel_planner"], 1e-300)) if len(outs) == 2 else None
                row["ok"] = (len(outs) == 2 and row["ratio"] <= sk.GATE_RATIO and row["planner_used"] == planner
                             and row["formula_used"] == formula and row["planner_replays_equal"]
                             and row["formula_replays_equal"])
                rec["cases"].append(row)
                save()
                if rank == 0:
                    print("SPLITSWEEP-GATE-CASE " + json.dumps(row), flush=True)
    except Exception:  # noqa: BLE001
        rec["error"] = traceback.format_exc()[-2000:]
        traceback.print_exc()
    save()
    try:
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001
        pass


def report_gate(path, world, t0, exitcodes) -> int:
    ranks = []
    for r in range(world):
        p = path.parent / f"splitsweep-gate-rank{r}.json"
        ranks.append(json.loads(p.read_text()) if p.is_file() else {"rank": r, "error": "no record"})
    table = [x for r in ranks for x in r.get("cases", [])]
    cases = sorted(sk.gate_cases(ranks[0].get("num_sm", sk.NUM_SM), tuple(ranks[0].get("cc", sk.CC))))
    for c in cases:
        rs = [x for x in table if (x["q_len"], x["B"], x["ctx"]) == c]
        print(f"SPLITSWEEP-GATE q={c[0]} B={c[1]} ctx={c[2]} formula={rs[0]['formula'] if rs else None} "
              f"ranks={len(rs)} ok={len(rs) == W and all(x['ok'] for x in rs)} "
              f"worst_ratio={max((x['ratio'] for x in rs if x['ratio'] is not None), default=None)} "
              f"rel_planner_max={max((x.get('rel_planner') or 0) for x in rs)}", flush=True)
    errors = {r["rank"]: r["error"] for r in ranks if "error" in r}
    stopped = [r.get("stopped_at") for r in ranks if r.get("stopped_at")]
    ok = bool(table) and not errors and not stopped and all(x["ok"] for x in table) and all(
        sum((x["q_len"], x["B"], x["ctx"]) == c for x in table) == W for c in cases)
    record = {"item": "expedition-D-split-gate", "mode": "gate", "pass": ok, "seconds": round(time.time() - t0, 1),
              "exitcodes": exitcodes, "rule": f"rel(formula) <= {sk.GATE_RATIO} x rel(planner) vs fp64 after the DCP "
              "combine, every gate case and rank; graph replays equal; hook used the intended splits",
              "cases": table, "errors": errors, "stopped": stopped}
    path.write_text(json.dumps(record, indent=1, default=str))
    print(f"SPLITSWEEP-GATE pass={ok} cases={len(cases)} rows={len(table)} errors={len(errors)} record={path}",
          flush=True)
    return 0 if ok else 1


# ---------------------------------------------------------------------------------------------------- isolate
def parse_case(s):
    q, B, ctx = (int(x) for x in s.split(","))
    return q, B, ctx


def isolate_jobs(cases, splits=None, shifts=(0,), coexist=False):
    """Jobs in run order: every (case, split of sk.arms, rank) on the N-01 layout; then, for cases whose N-01 block
    table is over-read on some rank, the same on the poison and the pad layouts. splits: run only these
    (--isolate-splits; the arms' order kept, a split above the tile count dropped), planner or not. shifts: every job
    once per P shift (--isolate-pshift; 0 = the stock kernel). coexist (--isolate-coexist): a shifted job also runs the
    stock kernel before and after the shifted one in its process (stock_equal, shift_differs, rel_ref_stock): the
    shadow's in-server use, mla_absorb.pshift_kernel."""
    jobs = []
    for lay in ("n01", "poison", "pad"):
        for q, B, ctx in cases:
            if lay == "n01" or any(bt_overrun(ctx, q, r) for r in range(W)):
                jobs += [{"q": q, "B": B, "ctx": ctx, "split": s, "rank": r, "layout": lay, "p_shift": h,
                          "coexist": bool(coexist and h), "overrun": lay != "pad" and bt_overrun(ctx, q, r)} for h in shifts
                         for s in (sk.arms(B, q, ctx) if splits is None else
                                   [x for x in dict.fromkeys(splits) if x <= sk.tiles(ctx, q)]) for r in range(W)]
    return jobs


def isolate_job(job) -> int:
    """Child (fresh interpreter, one visible GPU, CUDA_LAUNCH_BLOCKING=1): one split of one case on one rank's shard,
    N-01's pool of B_MAX requests; prints its SPLITSWEEP-ISOLATE-JOB record."""
    res, k = dict(job), None
    try:
        with torch.inference_mode():
            dev = torch.device("cuda", 0)
            torch.cuda.set_device(dev)
            k = Kernel(dev)
            h = job.get("p_shift", 0)
            if h:
                res["p_src"] = pshift_parts(h, results_path("x").parent)[2]
            q, B, ctx, s = job["q"], job["B"], job["ctx"], job["split"]
            k.alloc([(q, B, ctx, [s])])
            flush = torch.empty(FLUSH_BYTES, dtype=torch.int8, device=dev)
            pl = pool(dev, job["rank"], [ctx] * B_MAX, False, pad=job["layout"] == "pad", qmax=max(Q_MAX, q))
            inp = inputs(dev, job["rank"], pl, q, B, poison=job["layout"] == "poison")
            planner = ts_plan(B, q, H, L, k.num_sm, k.cc, inp.msl)[2]
            force = None if s == planner else s
            rows = sk.sample_rows(B, q, REF_ROWS)
            if h and job.get("coexist"):  # stock, P-shift, stock again in this one process (the shadow's use)
                _, o0, _ = run_arm(k, inp, force, flush)
                res["rel_ref_stock"] = rel(pick(o0, rows), ref_local(pl, inp, rows, {}))
            with pshift_kernel(h) if h else contextlib.nullcontext():
                r, o, _ = run_arm(k, inp, force, flush)
            if h and job.get("coexist"):
                _, o0b, _ = run_arm(k, inp, force, flush)
                res.update(stock_equal=torch.equal(o0, o0b), shift_differs=not torch.equal(o0, o))
            res.update(ok=True, planner=planner, used=r["used"], us=r["us"], replay_equal_eager=r["replay_equal_eager"],
                       first_call_s=r["first_call_s"],
                       finite=bool(torch.isfinite(o.float()).all()),
                       rel_ref=rel(pick(o, rows), ref_local(pl, inp, rows, {})))
    except Exception as ex:  # noqa: BLE001
        res.update(ok=False, phase=k.phase if k else "setup", error=short_error(ex))
    print("SPLITSWEEP-ISOLATE-JOB " + json.dumps(res, default=str), flush=True)
    return 0 if res["ok"] else 1


def job_result(job, out, err, rc):
    """The child's record; ok only if it says so and exited 0 (a child killed by the error leaves no record)."""
    line = next((x for x in reversed(out.splitlines()) if x.startswith("SPLITSWEEP-ISOLATE-JOB ")), None)
    res = json.loads(line.split(" ", 1)[1]) if line else dict(job)
    res.update(rc=rc, ok=bool(res.get("ok")) and rc == 0)
    if not res["ok"] and not res.get("error"):
        res["error"] = " | ".join([x.strip() for x in err.splitlines() if x.strip()][-2:])[-400:] or f"rc={rc}"
    return res


def launch(job, gpu, argv=None, timeout=JOB_TIMEOUT_S):
    """One job in a fresh interpreter on the parent's visible GPU `gpu`, CUDA_LAUNCH_BLOCKING=1."""
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    env = dict(os.environ, CUDA_LAUNCH_BLOCKING="1", CUDA_VISIBLE_DEVICES=vis.split(",")[gpu] if vis else str(gpu))
    t = time.time()
    try:
        p = subprocess.run(argv or [sys.executable, "-B", "-m", MODULE, "--isolate-job", json.dumps(job)], env=env,
                           capture_output=True, text=True, timeout=timeout)
        res = job_result(job, p.stdout, p.stderr, p.returncode)
    except subprocess.TimeoutExpired:
        res = dict(job, ok=False, rc=None, error=f"timeout {timeout} s")
    res["seconds"] = round(time.time() - t, 1)
    return res


def isolate_summary(done):
    """unsafe: splits that failed past setup (in or after the kernel calls, or killed without a record) on a layout the
    kernel does not over-read: the formula must exclude them; overrun_only: the other failed splits (harness
    artifact, or setup: import / allocation, listed in crashes)."""
    bad = [r for r in done if r["ok"] is False]
    groups = {}
    for r in bad:
        groups.setdefault((r["q"], r["B"], r["ctx"], r["layout"], r["split"]), []).append(r)
    unsafe = sorted({r["split"] for r in bad if not r["overrun"] and r.get("phase") != "setup"})
    return {"item": "expedition-D-split-02", "mode": "isolate", "jobs": done, "unsafe": unsafe,
            "overrun_only": sorted({r["split"] for r in bad} - set(unsafe)),
            "not_run": sum(r["ok"] is None for r in done), "done": all(r["ok"] is not None for r in done),
            "crashes": [{"case": c[:3], "layout": c[3], "split": c[4], "ranks": sorted(r["rank"] for r in rs),
                         "phase": rs[0].get("phase"), "error": rs[0].get("error")} for c, rs in sorted(groups.items())]}


def isolate_line(r):
    return (f"SPLITSWEEP-ISOLATE q={r['q']} B={r['B']} ctx={r['ctx']} rank={r['rank']} split={r['split']} "
            f"layout={r['layout']} p_shift={r.get('p_shift', 0)} overrun={r['overrun']} ok={r['ok']} used={r.get('used')} us={r.get('us')} "
            f"rel_ref={r.get('rel_ref')} finite={r.get('finite')} s={r.get('seconds')}"
            + (f" rel_ref_stock={r.get('rel_ref_stock')} stock_equal={r.get('stock_equal')} "
               f"shift_differs={r.get('shift_differs')}" if r.get("coexist") else "")
            + ("" if r["ok"] else f" phase={r.get('phase')} error={r.get('error')}"))


def run_isolate(cases, world, t_end, launch=launch, splits=None, shifts=(0,), coexist=False):
    """isolate_jobs on `world` lanes: lane g runs, in order, the jobs of ranks r % world == g, so the 8 shards of one
    (case, split, layout) run at once; no job starts after t_end. Record splitsweep-isolate.json."""
    jobs, t0 = isolate_jobs(cases, splits, shifts, coexist), time.time()
    done = [None] * len(jobs)

    def lane(g):
        for i, j in enumerate(jobs):
            if j["rank"] % world == g:
                done[i] = launch(j, g) if time.time() < t_end else dict(j, ok=None, error="not run: isolate budget")
                if done[i]["ok"] is not None:
                    print(isolate_line(done[i]), flush=True)

    with ThreadPoolExecutor(world) as ex:
        list(ex.map(lane, range(world)))
    rec = dict(isolate_summary(done), seconds=round(time.time() - t0, 1), cases=cases)
    path = results_path("splitsweep-isolate.json")
    path.write_text(json.dumps(rec, indent=1, default=str))
    for c in rec["crashes"]:
        print("SPLITSWEEP-ISOLATE-CRASH " + json.dumps(c, default=str), flush=True)
    print(f"SPLITSWEEP-ISOLATE-SUMMARY jobs={len(jobs)} crashed={sum(r['ok'] is False for r in done)} "
          f"not_run={rec['not_run']} unsafe={rec['unsafe']} overrun_only={rec['overrun_only']} "
          f"seconds={rec['seconds']} record={path}", flush=True)
    print("SPLITSWEEP-ISOLATE DONE" if rec["done"] else "SPLITSWEEP-ISOLATE FAIL", flush=True)
    return rec


# ------------------------------------------------------------------------------------------------ compositions
def comp_list(comps):
    """[(k, idx, measured us, steps, ctxs)] of a dedupe_compositions dict."""
    return [(int(k), i, e[0], e[2], e[1]) for k, es in comps.items() for i, e in enumerate(es)]


def comp_splits(planner, unsafe=(), cand=None):
    """The planner's split (kept even if unsafe: it is the served default), then cand (default COMP_SPLITS; --comp-splits)
    minus unsafe."""
    return list(dict.fromkeys([planner] + [s for s in (cand or sk.COMP_SPLITS) if s not in unsafe]))


def _comp_worker(rank, out_dir, t_end, world, todo, unsafe, cand=None):
    def body(k, flush, write):
        mine = [(c, comp_splits(ts_plan(len(c[4]), COMP_Q, H, L, k.num_sm, k.cc, 1)[2], unsafe, cand))
                for c in todo[rank::world]]
        k.alloc([(COMP_Q, len(c[4]), None, ss) for c, ss in mine])
        for (kk, idx, us, steps, ctxs), ss in mine:
            if time.time() > t_end:
                write({"stopped_at": [kk, idx], "rank": rank})
                break
            pl = pool(k.dev, rank, ctxs, False)
            rec = sweep_case(k, pl, inputs(k.dev, rank, pl, COMP_Q, len(ctxs)), ss, flush, {})
            write(dict(rec, k=kk, idx=idx, ctxs=ctxs, measured_served_us=us, steps=steps))

    _run_worker(rank, out_dir, "splitcomp", body)


def comp_picks(recs):
    """Per k: median us over the k's compositions of each split that ran on all of them, pick = argmin (smaller split
    on ties), ratio = us_pick / us_planner; worst rel_ref / rel_ref(planner) per split (the gate's data)."""
    out = {}
    for kk in sorted({x["k"] for x in recs}):
        rs = [x for x in recs if x["k"] == kk]
        us, rr = {}, {}
        for x in rs:
            ok = {a["split"]: a for a in x["arms"] if a.get("us") is not None}
            for s, a in ok.items():
                us.setdefault(s, []).append(a["us"])
                if x["planner"] in ok:
                    rr.setdefault(s, []).append(a["rel_ref"] / max(ok[x["planner"]]["rel_ref"], 1e-300))
        med = {s: statistics.median(v) for s, v in us.items() if len(v) == len(rs)}
        p = rs[0]["planner"]
        pick = min(med, key=lambda s: (med[s], s)) if med else None
        out[kk] = {"planner": p, "pick": pick, "compositions": len(rs), "median_us": med, "us_planner": med.get(p),
                   "us_pick": med.get(pick), "ratio": med[pick] / med[p] if pick is not None and p in med else None,
                   "worst_rel_ratio": {s: max(v) for s, v in rr.items()}}
    return out


def report_comp(path, world, t0, exitcodes, comps, unsafe, iso) -> int:
    recs = read_jsonl(path.parent, "splitcomp", world)
    rows = sorted((x for x in recs if "arms" in x), key=lambda x: (x["k"], x["idx"]))
    for x in rows:
        for a in x["arms"]:
            print(f"SPLITSWEEP-COMP k={x['k']} idx={x['idx']} ctxs={','.join(map(str, x['ctxs']))} "
                  f"measured_served_us={x['measured_served_us']} split={a['split']} planner={x['planner']} "
                  f"used={a.get('used')} us={a.get('us')} rel_ref={a.get('rel_ref')} ctas={a['ctas']} "
                  f"waves={a['waves']}" + (f" error={a['error']}" if "error" in a else ""), flush=True)
    picks = comp_picks(rows)
    for kk, p in picks.items():
        print(f"SPLITSWEEP-COMP-PICK k={kk} planner={p['planner']} pick={p['pick']} us_planner={p['us_planner']} "
              f"us_pick={p['us_pick']} ratio={p['ratio'] and round(p['ratio'], 3)} compositions={p['compositions']}",
              flush=True)
    errors = [x for x in recs if "error" in x] + [
        {"k": x["k"], "idx": x["idx"], "split": a["split"], "error": a["error"]} for x in rows for a in x["arms"]
        if "error" in a]
    ok = all(picks.get(int(k), {}).get("pick") is not None for k in comps)
    record = {"item": "expedition-D-split-02", "mode": "compositions", "q_len": COMP_Q,
              "seconds": round(time.time() - t0, 1), "exitcodes": exitcodes, "unsafe_excluded": unsafe, "isolate": iso and {x: iso[x] for x in (
                  "unsafe", "overrun_only", "crashes", "not_run", "done")}, "input": comps, "cases": rows,
              "picks": picks, "errors": errors, "stopped": [x for x in recs if "stopped_at" in x]}
    path.write_text(json.dumps(record, indent=1, default=str))
    print(f"SPLITSWEEP-COMP-SUMMARY compositions={len(rows)}/{len(comp_list(comps))} errors={len(errors)} "
          f"unsafe_excluded={unsafe} seconds={record['seconds']} record={path}", flush=True)
    print("SPLITSWEEP-COMP DONE" if ok else "SPLITSWEEP-COMP FAIL", flush=True)
    return 0 if ok else 1


# ---------------------------------------------------------------------------------------------------------- main
def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--deadline-s", type=float, default=600.0)
    ap.add_argument("--isolate", action="append", type=parse_case, metavar="q,B,ctx")
    ap.add_argument("--isolate-s", type=float, default=600.0, help="no isolation job starts after this")
    ap.add_argument("--isolate-splits", type=lambda v: [int(x) for x in v.split(",")], metavar="s,s,...",
                    help="isolate only these splits (q_len > 15 cases: the M absorbed-prefill numerics, H-acc)")
    ap.add_argument("--isolate-pshift", type=lambda v: [int(x) for x in v.split(",")], default=[0], metavar="h,h,...",
                    help="run every isolate job once per P shift (0 = stock kernel; H-P8 test, M-H02)")
    ap.add_argument("--isolate-coexist", action="store_true",
                    help="shifted jobs also run the stock kernel before and after, in one process (REF-locator prep)")
    ap.add_argument("--compositions", nargs="?", const=str(PKG_COMPS), metavar="PATH")
    ap.add_argument("--comp-splits", type=lambda v: [int(x) for x in v.split(",")], metavar="s,s,...",
                    help="forced splits of the composition sweep (default COMP_SPLITS); the planner's is always run")
    ap.add_argument("--comp-k", type=lambda v: {x for x in v.split(",")}, metavar="k,k,...",
                    help="sweep only these batch sizes k of the compositions file")
    ap.add_argument("--isolate-job", help=argparse.SUPPRESS)
    return ap


def spawn(target, world, out_dir, t0, deadline_s, *extra):
    """One spawned process per GPU: target(rank, out_dir, t_end, world, *extra); exit codes."""
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=target, args=(r, out_dir, t0 + deadline_s - 90, world, *extra)) for r in range(world)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=max(1.0, deadline_s - 20 - (time.time() - t0)))
    for p in procs:
        if p.is_alive():
            p.terminate()
    return [p.exitcode for p in procs]


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.isolate_job:
        return isolate_job(json.loads(args.isolate_job))
    t0 = time.time()
    world = W if args.gate else min(W, torch.cuda.device_count())
    if args.isolate or args.compositions:  # isolation first: its crashes stay in its own processes
        t_iso = t0 + min(args.isolate_s, args.deadline_s - 120)
        iso = run_isolate(args.isolate, world, t_iso, splits=args.isolate_splits, shifts=args.isolate_pshift,
                          coexist=args.isolate_coexist) \
            if args.isolate else None
        if not args.compositions:
            return 0 if iso["done"] else 1
        comps = sk.dedupe_compositions(json.loads(Path(args.compositions).read_text()))
        if args.comp_k:
            comps = {k: v for k, v in comps.items() if k in args.comp_k}
        unsafe, path = (iso["unsafe"] if iso else []), results_path("splitcomp.json")
        codes = spawn(_comp_worker, world, str(path.parent), t0, args.deadline_s, comp_list(comps), unsafe,
                      args.comp_splits)
        return report_comp(path, world, t0, codes, comps, unsafe, iso)
    target, report, name = ((_gate_rank, report_gate, "splitsweep-gate.json") if args.gate
                            else (_sweep_worker, report_sweep, "splitsweep.json"))
    path = results_path(name)
    return report(path, world, t0, spawn(target, world, str(path.parent), t0, args.deadline_s))


if __name__ == "__main__":
    sys.exit(main())
