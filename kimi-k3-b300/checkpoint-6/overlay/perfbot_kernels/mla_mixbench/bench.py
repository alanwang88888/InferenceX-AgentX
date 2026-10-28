"""Kernel-only bench of MLA prefill attention at the 12 M-03 mixed-step shapes (README.md). Run in the server env:

  python3 -B -m perfbot_kernels.mla_mixbench.bench [--variants image,image_split,absorbed] [--layers 24]
         [--deadline-s S] [--calib-mode synced|pipelined] [--no-graphs] [--shapes 0,3,...]
  python3 -B -m perfbot_kernels.mla_mixbench.bench --split-sweep    # image_split S in 1..24 on 4 shapes
  python3 -B -m perfbot_kernels.mla_mixbench.bench --gate           # image_split landing gate

8 processes (one per GPU, NCCL + symmetric memory). Per shape (one request: n new tokens after ctx cached tokens,
strided DCP 8 layout, 12 local heads, fp8 latent cache page 64) every variant runs on the same per-rank inputs.

BENCH (default). Pass 1 (eager): warm-up (its wall recorded as first_call_s: JIT compiles land here), outputs,
correctness (each variant vs image, rel-L2 over all ranks, against the variant's bound; each vs an fp64 reference
on <= 8 rows), then the wall of one layer call: 5 reps x 20 calls, median, two ways:
  synced     every call starts on an idle GPU (sync before, events around one call): the in-server shadow timing,
             which synced after each layer call (mla_absorb.shadow_report). The calibration uses this by default.
  pipelined  20 calls back to back between two events (CPU launch cost hidden behind the GPU).
Pass 2 (after every shape's eager record is saved): CUDA graph of one call, 5 x 20 replays, where every rank
captured it. Rank 0's values are the record; every rank's values are in the JSON.
CALIBRATION (the bench's validity gate): per variant with an M-03 wall (image, absorbed):
  ratio = us_per_layer x layers / m03_ms; ok when |ratio - 1| <= 0.25 on >= 10 of the 12 shapes.
  `PERFBOT-MIXBENCH-CALIBRATION ok=<bool>`; exit 0 iff ok. Record: $PERFBOT_WORKSPACE/$PERFBOT_RESULTS/mixbench.json.

SPLIT SWEEP (--split-sweep): image and image_split at S in SWEEP_S on SWEEP_SHAPES; us per layer per S, the grid
size (CTAs), rel-L2 and bit-equality vs image; knee = smallest S within 5 % of the best. Record mixbench-sweep.json.

GATE (--gate): gate_cases() = n in GATE_N x ctx in GATE_CTX, n in GATE_N_512K at ctx 512000, and the
multi-request steps GATE_MULTI; image_split (the server's mla_split backend) at S = default and S = 1, image and
both vs an fp64 reference per request (mla_absorb.ref_attention over the latent all-gathered from the cache shards
and placed with mla_absorb.global_positions; <= 16 rows per request): PASS iff on every case, request and rank
rel(image_split) <= 1.02 x rel(image) and the split path ran, plus a CUDA graph of the image_split chain at
GATE_GRAPH with 2 replays on new q bit-equal to eager.
`PERFBOT-MIXBENCH-GATE pass=<bool>`; exit 0 iff pass. Record mixbench-gate.json (per-case, per-rank table).
"""

import argparse
import json
import os
import statistics
import sys
import time
import traceback
import types
from pathlib import Path

import torch

import perfbot_kernels.mla_mixbench as mb
from perfbot_kernels.mla_mixbench import H, HL, I, L, P, PAGE, R, SCALE, V, W, WORKSPACE_ROWS

PORT = 29571
REPS, ITERS = 5, 20
BAND, MIN_OK = 0.25, 10
REF_ROWS = 8
FP8 = torch.float8_e4m3fn
DEFAULT_VARIANTS = "image,image_split,absorbed"
SWEEP_S = [1, 2, 4, 8, 12, 16, 24]
SWEEP_SHAPES = [(128, 124000), (256, 471000), (768, 171000), (2560, 115000)]  # matched to shapes.json by n, ctx
KNEE = 0.05
GATE_N = [120, 256, 512, 2048]
GATE_CTX = [30000, 100000, 300000]
GATE_N_512K, GATE_CTX_512K = [120, 512, 2048], 512000  # up to the router's ctx cap (mla_split.MAX_CTX_DEFAULT)
# Multi-request steps (rows in server order, every n > T = 512, so inside route A's domain), through the plan's
# q_index / perm path. With 65536 workspace rows per chunk: (a) 30k fits, the 100k request continues into chunk 2;
# (b) ctx < chunk next to a 300k request over 5 chunks; (c) 3 requests in one chunk, mixed n (chunk S from n 2048);
# (d) a ctx-0 row (no chunk, empty token slice), a 40k row and a 512k row.
GATE_MULTI = [[[640, 30000], [1100, 100000]],
              [[700, 2000], [900, 300000]],
              [[520, 8000], [2048, 20000], [768, 30000]],
              [[600, 0], [1500, 40000], [800, 512000]]]
GATE_RATIO = 1.02


def gate_cases() -> list[list[list[int]]]:
    """Every gate case as its request list [[n, ctx], ...]: the n x ctx grid, the 512k row, the multi-request steps."""
    single = [(n, c) for c in GATE_CTX for n in GATE_N] + [(n, GATE_CTX_512K) for n in GATE_N_512K]
    return [[[n, c]] for n, c in single] + GATE_MULTI
GATE_ROWS = 16
GATE_GRAPH = (120, 30000)


def results_path(name="mixbench.json") -> Path:
    root = Path(os.environ.get("PERFBOT_WORKSPACE", "/workspace"))
    out = root / os.environ.get("PERFBOT_RESULTS", "results/measure")
    out.mkdir(parents=True, exist_ok=True)
    return out / name


def sweep_indices(shapes) -> list[int]:
    """shapes.json index for each SWEEP_SHAPES entry: same n, nearest ctx."""
    out = []
    for n, ctx in SWEEP_SHAPES:
        cands = [i for i, s in enumerate(shapes) if s["n"] == n]
        out.append(min(cands, key=lambda i: abs(shapes[i]["ctx"] - ctx)))
    return out


class Env:
    """Per-rank resources shared by all shapes: the absorbed chain's workspaces (selftest.Rank), the image path's
    chunked-context workspace, DCP kv-gather workspace, prefill backends, and one layer's weights."""

    def __init__(self, rank, dev):
        import torch.distributed as dist

        import perfbot_kernels.mla_absorb.selftest as st

        self.rank, self.dev, self.dist = rank, dev, dist
        st.MAXT = mb.dcp_workspace_max_tokens()  # the server's DCP workspace (8192): pad 4096 rows must fit
        self.absorb = st.Rank(rank, dev)  # symm mem for WORLD, q-gather + a2a workspaces, tokenspeed workspace
        import vllm._custom_ops  # noqa: F401  (torch.ops._C / _C_cache_ops)
        from vllm.v1.attention.ops import dcp
        from vllm.v1.worker.workspace import init_workspace_manager

        from perfbot_kernels.mla_mixbench import variants

        # MLACommonMetadataBuilder.__init__ (pinned mla_attention.py:2203-2220): workspace of size + size / dcp rows
        # in the model dtype, and the manager's kv gather over it. init_kv_gather would build this same
        # DirectDCPKVGatherWorkspace (overlay dcp.py:1407); it is built directly on the WORLD group here.
        self.chunk_ws = torch.empty(WORKSPACE_ROWS + WORKSPACE_ROWS // W, L + R, dtype=torch.bfloat16, device=dev)
        self.kv_ws = dcp.DirectDCPKVGatherWorkspace(dist.group.WORLD, dev, WORKSPACE_ROWS, L + R, torch.bfloat16, 1)
        mgr = object.__new__(dcp.MLADCPManager)
        mgr._kv_gather = self.kv_ws.gather
        self.dcp_manager = mgr
        init_workspace_manager(dev)
        trt, split, fa4 = variants.backend_classes()
        kw = dict(num_heads=HL, scale=SCALE, kv_lora_rank=L, qk_nope_head_dim=P, qk_rope_head_dim=R, v_head_dim=V,
                  vllm_config=None)
        self.backends = {"image": trt(**kw), "image_split": split(**kw), "fa4_split": fa4(**kw)}
        self.backends["image_split"].splits_fn = mb.default_splits  # bench only: PERFBOT_MIXBENCH_SPLITS override
        g = torch.Generator(device=dev).manual_seed(11)
        w_all = (torch.randn(H * (P + V), L, device=dev, generator=g) * L ** -0.5).to(torch.bfloat16)
        # ColumnParallelLinear shard of kv_b_proj: this rank's 12 heads x [k_nope 128 | v 128] rows.
        self.kv_b_proj = variants.KvBProj(w_all[rank * HL * (P + V) : (rank + 1) * HL * (P + V)].contiguous())
        # W_UK_T / W_UV as process_weights_after_loading (overlay mla.py:430-448), all heads (the chain slices).
        w_uk, w_uv = w_all.T.reshape(L, H, P + V).split([P, V], dim=-1)
        self.absorb.w_uk_t = w_uk.permute(1, 2, 0).contiguous()
        self.absorb.w_uv = w_uv.transpose(0, 1).contiguous()
        self.one = torch.ones(1, dtype=torch.float32, device=dev)
        self.mine = slice(rank * HL, (rank + 1) * HL)
        mla_mod = sys.modules.get("vllm.models.kimi_k3.nvidia.mla")
        self.sources = {"dcp": dcp.__file__, "selftest": st.__file__, "mla": getattr(mla_mod, "__file__", None),
                        "trtllm_ragged": sys.modules[trt.__module__].__file__}

    def barrier(self):
        torch.cuda.synchronize()
        self.dist.barrier(device_ids=[self.dev.index])

    def all_sum(self, values):
        t = torch.tensor(values, dtype=torch.float64, device=self.dev)
        self.dist.all_reduce(t)
        return t.tolist()

    def any_flag(self, flag: bool) -> bool:
        t = torch.tensor([1 if flag else 0], dtype=torch.int32, device=self.dev)
        self.dist.all_reduce(t, op=self.dist.ReduceOp.MAX)
        return bool(t.item())


def make_case(env, shape, idx, page=PAGE):
    """Global inputs (identical on every rank: same seed) and this rank's paged shard of each request's context.

    shape: {"n", "ctx"} (one request) or {"reqs": [[n, ctx], ...]} (one prefill step of several requests, the
    server's row order). Request r owns physical pages [page_off, page_off + pages) (block table row r); its new
    tokens are rows [q_off, q_off + n) of q / kv_c / k_pe. One request draws the same random numbers as before."""
    dev, rank = env.dev, env.rank
    reqs = [types.SimpleNamespace(n=n, ctx=c) for n, c in (shape.get("reqs") or [[shape["n"], shape["ctx"]]])]
    g = torch.Generator(device=dev).manual_seed(5000 + 97 * idx)
    q_off = page_off = 0
    for r in reqs:
        r.G, r.q_off = r.ctx + r.n, q_off
        r.ctx_lat = torch.empty(r.ctx, L + R, dtype=FP8, device=dev)
        for a in range(0, r.ctx, 1 << 16):  # fp32 temp in 64k-row pieces
            b = min(r.ctx, a + (1 << 16))
            r.ctx_lat[a:b] = torch.randn(b - a, L + R, device=dev, generator=g).to(FP8)
        r.local = mb.shard_positions(rank, r.G, W, I, dev).numel()
        r.pages = -(-max(r.local, -(-r.G // W)) // page)
        r.page_off = page_off
        q_off, page_off = q_off + r.n, page_off + r.pages
    n = q_off
    kv_c = torch.randn(n, L, device=dev, generator=g).to(torch.bfloat16)
    k_pe = torch.randn(n, 1, R, device=dev, generator=g).to(torch.bfloat16)
    q_nope = torch.randn(n, H, P, device=dev, generator=g).to(torch.bfloat16)
    q_pe = torch.randn(n, H, R, device=dev, generator=g).to(torch.bfloat16)
    q = torch.cat([q_nope[:, env.mine], q_pe[:, env.mine]], -1).contiguous()  # this rank's heads [n, 12, 192]
    cache = torch.zeros(page_off * page, L + R, dtype=torch.uint8, device=dev)  # fp8 cache stored as uint8
    block_table = torch.zeros(len(reqs), max(r.pages for r in reqs), dtype=torch.int32, device=dev)
    slots = []
    for i, r in enumerate(reqs):
        base = r.page_off * page
        pos_ctx = mb.shard_positions(rank, r.ctx, W, I, dev)
        cache[base : base + pos_ctx.numel()] = r.ctx_lat[pos_ctx].view(torch.uint8)
        block_table[i, : r.pages] = torch.arange(r.page_off, r.page_off + r.pages, dtype=torch.int32, device=dev)
        p_new = torch.arange(r.ctx, r.G, device=dev)
        on_rank = (p_new // I) % W == rank
        local_idx = base + (p_new // (I * W)) * I + p_new % I
        slots.append(torch.where(on_rank, local_idx, torch.full_like(p_new, -1)).to(torch.int64))
    one = reqs[0] if len(reqs) == 1 else None  # the single-request fields the bench / absorbed variant read
    return types.SimpleNamespace(
        env=env, idx=idx, reqs=reqs, n=n, ctx=one and one.ctx, G=one and one.G, local=one and one.local,
        ctx_lat=one and one.ctx_lat, page=page, q=q, q_nope=q_nope, q_pe=q_pe, kv_c=kv_c, k_pe=k_pe,
        slots=torch.cat(slots), cache_u8=cache.view(page_off, page, L + R), block_table=block_table, memo={})


def _events():
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def time_eager(env, fn, synced: bool):
    reps = []
    for _ in range(REPS):
        env.barrier()
        if synced:
            total = 0.0
            for _ in range(ITERS):
                a, b = _events()
                a.record()
                fn()
                b.record()
                torch.cuda.synchronize()
                total += a.elapsed_time(b)
            reps.append(total * 1e3 / ITERS)
        else:
            a, b = _events()
            a.record()
            for _ in range(ITERS):
                fn()
            b.record()
            torch.cuda.synchronize()
            reps.append(a.elapsed_time(b) * 1e3 / ITERS)
    return round(statistics.median(reps), 1), [round(x, 1) for x in reps]


def capture(env, fn):
    """CUDA graph of one call after a side-stream warm-up; (graph, error). The graph is usable only if every rank
    captured (callers check env.any_flag(error))."""
    err, gr = None, torch.cuda.CUDAGraph()
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn()
        torch.cuda.current_stream().wait_stream(s)
        env.barrier()
        with torch.cuda.graph(gr):
            fn()
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"[:600]
    return gr, err


def time_graph(env, fn):
    """CUDA graph of one call, replayed 5 x 20; (us, reps, error)."""
    gr, err = capture(env, fn)
    if env.any_flag(err is not None):
        return None, [], err or "capture failed on another rank"
    reps = []
    for _ in range(REPS):
        env.barrier()
        a, b = _events()
        a.record()
        for _ in range(ITERS):
            gr.replay()
        b.record()
        torch.cuda.synchronize()
        reps.append(a.elapsed_time(b) * 1e3 / ITERS)
    del gr
    return round(statistics.median(reps), 1), [round(x, 1) for x in reps], None


def rel(x, ref):
    d = x.double() - ref.double()
    return [(d ** 2).sum().item(), (ref.double() ** 2).sum().item()]


def correctness(env, case, outs):
    """Per variant: rel-L2 vs image (this rank and all ranks; vs the variant's bound) and vs fp64 on <= REF_ROWS rows."""
    n = case.n
    sel = torch.unique(torch.linspace(0, n - 1, min(REF_ROWS, n), device=env.dev).round().long())
    new_lat = torch.cat([case.kv_c, case.k_pe.view(n, R)], -1)
    ref = mb.ref_rows(case.ctx_lat, new_lat, case.q_nope[sel][:, env.mine], case.q_pe[sel][:, env.mine],
                      env.absorb.w_uk_t[env.mine], env.absorb.w_uv[env.mine], sel, SCALE)
    img = outs.get("image")
    res = {}
    for name, out in outs.items():
        d_img = rel(out, img) if img is not None else [0.0, 1.0]
        d_ref = rel(out[sel], ref)
        tot = env.all_sum(d_img + d_ref)
        bound = mb.VARIANTS[name].rel_bound
        r_img = (tot[0] / max(tot[1], 1e-300)) ** 0.5 if img is not None else None
        res[name] = {
            "finite": bool(torch.isfinite(out).all()),
            "rel_vs_image": r_img,
            "rel_vs_image_rank": (d_img[0] / max(d_img[1], 1e-300)) ** 0.5 if img is not None else None,
            "rel_bound": bound, "rel_ok": None if (bound is None or r_img is None) else r_img <= bound,
            "ref_rel": (tot[2] / tot[3]) ** 0.5, "ref_rel_rank": (d_ref[0] / max(d_ref[1], 1e-300)) ** 0.5,
            "ref_rows": sel.tolist()}
    return res


def warm(env, fn, case):
    """First call (untimed; JIT compiles land here): (output clone, seconds, error). Failure agreed over ranks."""
    t, err, out = time.time(), None, None
    try:
        out = fn(case).clone()
        torch.cuda.synchronize()
    except Exception:  # noqa: BLE001
        err = traceback.format_exc()[-1500:]
        traceback.print_exc()
    if env.any_flag(err is not None):
        return None, round(time.time() - t, 2), err or "failed on another rank"
    return out, round(time.time() - t, 2), None


def unavailable(env, names):
    """{name: reason} for variants whose available(env) check fails on any rank."""
    out = {}
    for name in names:
        check = mb.VARIANTS[name].available
        reason = None if check is None else check(env)
        if env.any_flag(reason is not None):
            out[name] = reason or "unavailable on another rank"
    return out


# ------------------------------------------------------------------------------------------------- bench mode
def run_bench(env, args, rec, save, t0):
    shapes = mb.load_shapes()
    todo = [i for i in range(len(shapes)) if args.shapes is None or i in args.shapes]
    rec["unavailable"] = unavailable(env, args.variants)
    names = [v for v in args.variants if v not in rec["unavailable"]]
    for i in todo:  # pass 1: eager, correctness
        if env.any_flag(time.time() - t0 > args.deadline_s - 120):
            rec["stopped_at"] = {"pass": 1, "shape": i}
            break
        ts, sh = time.time(), shapes[i]
        case = make_case(env, sh, i)
        row = {"idx": i, "cls": sh["cls"], "n": sh["n"], "ctx": sh["ctx"], "variants": {}}
        outs = {}
        for name in names:
            fn = mb.VARIANTS[name].run
            v = row["variants"][name] = {}
            out, v["first_call_s"], err = warm(env, fn, case)
            if err:
                v["error"] = err
                continue
            outs[name] = out
            fn(case)
            v["eager_synced_us"], v["reps_synced"] = time_eager(env, lambda: fn(case), True)
            v["eager_pipelined_us"], v["reps_pipelined"] = time_eager(env, lambda: fn(case), False)
            if name in case.memo and hasattr(case.memo[name], "chunks"):
                row["chunks"] = case.memo[name].chunks
        if "image_split" in names:
            row["splits_default"] = mb.default_splits(sh["n"])
        for name, c in correctness(env, case, outs).items():
            row["variants"][name].update(c)
        row["seconds"] = round(time.time() - ts, 1)
        rec["shapes"][str(i)] = row
        save()
        if env.rank == 0:
            print("MIXBENCH-PROGRESS " + json.dumps({k: row.get(k) for k in ("idx", "n", "ctx", "chunks", "seconds")} | {
                n_: {k: v_.get(k) for k in ("eager_synced_us", "eager_pipelined_us", "rel_vs_image", "ref_rel")}
                for n_, v_ in row["variants"].items()}), flush=True)
        del case, outs
        torch.cuda.empty_cache()
    if args.graphs:
        for i in todo:  # pass 2: graphs (a capture failure cannot cost an eager record)
            if str(i) not in rec["shapes"] or env.any_flag(time.time() - t0 > args.deadline_s - 90):
                rec.setdefault("stopped_at", {"pass": 2, "shape": i})
                break
            case = make_case(env, shapes[i], i)
            for name in names:
                if "error" in rec["shapes"][str(i)]["variants"].get(name, {}):
                    continue
                fn = mb.VARIANTS[name].run
                fn(case)
                us, reps, err = time_graph(env, lambda: fn(case))
                rec["graphs"].setdefault(str(i), {})[name] = {"graph_us": us, "reps_graph": reps, "graph_error": err}
            save()
            del case
            torch.cuda.empty_cache()


def calibrate(rows, variants, layers, mode):
    """Per variant with an M-03 wall: ratio per shape and the >= MIN_OK-of-12 rule. Missing shapes count as misses."""
    out = {}
    shapes = mb.load_shapes()
    for name in variants:
        key = mb.VARIANTS[name].m03_key
        if key is None:
            continue
        per = []
        for i, sh in enumerate(shapes):
            us = ((rows.get(str(i)) or {}).get("variants", {}).get(name) or {}).get(f"eager_{mode}_us")
            ratio = None if us is None else us * layers / 1e3 / sh[key]
            per.append({"idx": i, "ratio": ratio, "ok": ratio is not None and abs(ratio - 1) <= BAND})
        n_ok = sum(p["ok"] for p in per)
        out[name] = {"ok": n_ok >= MIN_OK, "n_ok": n_ok, "of": len(shapes), "shapes": per}
    return out


def report_bench(args, ranks, path, t0, exitcodes) -> int:
    r0 = ranks[0].get("shapes", {})
    rows = []
    for i, sh in enumerate(mb.load_shapes()):
        row = r0.get(str(i))
        if row is None:
            continue
        for name in args.variants:
            if name in ranks[0].get("unavailable", {}):
                continue
            v = dict(row["variants"].get(name) or {})
            v.update((ranks[0].get("graphs", {}).get(str(i)) or {}).get(name) or {})
            for k in ("eager_synced_us", "eager_pipelined_us"):
                v[f"{k}_all_ranks"] = [(((rk.get("shapes", {}).get(str(i)) or {}).get("variants", {}).get(name) or {})
                                        .get(k)) for rk in ranks]
            v["graph_us_all_ranks"] = [((rk.get("graphs", {}).get(str(i)) or {}).get(name) or {}).get("graph_us")
                                       for rk in ranks]
            key = mb.VARIANTS[name].m03_key
            m03 = sh.get(key) if key else None
            us = v.get(f"eager_{args.calib_mode}_us")
            v["x_layers_ms"] = None if us is None else round(us * args.layers / 1e3, 3)
            v["m03_ms"] = m03
            v["ratio"] = None if (us is None or m03 is None) else round(v["x_layers_ms"] / m03, 3)
            rows.append({"idx": i, "cls": sh["cls"], "n": sh["n"], "ctx": sh["ctx"], "chunks": row.get("chunks"),
                         "splits_default": row.get("splits_default"), "variant": name, **v})
            rel_ = v.get("rel_vs_image")
            print(f"MIXBENCH cls={sh['cls']} n={sh['n']} ctx={sh['ctx']} variant={name} us_per_layer={us} "
                  f"x{args.layers}_ms={v['x_layers_ms']} m03_ms={m03} ratio={v['ratio']} "
                  f"rel_vs_image={'NA' if rel_ is None else f'{rel_:.4g}'} rel_ok={v.get('rel_ok')} "
                  f"ref_rel={v.get('ref_rel')} graph_us={v.get('graph_us')} pipelined_us={v.get('eager_pipelined_us')} "
                  f"first_call_s={v.get('first_call_s')}" + (" ERROR" if "error" in v else ""), flush=True)
    for name, reason in ranks[0].get("unavailable", {}).items():
        print(f"MIXBENCH variant={name} unavailable: {reason}", flush=True)
    cal = calibrate(r0, args.variants, args.layers, args.calib_mode)
    ok = bool(cal) and all(c["ok"] for c in cal.values())
    for name, c in cal.items():
        print(f"PERFBOT-MIXBENCH-CALIBRATION-{name} ok={c['ok']} n_ok={c['n_ok']}/{c['of']} band={BAND} mode="
              f"{args.calib_mode} ratios=" + ",".join("NA" if p["ratio"] is None else f"{p['ratio']:.3f}"
                                                      for p in c["shapes"]), flush=True)
    rel_checks = {name: [r.get("rel_ok") for r in rows if r["variant"] == name]
                  for name in args.variants if mb.VARIANTS[name].rel_bound is not None}
    for name, oks in rel_checks.items():
        print(f"MIXBENCH-REL variant={name} bound={mb.VARIANTS[name].rel_bound} ok_shapes={sum(o is True for o in oks)}"
              f"/{len(oks)}", flush=True)
    record = {"item": "expedition-M-mixbench", "mode": "bench", "args": vars(args), "exitcodes": exitcodes,
              "seconds": round(time.time() - t0, 1), "calibration": cal, "calibration_ok": ok, "rows": rows,
              "rel_checks": rel_checks, "unavailable": ranks[0].get("unavailable"),
              "sources": ranks[0].get("sources"), "setup_s": [r.get("setup_s") for r in ranks],
              "stopped_at": ranks[0].get("stopped_at"),
              "errors": {r["rank"]: r["error"][-1500:] for r in ranks if "error" in r}}
    path.write_text(json.dumps(record, indent=1, default=str))
    print(f"PERFBOT-MIXBENCH-CALIBRATION ok={ok} record={path}", flush=True)
    return 0 if ok else 1


# ------------------------------------------------------------------------------------------------- split sweep
def run_sweep(env, args, rec, save, t0):
    shapes = mb.load_shapes()
    backend = env.backends["image_split"]
    img_fn, split_fn = mb.VARIANTS["image"].run, mb.VARIANTS["image_split"].run
    for i in sweep_indices(shapes):
        if env.any_flag(time.time() - t0 > args.deadline_s - 120):
            rec["stopped_at"] = {"shape": i}
            break
        sh = shapes[i]
        case = make_case(env, sh, i)
        row = {"idx": i, "n": sh["n"], "ctx": sh["ctx"], "S": {}}
        img, row["image_first_call_s"], err = warm(env, img_fn, case)
        if err:
            row["error"] = err
            rec["shapes"][str(i)] = row
            save()
            continue
        row["image_synced_us"], _ = time_eager(env, lambda: img_fn(case), True)
        row["image_pipelined_us"], _ = time_eager(env, lambda: img_fn(case), False)
        chunk_kv = [int(c.seq_lens.sum()) for c in case.memo["image"].md.prefill.chunked_context.chunks]
        row["chunk_kv"] = chunk_kv
        for S in SWEEP_S:
            backend.splits_override = S
            out, first, err = warm(env, split_fn, case)
            s_eff = [min(S, k) for k in chunk_kv]
            v = row["S"][str(S)] = {"S_eff": s_eff, "ctas": [mb.ctas(sh["n"], 1, s) for s in s_eff], "first_call_s": first}
            if err:
                v["error"] = err
                continue
            d = env.all_sum(rel(out, img))
            v["rel_vs_image"] = (d[0] / max(d[1], 1e-300)) ** 0.5
            v["equal_image"] = not env.any_flag(not torch.equal(out, img))
            v["synced_us"], v["reps_synced"] = time_eager(env, lambda: split_fn(case), True)
            v["pipelined_us"], v["reps_pipelined"] = time_eager(env, lambda: split_fn(case), False)
        backend.splits_override = None
        rec["shapes"][str(i)] = row
        save()
        del case
        torch.cuda.empty_cache()


def report_sweep(args, ranks, path, t0, exitcodes) -> int:
    rows, done = [], True
    r0 = ranks[0].get("shapes", {})
    for i in sweep_indices(mb.load_shapes()):
        row = r0.get(str(i))
        if row is None or "error" in row:
            done = False
            continue
        for key in ("synced_us", "pipelined_us"):
            pts = {int(s): v[key] for s, v in row["S"].items() if v.get(key) is not None}
            if not pts:
                done = False
                continue
            best = min(pts.values())
            knee = min(s for s, us in pts.items() if us <= (1 + KNEE) * best)
            row[f"knee_{key}"] = {"knee_S": knee, "best_S": min(pts, key=pts.get), "best_us": best}
        for s, v in row["S"].items():
            print(f"MIXBENCH-SWEEP n={row['n']} ctx={row['ctx']} S={s} S_eff={v.get('S_eff')} ctas={v.get('ctas')} "
                  f"synced_us={v.get('synced_us')} pipelined_us={v.get('pipelined_us')} "
                  f"rel_vs_image={v.get('rel_vs_image')} equal_image={v.get('equal_image')}"
                  + (" ERROR" if "error" in v else ""), flush=True)
        k = row.get("knee_synced_us") or {}
        print(f"MIXBENCH-SWEEP-KNEE n={row['n']} ctx={row['ctx']} knee_S={k.get('knee_S')} best_S={k.get('best_S')} "
              f"best_us={k.get('best_us')} image_us={row.get('image_synced_us')} S_default="
              f"{mb.default_splits(row['n'])} pipelined_knee={(row.get('knee_pipelined_us') or {}).get('knee_S')}",
              flush=True)
        rows.append(row)
    record = {"item": "expedition-M-mixbench-sweep", "mode": "split_sweep", "args": vars(args), "exitcodes": exitcodes,
              "seconds": round(time.time() - t0, 1), "rows": rows, "all_ranks": [r.get("shapes") for r in ranks],
              "errors": {r["rank"]: r["error"][-1500:] for r in ranks if "error" in r}}
    path.write_text(json.dumps(record, indent=1, default=str))
    ok = done and not record["errors"]
    print(f"PERFBOT-MIXBENCH-SWEEP done={ok} record={path}", flush=True)
    return 0 if ok else 1


# ---------------------------------------------------------------------------------------------------- gate mode
def gate_reference(env, case, req, sel):
    """fp64 reference for request `req`'s rows `sel` (indices into its n new tokens) from the cache as it stands
    after the variants ran: this rank's padded shard of the request (its pages) all-gathered and placed with
    mla_absorb.global_positions (layout_ok: the context rows equal the generated latent bit for bit; map_rel: the
    new-token rows vs their bf16 source, ~fp8 rounding if the insert landed right), new tokens at their bf16
    values, then mla_absorb.ref_attention (un-absorbed, keys chunked)."""
    from perfbot_kernels import mla_absorb

    dev, n, ctx, G = env.dev, req.n, req.ctx, req.G
    lm = -(-G // W)
    base = req.page_off * case.page
    flat = case.cache_u8.view(-1, L + R)
    gathered = torch.empty(W * lm, L + R, dtype=torch.uint8, device=dev)
    env.dist.all_gather_into_tensor(gathered, flat[base : base + lm].contiguous())
    pos = mla_absorb.global_positions(W, lm, I, dev)
    keep = pos < G
    lat = torch.empty(G, L + R, dtype=torch.float64, device=dev)
    lat[pos[keep]] = gathered[keep].view(FP8).double()  # cache scale 1.0
    layout_ok = bool(torch.equal(lat[:ctx], req.ctx_lat.double()))
    rows = slice(req.q_off, req.q_off + n)
    new = torch.cat([case.kv_c[rows], case.k_pe[rows].view(n, R)], -1).double()
    map_rel = ((lat[ctx:] - new).norm() / new.norm()).item()
    lat[ctx:] = new
    g = req.q_off + sel
    ref = mla_absorb.ref_attention(lat, case.q_nope[g][:, env.mine].double(), case.q_pe[g][:, env.mine].double(),
                                   env.absorb.w_uk_t[env.mine], env.absorb.w_uv[env.mine], ctx + sel, SCALE, L)
    del lat, gathered
    return ref, layout_ok, map_rel


def run_gate(env, args, rec, save, t0):
    backend = env.backends["image_split"]
    img_fn, split_fn = mb.VARIANTS["image"].run, mb.VARIANTS["image_split"].run
    table = rec["gate"] = []
    for idx, reqs in enumerate(gate_cases(), start=1):
        if env.any_flag(time.time() - t0 > args.deadline_s - 60):
            rec["stopped_at"] = {"case": idx, "reqs": reqs}
            save()
            return
        case = make_case(env, {"reqs": reqs}, 1000 + idx)
        row = {"case": idx, "reqs": reqs, "rank": env.rank, "S_default": [mb.default_splits(n) for n, _ in reqs]}
        outs, errs = {}, []
        for key, fn, S in (("image", img_fn, None), ("split_default", split_fn, None), ("split_1", split_fn, 1)):
            backend.splits_override = S
            out, _, err = warm(env, fn, case)
            if err:
                errs.append(f"{key}: {err[-300:]}")
            else:
                outs[key] = out
            if key == "split_default":  # the split path really ran (a silent image fallback would tie trivially)
                row["split_ran"] = bool((getattr(backend, "_split_step", None) or (None, {}))[1])
        backend.splits_override = None
        ratios, row["layout_ok"], row["map_rel"] = [], [], []
        for key in outs:
            row[f"rel_{key}"] = []
        for req in case.reqs:  # per request: no averaging across the requests of a step
            sel = torch.unique(torch.linspace(0, req.n - 1, min(GATE_ROWS, req.n), device=env.dev).round().long())
            ref, layout_ok, map_rel = gate_reference(env, case, req, sel)
            row["layout_ok"].append(layout_ok)
            row["map_rel"].append(map_rel)
            for key, out in outs.items():
                d = rel(out[req.q_off + sel].view(sel.numel(), HL, V), ref)
                row[f"rel_{key}"].append((d[0] / max(d[1], 1e-300)) ** 0.5)
            e_img = row["rel_image"][-1] if "image" in outs else None
            ratios.append(max((row[f"rel_{k}"][-1] if k in outs else float("inf")) for k in ("split_default", "split_1"))
                          / max(e_img or 1e-300, 1e-300))
        row["worst_ratio"] = max(ratios)
        row["ok"] = (not errs and all(row["layout_ok"]) and "image" in outs and row.get("split_ran") is True
                     and row["worst_ratio"] <= GATE_RATIO)
        if errs:
            row["errors"] = errs
        table.append(row)
        save()
        if env.rank == 0:
            print("MIXBENCH-GATE-CASE " + json.dumps(row, default=str), flush=True)
        del case, outs
        torch.cuda.empty_cache()
    # CUDA graph of the image_split chain: 2 replays on new q, bit-equal to eager
    n, ctx = GATE_GRAPH
    case = make_case(env, {"n": n, "ctx": ctx}, 999)
    split_fn(case)
    gr, err = capture(env, lambda: split_fn(case))
    g = {"n": n, "ctx": ctx, "S": mb.default_splits(n), "capture_error": err, "replays_bit_equal": []}
    if not env.any_flag(err is not None):
        for k in range(2):
            gen = torch.Generator(device=env.dev).manual_seed(500 + k)
            case.q.copy_(torch.randn(case.q.shape, device=env.dev, generator=gen).to(case.q.dtype))
            env.barrier()
            gr.replay()
            env.barrier()
            got = case.memo["image_split"].out.clone()
            split_fn(case)
            env.barrier()
            g["replays_bit_equal"].append(bool(torch.equal(got, case.memo["image_split"].out)))
    g["ok"] = err is None and len(g["replays_bit_equal"]) == 2 and all(g["replays_bit_equal"])
    rec["gate_graph"] = g
    save()


def report_gate(args, ranks, path, t0, exitcodes) -> int:
    table = [row for r in ranks for row in (r.get("gate") or [])]
    graphs = [r.get("gate_graph") or {"ok": False, "rank": r["rank"]} for r in ranks]
    cases = gate_cases()
    expected = len(cases) * W
    cases_ok = len(table) == expected and all(row["ok"] for row in table)
    graph_ok = all(g.get("ok") for g in graphs)
    errors = {r["rank"]: r["error"][-1500:] for r in ranks if "error" in r}
    ok = cases_ok and graph_ok and not errors
    for idx, reqs in enumerate(cases, start=1):
        rs = [row for row in table if row["case"] == idx]
        worst = max((row["worst_ratio"] for row in rs), default=None)
        print(f"MIXBENCH-GATE case={idx} reqs={json.dumps(reqs)} ranks={len(rs)} "
              f"ok={all(r['ok'] for r in rs) and len(rs) == W} worst_ratio={worst} "
              f"split_ran={all(r.get('split_ran') for r in rs)} "
              f"rel_image_max={max((max(r.get('rel_image') or [0]) for r in rs), default=None)} "
              f"map_rel_max={max((max(r.get('map_rel') or [0]) for r in rs), default=None)}", flush=True)
    print(f"MIXBENCH-GATE-GRAPH ok={graph_ok} " + json.dumps([g.get("replays_bit_equal") for g in graphs]), flush=True)
    record = {"item": "expedition-M-mixbench-gate", "mode": "gate", "args": vars(args), "exitcodes": exitcodes,
              "seconds": round(time.time() - t0, 1), "pass": ok, "cases_ok": cases_ok, "graph_ok": graph_ok,
              "rule": f"rel(image_split, S default and 1) <= {GATE_RATIO} x rel(image) vs fp64, every case, request "
                      "and rank; the split path ran on every case",
              "cases": table, "graphs": graphs, "errors": errors}
    path.write_text(json.dumps(record, indent=1, default=str))
    print(f"PERFBOT-MIXBENCH-GATE pass={ok} record={path}", flush=True)
    return 0 if ok else 1


MODES = {"bench": (run_bench, report_bench, "mixbench.json"),
         "split_sweep": (run_sweep, report_sweep, "mixbench-sweep.json"),
         "gate": (run_gate, report_gate, "mixbench-gate.json")}


def _rank_main(rank, port, out_dir, args):
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=W, device_id=dev)
    t0 = time.time()
    rec = {"rank": rank, "shapes": {}, "graphs": {}}
    path = Path(out_dir, f"mixbench-{args.mode}-rank{rank}.json")
    save = lambda: path.write_text(json.dumps(rec, indent=1, default=str))
    try:
        with torch.inference_mode():
            env = Env(rank, dev)
            rec["sources"] = env.sources
            rec["setup_s"] = round(time.time() - t0, 1)
            MODES[args.mode][0](env, args, rec, save, t0)
    except Exception:  # noqa: BLE001
        rec["error"] = traceback.format_exc()
        traceback.print_exc()
    rec["seconds"] = round(time.time() - t0, 1)
    save()
    print(f"MIXBENCH rank={rank} mode={args.mode} done error={'error' in rec}", flush=True)
    try:
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001
        pass


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=DEFAULT_VARIANTS)
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--deadline-s", type=float, default=1500.0)
    ap.add_argument("--calib-mode", choices=("synced", "pipelined"), default="synced")
    ap.add_argument("--no-graphs", dest="graphs", action="store_false")
    ap.add_argument("--shapes", default=None, help="comma list of shape indices (default: all 12)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--split-sweep", dest="mode", action="store_const", const="split_sweep")
    mode.add_argument("--gate", dest="mode", action="store_const", const="gate")
    ap.set_defaults(mode="bench")
    args = ap.parse_args(argv)
    import perfbot_kernels.mla_mixbench.variants  # noqa: F401  (registers the variants)

    args.variants = [v for v in args.variants.split(",") if v]
    if args.mode != "bench":
        args.variants = ["image", "image_split"]
    unknown = [v for v in args.variants if v not in mb.VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants {unknown}; registered {sorted(mb.VARIANTS)}")
    args.shapes = None if args.shapes is None else [int(x) for x in args.shapes.split(",")]
    return args


def main() -> int:
    args = parse()
    _, report, name = MODES[args.mode]
    path = results_path(name)
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_rank_main, args=(r, PORT, str(path.parent), args)) for r in range(W)]
    t0 = time.time()
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=max(1.0, args.deadline_s + 60 - (time.time() - t0)))
    for p in procs:
        if p.is_alive():
            p.terminate()
    ranks = []
    for r in range(W):
        f = path.parent / f"mixbench-{args.mode}-rank{r}.json"
        ranks.append(json.loads(f.read_text()) if f.is_file() else {"rank": r, "error": "no record"})
    return report(args, ranks, path, t0, [p.exitcode for p in procs])


if __name__ == "__main__":
    sys.exit(main())
