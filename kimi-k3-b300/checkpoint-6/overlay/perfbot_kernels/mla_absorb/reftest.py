"""Kernel-only test of line M's fp64 reference (mla_absorb.reference_report) and of the q_len-13 decode crash of
M-04k. Run in the server env (PERFBOT_PRE_EXEC):

  python3 -B -m perfbot_kernels.mla_absorb.reftest [--deadline-s S] [--r7-deadline-s S] [--r7-only | --skip-r7]

Two stages in separate processes, so a CUDA error in R7 cannot reach R1-R6.
R7 (runs first): one process per GPU, no collectives. tokenspeed_mla_decode with the arguments
  TokenspeedMLAImpl.forward_mqa passes under DCP 8: fp8 q [1, q_len, 96, 576], page 32, seq_lens = this rank's keys,
  max_seq_len = ceil(G / 8), causal_seqs = G, cp_world 8, cp_rank = rank, return_lse, scale 192^-0.5, and the image's
  workspace (num_sms * 96 * 8 * 513 * 4 bytes, vllm tokenspeed_mla._get_workspace). Phases:
    A   q_len in R7_Q x ctx in R7_CTX x causal {1, 0}, in a fresh process;
    ABS one q_len-128 call at ctx 76800 (the absorbed path's bucket);  B  q_len in R7_Q at ctx 76800 after it;
    C   q_len {9, 13, 15} at ctx 76800 and every failed A/B case again, with a new workspace sized for
        q_len max(15, q_len) (num_sms * 96 * q * 513 * 4 bytes).
  Per case: the planner's tile M, fold, split_kv and required workspace (tokenspeed_mla's own functions, and ts_plan,
  this module's host mirror), the provided bytes, the error string or "ok", and a context probe after the call. A dead
  or stalled (R7_HANG_S) worker is restarted at the next case; records: reftest-r7-rank<r>.jsonl.
  R7 PASS = every A/B case has a result on every rank and ts_plan matches the library (diagnostic, not numerics).
R1-R6: 8 ranks (NCCL). Per case (n, ctx) in CASES at page 32 and per layer in LAYERS (kv_b_proj from the checkpoint
  under $PERFBOT_MODEL_DIR or /model when every rank can read it, else seeded random weights with the served dims;
  the record and the REFTEST-WEIGHTS line say which), random activations: the absorbed chain (selftest.Rank.chain)
  and the image prefill (mla_mixbench `image` variant) on the same inputs, then mla_absorb.reference_report on their
  outputs (its PERFBOT-MLAQ-REF line is the in-server format).
  R3 image path vs image_emulation (fp64, the image's quantization points) on layer IMAGE_LAYER at R3_CASE:
     err(image)/err(emulation) vs each truth (t8, mix) in R3_BAND.
  R1 map_rel in [MAP_REL_LO, MAP_REL_MAX] on every case, layer and rank; the gathered context equals the generated
     latent bit for bit (ctx 76800 cases).
  R2 reference_report's chunked truths and new-key mass vs dense_reference (one softmax over the materialized
     latent): rel <= R2_TOL (ctx 76800 cases, first layer).
  R4 injected outputs: image := truth mix, absorbed := truth mix x (1 + 0.01 (rank + 1)) on the sampled rows; every
     reported error equals its closed form within R4_TOL (R4_CASE, first layer).
  R5 sha256 of the cache, block table, metadata tensors, inputs and outputs: equal before and after every call.
  R6 peak allocated memory above the call's start and wall time per call, printed; PASS iff every peak <= R6_PEAK_MIB.
Final line `REFTEST OK` iff R1-R7 all PASS, else `REFTEST FAIL`; exit 0 iff OK.
Record: $PERFBOT_WORKSPACE/$PERFBOT_RESULTS/reftest.json.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
import traceback
import types
from pathlib import Path

import torch

import perfbot_kernels.mla_absorb as mla_absorb

W, HL, L, R, P, V = 8, 12, 512, 64, 128, 128
H = W * HL
PAGE = 32  # the served kernel block size (M-04k server.log: "Setting kv cache block size to 32 for TOKENSPEED_MLA")
SCALE = (P + R) ** -0.5
FP8 = torch.float8_e4m3fn
PORT = 29581
CASES = [(128, 76800), (128, 300000), (128, 500000), (13, 76800)]  # (n, ctx); (13, 76800) = the M-04k crash step
LAYERS = [3, 87, 91, 92]  # 0-based MLA layers (config full_attn_layers 4, 88, 92, 93 are 1-based)
IMAGE_LAYER, R3_CASE, R4_CASE = 91, (128, 76800), (13, 76800)
MAP_REL_LO, MAP_REL_MAX = 0.005, mla_absorb.MAP_REL_MAX  # below LO: the tail was not read from the cache
R2_TOL = R4_TOL = 1e-9
R3_BAND = (0.95, 1.05)
R6_PEAK_MIB = 2048
R7_Q = [5, 8, 9, 13, 15, 16, 32, 64, 73, 74]
R7_CTX = [9600, 76800, 300000]
R7_ABS = (128, 76800)
R7_C_Q = [9, 13, 15]
R7_IMAGE_MAX_Q = 8  # vllm tokenspeed_mla._TOKENSPEED_MAX_Q_LEN
R7_BIG_MAX_Q = 15  # 1 + 2 * 7: the decode threshold under dspark parallel drafting
R7_HANG_S = 240
MODEL = os.environ.get("PERFBOT_MODEL_DIR", "/model")


def results_path(name="reftest.json") -> Path:
    root = Path(os.environ.get("PERFBOT_WORKSPACE", "/workspace"))
    out = root / os.environ.get("PERFBOT_RESULTS", "results/measure")
    out.mkdir(parents=True, exist_ok=True)
    return out / name


# ------------------------------------------------------------------------------------- CPU-testable pure parts
def ts_plan(B, q_len, heads, kv_lora, num_sm, cc, max_seq_len):
    """Host mirror of tokenspeed_mla_decode's planning (mla_decode.py _get_split_kv_and_workspace_size with
    mla_helpers.select_mla_decode_tilers / get_mla_decode_fold_sq_factor): (tile_m, fold, split_kv, workspace bytes).
    The M128 path (every cc but (10, 0), or heads * q_len > 64) ignores the context length."""
    m = 64 if heads * q_len <= 64 and tuple(cc) == (10, 0) else 128
    fold = next((c for c in range(min(q_len, m // heads), 1, -1) if q_len % c == 0), 1)
    S, He = q_len // fold, heads * fold
    if m == 64:  # FP8 get_split_kv, one CTA per M tile
        tiles = -(-max_seq_len // 128)
        heur = min(tiles, max(1, num_sm // B // S))
        split = min(-(-tiles // -(-tiles // heur)), 64)
    else:  # get_split_kv_simplified
        split = min(max(1, num_sm // B // (S * 2)), 32)
    return m, fold, split, 0 if split == 1 else B * He * S * split * (kv_lora + 1) * 4


def ws_bound(heads, kv_lora, num_sm, cc, max_b, max_q, seq_lens=(1, 1 << 16, 1 << 19)):
    """Largest workspace ts_plan asks for over B <= max_b requests of q_len <= max_q (and the given local lengths)."""
    return max(ts_plan(b, q, heads, kv_lora, num_sm, cc, s)[3]
               for b in range(1, max_b + 1) for q in range(1, max_q + 1) for s in seq_lens)


def dense_reference(lat, qn, qp, w_uk_t, w_uv, limit, scale, L_, split):
    """R2's yardstick: fp64 attention over the materialized latent lat [G, L + R], one softmax per head (no chunks);
    also the softmax mass on keys >= split. Returns ([r, HL, V], [r, HL])."""
    G = lat.shape[0]
    cl, pe = lat[:, :L_], lat[:, L_:]
    mask = torch.arange(G, device=lat.device)[None, :] > limit[:, None]
    refs, mass = [], []
    for h in range(qn.shape[1]):
        s = (qn[:, h].double() @ (cl @ w_uk_t[h].double().T).T + qp[:, h].double() @ pe.T) * scale
        p = torch.softmax(s.masked_fill(mask, float("-inf")), -1)
        refs.append(p @ (cl @ w_uv[h].double()))
        mass.append(p[:, split:].sum(-1))
    return torch.stack(refs, 1), torch.stack(mass, 1)


def image_emulation(stored, ctx, bounds, new_c, new_pe, q, sel, w_kv, scale, k_scale=1.0):
    """fp64 emulation of the image prefill for rows `sel` of one request (overlay mla.py _perfbot_prefill_image and
    the pinned MLACommonBaseImpl._context_parallel_compute_prefill_context). Emulated quantization points:
      Q1 q -> fp8, unscaled (fused_mla_qkv_quant_kv_cache_fp8_insert, scale 1);
      Q2 new tokens: [k_nope | v] = bf16(kv_c @ W_kv^T) (kv_b_proj, bf16 out) -> fp8; k_pe (bf16) -> fp8;
      Q3 context: c = bf16(stored * k_scale) (gather_and_maybe_dequant_cache); [k_nope | v] = bf16(c @ W_kv^T) -> fp8
         (kv_nope.to(fp8)); k_pe = fp8(bf16(stored_pe * k_scale));
      Q4 per context chunk (bounds, non-causal) and for the new-token block (causal): softmax attention, output -> bf16
         (the fmha's out dtype), lse -> fp32;
      Q5 the chunk partials merge in order into a bf16 running partial (accumulate_mla_context_chunk ->
         merge_attn_states), then with the new-token partial into the bf16 output.
    Not emulated: fp32 accumulation order inside the GEMMs and the fmha, and the fmha's rounding of P before P.V.
    The second output also rounds P to fp8 as fp8(448 p) / 448 with p = exp(s - row max of the chunk): a diagnostic
    approximation (the kernel's per-tile max and P scale are not known here).
    stored(a, b) -> fp64 stored latent rows (unscaled); new_c [n, L], new_pe [n, R] bf16; q [n, HL, P + R] bf16;
    w_kv [HL * (P + V), L] bf16 (this rank's kv_b_proj shard). Returns two [r, HL, V] fp64 tensors of bf16 values."""
    f8 = lambda x: x.to(FP8).double()  # noqa: E731
    bf = lambda x: x.to(torch.bfloat16).double()  # noqa: E731
    hl, Lr, Rr = q.shape[1], new_c.shape[-1], new_pe.shape[-1]
    Pn = q.shape[2] - Rr
    w = w_kv.view(hl, -1, Lr)
    qs = f8(q[sel])
    mask_new = torch.arange(new_c.shape[0], device=q.device)[None, :] > sel[:, None]

    def merge(a, b):
        (o1, l1), (o2, l2) = a, b
        mx = torch.maximum(l1, l2)
        w1, w2 = torch.exp(l1 - mx), torch.exp(l2 - mx)
        return bf((o1 * w1[:, None] + o2 * w2[:, None]) / (w1 + w2)[:, None]), (mx + torch.log(w1 + w2)).float().double()

    run = [[None, None] for _ in range(hl)]
    blocks = [(a, b) for a, b in bounds] + [None]
    for blk in blocks:
        if blk is None:
            c, pe, mask = new_c.double(), new_pe.double(), mask_new
        else:
            lat = bf(stored(*blk) * k_scale)
            c, pe, mask = lat[:, :Lr], lat[:, Lr:], None
        kpe = f8(pe)
        for h in range(hl):
            kv = f8(bf(c @ w[h].double().T))
            s = (qs[:, h, :Pn] @ kv[:, :Pn].T + qs[:, h, Pn:] @ kpe.T) * scale
            if mask is not None:
                s = s.masked_fill(mask, float("-inf"))
            m = s.max(-1).values
            p = torch.exp(s - m[:, None])
            lsum = p.sum(-1)
            lse = (m + torch.log(lsum)).float().double()
            for i, pp in enumerate((p, f8(p * 448) / 448)):
                part = (bf((pp @ kv[:, Pn:]) / lsum[:, None]), lse)
                run[h][i] = part if run[h][i] is None else merge(run[h][i], part)
    return [torch.stack([run[h][i][0] for h in range(hl)], 1) for i in (0, 1)]


def case_id(phase, q, ctx, causal, tag=""):
    return f"{phase}:{q}:{ctx}:{causal}" + (f":{tag}" if tag else "")


def parse_id(cid):
    ph, q, ctx, causal = cid.split(":")[:4]
    return {"phase": ph, "q_len": int(q), "ctx": int(ctx), "causal": int(causal)}


def r7_cases():
    """(phase A ids, phase B ids) in run order; phase C comes from R7_C_Q and the A/B failures."""
    a = [case_id("A", q, ctx, c) for ctx in R7_CTX for q in R7_Q for c in (1, 0)]
    b = [case_id("B", q, R7_ABS[1], c) for q in R7_Q for c in (1, 0)]
    return a, b


def r7_c_cases(recs):
    """Phase C ids: R7_C_Q at ctx 76800 (both masks) plus every A/B (q, ctx, causal) that did not return ok."""
    keys = [(q, R7_ABS[1], c) for q in R7_C_Q for c in (1, 0)]
    for r in recs:
        if r.get("phase") in ("A", "B") and r.get("rc") not in (None, "ok"):
            k = (r["q_len"], r["ctx"], r["causal"])
            if k not in keys:
                keys.append(k)
    return [case_id("C", *k) for k in keys]


def r7_open(recs):
    """Ids that started but have no result (the worker died or stalled inside them)."""
    done = {r["id"] for r in recs if "rc" in r}
    return [r["id"] for r in recs if r.get("start") and r["id"] not in done]


def r7_read(path):
    recs = []
    if Path(path).is_file():
        for line in Path(path).read_text().splitlines():
            try:
                recs.append(json.loads(line))
            except ValueError:
                pass
    return recs


def r7_path(out_dir, rank):
    return Path(out_dir, f"reftest-r7-rank{rank}.jsonl")


def jd(v) -> str:
    """Compact JSON (no spaces): log lines stay whitespace-split key=value."""
    return json.dumps(v, default=str, separators=(",", ":"))


_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def short_error(e) -> str:
    return " ".join(_ANSI.sub("", f"{type(e).__name__}: {e}").split())[:400]


# ------------------------------------------------------------------------------------------------- R7 (per GPU)
def _r7_worker(rank, out_dir, spawn):
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    path = r7_path(out_dir, rank)
    f = path.open("a")

    def write(rec):
        f.write(json.dumps(rec, default=str) + "\n")
        f.flush()

    try:
        from tokenspeed_mla import get_num_sm, tokenspeed_mla_decode
        from tokenspeed_mla import mla_decode as tmd
        from tokenspeed_mla.mla_helpers import get_mla_decode_fold_sq_factor, select_mla_decode_tilers
    except Exception as e:  # noqa: BLE001
        write({"id": "import", "rc": short_error(e)})
        write({"id": "done"})
        return
    ended = {r["id"] for r in r7_read(path) if "rc" in r}
    num_sm, cc = get_num_sm(dev), tuple(torch.cuda.get_device_capability(dev))
    work = {"image": torch.empty(num_sm * H * R7_IMAGE_MAX_Q * (L + 1) * 4, dtype=torch.int8, device=dev)}
    caches = {}

    def inputs(ctx, q):
        if ctx not in caches:  # this rank's shard of ctx + 128 keys, random fp8, pages in order
            g = torch.Generator(device=dev).manual_seed(77 + ctx)
            local_max = -(-(ctx + R7_ABS[0] - rank) // W)
            pages = -(-local_max // PAGE) + 1
            cache = torch.empty(pages * PAGE, L + R, dtype=FP8, device=dev)
            for a in range(0, cache.shape[0], 1 << 16):
                b = min(cache.shape[0], a + (1 << 16))
                cache[a:b] = (torch.randn(b - a, L + R, device=dev, generator=g) * 0.5).to(FP8)
            caches[ctx] = cache.view(pages, PAGE, L + R), torch.arange(pages, dtype=torch.int32, device=dev).view(1, -1)
        g = torch.Generator(device=dev).manual_seed(q)
        return caches[ctx] + ((torch.randn(1, q, H, L + R, device=dev, generator=g) * 0.5).to(FP8),)

    def run(cid, ws):
        c = parse_id(cid)
        q, ctx, causal = c["q_len"], c["ctx"], c["causal"]
        write({"id": cid, "start": True, "spawn": spawn})
        G = ctx + q
        local, msl = -(-(G - rank) // W), -(-G // W)
        rec = {"id": cid, **c, "rank": rank, "G": G, "local": local, "max_seq_len": msl, "num_sm": num_sm, "cc": cc,
               "spawn": spawn, "ws": ws, "ws_provided": work[ws].numel()}
        try:
            tiler, _ = select_mla_decode_tilers(H, q, is_fp8=True, compute_capability=cc)
            fold = get_mla_decode_fold_sq_factor(H, q, tiler[0])
            split, need = tmd._get_split_kv_and_workspace_size(1, q // fold, H * fold, L, num_sm, msl, FP8, tuple(tiler))
            rec.update(tile_m=tiler[0], fold=fold, split_kv=split, ws_required=need)
        except Exception as e:  # noqa: BLE001
            rec["planner_error"] = short_error(e)
        rec["mirror"] = list(ts_plan(1, q, H, L, num_sm, cc, msl))
        rec["mirror_ok"] = [rec.get(k) for k in ("tile_m", "fold", "split_kv", "ws_required")] == rec["mirror"]
        t0 = time.time()
        try:
            cache, bt, query = inputs(ctx, q)
            o, lse = tokenspeed_mla_decode(
                query=query, kv_cache=cache, workspace_buffer=work[ws], kv_lora_rank=L, qk_rope_head_dim=R,
                block_tables=bt, seq_lens=torch.tensor([local], dtype=torch.int32, device=dev), max_seq_len=msl,
                softmax_scale=SCALE, output_scale=1.0, enable_pdl=False, return_lse=True, causal_mask=bool(causal),
                causal_seqs=torch.tensor([G], dtype=torch.int32, device=dev), cp_world=W, cp_rank=rank)
            torch.cuda.synchronize(dev)
            rec["rc"] = "ok"
            rec["finite"] = bool(torch.isfinite(o).all()) and bool(torch.isfinite(lse).all())
        except Exception as e:  # noqa: BLE001
            rec["rc"] = short_error(e)
            try:
                torch.cuda.synchronize(dev)
            except Exception as e2:  # noqa: BLE001
                rec["sync_error"] = short_error(e2)
        rec["s"] = round(time.time() - t0, 2)
        try:
            rec["context_ok"] = torch.ones(4, device=dev).sum().item() == 4.0
        except Exception as e3:  # noqa: BLE001
            rec["context_ok"], rec["probe_error"] = False, short_error(e3)
        write(rec)
        if not rec["context_ok"]:  # a sticky error: the parent restarts this rank at the next case
            f.close()
            os._exit(3)

    a_ids, b_ids = r7_cases()
    for cid in a_ids:
        if cid not in ended:
            run(cid, "image")
    absorbed = False
    for cid in b_ids:
        if cid in ended:
            continue
        if not absorbed:
            run(case_id("ABS", *R7_ABS, 1, f"spawn{spawn}"), "image")
            absorbed = True
        run(cid, "image")
    c_ids = [c for c in r7_c_cases(r7_read(path)) if c not in ended]
    if c_ids:
        q_max = max([R7_BIG_MAX_Q] + [parse_id(c)["q_len"] for c in c_ids])
        work["big"] = torch.empty(num_sm * H * q_max * (L + 1) * 4, dtype=torch.int8, device=dev)
        for cid in c_ids:
            run(cid, "big")
    write({"id": "done"})
    f.close()


def run_r7(out_dir, deadline_s) -> None:
    """Supervise one R7 worker per GPU: restart a dead or stalled worker at the next case until done or deadline."""
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    t_end = time.time() + deadline_s
    procs, spawns, started = [None] * W, [0] * W, [0.0] * W

    def start(r):
        procs[r] = ctx.Process(target=_r7_worker, args=(r, out_dir, spawns[r]))
        procs[r].start()
        started[r] = time.time()
        spawns[r] += 1

    def close(r, why):
        path = r7_path(out_dir, r)
        with path.open("a") as f:
            for cid in r7_open(r7_read(path)):
                f.write(json.dumps({"id": cid, **parse_id(cid), "rank": r, "rc": why, "died": True}) + "\n")

    for r in range(W):
        r7_path(out_dir, r).write_text("")
        start(r)
    while any(p is not None for p in procs):
        time.sleep(1.0)
        for r, p in enumerate(procs):
            if p is None:
                continue
            path = r7_path(out_dir, r)
            last = max(started[r], path.stat().st_mtime if path.exists() else 0.0)
            if p.is_alive():
                if time.time() - last <= R7_HANG_S and time.time() <= t_end:
                    continue
                p.kill()
                p.join()
                close(r, "stalled" if time.time() <= t_end else "deadline")
            else:
                p.join()
                close(r, f"worker died (exit {p.exitcode})")
            if time.time() > t_end or any(x["id"] == "done" for x in r7_read(path)):
                procs[r] = None
            else:
                start(r)


def report_r7(out_dir) -> dict:
    recs = {r: [x for x in r7_read(r7_path(out_dir, r)) if "rc" in x] for r in range(W)}
    a_ids, b_ids = r7_cases()
    ids = a_ids + sorted({x["id"] for rr in recs.values() for x in rr if x["id"].startswith("ABS")}) + b_ids
    ids += [c for c in dict.fromkeys(x["id"] for rr in recs.values() for x in rr if x["id"].startswith("C:"))]
    table, fails, mirror_ok = [], [], True
    for cid in ids:
        per = [next((x for x in reversed(recs[r]) if x["id"] == cid), None) for r in range(W)]
        any_ = next((x for x in per if x and "split_kv" in x), per[0] or {})
        rcs = ["-" if x is None else ("ok" if x["rc"] == "ok" else "ERR") for x in per]
        mirror_ok = mirror_ok and all(x["mirror_ok"] for x in per if x and "mirror_ok" in x)  # planner_error: False
        row = {"id": cid, "split_kv": any_.get("split_kv"), "tile_m": any_.get("tile_m"), "fold": any_.get("fold"),
               "ws_required": any_.get("ws_required"), "ws_provided": any_.get("ws_provided"), "rc": rcs}
        table.append(row)
        print(f"R7-ROW id={cid} tile_m={row['tile_m']} fold={row['fold']} split_kv={row['split_kv']} "
              f"ws_required={row['ws_required']} ws_provided={row['ws_provided']} rc={','.join(rcs)}", flush=True)
        fails += [(cid, r, x["rc"]) for r, x in enumerate(per) if x is not None and x["rc"] != "ok"]
    for cid, r, rc in fails[:40]:
        print(f"R7-ERR id={cid} rank={r} rc={rc}", flush=True)
    complete = all(any(x["id"] == cid for x in recs[r]) for cid in a_ids + b_ids for r in range(W))
    imports = {r: x["rc"] for r in range(W) for x in recs[r] if x["id"] == "import"}
    failing = sorted({(parse_id(c)["phase"], parse_id(c)["q_len"], parse_id(c)["ctx"], parse_id(c)["causal"])
                      for c, _, _ in fails})
    ok = complete and mirror_ok and not imports
    print(f"R7 {'PASS' if ok else 'FAIL'} complete={complete} mirror_ok={mirror_ok} failures={len(fails)} "
          f"failing_cases={jd(failing)}" + (f" import_errors={jd(imports)}" if imports else ""),
          flush=True)
    return {"ok": ok, "complete": complete, "mirror_ok": mirror_ok, "failures": len(fails), "failing_cases": failing,
            "table": table, "errors": [list(x) for x in fails], "import_errors": imports}


# ------------------------------------------------------------------------------------------------ R1-R6 (8 ranks)
def load_weights(layer, dev):
    """(kv_b_proj weight [H * (P + V), L] bf16, source) of `layer` from the checkpoint's safetensors index under
    MODEL, or (seeded random weights with the served dims, "random: <reason>")."""
    try:
        from safetensors import safe_open

        root = Path(MODEL)
        index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
        names = [k for k in index if k.endswith(f"layers.{layer}.self_attn.kv_b_proj.weight")]
        assert len(names) == 1, names
        with safe_open(str(root / index[names[0]]), "pt", device=str(dev)) as f:
            w = f.get_tensor(names[0])
        assert tuple(w.shape) == (H * (P + V), L) and w.dtype == torch.bfloat16, (tuple(w.shape), w.dtype)
        return w.contiguous(), f"real: {names[0]}"
    except Exception as e:  # noqa: BLE001
        return random_weights(layer, dev), f"random: {short_error(e)}"


def random_weights(layer, dev):
    g = torch.Generator(device=dev).manual_seed(4000 + layer)
    return (torch.randn(H * (P + V), L, device=dev, generator=g) * L ** -0.5).to(torch.bfloat16)


def set_weights(env, w_all):
    """In place (the variants' memo keeps the objects): the image's kv_b_proj shard and the chain's W_UK_T / W_UV
    (process_weights_after_loading's split, all heads)."""
    r = env.rank
    env.kv_b_proj.weight.copy_(w_all[r * HL * (P + V) : (r + 1) * HL * (P + V)])
    w_uk, w_uv = w_all.T.reshape(L, H, P + V).split([P, V], dim=-1)
    env.absorb.w_uk_t.copy_(w_uk.permute(1, 2, 0))
    env.absorb.w_uv.copy_(w_uv.transpose(0, 1))


def duck(env, case, layer):
    """(layer, info) as reference_report reads them from MultiHeadLatentAttention and the builder hook."""
    dev, n, G = env.dev, case.n, case.G
    cache = case.cache_u8.view(FP8)
    lay = types.SimpleNamespace(
        layer_name=f"layers.{layer}", dcp_world_size=W, impl=types.SimpleNamespace(dcp_rank=env.rank), _k_scale=1.0,
        kv_lora_rank=L, qk_nope_head_dim=P, v_head_dim=V, scale=SCALE, W_UK_T=env.absorb.w_uk_t[env.mine],
        W_UV=env.absorb.w_uv[env.mine], _attn_read_kv_cache=lambda: cache)
    i32 = dict(dtype=torch.int32, device=dev)
    info = types.SimpleNamespace(
        q_lens=[n], seq_lens_cpu=[G], interleave=1, local_max=[-(-G // W)], block_table=case.block_table,
        seq_lens=torch.tensor([G], **i32), local_seq_lens=torch.tensor([case.local], **i32),
        padded_query_start_loc=torch.tensor([0, mla_absorb.bucket(n)], **i32))
    return lay, info


def digest(tensors):
    return {k: hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]
            for k, t in tensors.items()}


def full_latent(env, case):
    """fp64 [G, L + R] stored latent from the cache shards (all-gathered, placed with global_positions: the forward
    map, independent of reference_report's shard_rows) and whether its context rows equal the generated latent."""
    G, lm = case.G, -(-case.G // W)
    flat = case.cache_u8.view(-1, L + R)
    gathered = torch.empty(W * lm, L + R, dtype=torch.uint8, device=env.dev)
    env.dist.all_gather_into_tensor(gathered, flat[:lm].contiguous())
    pos = mla_absorb.global_positions(W, lm, 1, env.dev)
    keep = pos < G
    lat = torch.empty(G, L + R, dtype=torch.float64, device=env.dev)
    lat[pos[keep]] = gathered[keep].view(FP8).double()
    return lat, bool(torch.equal(lat[: case.ctx], case.ctx_lat.double()))


def rel(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-300)).item()


def check_r2(env, case, res, lat8):
    n, ctx, sel = case.n, case.ctx, res["sel"]
    latm = lat8.clone()
    latm[ctx:] = torch.cat([case.kv_c, case.k_pe.view(n, R)], -1).double()
    qd = case.q[sel].double()
    out = {}
    for t, lat in (("t8", lat8), ("mix", latm)):
        d, mass = dense_reference(lat, qd[..., :P], qd[..., P:], env.absorb.w_uk_t[env.mine],
                                  env.absorb.w_uv[env.mine], ctx + sel, SCALE, L, ctx)
        out[f"rel_{t}"] = rel(res["ref"][t], d)
        if t == "mix":
            m = mass.sum(1)
            env.dist.all_reduce(m)
            out["pnew_diff"] = (m / H - torch.tensor(res["pnew_rows"], dtype=torch.float64, device=env.dev)).abs().max().item()
    out["ok"] = max(out["rel_t8"], out["rel_mix"], out["pnew_diff"]) <= R2_TOL
    return out


def check_r3(env, case, res, lat8, image_out):
    n, ctx, sel = case.n, case.ctx, res["sel"]
    try:
        sizes = [int(c.seq_lens.sum()) for c in case.memo["image"].md.prefill.chunked_context.chunks]
        src = "image metadata"
        assert sum(sizes) == ctx, sizes
    except Exception as e:  # noqa: BLE001
        sizes, src = [min(65536, ctx - a) for a in range(0, ctx, 65536)], f"65536 steps ({short_error(e)[:120]})"
    bounds, a = [], 0
    for s in sizes:
        bounds.append((a, a + s))
        a += s
    emu, emu_pq = image_emulation(lambda x, y: lat8[x:y], ctx, bounds, case.kv_c, case.k_pe.view(n, R), case.q, sel,
                                  env.kv_b_proj.weight, SCALE)
    img = image_out[sel].double()
    sums = []
    for t in mla_absorb.TRUTHS:
        ref = res["ref"][t]
        sums += [(img - ref).pow(2).sum(), (emu - ref).pow(2).sum(), (emu_pq - ref).pow(2).sum(), ref.pow(2).sum()]
    sums += [(img - emu).pow(2).sum(), (img - emu_pq).pow(2).sum(), emu.pow(2).sum()]
    v = torch.stack(sums)
    env.dist.all_reduce(v)
    out = {"bounds": bounds, "bounds_source": src, "image_vs_emul": (v[8] / v[10]).sqrt().item(),
           "image_vs_emul_pq": (v[9] / v[10]).sqrt().item()}
    for i, t in enumerate(mla_absorb.TRUTHS):
        e = [(v[4 * i + k] / v[4 * i + 3]).sqrt().item() for k in range(3)]
        out.update({f"image_rel_{t}": e[0], f"emul_rel_{t}": e[1], f"emul_pq_rel_{t}": e[2],
                    f"ratio_{t}": e[0] / max(e[1], 1e-300), f"ratio_pq_{t}": e[0] / max(e[2], 1e-300)})
    out["ok"] = all(R3_BAND[0] <= out[f"ratio_{t}"] <= R3_BAND[1] for t in mla_absorb.TRUTHS)
    return out


def check_r4(env, case, res, lay, info):
    n, sel = case.n, res["sel"]
    r = sel.numel()
    mix, t8 = res["ref"]["mix"], res["ref"]["t8"]
    eps = 1e-2 * (env.rank + 1)
    img = torch.zeros(n, HL * V, dtype=torch.float64, device=env.dev)
    ab = torch.zeros_like(img)
    img[sel] = mix.reshape(r, -1)
    ab[sel] = (mix * (1 + eps)).reshape(r, -1)
    got = mla_absorb.reference_report(lay, info, case.q, case.kv_c, case.k_pe, img, ab, group=env.dist.group.WORLD,
                                      force=True)
    v = torch.stack([eps ** 2 * mix.pow(2).sum(), mix.pow(2).sum(), (mix - t8).pow(2).sum(),
                     (mix * (1 + eps) - t8).pow(2).sum(), t8.pow(2).sum()])
    env.dist.all_reduce(v)
    want = {"image_rel_mix": 0.0, "absorbed_rel_mix": (v[0] / v[1]).sqrt().item(),
            "image_rel_t8": (v[2] / v[4]).sqrt().item(), "absorbed_rel_t8": (v[3] / v[4]).sqrt().item(),
            "map_rel": res["map_rel"]}
    diff = {k: abs(got[k] - w) / max(abs(w), 1e-3) for k, w in want.items()}
    diff["pnew_rows"] = max(abs(a - b) for a, b in zip(got["pnew_rows"], res["pnew_rows"]))
    return {"want": want, "got": {k: got[k] for k in want}, "diff": diff, "ok": max(diff.values()) <= R4_TOL}


def _rank_main(rank, port, out_dir, deadline_s):
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=W, device_id=dev)
    t0 = time.time()
    rec = {"rank": rank, "rows": [], "r2": [], "r3": None, "r4": None}
    path = Path(out_dir, f"reftest-rank{rank}.json")
    save = lambda: path.write_text(json.dumps(rec, indent=1, default=str))  # noqa: E731
    try:
        with torch.inference_mode():
            import perfbot_kernels.mla_mixbench as mb
            from perfbot_kernels.mla_mixbench import bench

            env = bench.Env(rank, dev)
            img_fn, abs_fn = mb.VARIANTS["image"].run, mb.VARIANTS["absorbed"].run
            weights = {layer: load_weights(layer, dev) for layer in LAYERS}
            if env.any_flag(any(src.startswith("random") for _, src in weights.values())):  # one source on all ranks
                weights = {k: (random_weights(k, dev), src if src.startswith("random") else "random: another rank")
                           for k, (_, src) in weights.items()}
            rec["weights"] = {k: src for k, (_, src) in weights.items()}
            if rank == 0:
                print(f"REFTEST-WEIGHTS {jd(rec['weights'])}", flush=True)
            for idx, (n, ctx) in enumerate(CASES):
                if env.any_flag(time.time() - t0 > deadline_s - 90):
                    rec["stopped_at"] = [n, ctx]
                    break
                case = bench.make_case(env, {"n": n, "ctx": ctx}, 3000 + idx, page=PAGE)
                for li, layer in enumerate(LAYERS):
                    set_weights(env, weights[layer][0])
                    absorbed = abs_fn(case).clone()  # the shadow order: absorbed first, then the image path
                    image = img_fn(case).clone()
                    env.barrier()
                    lay, info = duck(env, case, layer)
                    watch = {"cache": case.cache_u8, "block_table": case.block_table, "seq_lens": info.seq_lens,
                             "local_seq_lens": info.local_seq_lens, "padded_qsl": info.padded_query_start_loc,
                             "q": case.q, "kv_c": case.kv_c, "k_pe": case.k_pe, "image": image, "absorbed": absorbed}
                    h0 = digest(watch)
                    torch.cuda.synchronize(dev)
                    torch.cuda.reset_peak_memory_stats(dev)
                    base, ts = torch.cuda.memory_allocated(dev), time.time()
                    res = mla_absorb.reference_report(lay, info, case.q, case.kv_c, case.k_pe, image.view(n, -1),
                                                      absorbed.view(n, -1), group=dist.group.WORLD, force=True)
                    torch.cuda.synchronize(dev)
                    wall, peak = time.time() - ts, (torch.cuda.max_memory_allocated(dev) - base) / 2 ** 20
                    h1 = digest(watch)
                    row = {k: v for k, v in res.items() if k not in ("ref", "sel", "pnew_rows")}
                    row.update(layer=layer, n=n, ctx=ctx, r6_peak_mib=round(peak, 1), r6_s=round(wall, 3),
                               r5_ok=h0 == h1, r5_changed=[k for k in h0 if h0[k] != h1[k]],
                               r1_ok=MAP_REL_LO <= res["map_rel"] <= MAP_REL_MAX)
                    if ctx == R3_CASE[1] and (li == 0 or (n, ctx, layer) == (*R3_CASE, IMAGE_LAYER)):
                        lat8, row["layout_ok"] = full_latent(env, case)
                        row["r1_ok"] = row["r1_ok"] and row["layout_ok"]
                        if li == 0:
                            rec["r2"].append({"n": n, "ctx": ctx, "layer": layer, **check_r2(env, case, res, lat8)})
                        if (n, ctx, layer) == (*R3_CASE, IMAGE_LAYER):
                            rec["r3"] = {"n": n, "ctx": ctx, "layer": layer, **check_r3(env, case, res, lat8, image)}
                        del lat8
                    if (n, ctx) == R4_CASE and li == 0:
                        rec["r4"] = {"n": n, "ctx": ctx, "layer": layer, **check_r4(env, case, res, lay, info)}
                    rec["rows"].append(row)
                    save()
                del case
                torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        rec["error"] = traceback.format_exc()
        traceback.print_exc()
    rec["seconds"] = round(time.time() - t0, 1)
    save()
    print(f"REFTEST rank={rank} done error={'error' in rec}", flush=True)
    try:
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001
        pass


def run_numerics(out_dir, deadline_s):
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_rank_main, args=(r, PORT, out_dir, deadline_s)) for r in range(W)]
    t0 = time.time()
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=max(1.0, deadline_s + 60 - (time.time() - t0)))
    for p in procs:
        if p.is_alive():
            p.terminate()
    ranks = []
    for r in range(W):
        f = Path(out_dir, f"reftest-rank{r}.json")
        ranks.append(json.loads(f.read_text()) if f.is_file() else {"rank": r, "error": "no record"})
    return ranks, [p.exitcode for p in procs]


def verdict(name, ok, **kw):
    print(f"{name} {'PASS' if ok else 'FAIL'} " + " ".join(f"{k}={jd(v)}" for k, v in kw.items()),
          flush=True)
    return ok


def report_numerics(ranks) -> dict:
    rows = [row for r in ranks for row in r.get("rows", [])]
    full = all(len(r.get("rows", [])) == len(CASES) * len(LAYERS) for r in ranks)
    errors = {r["rank"]: r["error"][-1500:] for r in ranks if "error" in r}
    parts = {}
    r3 = [r.get("r3") for r in ranks]
    parts["R3"] = verdict("R3", all(x and x["ok"] for x in r3), **{k: (r3[0] or {}).get(k) for k in (
        "ratio_t8", "ratio_mix", "ratio_pq_t8", "ratio_pq_mix", "image_rel_mix", "emul_rel_mix", "image_vs_emul",
        "image_vs_emul_pq", "bounds_source")})
    bad = [(x["rank"], x["layer"], x["n"], x["ctx"], x["map_rel"], x.get("layout_ok"))
           for r in ranks for x in [dict(row, rank=r["rank"]) for row in r.get("rows", [])] if not x["r1_ok"]]
    parts["R1"] = verdict("R1", full and not bad, rows=len(rows), complete=full, band=[MAP_REL_LO, MAP_REL_MAX],
                          map_rel_max=max((x["map_rel"] for x in rows), default=None), bad=bad[:8])
    r2 = [x for r in ranks for x in r.get("r2", [])]
    parts["R2"] = verdict("R2", len(r2) == 2 * W and all(x["ok"] for x in r2), entries=len(r2),
                          rel_max=max((max(x["rel_t8"], x["rel_mix"]) for x in r2), default=None),
                          pnew_diff_max=max((x["pnew_diff"] for x in r2), default=None))
    r4 = [r.get("r4") for r in ranks]
    parts["R4"] = verdict("R4", all(x and x["ok"] for x in r4),
                          diff_max=max((max(x["diff"].values()) for x in r4 if x), default=None),
                          got=(r4[0] or {}).get("got"), want=(r4[0] or {}).get("want"))
    changed = sorted({k for x in rows for k in x["r5_changed"]})
    parts["R5"] = verdict("R5", full and all(x["r5_ok"] for x in rows), calls=len(rows), changed=changed)
    for row in ranks[0].get("rows", []):
        peaks = [x["r6_peak_mib"] for x in rows if (x["layer"], x["n"], x["ctx"]) == (row["layer"], row["n"], row["ctx"])]
        print(f"R6-ROW layer={row['layer']} n={row['n']} ctx={row['ctx']} peak_mib_max={max(peaks)} s={row['r6_s']}",
              flush=True)
    parts["R6"] = verdict("R6", full and all(x["r6_peak_mib"] <= R6_PEAK_MIB for x in rows), bound_mib=R6_PEAK_MIB,
                          peak_mib_max=max((x["r6_peak_mib"] for x in rows), default=None),
                          s_max=max((x["r6_s"] for x in rows), default=None))
    if errors:
        print(f"REFTEST-ERRORS {json.dumps({k: v[-400:] for k, v in errors.items()})}", flush=True)
    return {"parts": parts, "errors": errors, "complete": full, "weights": ranks[0].get("weights")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deadline-s", type=float, default=1100.0)
    ap.add_argument("--r7-deadline-s", type=float, default=480.0)
    only = ap.add_mutually_exclusive_group()
    only.add_argument("--r7-only", action="store_true")
    only.add_argument("--skip-r7", action="store_true")
    args = ap.parse_args(argv)
    path = results_path()
    out_dir, t0 = str(path.parent), time.time()
    record = {"item": "expedition-M-reftest", "args": vars(args)}
    parts = {}
    if not args.skip_r7:
        run_r7(out_dir, min(args.r7_deadline_s, args.deadline_s - 60))
        record["r7"] = report_r7(out_dir)
        parts["R7"] = record["r7"]["ok"]
        path.write_text(json.dumps(record, indent=1, default=str))
    if not args.r7_only:
        ranks, record["exitcodes"] = run_numerics(out_dir, args.deadline_s - (time.time() - t0) - 30)
        record["numerics"] = report_numerics(ranks)
        parts.update(record["numerics"]["parts"])
    record["parts"], record["seconds"] = parts, round(time.time() - t0, 1)
    ok = bool(parts) and all(parts.values())
    record["ok"] = ok
    path.write_text(json.dumps(record, indent=1, default=str))
    print(f"REFTEST-PARTS {jd(parts)} seconds={record['seconds']} record={path}", flush=True)
    print(f"REFTEST {'OK' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
