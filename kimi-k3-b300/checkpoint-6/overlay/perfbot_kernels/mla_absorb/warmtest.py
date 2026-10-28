"""Kernel-only gate of line M's warm(): after it, the first absorbed call per bucket must not JIT. Server env:

  python3 -B -m perfbot_kernels.mla_absorb.warmtest [--deadline-s S]

Three arms run one after another, each as 8 processes (one per GPU, cp_rank = GPU, no collectives):
  cold     no warm;  stale16  warm() at page 16 only (the page P-01c's warm compiled);
  warm     warm() as the server calls it at weight load (every page in mla_absorb.served_pages()).
warm() runs on a stub layer per gathered head count in HEADS (target 96, dspark draft 64; P-01c warm lines).
Then, per head count and per page in the backend's int kernel block sizes, the absorbed path runs for each n in CASES
through the served code: mla_absorb.attend (bucket padding, metadata) -> the image's TokenspeedMLAImpl.forward_mqa
(its static args) -> tokenspeed_mla_decode. Stubs only replace the q-concat/cache insert (seeded fp8 query of the
gathered heads), the DCP gather (identity), the combine (records o, lse) and W_UV. Context CTX, DCP 8, interleave 1.
JIT detection per forward_mqa call: (1) the image's jit monitor (vllm.utils.jit_monitor.activate, the hook the server
logs as "Kernel JIT monitor activated"; its _handle_jit_event counted per CuTeDSL event), (2) misses of
tokenspeed_mla.mla_decode._get_compiled_mla_kernel (the functools.cache holding compiled kernels; its arguments are
the compile key and are recorded per call), (3) wall time of the call (sync before and after).
The served page: the image's select_common_block_size(MANAGER_BLOCK, [TokenspeedMLABackend]). K3 is hybrid:
P-01c server.log interface.py:911 "Setting attention block size to 1536 tokens" (after interface.py:635's 32), and
the kv cache is viewed in kernel blocks of that selection (vllm/v1/worker/utils.py select_common_block_size,
kv_cache_interface.py create_kv_cache_views) = 64. Only the served page is gated; the other page is printed.
  A  warm arm, served page: every call has 0 monitor events, 0 cache misses, wall < WALL_BOUND_S, on every rank.
  S  sensitivity (cold, stale16 arms, served page): every first call per (heads, bucket) is flagged by both the
     monitor and the cache, every repeated bucket (CASES' last three) by neither.
  B  warm vs cold arm, n = SAVE_N: o and lse torch.equal (separate processes, same seeded inputs, saved tensors) at
     every page where the warm arm's call did not compile (a warmed kernel); at least one such page per head count.
  C  every compile key of the warm arm's served calls is one warm() compiled; WARMTEST-KEYDIFF lines name the
     differing static arguments.
Final line `WARMTEST OK` iff A, S and B, C pass, else `WARMTEST FAIL`; exit 0 iff OK.
Record: $PERFBOT_WORKSPACE/$PERFBOT_RESULTS/warmtest.json (tensors and per-rank records beside it).
"""

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
import types
from pathlib import Path

import torch

import perfbot_kernels.mla_absorb as mla_absorb
from perfbot_kernels.mla_absorb.reftest import jd, results_path, verdict

W, L, R, P, V = 8, 512, 64, 128, 128
HEADS = (96, 64)  # gathered heads: target, dspark draft (P-01c "warmed buckets ... (heads 96 / 64, page 16)")
CTX = 9600
CASES = [13, 130, 300, 8, 32, 64, 128, 16, 256, 512]  # 13, 130, 300 pad to 16, 256, 512; the last three repeat
SAVE_N = 13
MANAGER_BLOCK = 1536  # P-01c server.log interface.py:911 (hybrid alignment of the attention block size)
WALL_BOUND_S = 0.05  # a CuTe DSL compile of this kernel takes ~1.4 s
ARMS = ("cold", "stale16", "warm")
FP8, BF16 = torch.float8_e4m3fn, torch.bfloat16
SCALE = (P + R) ** -0.5


# ------------------------------------------------------------------------------------- CPU-testable pure parts
def local_len(G: int, rank: int, w: int = W, i: int = 1) -> int:
    """Keys of a G-token sequence on DCP rank `rank`: blocks of i tokens dealt round-robin over w ranks."""
    full, rem = divmod(G, w * i)
    return full * i + min(max(rem - rank * i, 0), i)


def mark_first(rows) -> list:
    """Per row (call order): is it the first call of its (heads, page, pad) in this process?"""
    seen, out = set(), []
    for x in rows:
        k = (x["heads"], x["page"], x["pad"])
        out.append(k not in seen)
        seen.add(k)
    return out


def key_diff(key: dict, warm_keys: list) -> dict:
    """{} if warm() compiled `key`; else {arg: [warm value, served value]} against the warm key with the same heads and
    q_len, or {"_": ...} when warm() compiled none."""
    if key in warm_keys:
        return {}
    for wk in warm_keys:
        if (wk.get("num_heads"), wk.get("seq_len_q")) == (key.get("num_heads"), key.get("seq_len_q")):
            return {k: [wk.get(k), v] for k, v in key.items() if wk.get(k) != v}
    return {"_": "no warm key with these heads and q_len"}


def judge(recs: dict, equal: dict) -> dict:
    """recs {arm: [rank records]}; equal {(rank, heads, page): bool}, warm vs cold n = SAVE_N tensors."""
    errors = {f"{a}:{r.get('rank')}": str(r["error"])[-600:] for a in ARMS for r in recs.get(a, []) if "error" in r}

    def rows(rec, served=True):
        xs = [dict(x, first=f) for x, f in zip(rec.get("rows", []), mark_first(rec.get("rows", [])))]
        return [x for x in xs if (x["page"] == rec.get("served_page")) == served]

    complete = not errors and all(len(recs.get(a, [])) == W and all(len(rows(r)) == len(HEADS) * len(CASES)
                                                                     for r in recs[a]) for a in ARMS)
    out = {"complete": complete, "errors": errors}
    warm = [x for r in recs.get("warm", []) for x in rows(r)]
    out["A"] = complete and all(x["jit"] == 0 and x["misses"] == 0 and x["wall_s"] < WALL_BOUND_S for x in warm)
    out["S"] = complete and all((x["jit"] > 0) == (x["misses"] > 0) == x["first"]
                                for a in ("cold", "stale16") for r in recs[a] for x in rows(r))
    pairs = {}
    for r in recs.get("warm", []):
        for x in rows(r) + rows(r, False):
            if x["n"] == SAVE_N:
                pairs[(r["rank"], x["heads"], x["page"])] = x["jit"] == 0 and x["misses"] == 0
    warmed = [k for k, v in pairs.items() if v]
    out["B_pairs"] = {f"{k[0]}:{k[1]}:{k[2]}": equal.get(k) for k in warmed}
    out["B"] = complete and all(equal.get(k) is True for k in warmed) and all(
        any((r, h) == k[:2] for k in warmed) for r in range(W) for h in HEADS)
    diffs = {}
    for r in recs.get("warm", []):
        for x in rows(r):
            d = key_diff(x["key"], r.get("warm_keys", []))
            if d:
                diffs.setdefault((x["heads"], x["page"], x["pad"]), d)
    out["keydiff"] = {f"{k[0]}:{k[1]}:{k[2]}": v for k, v in diffs.items()}
    out["C"] = complete and not diffs
    out["ok"] = all(out[k] for k in ("A", "S", "B", "C"))
    return out


# ------------------------------------------------------------------------------------------------ GPU worker
def key_of(kw: dict) -> dict:
    return {k: v if isinstance(v, (bool, int, float, str)) else str(v) for k, v in sorted(kw.items())}


def _arm(arm, rank, dev, out_dir, rec):
    import tokenspeed_mla.mla_decode as tmd
    from vllm.utils import jit_monitor as jm
    from vllm.v1.attention.backends.mla.tokenspeed_mla import TokenspeedMLABackend, TokenspeedMLAImpl
    from vllm.v1.worker.utils import select_common_block_size

    served = select_common_block_size(MANAGER_BLOCK, [TokenspeedMLABackend])
    pages = sorted({s for s in TokenspeedMLABackend.get_supported_kernel_block_sizes() if isinstance(s, int)} | {served})
    rec.update(served_page=served, pages=pages)
    keys, events, cur = [], [], {}
    compiled = tmd._get_compiled_mla_kernel

    def record(**kw):  # tokenspeed_mla_decode passes every argument by keyword
        keys.append(key_of(kw))
        return compiled(**kw)

    tmd._get_compiled_mla_kernel = record
    if arm == "stale16":
        mla_absorb.served_pages = lambda backend=None: [16]

    def forward_mqa(self, *a):
        torch.cuda.synchronize(dev)
        m0, e0, t0 = compiled.cache_info().misses, len(events), time.perf_counter()
        res = TokenspeedMLAImpl.forward_mqa(self, *a)
        torch.cuda.synchronize(dev)
        cur.update(wall_s=time.perf_counter() - t0, misses=compiled.cache_info().misses - m0,
                   jit=events[e0:].count("CuTeDSL"), key=keys[-1])
        return res

    layers = {}
    for heads in HEADS:
        hl = heads // W
        impl = types.SimpleNamespace(
            kv_lora_rank=L, qk_rope_head_dim=R, scale=SCALE, softmax_scale=None, output_scale=None,
            _workspace_buffer=None, dcp_world_size=W, dcp_rank=rank,
            need_to_return_lse_for_decode=W > 1 and TokenspeedMLAImpl.can_return_lse_for_decode)
        impl.forward_mqa = types.MethodType(forward_mqa, impl)
        lay = types.SimpleNamespace(
            layer_name=f"warmtest.heads{heads}", num_local_heads=hl, dcp_world_size=W, q_pad_num_heads=None,
            kv_lora_rank=L, qk_rope_head_dim=R, qk_nope_head_dim=P, v_head_dim=V, impl=impl, _q_scale_float=1.0,
            _k_scale_float=1.0, W_UK_T=torch.zeros(hl, P, L, dtype=BF16, device=dev),
            W_UV=torch.zeros(hl, L, V, dtype=BF16, device=dev),
            kv_cache=torch.zeros(2, served, L + R, dtype=torch.uint8, device=dev))  # the served cache's page
        lay._decode_concat_cache = lambda *a, lay=lay: lay.next_q
        lay._attn_read_kv_cache = lambda: cur["cache"]
        lay._v_up_proj = lambda x, out: None
        lay.dcp_manager = types.SimpleNamespace(
            query_gather=lambda x: x, combine=lambda o, lse, **kw: (cur.update(o=o, lse=lse), o)[1])
        layers[heads] = lay
        if arm != "cold":
            mla_absorb.warm(lay)
    rec["warm_keys"] = [k for i, k in enumerate(keys) if k not in keys[:i]]
    jm.activate(mode="warn")
    handle = jm._handle_jit_event

    def counted(**kw):
        events.append(kw.get("backend"))
        return handle(**kw)

    jm._handle_jit_event = counted
    i32 = lambda v: torch.tensor(v, dtype=torch.int32, device=dev)  # noqa: E731
    for heads, lay in layers.items():
        for page in pages:
            g = torch.Generator(device=dev).manual_seed(1000 * rank + page)
            npages = -(-local_len(CTX + max(CASES), 0) // page) + 1  # rank 0 holds the most keys
            cur["cache"] = (torch.randn(npages, page, L + R, device=dev, generator=g) * 0.5).to(FP8)
            bt = torch.arange(npages, dtype=torch.int32, device=dev).view(1, -1)
            for n in CASES:
                G, pad = CTX + n, mla_absorb.bucket(n)
                g = torch.Generator(device=dev).manual_seed(heads * 100003 + page * 1009 + n)
                lay.next_q = (torch.randn(n, heads, L + R, device=dev, generator=g) * 0.5).to(FP8)
                q = torch.randn(n, heads // W, P + R, device=dev, generator=g).to(BF16)
                info = types.SimpleNamespace(
                    q_lens=[n], pads=[pad], seq_lens_cpu=[G], local_max=[-(-G // W)], interleave=1,
                    seq_lens=i32([G]), local_seq_lens=i32([local_len(G, rank)]), padded_query_start_loc=i32([0, pad]),
                    block_table=bt)
                out = torch.empty(n, heads // W * V, dtype=BF16, device=dev)
                mla_absorb.attend(lay, q, None, None, None, None, None, info, out)
                o, lse = cur["o"].contiguous(), cur["lse"].contiguous()
                h = hashlib.sha256(o.view(torch.uint8).cpu().numpy().tobytes())
                h.update(lse.view(torch.uint8).cpu().numpy().tobytes())
                if n == SAVE_N:
                    torch.save({"o": o.cpu(), "lse": lse.cpu()}, Path(out_dir, f"warmtest-{arm}-rank{rank}-h{heads}-p{page}.pt"))
                rec["rows"].append({"heads": heads, "page": page, "n": n, "pad": pad, "digest": h.hexdigest()[:16],
                                    **{k: cur[k] for k in ("wall_s", "misses", "jit", "key")}})


def _worker(arm, rank, out_dir):
    os.environ.setdefault("PERFBOT_EXP_MLAQ", "1")
    torch.cuda.set_device(rank)
    rec = {"arm": arm, "rank": rank, "rows": []}
    try:
        with torch.inference_mode():
            _arm(arm, rank, torch.device("cuda", rank), out_dir, rec)
    except Exception:  # noqa: BLE001
        rec["error"] = traceback.format_exc()
        traceback.print_exc()
    Path(out_dir, f"warmtest-{arm}-rank{rank}.json").write_text(json.dumps(rec, default=str))


def run_arm(arm, out_dir, timeout_s):
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_worker, args=(arm, r, out_dir)) for r in range(W)]
    t0 = time.time()
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=max(1.0, timeout_s - (time.time() - t0)))
    for p in procs:
        if p.is_alive():
            p.terminate()
    recs = []
    for r in range(W):
        f = Path(out_dir, f"warmtest-{arm}-rank{r}.json")
        recs.append(json.loads(f.read_text()) if f.is_file() else {"rank": r, "error": f"no record (exit {procs[r].exitcode})"})
    return recs


def bit_equal(out_dir) -> dict:
    eq = {}
    for r in range(W):
        for f in Path(out_dir).glob(f"warmtest-warm-rank{r}-h*-p*.pt"):
            c = Path(out_dir, f.name.replace("-warm-", "-cold-"))
            if c.is_file():
                a, b = torch.load(f), torch.load(c)
                h, p = (int(s[1:]) for s in f.stem.split("-")[-2:])
                eq[(r, h, p)] = torch.equal(a["o"], b["o"]) and torch.equal(a["lse"], b["lse"])
    return eq


def report(recs, res):
    for arm in ARMS:
        rows = {}
        for r in recs[arm]:
            for x, first in zip(r.get("rows", []), mark_first(r.get("rows", []))):
                rows.setdefault((x["heads"], x["page"], x["n"]), []).append(dict(x, first=first))
        for (heads, page, n), xs in rows.items():
            print(f"WARMTEST-ROW arm={arm} heads={heads} page={page} served={int(page == recs[arm][0].get('served_page'))} "
                  f"n={n} pad={xs[0]['pad']} first={int(xs[0]['first'])} ranks={len(xs)} jit_max={max(x['jit'] for x in xs)} "
                  f"misses_max={max(x['misses'] for x in xs)} wall_max_s={max(x['wall_s'] for x in xs):.4f} "
                  f"digests={len({x['digest'] for x in xs})}", flush=True)
    for k, d in res["keydiff"].items():
        print(f"WARMTEST-KEYDIFF heads:page:pad={k} diff={jd(d)}", flush=True)
    w = recs["warm"][0]
    print(f"WARMTEST-PAGES served={w.get('served_page')} pages={jd(w.get('pages'))} manager_block={MANAGER_BLOCK} "
          f"warm_pages={jd(sorted({k.get('page_size') for k in w.get('warm_keys', [])}))}", flush=True)
    if res["errors"]:
        print(f"WARMTEST-ERRORS {jd(res['errors'])}", flush=True)
    verdict("A", res["A"], complete=res["complete"], bound_s=WALL_BOUND_S)
    verdict("S", res["S"], arms=["cold", "stale16"])
    verdict("B", res["B"], pairs=res["B_pairs"])
    verdict("C", res["C"], keydiff=len(res["keydiff"]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deadline-s", type=float, default=1100.0)
    args = ap.parse_args(argv)
    path = results_path("warmtest.json")
    out_dir, t0 = str(path.parent), time.time()
    recs = {}
    for i, arm in enumerate(ARMS):
        recs[arm] = run_arm(arm, out_dir, (args.deadline_s - (time.time() - t0)) / (len(ARMS) - i))
    res = judge(recs, bit_equal(out_dir))
    report(recs, res)
    res["seconds"] = round(time.time() - t0, 1)
    path.write_text(json.dumps({"item": "expedition-M-warmtest", "result": res, "arms": recs}, indent=1, default=str))
    print(f"WARMTEST-PARTS {jd({k: res[k] for k in 'ASBC'})} seconds={res['seconds']} record={path}", flush=True)
    print(f"WARMTEST {'OK' if res['ok'] else 'FAIL'}", flush=True)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
