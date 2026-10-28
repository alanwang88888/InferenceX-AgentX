"""GPU judge of mla_ctx_segments (kernel only; served image, one GPU): fp64 envelope + timing.

  python3 -B selftest.py [--out result.md]     (PYTHONPATH = the overlay root)

Per shape (q new tokens; kv context, cut as served into 65,536-row chunks + the rest), random fp8 q/k/v (12 heads,
qk 192, v 128, scale 1/sqrt(192)). one = the image body's call (trtllm_ragged_attention_deepseek, batch 1); seg =
mla_ctx_segments.run at S = auto and fixed 2 / 4 / 12. ref = fp64 attention of the same fp8 values, one head at a time.
Per context call and for the whole context (chunks folded by the served merge_attn_states):
  PASS iff max|seg - ref| <= 1.02 x max|one - ref| + 1 bf16 ulp of max|ref|, and the same for the ln-lse with floor
  1e-5, and every value is finite. Timing: median of 3 x 50 CUDA-event loops, one vs seg (S = auto).
"""
import argparse
import json
import math
import statistics as st
import sys
from types import SimpleNamespace

import torch
from flashinfer.prefill import trtllm_ragged_attention_deepseek as fmha
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states

import perfbot_kernels.mla_ctx_segments as seg

H, QK, V, CHUNK = 12, 192, 128, 65536
SCALE = 1 / math.sqrt(QK)
LN2 = math.log(2.0)
SHAPES = [(q, kv) for kv in (80000, 100000) for q in (183, 512, 1408, 2100)]
MODES = ("auto", "2", "4", "12")
dev = torch.device("cuda")
ws = torch.zeros(394 * 1024 * 1024, dtype=torch.uint8, device=dev)
BACKEND = SimpleNamespace(scale=SCALE, _workspace_buffer=ws)


def one_call(q, k, v):  # the image body (trtllm_ragged.py run_prefill_context_chunk) for one request
    n, kv = q.shape[0], k.shape[0]
    t = lambda x: torch.tensor(x, dtype=torch.int32, device=dev)
    out = torch.empty(n, H, V, dtype=torch.bfloat16, device=dev)
    o, lse = fmha(query=q.contiguous(), key=k, value=v, workspace_buffer=ws, seq_lens=t([kv]), max_q_len=n, max_kv_len=kv,
                  bmm1_scale=SCALE, bmm2_scale=1.0, o_sf_scale=1.0, batch_size=1, window_left=-1, cum_seq_lens_q=t([0, n]),
                  cum_seq_lens_kv=t([0, kv]), enable_pdl=False, is_causal=False, return_lse=True, out=out,
                  skip_all_rows_active_check=True)
    return o, (lse.transpose(0, 1) * LN2)


def seg_call(q, k, v, mode):
    seg.MODE = mode
    r = seg.run(BACKEND, SimpleNamespace(num_requests=1, all_rows_active=True), q, k, v, None, torch.bfloat16)
    if r is None:  # S = 1: the image body runs
        return one_call(q, k, v) + (1,)
    return r[0], (r[1].transpose(0, 1) * LN2), seg.plan(q.shape[0], k.shape[0], H, seg._sms(dev), mode)


def ref64(q, k, v):
    outs, lses = [], []
    for h in range(H):
        s = (q[:, h].double() @ k[:, h].double().T) * SCALE
        lses.append(torch.logsumexp(s, dim=-1))
        outs.append(torch.softmax(s, dim=-1) @ v[:, h].double())
    return torch.stack(outs, 1), torch.stack(lses, 0)


def fold(parts, n):  # the served accumulate: merge_attn_states over the chunk partials
    o, l = parts[0]
    for po, pl in parts[1:]:
        out, ol = torch.empty_like(o), torch.empty_like(l)
        merge_attn_states(out, o, l.contiguous(), po, pl.contiguous(), output_lse=ol)
        o, l = out, ol
    return o, l


def judge(one, sg, ref):
    (o1, l1), (o2, l2), (orf, lrf) = one, sg, ref
    e1, e2 = (o1.double() - orf).abs().max().item(), (o2.double() - orf).abs().max().item()
    f1, f2 = (l1.double() - lrf).abs().max().item(), (l2.double() - lrf).abs().max().item()
    mx = orf.abs().max().item()
    ulp = 2.0 ** (math.floor(math.log2(mx)) - 7) if mx > 0 else 0.0
    finite = bool(torch.isfinite(o2).all() and torch.isfinite(l2).all())
    ok = finite and e2 <= 1.02 * e1 + ulp and f2 <= 1.02 * f1 + 1e-5
    return ok, dict(err_one=e1, err_seg=e2, lse_err_one=f1, lse_err_seg=f2, ulp=ulp, finite=finite)


def timeit(fn, iters=50, reps=3):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    ms = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(iters):
            fn()
        b.record()
        torch.cuda.synchronize()
        ms.append(a.elapsed_time(b) * 1e3 / iters)
    return st.median(ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="selftest_result.md")
    a = ap.parse_args()
    torch.manual_seed(0)
    f8 = torch.float8_e4m3fn
    rows, allok = [], True
    for q_len, kv in SHAPES:
        q = (torch.randn(q_len, H, QK, device=dev) * 0.5).to(f8)
        k = (torch.randn(kv, H, QK, device=dev) * 0.5).to(f8)
        v = (torch.randn(kv, H, V, device=dev) * 0.5).to(f8)
        cuts = list(range(0, kv, CHUNK)) + [kv]
        chunks = [(k[s:e], v[s:e]) for s, e in zip(cuts, cuts[1:])]
        refs = [ref64(q, kc, vc) for kc, vc in chunks]
        rfull = ref64(q, k, v)
        ones = [one_call(q, kc, vc) for kc, vc in chunks]
        for mode in MODES:
            sgs = [seg_call(q, kc, vc, mode) for kc, vc in chunks]
            for i, ((kc, _), r, o, s) in enumerate(zip(chunks, refs, ones, sgs)):
                ok, d = judge(o, s[:2], r)
                rows.append(dict(q=q_len, kv=kv, part=f"chunk {i} ({kc.shape[0]})", mode=mode, S=s[2], ok=ok, **d))
                allok &= ok
            ok, d = judge(fold(ones, q_len), fold([s[:2] for s in sgs], q_len), rfull)
            rows.append(dict(q=q_len, kv=kv, part="whole context (served fold)", mode=mode, S="/".join(str(s[2]) for s in sgs), ok=ok, **d))
            allok &= ok
        kc, vc = chunks[0]
        t1 = timeit(lambda: one_call(q, kc, vc))
        t2 = timeit(lambda: seg_call(q, kc, vc, "auto"))
        rows.append(dict(q=q_len, kv=kv, part=f"timing chunk 0 ({kc.shape[0]})", mode="auto",
                         S=seg.plan(q_len, kc.shape[0], H, seg._sms(dev), "auto"), ok=True, one_us=t1, seg_us=t2, ratio=t2 / t1))
        print(json.dumps(rows[-1]), flush=True)
    dev_name = torch.cuda.get_device_name(0)
    lines = [f"mla_ctx_segments selftest on {dev_name}; verdict {'PASS' if allok else 'FAIL'}", "",
             "| q | kv | part | mode | S | err one | err seg | lse err one | lse err seg | ulp | result |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "err_one" in r:
            lines.append(f"| {r['q']} | {r['kv']} | {r['part']} | {r['mode']} | {r['S']} | {r['err_one']:.3e} | {r['err_seg']:.3e} | "
                         f"{r['lse_err_one']:.2e} | {r['lse_err_seg']:.2e} | {r['ulp']:.1e} | {'PASS' if r['ok'] else 'FAIL'} |")
    lines += ["", "| q | kv chunk | S (auto) | one-call us | segmented us | ratio |", "|---|---|---|---|---|---|"]
    for r in rows:
        if "one_us" in r:
            lines.append(f"| {r['q']} | {r['part'].split('(')[1][:-1]} | {r['S']} | {r['one_us']:.0f} | {r['seg_us']:.0f} | {r['ratio']:.2f} |")
    open(a.out, "w").write("\n".join(lines) + "\n")
    json.dump(rows, open(a.out.replace(".md", ".json"), "w"), indent=1)
    print("\n".join(lines))
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
