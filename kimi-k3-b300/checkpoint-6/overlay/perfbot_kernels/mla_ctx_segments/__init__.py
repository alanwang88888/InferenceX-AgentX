"""Adaptive context segments for the MLA chunked-context prefill fmha (item 13 lever L1).

The served context call (trtllm_ragged.py run_prefill_context_chunk, image 49cb2bbf) runs one
trtllm_ragged_attention_deepseek per context chunk with a grid of ceil(q/256) q tiles x H local heads. Each CTA streams
the whole chunk's keys, so a short query (a collider turn's 183 / 512 / 1,408 new tokens) fills 12 / 24 / 72 of the
148 SMs and the call costs ~1.0 ms per layer at 80k keys for any q (B200 L1 bench, item-13 data/l1_bench.json).
run() cuts the request's keys into S segments, runs them as S rows of ONE call (q repeated per row; the segment
bounds are cu_seq_lens_kv offsets into the same k/v, no copy), then merges the S partials by their log-sum-exp
(Flash-Decoding; the same math as vLLM's merge_attn_states).

  S = argmin over 1..S_max of waves(S) / S, waves(S) = ceil(ceil(q/256) * H * S / SMs) (tie: smaller S),
  S_max = min(8, 16384 // q, kv // 4096)
The ceil(SMs / tiles) form overshoots into a partial second wave (q 2,100: S 2 = 1.09x one call, B200 probe).
S_CAP 8: the kernel writes no fp32 output, so each partial is rounded to bf16 before the merge; at S 12-13 the
max-element error passed the selftest bound by 5 % (mean error stays below one call). S <= 8 passed every row.

Switch PERFBOT_EXP_MLA_CTX_SEGMENTS, read once at import: unset / 0 = off; auto = the formula; <N> = N, still clamped
by S_max. run() -> (out (q, H, V), lse log2 (q, H)) as the kernel returns them, or None (decline: the caller runs the
image body unchanged). Declines: switch off, S = 1, more than one request in the chunk, an inactive row.
Numerics: not bit-equal to one call (key order of the softmax sums + one merge); judged in an fp64 envelope (selftest.py).
"""

import logging
import math
import os

MODE = os.environ.get("PERFBOT_EXP_MLA_CTX_SEGMENTS", "0").strip().lower()
Q_TILE, S_CAP, ROW_CAP, MIN_SEG, LOG_CAP = 256, 8, 16384, 4096, 4096
_SMS: dict = {}
_BOUNDS: dict = {}
_LOGGED: set = set()
_log = logging.getLogger("vllm.perfbot.mla_ctx_segments")  # a child of the vllm logger: INFO lands in server.log


def _log_decision(q_len: int, kv_len: int, s: int) -> None:
    """One INFO line per distinct (q, kv, S), so the arm log proves the segments took effect (capped at LOG_CAP)."""
    key = (q_len, kv_len, s)
    if key in _LOGGED or len(_LOGGED) > LOG_CAP:
        return
    _LOGGED.add(key)
    _log.info("PERFBOT-MLASEG mode=%s q=%d kv=%d S=%d%s", MODE, q_len, kv_len, s,
              " (log cap reached)" if len(_LOGGED) > LOG_CAP else "")


def plan(q_len: int, kv_len: int, heads: int, sms: int, mode: str | None = None) -> int:
    """Segment count for one single-request context call; 1 = run the image body. mode None = the switch (MODE)."""
    mode = MODE if mode is None else mode
    if mode in ("", "0", "off") or q_len <= 0 or kv_len <= 0:
        return 1
    s_max = max(1, min(S_CAP, ROW_CAP // q_len, kv_len // MIN_SEG))
    if mode != "auto":
        return max(1, min(int(mode), s_max))
    tiles = math.ceil(q_len / Q_TILE) * heads
    return min(range(1, s_max + 1), key=lambda s: (math.ceil(tiles * s / sms) / s, s))


def segment_bounds(kv_len: int, s: int) -> list[int]:
    """cu_seq_lens_kv of the S rows: the request's keys cut at kv * j // S (every row non-empty when S <= kv)."""
    return [kv_len * j // s for j in range(s + 1)]


def merge_log2(partial, lse2, s: int, q_len: int):
    """Merge S partials (S*q, H, V) with base-2 lse (S*q, H) -> (q, H, V) fp32, lse2 (q, H). Every row has keys."""
    import torch
    l = lse2.view(s, q_len, -1).float()
    m = l.amax(dim=0)
    w = torch.exp2(l - m)
    den = w.sum(dim=0)
    o = (partial.view(s, q_len, *partial.shape[1:]).float() * w.unsqueeze(-1)).sum(dim=0) / den.unsqueeze(-1)
    return o, m + torch.log2(den)


def _device_tensors(dev, q_len: int, kv_len: int, s: int):
    key = (dev, q_len, kv_len, s)
    if key not in _BOUNDS:
        import torch
        if len(_BOUNDS) > 256:
            _BOUNDS.clear()
        b = segment_bounds(kv_len, s)
        _BOUNDS[key] = (torch.tensor([q_len * j for j in range(s + 1)], dtype=torch.int32, device=dev),
                        torch.tensor(b, dtype=torch.int32, device=dev),
                        torch.tensor([b[j + 1] - b[j] for j in range(s)], dtype=torch.int32, device=dev),
                        max(b[j + 1] - b[j] for j in range(s)))
    return _BOUNDS[key]


def _sms(dev) -> int:
    if dev not in _SMS:
        import torch
        _SMS[dev] = torch.cuda.get_device_properties(dev).multi_processor_count
    return _SMS[dev]


def run(backend, chunk, q, k, v, out, output_dtype):
    """Segmented context call, or None (decline)."""
    if MODE in ("", "0", "off"):
        return None
    if chunk.num_requests != 1 or not chunk.all_rows_active:
        return None
    q_len, kv_len, heads = q.shape[0], k.shape[0], q.shape[1]
    s = plan(q_len, kv_len, heads, _sms(q.device))
    _log_decision(q_len, kv_len, s)
    if s <= 1:
        return None
    import torch
    from flashinfer.prefill import trtllm_ragged_attention_deepseek
    cu_q, cu_kv, seq_lens, max_kv = _device_tensors(q.device, q_len, kv_len, s)
    partial = torch.empty(s * q_len, heads, v.shape[2], device=q.device, dtype=output_dtype)
    _, lse2 = trtllm_ragged_attention_deepseek(
        query=q.contiguous().repeat(s, 1, 1), key=k, value=v, workspace_buffer=backend._workspace_buffer,
        seq_lens=seq_lens, max_q_len=q_len, max_kv_len=max_kv, bmm1_scale=backend.scale, bmm2_scale=1.0,
        o_sf_scale=1.0, batch_size=s, window_left=-1, cum_seq_lens_q=cu_q, cum_seq_lens_kv=cu_kv, enable_pdl=False,
        is_causal=False, return_lse=True, out=partial, skip_all_rows_active_check=True)
    o, lse = merge_log2(partial, lse2, s, q_len)
    if out is None:
        out = o.to(output_dtype)
    else:
        out.copy_(o)
    return out, lse
