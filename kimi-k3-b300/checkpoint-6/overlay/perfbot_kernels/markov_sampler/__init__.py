"""Fused DSpark Markov sampling rounds (card P-10 iii-A). Kernels in kernel.py.

Replaces the loop of speculator.py `_sample_sequential` (image 49cb2bbf, l.173-186). Per call: 1 embed launch,
then per round i: `spec.model.markov_bias(ebuf[:M])` (the image's own cuBLAS call, same shape and strides) and ONE
`_round_kernel` launch (add, cache store, the image's Gumbel argmax, token write, next embed).

`run(spec, base_logits)` -> spec.draft_tokens with [:M, :n_spec] and spec.draft_logits written bit for bit as the
image loop, or None (decline; the caller runs the image loop). `warmup(spec)` compiles eagerly before capture.
Switch PERFBOT_EXP_MARKOV=1 (default off). One workspace per device; calls on one device must not overlap.
"""

import os
import sys
from types import SimpleNamespace

import torch

MAX_M, BLOCK, NUM_WARPS = 16, 1024, 4  # BLOCK and num_warps as the image's gumbel_sample (default warps)
_WS: dict = {}
_WARM: set = set()
_LOGGED: set = set()


def _log_once(msg: str) -> None:
    if msg not in _LOGGED:
        _LOGGED.add(msg)
        print(f"PERFBOT-MARKOV {msg}", file=sys.stderr, flush=True)


def _w1(spec):
    return spec.model.model.markov_head.markov_w1.weight


def _decline(spec, base_logits) -> str | None:
    """None when the fused path computes exactly what the image loop does, else the reason."""
    if os.environ.get("PERFBOT_EXP_MARKOV", "0") != "1":
        return "switch off"
    lp = getattr(spec.model, "logits_processor", None)
    w1, dl, dt = _w1(spec), spec.draft_logits, spec.draft_tokens
    m, n, v = base_logits.shape
    checks = (
        (1 <= m <= MAX_M, f"M={m}"),
        (n == spec.num_speculative_steps, "n_spec"),
        (dl is not None and dl.dtype == torch.bfloat16 and dl.size(-1) >= v and dl.stride(-1) == 1, "draft_logits"),
        (spec._d2t_scatter_index is None, "d2t"),
        (getattr(spec, "_draft_topk", None) is None, "top-k"),
        (not spec.use_fp64_gumbel, "fp64 gumbel"),
        (getattr(spec, "draft_watermarker", None) is None, "watermarker"),
        (getattr(spec, "acceptance_estimator", None) is None, "acceptance estimator"),
        (not spec.enable_adaptive_verification and not getattr(spec, "use_confidence_head", False), "adaptive"),
        (lp is not None and lp.scale == 1.0 and lp.soft_cap is None, "logits processor"),
        (base_logits.dtype == w1.dtype == torch.bfloat16, "dtype"),
        (base_logits.stride(2) == 1 and base_logits.stride(1) == v and w1.is_contiguous(), "layout"),
        (dt.dtype == torch.int64 and dt.stride(1) == 1, "draft_tokens"),
        (spec.sample_idx_mapping.is_contiguous() and spec.sample_pos.is_contiguous(), "idx/pos"),
        (w1.shape[1] & (w1.shape[1] - 1) == 0, f"R={w1.shape[1]}"),
    )
    return next((why for ok, why in checks if not ok), None)


def _workspace(dev: torch.device, r: int, nbp: int) -> dict:
    key = (dev.index, r, nbp)
    if key not in _WS:
        _WS[key] = {"e": torch.zeros(MAX_M, r, dtype=torch.bfloat16, device=dev),
                    "pval": torch.empty(MAX_M, nbp, dtype=torch.float32, device=dev),
                    "pidx": torch.empty(MAX_M, nbp, dtype=torch.int32, device=dev),
                    "cnt": torch.zeros(MAX_M, dtype=torch.int32, device=dev)}  # self-resetting
    return _WS[key]


def run(spec, base_logits: torch.Tensor):
    why = _decline(spec, base_logits)
    capturing = torch.cuda.is_current_stream_capturing()
    if why is None and capturing and base_logits.device.index not in _WARM:
        why = "first call inside graph capture (no warmup)"
    if why is not None:
        _log_once(f"decline {why}")
        return None
    from .kernel import _embed_kernel, _round_kernel

    m, n, v = base_logits.shape
    w1 = _w1(spec)
    r = w1.shape[1]
    nb = -(-v // BLOCK)
    nbp = 1 << (nb - 1).bit_length()
    ws = _workspace(base_logits.device, r, nbp)
    ebuf, cache, dt = ws["e"], spec.draft_logits, spec.draft_tokens
    _embed_kernel[(m,)](spec.input_buffers.input_ids, spec._anchor_idx, w1, ebuf, R=r)
    for i in range(n):
        bias = spec.model.markov_bias(ebuf[:m])  # the image's call: same op, shape, strides
        if i == 0 and (bias.dtype != torch.bfloat16 or bias.shape != (m, v) or bias.stride(1) != 1):
            _log_once(f"decline markov_bias {bias.dtype} {tuple(bias.shape)} {bias.stride()}")
            return None  # nothing but the workspace was written
        _round_kernel[(m, nb)](
            base_logits, base_logits.stride(0), bias, bias.stride(0),
            spec.sample_idx_mapping, spec.sample_pos, spec.temperature, spec.seeds,
            cache, cache.stride(0), cache.stride(1), dt, dt.stride(0),
            w1, ebuf, ws["pval"], ws["pidx"], ws["cnt"], i,
            V=v, R=r, NSPEC=n, BLOCK=BLOCK, NB=nb, NBP=nbp, num_warps=NUM_WARPS)
    if not capturing:
        _WARM.add(base_logits.device.index)
    return dt


def warmup(spec) -> None:
    """Compile every launch eagerly on scratch state (all rows inert: draft_logits is never written)."""
    if os.environ.get("PERFBOT_EXP_MARKOV", "0") != "1" or torch.cuda.is_current_stream_capturing():
        return
    w1 = _w1(spec)
    dev, n = w1.device, spec.num_speculative_steps
    v = spec.draft_logits.size(-1) if spec.draft_logits is not None else 0
    ns = SimpleNamespace(**{k: getattr(spec, k, None) for k in (
        "model", "num_speculative_steps", "draft_logits", "_d2t_scatter_index", "_draft_topk", "use_fp64_gumbel",
        "draft_watermarker", "acceptance_estimator", "enable_adaptive_verification", "use_confidence_head")})
    ns.draft_tokens = torch.zeros_like(spec.draft_tokens)
    ns.input_buffers = SimpleNamespace(input_ids=torch.zeros_like(spec.input_buffers.input_ids))
    ns._anchor_idx = torch.zeros_like(spec._anchor_idx)
    ns.sample_idx_mapping = torch.full_like(spec.sample_idx_mapping, -1)
    ns.sample_pos = torch.ones_like(spec.sample_pos)
    ns.temperature, ns.seeds = torch.ones_like(spec.temperature), torch.zeros_like(spec.seeds)
    out = run(ns, torch.zeros(1, n, v, dtype=w1.dtype, device=dev))
    torch.cuda.synchronize(dev)
    _log_once(f"warmup ok={out is not None}")
