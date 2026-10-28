"""Triton kernels of the fused DSpark Markov round (card P-10 iii-A). Driver in __init__.py.

`_round_kernel`, grid (row r, vocab block b of BLOCK): one round of speculator.py `_sample_sequential` minus the
cuBLAS bias GEMM. x = bf16(base[r, i] + bias[r]); cache store; the image's `gumbel_noised_argmax` on the image's
1-D layout (gumbel.py `_gumbel_sample_kernel`); per-block winners to a workspace; the last block of row r reduces
them with torch argmax semantics (first max, NaN is max), writes draft_tokens[r, i] and the next embed w1[token].
"""

import triton
import triton.language as tl

from vllm.v1.worker.gpu.sample.gumbel import gumbel_noised_argmax, spec_step_seed


@triton.jit
def _embed_kernel(ids_ptr, anchor_ptr, w1_ptr, e_ptr, R: tl.constexpr):
    """e[r] = w1[input_ids[anchor[r]]]: the image's markov_embed of the anchor token."""
    r = tl.program_id(0)
    tok = tl.load(ids_ptr + tl.load(anchor_ptr + r)).to(tl.int64)
    k = tl.arange(0, R)
    tl.store(e_ptr + r * R + k, tl.load(w1_ptr + tok * R + k))


@triton.jit(do_not_specialize=["step"])
def _round_kernel(
    base_ptr, base_s0,  # base logits [M, NSPEC, V] bf16, unit stride on V; step stride V
    bias_ptr, bias_s0,  # markov_bias [M, V] bf16
    idx_ptr, pos_ptr,  # sample_idx_mapping, sample_pos: flat (row, step)
    temp_ptr, seed_ptr,  # per request state
    cache_ptr, cache_s0, cache_s1,  # draft_logits [max_reqs, NSPEC, >=V] bf16
    dt_ptr, dt_s0,  # draft_tokens [max_reqs, NSPEC] int64
    w1_ptr, e_ptr,  # markov_w1 [V1, R]; next round's embed [16, R]
    pval_ptr, pidx_ptr, cnt_ptr,  # workspace: block winners [16, NBP] fp32 / int32; per-row counters int32
    step,
    V: tl.constexpr, R: tl.constexpr, NSPEC: tl.constexpr,
    BLOCK: tl.constexpr, NB: tl.constexpr, NBP: tl.constexpr,
):
    r = tl.program_id(0).to(tl.int64)
    b = tl.program_id(1)
    block = b * BLOCK + tl.arange(0, BLOCK)  # int32 keys, as the image's `block`
    mask = block < V

    # gumbel.py gumbel_block_argmax, with col = step and pos = sample_pos - 1 read in place.
    req = tl.load(idx_ptr + r * NSPEC + step).to(tl.int64)
    valid = req >= 0
    temp = tl.load(temp_ptr + req, mask=valid, other=0.0).to(tl.float32)
    base = tl.load(base_ptr + r * base_s0 + step * V + block, mask=mask)
    bias = tl.load(bias_ptr + r * bias_s0 + block, mask=mask)
    x = (base.to(tl.float32) + bias.to(tl.float32)).to(tl.bfloat16)  # torch bf16 add: fp32 sum, RNE
    tl.store(cache_ptr + req * cache_s0 + step * cache_s1 + block, x, mask=mask & valid)
    seed = tl.load(seed_ptr + req, mask=valid, other=0)
    pos = tl.load(pos_ptr + r * NSPEC + step) - 1
    seed = spec_step_seed(seed, step)
    logits = tl.where(mask, x.to(tl.float32), float("-inf"))
    value, idx = gumbel_noised_argmax(logits, block, mask, seed, pos, temp,
                                      IS_DRAFTING=True, USE_FP64=False, APPLY_TEMPERATURE=True)
    tl.store(pval_ptr + r * NBP + b, value)
    tl.store(pidx_ptr + r * NBP + b, b * BLOCK + idx)

    # Last block of row r: torch argmax over the NB block winners, token out, next embed, counter reset.
    tl.debug_barrier()
    done = tl.atomic_add(cnt_ptr + r, 1, sem="acq_rel", scope="gpu")
    if done == NB - 1:
        nb = tl.arange(0, NBP)
        pv = tl.load(pval_ptr + r * NBP + nb, mask=nb < NB, other=float("-inf"), volatile=True)
        first_nan = tl.min(tl.where(pv != pv, nb, NBP))
        _, imax = tl.max(pv, axis=0, return_indices=True, return_indices_tie_break_left=True)
        pi = tl.load(pidx_ptr + r * NBP + nb, mask=nb < NB, other=0, volatile=True)  # parallel to pv, not after
        bsel = tl.where(first_nan < NBP, first_nan, imax)
        tok = tl.sum(tl.where(nb == bsel, pi, 0)).to(tl.int64)
        tl.store(dt_ptr + r * dt_s0 + step, tok)
        if step < NSPEC - 1:
            k = tl.arange(0, R)
            tl.store(e_ptr + r * R + k, tl.load(w1_ptr + tok * R + k))
        tl.store(cnt_ptr + r, 0)  # the next launch is stream-ordered after this one
