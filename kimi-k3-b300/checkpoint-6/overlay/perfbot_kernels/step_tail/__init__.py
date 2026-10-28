"""Fused rejection-sampler tail: one CuTe DSL kernel for gather_draft_sampled, the cu slice, the 4 launches of
rejection_sample (stats, rejection, resample, insert) and get_num_sampled_and_rejected. Bit-equal to the image path.

rejection_tail(**inputs) -> (sampled i64 [R, k+1], num_sampled i32 [R], num_rejected i32 [R], draft_sampled i32 [L],
pos i64 [L], target_rejected_logsumexp f32 [R], draft_rejected_logsumexp f32 [R]).
Declines (calls the image functions) on: no draft logits, block verification, fp64, watermark, V % 8192 != 0,
more than 32 vocab blocks, R > 16, L > 128, k > 7, an unexpected dtype or a non-contiguous input.
FUSED_CALLS counts the calls that took the fused kernel.
The kernel keeps a small per-device workspace (counters reset by the kernel itself), so two launches must not run
concurrently on two streams of one device.
"""

import torch

from . import ptx as _ptx  # noqa: F401
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass import Int32, Int64

from . import kernel as _k

FUSED_CALLS = 0
_compiled = {}
_ws = {}
_sms = {}
_streams = {}  # raw stream handle -> CUstream

_DT = dict(input_ids=torch.int32, positions=torch.int64, logits_indices=torch.int64, expanded_idx_mapping=torch.int64,
           expanded_local_pos=torch.int32, prefill_len=torch.int32, cu_num_logits=torch.int32,
           idx_mapping=torch.int64, seq_lens=torch.int32, temperature=torch.float32, seed=torch.int64)


def _fusable(a, rates, use_fp64, use_block_verification, extra):
    lg, dl = a["logits"], a["draft_logits"]
    if dl is None or use_fp64 or use_block_verification or extra:
        return False
    L, V = lg.shape
    R, K = a["idx_mapping"].shape[0], a["num_speculative_steps"]
    ok = (lg.dtype in (torch.bfloat16, torch.float32) and dl.dtype == torch.bfloat16 and dl.dim() == 3
          and lg.stride(1) == 1 and dl.stride(2) == 1 and dl.size(2) == V and V % 8192 == 0 and V // 8192 <= 32
          and 1 <= R <= 16 and L <= 128 and 1 <= K <= 7 and dl.size(1) >= K and lg.is_cuda
          and (lg.stride(0) * lg.element_size()) % 16 == 0 and lg.data_ptr() % 16 == 0
          and (dl.stride(0) * 2) % 16 == 0 and (dl.stride(1) * 2) % 16 == 0 and dl.data_ptr() % 16 == 0
          and a["cu_num_logits"].shape[0] >= R + 1)
    ok = ok and all(a[k].dtype == d and a[k].is_contiguous() for k, d in _DT.items())
    return ok and (rates is None or (rates.dtype == torch.float32 and rates.is_contiguous() and rates.numel() >= K))


def _image_path(a, rates, use_fp64, use_block_verification, extra):
    from vllm.triton_utils import triton
    from vllm.v1.worker.gpu.input_batch import get_num_sampled_and_rejected
    from vllm.v1.worker.gpu.spec_decode import rejection_sampler_utils as rsu
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import gather_draft_sampled

    R = a["idx_mapping"].shape[0]
    ds, pos = gather_draft_sampled(a["input_ids"], a["positions"], a["logits_indices"], a["expanded_idx_mapping"],
                                   a["expanded_local_pos"], a["prefill_len"])
    cu = a["cu_num_logits"][0:R + 1] - 0
    sampled, ns = rsu.rejection_sample(a["logits"], a["draft_logits"], ds, cu, pos, a["idx_mapping"],
                                       a["expanded_idx_mapping"], a["expanded_local_pos"], a["temperature"],
                                       a["seed"], a["num_speculative_steps"], rates, use_fp64=use_fp64,
                                       use_block_verification=use_block_verification, **extra)
    # the two rejected-row logsumexps: rerun the image's stats + rejection kernels (rejection_sample, non-block)
    lg, dl = a["logits"], a["draft_logits"]
    L, V = lg.shape
    tl_ = lg.new_zeros(R, dtype=torch.float32)
    dl_ = lg.new_zeros(R, dtype=torch.float32)
    if dl is not None and not use_block_verification:
        V = min(V, dl.size(-1))
        nb = triton.cdiv(V, 8192)
        tam = lg.new_empty(L, nb, dtype=torch.int64)
        tmx, tse, dmx, dse = (lg.new_empty(L, nb, dtype=torch.float32) for _ in range(4))
        rsu._compute_local_logits_stats_kernel[(L, nb)](
            tam, nb, tmx, nb, tse, nb, dmx, nb, dse, nb, lg, lg.stride(0), dl, dl.stride(0), dl.stride(1),
            a["expanded_idx_mapping"], a["expanded_local_pos"], a["temperature"], V, a["num_speculative_steps"],
            BLOCK_SIZE=8192, HAS_DRAFT_LOGITS=True)
        s2 = sampled.new_empty(sampled.shape)
        n2 = ns.new_empty(R)
        rsu._rejection_kernel[(R,)](
            s2, s2.stride(0), n2, tl_, dl_, lg, lg.stride(0), tam, nb, tmx, nb, tse, nb, ds, dl, dl.stride(0),
            dl.stride(1), dmx, nb, dse, nb, cu, a["idx_mapping"], a["temperature"], a["seed"], pos, rates, None, None,
            0, nb, PADDED_VOCAB_NUM_BLOCKS=triton.next_power_of_2(nb), HAS_DRAFT_LOGITS=True,
            SYNTHETIC_MODE=rates is not None, USE_BLOCK_VERIFICATION=False, num_warps=1)
    ns, nr = get_num_sampled_and_rejected(ns, a["seq_lens"], a["cu_num_logits"], a["idx_mapping"], a["prefill_len"])
    return sampled, ns, nr, ds, pos, tl_, dl_


def rejection_tail(*, synthetic_conditional_rates=None, use_fp64=False, use_block_verification=False, contexts=None,
                   watermarking=None, watermarking_skip_mask=None, watermark_key=None, **a):
    global FUSED_CALLS
    rates = synthetic_conditional_rates
    extra = {k: v for k, v in dict(contexts=contexts, watermarking=watermarking,
                                   watermarking_skip_mask=watermarking_skip_mask, watermark_key=watermark_key).items()
             if v is not None}
    if not _fusable(a, rates, use_fp64, use_block_verification, extra):
        return _image_path(a, rates, use_fp64, use_block_verification, extra)
    FUSED_CALLS += 1
    lg, dl = a["logits"], a["draft_logits"]
    L, V = lg.shape
    R, K = a["idx_mapping"].shape[0], a["num_speculative_steps"]
    dev = lg.device
    if dev not in _ws:
        _ws[dev] = torch.zeros(_k.W_WORDS + (_k.W_TRACE_CTAS * 32 if _k.TRACE else 0), dtype=torch.int32, device=dev)
        _sms[dev] = torch.cuda.get_device_properties(dev).multi_processor_count
    NB, NB1 = V // 8192, V // 1024
    G = min(2 * _sms[dev], L * NB + R * NB1)
    # 7 torch.empty: measured 10.0 us; one buffer cut into 7 views was not cheaper (9.8 to 13.1 us, dev/host_split2.py)
    e = torch.empty
    sampled, ns, nr = e(R, K + 1, dtype=torch.int64, device=dev), e(R, dtype=torch.int32, device=dev), \
        e(R, dtype=torch.int32, device=dev)
    ds, pos = e(L, dtype=torch.int32, device=dev), e(L, dtype=torch.int64, device=dev)
    tl_, dl_ = e(R, dtype=torch.float32, device=dev), e(R, dtype=torch.float32, device=dev)
    f32, synth = lg.dtype == torch.float32, rates is not None
    h = torch.cuda.current_stream(dev).cuda_stream
    if h not in _streams:
        _streams[h] = cuda.CUstream(h)
    args = [L, R, V, NB, NB1, K, G,
            lg.data_ptr(), lg.stride(0) * lg.element_size(), dl.data_ptr(), dl.stride(0) * 2, dl.stride(1) * 2,
            *[a[n].data_ptr() for n in _PTRS], rates.data_ptr() if synth else 0,
            *[x.data_ptr() for x in (sampled, ns, nr, ds, pos, tl_, dl_, _ws[dev])], _streams[h]]
    key = (f32, synth)
    if key not in _compiled:  # TVM-FFI: the compiled call takes plain Python ints
        typed = [t(x) for t, x in zip(_TYPES, args[:-1])] + [args[-1]]
        _compiled[key] = cute.compile(_k.tail_launch, *typed, f32, synth, options="--enable-tvm-ffi")
    _compiled[key](*args)
    return sampled, ns, nr, ds, pos, tl_, dl_


_PTRS = ("input_ids", "positions", "logits_indices", "expanded_idx_mapping", "expanded_local_pos", "prefill_len",
         "cu_num_logits", "idx_mapping", "seq_lens", "temperature", "seed")
_TYPES = [Int32] * 7 + [Int64] * 25  # tail_launch argument types, stream excluded

# unit C: fused draft prep (launches 1-4 of the DFlash/DSpark draft prep); module prep.py.
# DRAFT_PREP_FUSED_CALLS counts the draft_prep calls that took the fused kernel (prep.py increments it).
DRAFT_PREP_FUSED_CALLS = 0
from .prep import draft_prep  # noqa: E402,F401

# unit E: fused next-step prologue (launches #3-#9, #11-#20 of the unit E map); module prologue.py.
# STEP_PROLOGUE_FUSED_CALLS counts the step_prologue calls that took the fused kernel (prologue.py increments it).
STEP_PROLOGUE_FUSED_CALLS = 0
from .prologue import step_prologue  # noqa: E402,F401
