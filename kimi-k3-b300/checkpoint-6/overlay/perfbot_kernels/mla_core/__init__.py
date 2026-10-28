# SPDX-License-Identifier: Apache-2.0
# PERFBOT item 03 (roadmap v0 up-mla-core-mma-efficiency): the MLA decode core under DCP at q 8 x 96 heads.
"""perfbot_kernels.mla_core: ``stock_decode`` with the signature and contract of vLLM 49cb2bbf
``specialized_decode/cutedsl/mla/dcp.py:281`` (FP8 split-KV MLA decode into caller-owned BF16 rows and log2 LSE),
running the forked kernel in ``kernel.py`` (DESIGN.md). The split planner, tilers and workspace sizing are
tokenspeed_mla's own functions, so the split is the served planner's split."""

import functools
import logging  # PERFBOT item-03 MLA-LOOP
import os
import sys  # PERFBOT item-03 MLA-LOOP

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32

from .kernel import BlackwellMultiHeadLatentAttentionForwardFP8

NAME = "mla_core"
# Compile contract (P-08 judge kernels.PINNED); the judge asserts these keys. kernel.py serves this contract only:
# the options it names (local visible lens, var split, window, fp16 partials) are cut from the kernel.
# Exception: item 01's var split (L6 + 01d) under item 01's own argument: stock_decode(splits=...) compiles  # PERFBOT item-03
# is_var_split_kv=True (perfbot_kernels.mla_varsplit; switch PERFBOT_EXP_B35); splits=None keeps this contract.  # PERFBOT item-03
COMPILE_CONTRACT = dict(has_local_visible_lens=False, is_var_split_kv=False, skip_correction_threshold=8.0,
                        window_left=-1, partial_fp16=False)

_WORKSPACE: dict = {}
# PERFBOT_MLA_CORE_IKET=1: compile with IKET phase ranges (run-iket readings only; never served). Default off.
IKET = os.environ.get("PERFBOT_MLA_CORE_IKET") == "1"
SPLIT_MMA = os.environ.get("PERFBOT_MLA_CORE_SPLITMMA", "1") == "1"  # QK and PV on two issuer warps. Default ON (freeze after B300 LAND_READING); "0" = E2b  # PERFBOT item-03 MLA-LOOP
_log = logging.getLogger("vllm.perfbot.mla_core")  # a child of the vllm logger: INFO lands in server.log  # PERFBOT item-03 MLA-LOOP
_mark = getattr(sys.modules.get("sitecustomize"), "mark", None)  # PERFBOT liveness: overlay 0dcc27a+; older = no-op  # PERFBOT item-03 MLA-LOOP
if _mark is not None:  # PERFBOT item-03 MLA-LOOP
    _mark("PERFBOT_MLA_CORE_SPLITMMA", "1" if SPLIT_MMA else "0")  # PERFBOT item-03 MLA-LOOP
NO_CACHE = os.environ.get("PERFBOT_MLA_CORE_NOCACHE") == "1"  # null arm: bypass the DSL JIT/file cache (a true recompile)  # PERFBOT item-03 MLA-LOOP
MAX_REQS = 5  # requests per call served by this kernel (B1-B5); B >= 6 keeps the incumbent bit for bit


@functools.cache
def compile_kernel(page_size, arch, heads, q_len, cp, pdl, return_lse, reducer_d_tiles, reducer_max_splits, pack,
                   workspace, iket=False,  # PERFBOT item-03
                   var_split=False, unit_table=False,  # PERFBOT item-03
                   split_mma=False):  # PERFBOT item-03 MLA-LOOP
    """FP8 kernel of one served shape; the fake-tensor signature of tokenspeed_mla.mla_decode._get_compiled_mla_kernel
    (FP8, paged, var-seq, non-persistent, packed Q, causal)."""
    from tokenspeed_mla import mla_decode as md

    qk, pv = md.select_mla_decode_tilers(heads, q_len, is_fp8=True, compute_capability=arch)
    assert pack and qk[0] == 128, "mla_core serves packed Q on the M128 2-CTA tile only"
    kernel = BlackwellMultiHeadLatentAttentionForwardFP8(
        compute_capability=arch, acc_dtype=cutlass.Float32, lse_dtype=cutlass.Float32, mma_qk_tiler_mn=qk,
        mma_pv_tiler_mn=pv, page_size=page_size,
        skip_correction_threshold=COMPILE_CONTRACT["skip_correction_threshold"], num_heads=heads,
        seq_len_q=q_len, cp_world=cp[0], cp_rank=cp[1], cp_interleave_size=cp[2],
        reducer_d_tiles=reducer_d_tiles, reducer_max_splits=reducer_max_splits, iket=iket,  # PERFBOT item-03
        is_var_split_kv=var_split, unit_table=unit_table,  # PERFBOT item-03
        split_mma=split_mma)  # PERFBOT item-03 MLA-LOOP
    sym = cute.sym_int
    fp8 = cutlass.Float8E4M3FN
    batch, seq_q, h, latent, rope = sym(), sym(), sym(), sym(divisibility=16), sym(divisibility=16)
    kv_batch, seq_kv = sym(), sym()

    def fake(dtype, shape, stride=None, align=16, order=None):
        if stride is not None:
            return cute.runtime.make_fake_tensor(dtype, shape, stride=stride, assumed_align=align)
        return cute.runtime.make_fake_compact_tensor(dtype, shape, stride_order=order, assumed_align=align)

    q4 = (sym(), sym(), sym(), 1)
    args = [
        kernel,
        fake(fp8, (batch, seq_q, h, latent), q4),
        fake(fp8, (batch, seq_q, h, rope), q4),
        fake(fp8, (kv_batch, seq_kv, latent), (sym(), sym(), 1)),
        fake(fp8, (kv_batch, seq_kv, rope), (sym(), sym(), 1)),
        fake(cutlass.Int32, (batch, sym()), order=(1, 0), align=4),
        fake(cutlass.BFloat16, (batch, seq_q, h, latent), order=(3, 2, 1, 0)),
        fake(cutlass.Float32, (batch, seq_q, h), order=(2, 1, 0), align=4) if return_lse else None,
        fake(cutlass.Int8, (sym(),), order=(0,), align=32) if workspace else None,
        Int32(1),
        fake(cutlass.Int32, (batch,), order=(0,), align=4),
        fake(cutlass.Int32, (batch,), order=(0,), align=4),
        fake(cutlass.Int32, (batch,), order=(0,), align=4) if var_split else None,  # block_split_kvs  # PERFBOT item-03
        Float32(1.0),
        Float32(1.0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        pdl,
    ]
    arch_name = md.get_mla_decode_arch(arch)
    _log.info("PERFBOT-MLALOOP split_mma=%d q=%d heads=%d var_split=%d", SPLIT_MMA, q_len, heads, bool(var_split))  # once per compiled shape  # PERFBOT item-03 MLA-LOOP
    return cute.compile(*args, options=f"--enable-tvm-ffi --opt-level 2 --gpu-arch={arch_name}a",
                        **({"no_cache": True} if NO_CACHE else {}))  # PERFBOT item-03 MLA-LOOP


def stock_decode(query, cache, block_table, lengths, max_seq, scales, out, lse=None, min_split=1, pdl=True,
                 cp=(1, 0, 1), totals=None, pack=False,  # PERFBOT item-03
                 splits=None):  # item 01 L6: Int32[B] per-request splits (mla_varsplit.decode_args), None = served  # PERFBOT item-03
    """dcp.stock_decode's contract on the forked kernel: ``query`` [B, q, H, 576] FP8, ``out`` [B, q, H, 512] BF16,
    ``lse`` [B, q, H] FP32 log2 (or None); split from (B, max_seq, min_split) by tokenspeed's planner."""
    import tvm_ffi
    from tokenspeed_mla import mla_decode as md

    batch, q_len, heads, _ = query.shape
    if batch > MAX_REQS:  # the served gate (prepared.py: 8 <= rows <= 40, q 8) sends these to the incumbent
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.mla.dcp import stock_decode as inc
        return inc(query, cache, block_table, lengths, max_seq, scales, out, lse, min_split, pdl, cp, totals, pack)
    fp8, device = torch.float8_e4m3fn, query.device
    arch = torch.cuda.get_device_capability(device)
    tiler, _ = md.select_mla_decode_tilers(heads, q_len, is_fp8=True, compute_capability=arch)
    # the cut kernel (DESIGN.md section 3) serves sm_100 / sm_103, M128 2-CTA, packed Q only
    assert pack and tiler[0] == 128, "mla_core serves packed Q on the M128 2-CTA tile only"
    rows = (md.compute_q_tile_layout(heads, q_len, 128)[1], 128)
    sms = md.get_num_sm(device)
    if splits is not None:  # item 01's grid split floor (dcp.stock_decode's P-14 lines)  # PERFBOT item-03
        from perfbot_kernels import mla_varsplit  # PERFBOT item-03
        min_split = mla_varsplit.min_split(min_split, batch, splits)  # PERFBOT item-03
    split, size = md._get_split_kv_and_workspace_size(batch, *rows, 512, sms, max_seq, fp8, tiler, min_split)
    assert splits is None or split == mla_varsplit.s_grid(batch), (split, batch)  # mla_varsplit.compiler  # PERFBOT item-03
    key = device, heads
    if key not in _WORKSPACE or _WORKSPACE[key].numel() < size:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("the MLA split-KV workspace must be sized before capture")
        bound = max(size, sms * heads * 513 * 4)
        _WORKSPACE[key] = torch.empty(bound, dtype=torch.int8, device=device)
    kernel = compile_kernel(
        cache.shape[1], arch, heads, q_len, tuple(cp), pdl, lse is not None,
        md._get_reducer_d_tiles(batch, q_len, heads, sms, split), md._get_reducer_max_splits(split), pack, size > 0, IKET,  # PERFBOT item-03
        splits is not None, splits is not None and mla_varsplit.ORDER,  # PERFBOT item-03
        SPLIT_MMA)  # PERFBOT item-03 MLA-LOOP
    cache = cache.view(fp8)
    softmax, output_scale = scales
    args = [query[..., :512], query[..., 512:], cache[..., :512], cache[..., 512:]]
    args += [block_table, out, lse, _WORKSPACE[key][:size] if size else None]
    args += [Int32(split), lengths, lengths if totals is None else totals]
    args += [splits]  # the block_split_kvs slot  # PERFBOT item-03
    args += [Float32(softmax), Float32(output_scale)]
    with tvm_ffi.use_torch_stream():
        kernel(*args)
