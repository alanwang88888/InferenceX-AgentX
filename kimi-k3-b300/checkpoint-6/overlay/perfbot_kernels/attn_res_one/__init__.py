"""r3-307: one AttnRes kernel for every k (Gluon, Triton 3.7.1 in the image).

Replaces, on the decode path, the eight native `sm100::fwd_prod_v2::attn_res_fwd_online_v2_kernel<7168, k, ...>`
instantiations (k = 2..9; `torch.ops._C.kimi_k3_attn_res`) and the image's Triton `_attn_res_kernel` with ONE
kernel. The number of committed blocks k (`num_blocks`) is a runtime argument; the bank capacity KMAX
(`blocks.shape[1]`, one value per model) only sizes shared memory.

Op (per token row m; the image's attn_res, vllm/models/kimi_k3/nvidia/ops/attn_res.py at 3696c772):
  p   = bf16(prefix[m] + delta[m])                 (HAS_DELTA; stored back into prefix; else p = prefix[m])
  blocks[m, block_write_idx] = p                   (WRITE_BLOCK)
  sources = blocks[m, 0..k-1], p                   (k + 1 rows of HIDDEN bf16)
  logit_s = sum(s * norm_w * qk_w) * rsqrt(mean(s^2) + eps)
  mixed   = softmax(logits) @ sources
  out[m]  = rmsnorm(mixed, output_norm_w, output_norm_eps)  (APPLY_OUTPUT_NORM; else mixed)

Schedule: one CTA per token row (grid M <= 255, one wave on 148 SMs); no cross-CTA reduction.
  1. Before the PDL wait: cp.async the row's k committed blocks into shared memory (all in flight at once),
     load norm_w * qk_w, and run the online softmax over the bank: m_b, d_b and U = sum_j e^(l_j - m_b) b_j.
     The bank and the weights are not written by the immediately preceding kernel (a block is written by the
     pre-attention AttnRes of its own layer, many launches earlier), so this part overlaps the predecessor.
  2. After the PDL wait: load prefix (+ delta), store the bf16 prefix (and the block), its logit, then
     mixed = (U e^(m_b - m) + e^(l_p - m) p) / (d_b e^(m_b - m) + e^(l_p - m)), m = max(m_b, l_p), then the norm.
A row is held as an [8, 1024] tile (HIDDEN <= 8192, HIDDEN % 1024 == 0; row 7 masked for 7168).

Served (serves()): CUDA bf16, HIDDEN in {1024..8192} step 1024, M < 256 (the image's own decode/prefill split),
blocks.shape[1] <= KMAX_CAP, block_write_idx < 0 or >= num_blocks (a written block is never a source here),
16-byte aligned rows. Everything else runs the image path unchanged.
Switch: PERFBOT_R3_307_ATTN_RES=0 (arm env) restores the image path; on by default.
Sample inputs (selftest.py): prefix/delta [48, 7168] bf16, blocks [48, 9 or 10, 7168] bf16, weights [7168] bf16.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp

COLS = 1024          # tile columns; a row of HIDDEN is [8, COLS] with HIDDEN // COLS rows used
ROWS = 8
MAX_M = 255          # the image routes num_tokens >= 256 to its prefill tiling; we keep that split
KMAX_CAP = 12        # 12 x 16 KiB of bank in shared memory
NUM_WARPS = 8        # one warp per tile row of the [8, 1024] row tile

_SEEN: set = set()
CALLS = {"served": 0}


def enabled() -> bool:
    return os.environ.get("PERFBOT_R3_307_ATTN_RES", "1") != "0"


def source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@gluon.jit
def _add2(a0, b0, a1, b1):
    return a0 + a1, b0 + b1


@gluon.jit
def _row_logit(x, w, eps, HIDDEN: gl.constexpr):
    """x, w: [ROWS, COLS] fp32 (masked lanes 0) -> scalar dot(x, w) * rsqrt(mean(x^2) + eps)."""
    ss, dt = gl.reduce((x * x, x * w), 1, _add2)
    ss, dt = gl.reduce((ss, dt), 0, _add2)
    return dt * gl.rsqrt(ss * (1.0 / HIDDEN) + eps)


@gluon.jit(do_not_specialize=["num_blocks", "block_write_idx"])
def attn_res_one_kernel(
    prefix_ptr,
    delta_ptr,
    blocks_ptr,
    norm_w_ptr,
    qk_w_ptr,
    out_norm_w_ptr,
    out_ptr,
    stride_prefix_m,
    stride_delta_m,
    stride_block_m,
    stride_block_r,
    stride_out_m,
    num_blocks,
    block_write_idx,
    eps,
    out_eps,
    HIDDEN: gl.constexpr,
    KMAX: gl.constexpr,
    HAS_DELTA: gl.constexpr,
    WRITE_BLOCK: gl.constexpr,
    APPLY_OUTPUT_NORM: gl.constexpr,
    WARPS_R: gl.constexpr,
    WARPS_C: gl.constexpr,
    USE_PDL: gl.constexpr,
):
    # A row as [8, 1024]: warp r owns tile row r, 8 contiguous bf16 per thread per 256 columns.
    L: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8], threads_per_warp=[1, 32], warps_per_cta=[WARPS_R, WARPS_C], order=[1, 0]
    )
    SH: gl.constexpr = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    row = gl.program_id(0).to(gl.int64)
    r = gl.arange(0, 8, layout=gl.SliceLayout(1, L))
    c = gl.arange(0, 1024, layout=gl.SliceLayout(0, L))
    offs = gl.expand_dims(r, 1) * 1024 + gl.expand_dims(c, 0)
    dmask = offs < HIDDEN

    # 1. Bank: every committed block of this row in flight at once.
    bank = gl.allocate_shared_memory(gl.bfloat16, [KMAX, 8, 1024], SH)
    bank_row = blocks_ptr + row * stride_block_m
    for j in range(num_blocks):
        cp.async_copy_global_to_shared(bank.index(j), bank_row + j * stride_block_r + offs, mask=dmask)
    cp.commit_group()
    w = gl.load(norm_w_ptr + offs, mask=dmask, other=0.0).to(gl.float32) * gl.load(
        qk_w_ptr + offs, mask=dmask, other=0.0
    ).to(gl.float32)
    cp.wait_group(0)
    gl.barrier()
    # Online softmax over the bank: m_b, d_b and U = sum_j e^(l_j - m_b) b_j.
    m_b = -float("inf")
    d_b = 0.0
    u = gl.zeros([8, 1024], gl.float32, layout=L)
    for j in range(num_blocks):
        x = gl.where(dmask, bank.index(j).load(L).to(gl.float32), 0.0)
        logit = _row_logit(x, w, eps, HIDDEN)
        m_new = gl.maximum(m_b, logit)
        a0 = gl.exp(m_b - m_new)
        p0 = gl.exp(logit - m_new)
        d_b = d_b * a0 + p0
        u = u * a0 + p0 * x
        m_b = m_new

    # 2. The prefix: written by the preceding kernels, so after the PDL wait.
    if USE_PDL:
        gl.inline_asm_elementwise("griddepcontrol.wait; // dummy $0", "=r", [], dtype=gl.int32, is_pure=False,
                                  pack=1)
        gl.inline_asm_elementwise("griddepcontrol.launch_dependents; // dummy $0", "=r", [], dtype=gl.int32,
                                  is_pure=False, pack=1)
    pre = gl.load(prefix_ptr + row * stride_prefix_m + offs, mask=dmask, other=0.0)
    if HAS_DELTA:
        dl = gl.load(delta_ptr + row * stride_delta_m + offs, mask=dmask, other=0.0)
        pre = (pre.to(gl.float32) + dl.to(gl.float32)).to(gl.bfloat16)
        gl.store(prefix_ptr + row * stride_prefix_m + offs, pre, mask=dmask)
    if WRITE_BLOCK:
        gl.store(bank_row + block_write_idx * stride_block_r + offs, pre, mask=dmask)
    pf = pre.to(gl.float32)
    lp = _row_logit(pf, w, eps, HIDDEN)
    m = gl.maximum(m_b, lp)
    a = gl.exp(m_b - m)
    pp = gl.exp(lp - m)
    mixed = (u * a + pp * pf) / (d_b * a + pp)
    if APPLY_OUTPUT_NORM:
        ms = gl.sum(gl.sum(mixed * mixed, 1), 0)
        onw = gl.load(out_norm_w_ptr + offs, mask=dmask, other=0.0).to(gl.float32)
        mixed = mixed * gl.rsqrt(ms * (1.0 / HIDDEN) + out_eps) * onw
    gl.store(out_ptr + row * stride_out_m + offs, mixed.to(gl.bfloat16), mask=dmask)


def _aligned(t: torch.Tensor) -> bool:
    return t.data_ptr() % 16 == 0 and all(s % 8 == 0 for s in t.stride()[:-1]) and t.stride(-1) == 1


def serves(prefix, delta, blocks, norm_weight, qk_weight, output_norm_weight, num_blocks, block_write_idx) -> bool:
    if not enabled() or not prefix.is_cuda or prefix.dim() != 2:
        return False
    m, h = prefix.shape
    if not (0 < m <= MAX_M and h % COLS == 0 and h <= ROWS * COLS):
        return False
    if blocks.dim() != 3 or blocks.shape[0] < m or blocks.shape[2] != h or not 0 < blocks.shape[1] <= KMAX_CAP:
        return False
    if not 0 <= num_blocks <= blocks.shape[1]:
        return False
    if block_write_idx >= 0 and (block_write_idx < num_blocks or block_write_idx >= blocks.shape[1]):
        return False
    ts = [prefix, blocks, norm_weight, qk_weight] + [t for t in (delta, output_norm_weight) if t is not None]
    if any(t.dtype != torch.bfloat16 or not _aligned(t) for t in ts):
        return False
    if delta is not None and tuple(delta.shape) != (m, h):
        return False
    return torch.cuda.get_device_capability(prefix.device)[0] == 10


def _pdl() -> bool:
    try:
        from vllm.platforms import current_platform

        return bool(current_platform.is_arch_support_pdl())
    except Exception:
        return torch.cuda.get_device_capability()[0] >= 9


def launch(prefix, delta, blocks, norm_weight, qk_weight, output_norm_weight, num_blocks, block_write_idx, eps,
           output_norm_eps, out=None):
    m, h = prefix.shape
    out = torch.empty_like(prefix) if out is None else out
    nw = NUM_WARPS
    wr = 8
    pdl = _pdl()
    attn_res_one_kernel[(m,)](
        prefix, prefix if delta is None else delta, blocks, norm_weight, qk_weight,
        norm_weight if output_norm_weight is None else output_norm_weight, out,
        prefix.stride(0), 0 if delta is None else delta.stride(0), blocks.stride(0), blocks.stride(1), out.stride(0),
        num_blocks, block_write_idx, eps, output_norm_eps,
        HIDDEN=h, KMAX=blocks.shape[1], HAS_DELTA=delta is not None, WRITE_BLOCK=block_write_idx >= 0,
        APPLY_OUTPUT_NORM=output_norm_weight is not None, WARPS_R=wr, WARPS_C=nw // wr, USE_PDL=pdl,
        num_warps=nw, launch_pdl=pdl,
    )
    return out


_LIB = torch.library.Library("perfbot_attn_res", "DEF")
_LIB.define(
    "run(Tensor(a!) prefix, Tensor? delta, Tensor(b!) blocks, Tensor norm_weight, Tensor qk_weight, "
    "Tensor? output_norm_weight, int num_blocks, int block_write_idx, float eps, float output_norm_eps) -> Tensor"
)


def _run_cuda(prefix, delta, blocks, norm_weight, qk_weight, output_norm_weight, num_blocks, block_write_idx, eps,
              output_norm_eps):
    return launch(prefix, delta, blocks, norm_weight, qk_weight, output_norm_weight, num_blocks, block_write_idx,
                  eps, output_norm_eps)


_LIB.impl("run", _run_cuda, "CUDA")


def run(prefix, delta, blocks, norm_weight, qk_weight, output_norm_weight, num_blocks, block_write_idx, eps,
        output_norm_eps):
    """The served call: torch.ops.perfbot_attn_res.run; one log line per new (M, KMAX, k, flags) signature."""
    CALLS["served"] += 1
    key = (prefix.shape[0], blocks.shape[1], num_blocks, delta is not None, block_write_idx >= 0,
           output_norm_weight is not None)
    if key not in _SEEN:
        _SEEN.add(key)
        print(f"PERFBOT-ATTN-RES run sha={source_sha256()[:16]} M={key[0]} KMAX={key[1]} k={key[2]} "
              f"delta={int(key[3])} write={int(key[4])} norm={int(key[5])} warps={NUM_WARPS}", file=sys.stderr,
              flush=True)
    return torch.ops.perfbot_attn_res.run(prefix, delta, blocks, norm_weight, qk_weight, output_norm_weight,
                                          int(num_blocks), int(block_write_idx), float(eps), float(output_norm_eps))


def warmup_profiles(max_blocks: int) -> tuple[tuple[int, bool, int, bool], ...]:
    """Every (has_delta, write, norm) combination once: k is a runtime argument, so one call compiles each."""
    if max_blocks < 2:
        return ()
    return tuple((1, d, 1 if wr else -1, n) for d in (False, True) for wr in (False, True) for n in (False, True))
