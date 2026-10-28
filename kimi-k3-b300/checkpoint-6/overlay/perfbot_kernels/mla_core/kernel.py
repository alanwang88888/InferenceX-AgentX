# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# PERFBOT item 03 (mla_core): forked from tokenspeed_mla 2.0.2 mla_decode_fp8.py (sha256 4a8e1055...,
# https://github.com/lightseekorg/tokenspeed, MIT) as installed in vllm/vllm-openai:nightly-49cb2bbf. The standalone
# benchmark harness (run / __main__) is removed; every lever after the fork is its own commit (DESIGN.md).

import math
from types import SimpleNamespace
from typing import Optional, Tuple, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.cpasync as cpasync
import cutlass.memory
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import OperandMajorMode, tcgen05
from cutlass.cute.typing import Int32
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass._mlir.dialects import nvvm  # PERFBOT item-03
# item 01 L6 / 01d interface (perfbot_kernels/mla_varsplit/core.py): the empty-unit flag of the unit table  # PERFBOT item-03
PERFBOT_EMPTY = 1 << 16  # PERFBOT item-03

try:  # IKET phase ranges (iket=True only)
    from cutlass.cute.experimental import iket as _iket
except ImportError:  # pragma: no cover
    _iket = None

try:
    from tokenspeed_mla.mla_helpers import (
        LOG2_E,
        MAX_SPLITS,
        ceil_div,
        compute_q_tile_layout,
        get_mla_decode_arch,
    )
except ImportError:
    from mla_helpers import (
        LOG2_E,
        MAX_SPLITS,
        ceil_div,
        compute_q_tile_layout,
        get_mla_decode_arch,
    )

# TMEM columns per CTA on sm_100 / sm_103 (cute.arch.get_max_tmem_alloc_cols)
_TMEM_ALLOC_COLS = 512


"""
MLA decode with FP8 inputs on SM100 / SM103, cut to the one served path: packed (query row, head) M128 tiles on
2-CTA clusters (cta_group::2), FP8 Q x FP8 KV, paged KV (variable length), causal in global positions under DCP,
split-KV over a non-persistent grid (2, num_q_tiles x B, split) and the split reducer. Warp roles: load K, load Q + V,
MMA, softmax (4 warps), correction + epilogue (4 warps).
"""


class BlackwellMultiHeadLatentAttentionForwardFP8:
    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        lse_dtype: Type[cutlass.Numeric],
        mma_qk_tiler_mn: Tuple[int, int],
        mma_pv_tiler_mn: Tuple[int, int],
        page_size: int,
        skip_correction_threshold: float,
        num_heads: int,
        seq_len_q: int,
        cp_world: int = 1,  # DCP world size; >1 enables strided global-coord causal masking
        cp_rank: int = 0,
        cp_interleave_size: int = 1,
        reducer_d_tiles: int = 1,
        *,
        compute_capability: tuple[int, int],
        reducer_max_splits: int,
        iket: bool = False,
        is_var_split_kv: bool = False,  # item 01 L6: block_split_kvs[b] = request b's split (<= the grid split)  # PERFBOT item-03
        unit_table: bool = False,  # item 01d: block_split_kvs[B:] = the unit table; empty units exit  # PERFBOT item-03
        split_mma: bool = False,  # warp 8 issues QK only, warp 11 (the empty warp) issues PV only  # PERFBOT item-03 MLA-LOOP
    ):
        """The served MLA decode configuration (sm_100 / sm_103, M128 2-CTA, packed Q, causal).

        :param mma_qk_tiler_mn: (H, K) tile of the Q*K^T MMA; H must be 128
        :param mma_pv_tiler_mn: (H, D) tile of the P*V MMA
        :param reducer_d_tiles: Number of disjoint D512 output bands (1, 2 or 4)
        :param reducer_max_splits: Static reducer capacity; must cover every split
        :param iket: emit IKET phase ranges (run-iket); False emits no IR, so the kernel is unchanged
        """
        self.iket = bool(iket)
        self.is_var_split_kv, self.unit_table = is_var_split_kv, unit_table and is_var_split_kv  # PERFBOT item-03
        self.split_mma = bool(split_mma)  # PERFBOT item-03 MLA-LOOP
        if self.iket and _iket is None:
            raise ImportError("iket=True needs cutlass.cute.experimental.iket")
        self.arch = get_mla_decode_arch(compute_capability)
        if self.arch not in ("sm_100", "sm_103"):
            raise ValueError(f"mla_core serves sm_100 / sm_103 only, got {self.arch}")
        if mma_qk_tiler_mn[0] != 128 or mma_pv_tiler_mn[0] != 128:
            raise ValueError("mla_core serves the M128 2-CTA tile only")
        self.latent_dim = 512
        if reducer_d_tiles not in (1, 2, 4):
            raise ValueError("reducer_d_tiles must be 1, 2 or 4")
        if not 1 <= reducer_max_splits <= MAX_SPLITS:
            raise ValueError(f"reducer_max_splits must be in [1, {MAX_SPLITS}]")
        self.reducer_d_tiles = reducer_d_tiles
        self.reducer_d_tile = self.latent_dim // reducer_d_tiles
        # the reducer keeps a row's splits in registers: capacity is a multiple of 4,
        # loaded in sets of up to 64
        self.reducer_accumulate_group = 4
        self.reducer_max_splits = (
            ceil_div(reducer_max_splits, self.reducer_accumulate_group)
            * self.reducer_accumulate_group
        )
        self.reducer_batch_splits = min(self.reducer_max_splits, 64)
        self.rope_dim = 64
        self.acc_dtype = acc_dtype
        self.lse_dtype = lse_dtype
        self.mma_qk_tiler_mn = mma_qk_tiler_mn
        self.mma_pv_tiler_mn = mma_pv_tiler_mn
        self.skip_correction_threshold = skip_correction_threshold
        self.page_size = page_size
        # Packed Q: row r of the flat (query row, head) order is query r // num_heads, head r % num_heads.
        self.total_q_rows, self.num_q_tiles, self.tail_q_rows = compute_q_tile_layout(
            num_heads, seq_len_q, mma_qk_tiler_mn[0]
        )
        self.num_heads = num_heads
        self.seq_len_q = seq_len_q
        self.cp_world = cp_world
        self.cp_rank = cp_rank
        self.cp_interleave_size = cp_interleave_size
        self.cluster_shape_mnk = (2, 1, 1)
        # Warps 0-1 handle first N half; warps 2-3 handle second N half.
        self.warps_in_n = 2
        self.num_compute_warps = 4
        self.threads_per_warp = 32
        mma_qk_tiler_k = self.rope_dim * 2
        self.mma_qk_tiler = (
            self.mma_qk_tiler_mn[0],
            self.mma_qk_tiler_mn[1],
            mma_qk_tiler_k,
        )
        self.mma_qk_rope_tiler = (
            self.mma_qk_tiler_mn[0],
            self.mma_qk_tiler_mn[1],
            self.rope_dim,
        )
        self.mma_pv_tiler = (
            self.mma_pv_tiler_mn[0],
            self.mma_pv_tiler_mn[1],
            self.mma_qk_tiler[1] * self.mma_qk_tiler[2] // self.mma_pv_tiler_mn[1],
        )
        self.iterations_qk_latent = self.latent_dim // self.mma_qk_tiler[2]
        self.iterations_qk_rope = 1
        self.iterations_qk = self.iterations_qk_latent + self.iterations_qk_rope
        self.iterations_pv_k = self.mma_qk_tiler[1] // self.mma_pv_tiler[2]
        self.iterations_pv_n = self.latent_dim // self.mma_pv_tiler[1]

        # Set specialized warp ids
        self.compute_warp_ids = (0, 1, 2, 3)
        self.correction_warp_ids = (4, 5, 6, 7)
        self.mma_warp_id = 8
        self.load_tma_k_warp_id = 9
        self.load_tma_v_warp_id = 10
        self.empty_warp_ids = (11,)
        self.pv_warp_id = 11  # split_mma: the empty warp issues PV  # PERFBOT item-03 MLA-LOOP
        self.threads_per_cta = self.threads_per_warp * len(
            (
                self.mma_warp_id,
                self.load_tma_k_warp_id,
                self.load_tma_v_warp_id,
                *self.compute_warp_ids,
                *self.correction_warp_ids,
                *self.empty_warp_ids,
            )
        )

        # register settings
        self.softmax_reg_num = 208
        self.correction_reg_num = 256
        self.other_reg_num = 40
        # Named barriers
        self.tmem_ptr_sync_bar = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=(
                self.threads_per_warp
                + self.threads_per_warp * self.num_compute_warps * 2
                + (self.threads_per_warp if self.split_mma else 0)  # the PV warp retrieves the TMEM ptr  # PERFBOT item-03 MLA-LOOP
            ),
        )
        # E2b: running max hand-off between the softmax WGs (one arrives, the other waits) and WG0's final merge
        wg_pair = self.threads_per_warp * self.num_compute_warps * 2
        self.wg_pair_threads = wg_pair
        self.handoff_bar_ids = (6, 7)  # [wg] = the barrier WG wg arrives on after it publishes its max
        self.pair_bar_ids = ((2, 3), (4, 5))  # [wg] = row-max exchange pairs {w0, w2}, {w1, w3} of WG wg
        self.final_bar = pipeline.NamedBarrier(barrier_id=8, num_threads=wg_pair)
        # the epilogue WG (warps 4-7) arrives after its last TMEM read; the MMA warp waits before tmem.free
        self.tmem_free_bar = pipeline.NamedBarrier(barrier_id=9, num_threads=5 * self.threads_per_warp)
        # Split exchange barriers: warp0↔warp2 and warp1↔warp3 are independent
        # exchange pairs. Using separate barriers avoids cross-pair waiting.
        self.softmax_exchange_sync_bar_pair02 = pipeline.NamedBarrier(
            barrier_id=2, num_threads=(self.threads_per_warp * 2)
        )
        self.softmax_exchange_sync_bar_pair13 = pipeline.NamedBarrier(
            barrier_id=3, num_threads=(self.threads_per_warp * 2)
        )
        self.epilogue_exchange_sync_bar_pair02 = pipeline.NamedBarrier(
            barrier_id=4, num_threads=(self.threads_per_warp * 2)
        )
        self.epilogue_exchange_sync_bar_pair13 = pipeline.NamedBarrier(
            barrier_id=5, num_threads=(self.threads_per_warp * 2)
        )

        self.load_q_stage = 1
        self.load_k_stage = 3
        self.load_v_stage = 2
        self.mma_s_stage = 2
        self.p_mma_stage = 2
        self.mma_o_stage = 2 * self.iterations_pv_n  # (tile parity, O column half); full barriers only

        self.tmem_o_offset = self.mma_s_stage * self.mma_qk_tiler[1] // self.warps_in_n
        self.correction_factor_offset = (
            self.tmem_o_offset + self.latent_dim // self.warps_in_n
        )
        self.pv_acc_cols = self.mma_pv_tiler[1] // self.warps_in_n

    def _ik(self, name=None):
        """IKET phase range at trace time: push ``name``, or pop when None. Emits nothing unless iket=True.
        Per warp; every push and its pop are on one warp-uniform path."""
        if self.iket:
            if name is None:
                _iket.range_pop()
            else:
                _iket.range_push(name)

    @cute.jit
    def _get_dcp_local_bound(self, global_bound: Int32) -> Int32:
        """Map global causal bound to this rank's local bound."""
        if cutlass.const_expr(self.cp_interleave_size == 1):
            return (global_bound - self.cp_rank + self.cp_world - 1) // self.cp_world

        cycle_width = self.cp_world * self.cp_interleave_size
        full_cycles = global_bound // cycle_width
        cycle_offset = global_bound - full_cycles * cycle_width
        local_tail = cycle_offset - self.cp_rank * self.cp_interleave_size
        if cute.elem_less(local_tail, 0):
            local_tail = Int32(0)
        local_tail = cutlass.min(local_tail, self.cp_interleave_size)
        return full_cycles * self.cp_interleave_size + local_tail

    @cute.jit
    def __call__(
        self,
        q_latent: cute.Tensor,
        q_rope: cute.Tensor,
        c_latent: cute.Tensor,
        c_rope: cute.Tensor,
        page_table: cute.Tensor,
        o: cute.Tensor,
        lse: Optional[cute.Tensor],
        workspace: cute.Tensor,
        split_kv: cutlass.Int32,
        cache_seqs: Optional[cute.Tensor],
        causal_seqs: Optional[cute.Tensor],  # per-request global causal bound (DCP)
        block_split_kvs: Optional[cute.Tensor],  # item 01 L6 / 01d; None unless is_var_split_kv  # PERFBOT item-03
        softmax_scale: cutlass.Float32,
        output_scale: cutlass.Float32,
        stream: cuda.CUstream,
        use_pdl: cutlass.Constexpr = True,
    ):
        """Execute the Multi-Head Latent Attention operation on the provided tensors.

        The method handles:
        1. Initialization of workspace for split-KV buffers
        2. Validation of tensor data types
        3. Initialization of hardware-specific parameters and memory layouts
        4. Configuration of TMA (Tensor Memory Access) operations
        5. Grid and work scheduling computation
        6. Kernel launch(split KV kernel and reduction kernel) with appropriate parameters

        :param q_latent: The query tensor with shape [batch_size, seq_len_q, num_head, latent_dim] (contiguous)
        :type q_latent: cute.Tensor
        :param q_rope: The query RoPE tensor with shape [batch_size, seq_len_q, num_head, rope_dim] (contiguous)
        :type q_rope: cute.Tensor
        :param c_latent: The key tensor with shape [num_pages, page_size, latent_dim] (contiguous)
        :type c_latent: cute.Tensor
        :param c_rope: The key RoPE tensor with shape [num_pages, page_size, rope_dim] (contiguous)
        :type c_rope: cute.Tensor
        :param page_table: The page table tensor with shape [batch_size, page_count] (contiguous)
        :type page_table: cute.Tensor
        :param o: The output tensor with shape [batch_size, seq_len_q, num_head, latent_dim] (contiguous)
        :type o: cute.Tensor
        :param lse: The LSE tensor with shape [batch_size, seq_len_q, num_head] (contiguous)
        :type lse: cute.Tensor
        :param workspace: The workspace tensor with 1-d shape prepared for acc_o and acc_lse
        :type workspace: cute.Tensor
        :param split_kv: The scalar factor for split KV
        :type split_kv: cutlass.Int32
        :param cache_seqs: The cache sequences tensor with shape [batch_size]
        :type cache_seqs: cute.Tensor
        :param softmax_scale: The scale factor for softmax
        :type softmax_scale: cutlass.Float32
        :param output_scale: The scale factor for the output
        :type output_scale: cutlass.Float32
        :param stream: The CUDA stream to execute the kernel on
        :type stream: cuda.CUstream

        :raises TypeError: If tensor data types don't match or aren't supported
        """

        # setup static attributes before smem/grid/tma computation
        self.q_dtype = q_latent.element_type
        self.k_dtype = c_latent.element_type
        self.v_dtype = c_latent.element_type
        self.o_dtype = o.element_type
        self.skip_lse = lse is None

        # check type consistency
        if cutlass.const_expr(
            self.q_dtype != self.k_dtype or self.q_dtype != self.v_dtype
        ):
            raise TypeError(
                f"Type mismatch: {self.q_dtype} != {self.k_dtype} or {self.q_dtype} != {self.v_dtype}"
            )

        # Reinterpret contiguous [B, S_q, H, D] as [H, D, S_q, B]
        # Input stride: (S_q*H*D, H*D, D, 1) → Target: (D, 1, H*D, S_q*H*D)
        def _reinterpret_4d(t):
            return cute.make_tensor(
                t.iterator,
                cute.make_layout(
                    (t.shape[2], t.shape[3], t.shape[1], t.shape[0]),
                    stride=(t.stride[2], t.stride[3], t.stride[1], t.stride[0]),
                ),
            )

        q_latent = _reinterpret_4d(q_latent)
        q_rope = _reinterpret_4d(q_rope)
        o = _reinterpret_4d(o)

        # Reinterpret contiguous [num_pages, page_size, D] as [page_size, D, num_pages]
        # Input stride: (PS*D, D, 1) → Target: (D, 1, PS*D)
        def _reinterpret_3d_kv(t):
            return cute.make_tensor(
                t.iterator,
                cute.make_layout(
                    (t.shape[1], t.shape[2], t.shape[0]),
                    stride=(t.stride[1], t.stride[2], t.stride[0]),
                ),
            )

        c_latent = _reinterpret_3d_kv(c_latent)
        c_rope = _reinterpret_3d_kv(c_rope)

        # Reinterpret contiguous [B, page_count] as [page_count, B]
        page_table = cute.make_tensor(
            page_table.iterator,
            cute.make_layout(
                (page_table.shape[1], page_table.shape[0]),
                stride=(page_table.stride[1], page_table.stride[0]),
            ),
        )

        # Reinterpret contiguous [B, S_q, H] as [H, S_q, B]
        # Input stride: (S_q*H, H, 1) → Target: (1, H, S_q*H)
        lse = (
            cute.make_tensor(
                lse.iterator,
                cute.make_layout(
                    (lse.shape[2], lse.shape[1], lse.shape[0]),
                    stride=(lse.stride[2], lse.stride[1], lse.stride[0]),
                ),
            )
            if not self.skip_lse
            else None
        )

        o_reduction = o
        lse_reduction = lse
        # Packed Q: bound Q's TMA row extent by the real rows; only the final
        # tile can be partial. Keep a singleton S mode for the existing loader.
        def _flatten_q(t):
            return cute.make_tensor(
                t.iterator,
                cute.make_layout(
                    (self.total_q_rows, t.shape[1], 1, t.shape[3]),
                    stride=(t.stride[0], t.stride[1], t.stride[3], t.stride[3]),
                ),
            )

        q_latent = _flatten_q(q_latent)
        q_rope = _flatten_q(q_rope)
        m_tile = self.mma_qk_tiler_mn[0]
        num_q_tiles = cute.ceil_div(o.shape[0] * o.shape[2], m_tile)
        o = cute.make_tensor(
            o.iterator,
            cute.make_layout(
                (m_tile, o.shape[1], num_q_tiles, o.shape[3]),
                stride=(
                    o.stride[0],
                    o.stride[1],
                    m_tile * o.stride[0],
                    o.stride[3],
                ),
            ),
        )
        if cutlass.const_expr(not self.skip_lse):
            lse = cute.make_tensor(
                lse.iterator,
                cute.make_layout(
                    (m_tile, num_q_tiles, lse.shape[2]),
                    stride=(lse.stride[0], m_tile * lse.stride[0], lse.stride[2]),
                ),
            )

        acc_o, acc_lse = self.initialize_workspace(
            o.shape[0],
            o.shape[1],
            o.shape[2],
            o.shape[3],
            split_kv,
            self.acc_dtype,
            workspace,
        )

        c_latent_tranpose_layout = cute.select(c_latent.layout, mode=[1, 0, 2])
        c_latent_transpose = cute.make_tensor(
            c_latent.iterator, c_latent_tranpose_layout
        )

        self.q_major_mode = OperandMajorMode.K
        self.k_major_mode = OperandMajorMode.K
        self.v_major_mode = OperandMajorMode.MN

        cta_group = tcgen05.CtaGroup.TWO
        # the intermediate tensor p is from smem & k-major
        p_major_mode = OperandMajorMode.K
        qk_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            self.q_dtype,
            self.q_major_mode,
            self.k_major_mode,
            self.acc_dtype,
            cta_group,
            self.mma_qk_tiler[:2],
        )
        pv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.v_dtype,
            self.v_dtype,
            p_major_mode,
            self.v_major_mode,
            self.acc_dtype,
            cta_group,
            self.mma_pv_tiler[:2],
        )
        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (qk_tiled_mma.thr_id.shape,),
        )

        self.epi_tile = self.mma_pv_tiler[:2]

        q_latent_smem_layout_staged = sm100_utils.make_smem_layout_a(
            qk_tiled_mma,
            self.mma_qk_tiler,
            self.q_dtype,
            (self.iterations_qk_latent * self.load_q_stage),
        )
        q_latent_smem_layout_staged = cute.logical_divide(
            q_latent_smem_layout_staged, (None, None, None, self.iterations_qk_latent)
        )
        q_rope_smem_layout_staged = sm100_utils.make_smem_layout_a(
            qk_tiled_mma,
            self.mma_qk_rope_tiler,
            self.q_dtype,
            self.load_q_stage,
        )

        kc_latent_smem_layout_staged = sm100_utils.make_smem_layout_b(
            qk_tiled_mma,
            self.mma_qk_tiler,
            self.k_dtype,
            (self.iterations_qk_latent * self.load_k_stage),
        )
        kc_page_tile_size = min(
            self.page_size, qk_tiled_mma.op.shape_mnk[0] // qk_tiled_mma.thr_id.shape
        )
        kc_latent_smem_layout_staged = cute.logical_divide(
            kc_latent_smem_layout_staged, (None, None, None, self.iterations_qk_latent)
        )

        kc_latent_smem_layout_for_tma = sm100_utils.make_smem_layout(
            OperandMajorMode.K,
            (self.mma_qk_tiler[0] // qk_tiled_mma.thr_id.shape, self.mma_qk_tiler[2]),
            self.k_dtype,
            (self.iterations_qk_latent * self.load_k_stage),
        )
        kc_latent_smem_layout_for_tma = cute.tiled_divide(
            kc_latent_smem_layout_for_tma, (kc_page_tile_size, self.mma_qk_tiler[2])
        )
        kc_latent_smem_layout_for_tma = cute.logical_divide(
            kc_latent_smem_layout_for_tma, (None, None, None, self.iterations_qk_latent)
        )

        kc_rope_smem_layout_staged = sm100_utils.make_smem_layout_b(
            qk_tiled_mma,
            self.mma_qk_rope_tiler,
            self.k_dtype,
            self.load_k_stage,
        )
        kc_rope_smem_layout_for_tma = sm100_utils.make_smem_layout(
            OperandMajorMode.K,
            (
                self.mma_qk_rope_tiler[0] // qk_tiled_mma.thr_id.shape,
                self.mma_qk_rope_tiler[2],
            ),
            self.k_dtype,
            (self.iterations_qk_rope * self.load_k_stage),
        )
        kc_rope_smem_layout_for_tma = cute.tiled_divide(
            kc_rope_smem_layout_for_tma, (kc_page_tile_size, self.mma_qk_rope_tiler[2])
        )

        p_smem_layout_staged = sm100_utils.make_smem_layout_a(
            pv_tiled_mma,
            self.mma_pv_tiler,
            self.q_dtype,
            (self.iterations_pv_k * self.p_mma_stage),
        )
        p_smem_layout_staged = cute.logical_divide(
            p_smem_layout_staged, (None, None, None, self.iterations_pv_k)
        )

        vc_smem_layout_staged = sm100_utils.make_smem_layout_b(
            pv_tiled_mma,
            self.mma_pv_tiler,
            self.v_dtype,
            (self.iterations_pv_k * self.iterations_pv_n * self.load_v_stage),
        )
        vc_smem_layout_staged = cute.logical_divide(
            cute.logical_divide(
                vc_smem_layout_staged,
                (None, None, None, self.iterations_pv_k * self.iterations_pv_n),
            ),
            (None, None, None, (self.iterations_pv_n, None)),
        )
        vc_page_tile_size = min(self.page_size, self.mma_pv_tiler[2])
        vc_smem_layout_for_tma = sm100_utils.make_smem_layout(
            OperandMajorMode.MN,
            (self.mma_pv_tiler[1] // pv_tiled_mma.thr_id.shape, self.mma_pv_tiler[2]),
            self.v_dtype,
            (self.iterations_pv_k * self.iterations_pv_n * self.load_v_stage),
        )
        vc_smem_layout_for_tma = cute.tiled_divide(
            vc_smem_layout_for_tma,
            (
                pv_tiled_mma.op.shape_mnk[1] // pv_tiled_mma.thr_id.shape,
                vc_page_tile_size,
            ),
        )
        vc_smem_layout_for_tma = cute.logical_divide(
            cute.logical_divide(
                vc_smem_layout_for_tma,
                (None, None, None, self.iterations_pv_k * self.iterations_pv_n),
            ),
            (None, None, None, (self.iterations_pv_n, None)),
        )
        # TMA load for Q latent and rope
        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(cta_group)

        q_smem_layout = cute.select(q_latent_smem_layout_staged, mode=[0, 1, 2])

        tma_atom_q_latent, tma_tensor_q_latent = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            q_latent,
            q_smem_layout,
            self.mma_qk_tiler,
            qk_tiled_mma,
            cta_layout_vmnk.shape,
        )
        q_rope_smem_layout = cute.select(q_rope_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_q_rope, tma_tensor_q_rope = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            q_rope,
            q_rope_smem_layout,
            self.mma_qk_rope_tiler,
            qk_tiled_mma,
            cta_layout_vmnk.shape,
        )
        # TMA load for c latent and k rope
        kc_smem_layout = cute.select(kc_latent_smem_layout_for_tma, mode=[0])
        tma_atom_c_latent, tma_tensor_c_latent = self.make_paged_tiled_tma_atom(
            tma_load_op,
            c_latent,
            kc_smem_layout,
            (self.mma_qk_tiler[1], self.mma_qk_tiler[2]),
            qk_tiled_mma,
            is_k_load=True,
        )
        kc_rope_smem_layout = cute.select(kc_rope_smem_layout_for_tma, mode=[0])
        tma_atom_c_rope, tma_tensor_c_rope = self.make_paged_tiled_tma_atom(
            tma_load_op,
            c_rope,
            kc_rope_smem_layout,
            (self.mma_qk_rope_tiler[1], self.mma_qk_rope_tiler[2]),
            qk_tiled_mma,
            is_k_load=True,
        )

        # TMA load for c latent transpose
        vc_smem_layout = cute.select(vc_smem_layout_for_tma, mode=[0])
        tma_atom_c_latent_transpose, tma_tensor_c_latent_transpose = (
            self.make_paged_tiled_tma_atom(
                tma_load_op,
                c_latent_transpose,
                vc_smem_layout,
                (self.mma_pv_tiler[1], self.mma_pv_tiler[2]),
                pv_tiled_mma,
                is_k_load=False,
            )
        )

        q_latent_copy_size = (
            cute.size_in_bytes(self.q_dtype, q_smem_layout)
            * cute.size(qk_tiled_mma.thr_id.shape)
            * self.iterations_qk_latent
        )
        q_rope_copy_size = (
            cute.size_in_bytes(self.q_dtype, q_rope_smem_layout)
            * cute.size(qk_tiled_mma.thr_id.shape)
            * self.iterations_qk_rope
        )
        kc_latent_copy_size = (
            cute.size_in_bytes(
                self.k_dtype,
                cute.select(kc_latent_smem_layout_staged, mode=[0, 1, 2]),
            )
            * cute.size(qk_tiled_mma.thr_id.shape)
            * self.iterations_qk_latent
        )
        kc_rope_copy_size = (
            cute.size_in_bytes(
                self.k_dtype,
                cute.select(kc_rope_smem_layout_staged, mode=[0, 1, 2]),
            )
            * cute.size(qk_tiled_mma.thr_id.shape)
            * self.iterations_qk_rope
        )
        vc_copy_size = (
            cute.size_in_bytes(
                self.v_dtype, cute.select(vc_smem_layout_staged, mode=[0, 1, 2])
            )
            * cute.size(pv_tiled_mma.thr_id.shape)
            * self.iterations_pv_n
            * self.iterations_pv_k
        )

        self.tma_copy_q_bytes = q_latent_copy_size + q_rope_copy_size
        self.tma_copy_kc_bytes = kc_latent_copy_size + kc_rope_copy_size
        self.tma_copy_vc_bytes = vc_copy_size

        # non-persistent: one work tile per CTA, blk_coord = (cta in cluster, q tile, b, split)
        grid = (self.cluster_shape_mnk[0], o.shape[3] * o.shape[2], split_kv)

        @cute.struct
        class SplitKVKernelSharedStorage:
            # Pipeline barriers
            load_q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_q_stage * 2]
            load_k_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_k_stage * 2]
            load_v_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_v_stage * 2]
            mma_s_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mma_s_stage * 2]
            p_mma_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.p_mma_stage * 2]
            mma_o_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mma_o_stage * 2]
            # Tmem dealloc cluster barrier
            tmem_dealloc_mbar_ptr: cutlass.Int64
            # Tmem holding buffer
            tmem_holding_buf: cutlass.Int32
            # E2b: WG0's final row sum. It sits in the pad before the 1024-aligned smem_p
            # (28 mbarriers + 16 B + 512 B < 1024 B), so the E2b buffers add no 1 KB block.
            final_sum_smem: cute.struct.MemRange[
                self.acc_dtype, self.num_compute_warps * self.threads_per_warp
            ]

            # Smem tensors
            smem_p: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, cute.cosize(p_smem_layout_staged)],
                1024,
            ]
            smem_kc_latent: cute.struct.Align[
                cute.struct.MemRange[
                    self.k_dtype, cute.cosize(kc_latent_smem_layout_staged)
                ],
                1024,
            ]

            smem_kc_rope: cute.struct.Align[
                cute.struct.MemRange[
                    self.k_dtype, cute.cosize(kc_rope_smem_layout_staged)
                ],
                1024,
            ]
            smem_q_latent: cute.struct.Align[
                cute.struct.MemRange[
                    self.q_dtype, cute.cosize(q_latent_smem_layout_staged)
                ],
                1024,
            ]
            smem_q_rope: cute.struct.Align[
                cute.struct.MemRange[
                    self.q_dtype, cute.cosize(q_rope_smem_layout_staged)
                ],
                1024,
            ]
            smem_vc: cute.struct.Align[
                cute.struct.MemRange[self.v_dtype, cute.cosize(vc_smem_layout_staged)],
                1024,
            ]
            softmax_smem_exchange: cute.struct.MemRange[
                self.acc_dtype, self.num_compute_warps * self.threads_per_warp
            ]
            epilogue_smem_exchange: cute.struct.MemRange[
                self.acc_dtype, self.num_compute_warps * self.threads_per_warp
            ]
            handoff_smem: cute.struct.MemRange[
                self.acc_dtype, 2 * self.num_compute_warps * self.threads_per_warp
            ]

        softmax_scale_log2 = softmax_scale * LOG2_E

        self.split_kv_kernel(
            qk_tiled_mma,
            pv_tiled_mma,
            tma_atom_q_latent,
            tma_tensor_q_latent,
            tma_atom_q_rope,
            tma_tensor_q_rope,
            tma_atom_c_latent,
            tma_tensor_c_latent,
            tma_atom_c_rope,
            tma_tensor_c_rope,
            tma_atom_c_latent_transpose,
            tma_tensor_c_latent_transpose,
            page_table,
            o,
            lse if not self.skip_lse else None,
            acc_o,
            acc_lse,
            split_kv,
            cache_seqs,
            causal_seqs,
            block_split_kvs,  # PERFBOT item-03
            softmax_scale_log2,
            output_scale,
            q_latent_smem_layout_staged,
            q_rope_smem_layout_staged,
            kc_latent_smem_layout_staged,
            kc_rope_smem_layout_staged,
            p_smem_layout_staged,
            vc_smem_layout_staged,
            kc_latent_smem_layout_for_tma,
            kc_rope_smem_layout_for_tma,
            vc_smem_layout_for_tma,
            cta_layout_vmnk,
            SplitKVKernelSharedStorage,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            smem=SplitKVKernelSharedStorage.size_in_bytes(),  # type: ignore[attr-defined]
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=use_pdl,
        )
        if cutlass.const_expr(acc_o is not None):
            self.reduction_kernel(
                o_reduction,
                lse_reduction,
                acc_o,
                acc_lse,
                split_kv,
                cache_seqs,
                block_split_kvs,  # PERFBOT item-03
                output_scale,
            ).launch(
                grid=(
                    o_reduction.shape[0] * self.reducer_d_tiles,
                    o_reduction.shape[2],
                    o_reduction.shape[3],
                ),
                block=[self.threads_per_warp * self.num_compute_warps, 1, 1],
                smem=ceil_div(self.reducer_max_splits, self.threads_per_warp)
                * self.threads_per_warp
                * self.acc_dtype.width
                // 8,
                stream=stream,
                min_blocks_per_mp=1,
                use_pdl=use_pdl,
            )

    @cute.jit
    def make_paged_tiled_tma_atom(
        self,
        tma_load_op: cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp,
        gmem: cute.Tensor,
        smem_layout: cute.Layout,
        mma_tiler,
        tiled_mma: cute.TiledMma,
        is_k_load: bool,
    ):
        ident = cute.make_identity_layout(gmem.shape)
        g_tile = cute.composition(ident, mma_tiler)
        cta_mn = mma_tiler[0] // tiled_mma.thr_id.shape
        cta_v_map = cute.flat_divide(g_tile, (cta_mn,))
        cta_v_map = cute.select(cta_v_map, mode=[0, 2])
        page_tile_size = (
            min(self.page_size, cta_mn)
            if is_k_load
            else min(self.page_size, mma_tiler[1])
        )
        cta_v_map = cute.zipped_divide(
            cta_v_map,
            (page_tile_size, mma_tiler[1]) if is_k_load else (cta_mn, page_tile_size),
        )
        cta_v_map = cute.select(cta_v_map, mode=[0])
        from cutlass._mlir.dialects import cute_nvgpu as _cute_nvgpu_ir

        res = _cute_nvgpu_ir.atom_make_non_exec_tiled_tma_load(
            gmem.value,
            smem_layout.value,
            cta_v_map,
            tma_load_op._to_ir(),
            num_multicast=1,
        )
        return (
            cute.CopyAtom(
                tma_load_op, cpasync.CopyBulkTensorTileG2SNonExecTrait(res[0])
            ),
            res[1],
        )

    @cute.kernel
    def split_kv_kernel(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tma_atom_q_latent: Optional[cute.CopyAtom],
        mQL: cute.Tensor,
        tma_atom_q_rope: Optional[cute.CopyAtom],
        mQR: cute.Tensor,
        tma_atom_c_latent: Optional[cute.CopyAtom],
        mCL: cute.Tensor,
        tma_atom_c_rope: Optional[cute.CopyAtom],
        mKR: cute.Tensor,
        tma_atom_c_latent_transpose: Optional[cute.CopyAtom],
        mCLT: cute.Tensor,
        mPT: cute.Tensor,
        mO: Optional[cute.Tensor],
        mLSE: Optional[cute.Tensor],
        mAccO: Optional[cute.Tensor],
        mAccLSE: Optional[cute.Tensor],
        split_kv: cutlass.Int32,
        cache_seqs: cute.Tensor,
        causal_seqs: cute.Tensor,  # per-request global causal bound (DCP)
        block_split_kvs: Optional[cute.Tensor],  # PERFBOT item-03
        softmax_scale_log2: cutlass.Float32,
        output_scale: cutlass.Float32,
        q_latent_smem_layout_staged: cute.ComposedLayout,
        q_rope_smem_layout_staged: cute.ComposedLayout,
        kc_latent_smem_layout_staged: cute.ComposedLayout,
        kc_rope_smem_layout_staged: cute.ComposedLayout,
        p_smem_layout_staged: cute.ComposedLayout,
        vc_smem_layout_staged: cute.ComposedLayout,
        kc_latent_smem_layout_for_tma: Optional[cute.ComposedLayout],
        kc_rope_smem_layout_for_tma: Optional[cute.ComposedLayout],
        vc_smem_layout_for_tma: Optional[cute.ComposedLayout],
        cta_layout_vmnk: cute.Layout,
        SharedStorage: cutlass.Constexpr,
    ):
        """One (q tile, request, split) work tile per 2-CTA cluster. Warp roles: load K (warp 9), load Q + V
        (warp 10), MMA (warp 8, leader CTA issues), softmax (warps 0-3), correction + epilogue (warps 4-7).
        With split_kv > 1 the epilogue writes normalized fp32 partials + LSE to the workspace (mAccO / mAccLSE);
        with split_kv == 1 it writes the final rows to mO / mLSE."""

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma_qk.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        # grid (2, num_q_tiles * B, split): y = q_tile * B + b
        num_b = mO.shape[3]
        blk_coord = (bidx, bidy // num_b, bidy % num_b, bidz)
        if cutlass.const_expr(self.unit_table):  # table entry B + z * B + b: the unit b' + B * z' (+ EMPTY)  # PERFBOT item-03
            perfbot_entry = block_split_kvs[num_b + bidz * num_b + bidy % num_b]  # PERFBOT item-03
            perfbot_unit = perfbot_entry % PERFBOT_EMPTY  # PERFBOT item-03
            blk_coord = (bidx, blk_coord[1], perfbot_unit % num_b, perfbot_unit // num_b)  # PERFBOT item-03
        if cutlass.const_expr(self.is_var_split_kv):  # request b's split partitions its k tiles  # PERFBOT item-03
            split_kv = block_split_kvs[blk_coord[2]]  # PERFBOT item-03

        # Prefetch TMA descriptors from load warps (not MMA warp) so the
        # MMA warp can proceed to TMEM allocation sooner.  The prefetch
        # populates the L2 cache and is not warp-specific.
        if warp_idx == self.load_tma_k_warp_id:
            cpasync.prefetch_descriptor(tma_atom_c_latent)
            cpasync.prefetch_descriptor(tma_atom_c_rope)
        if warp_idx == self.load_tma_v_warp_id:
            cpasync.prefetch_descriptor(tma_atom_q_latent)
            cpasync.prefetch_descriptor(tma_atom_q_rope)
            cpasync.prefetch_descriptor(tma_atom_c_latent_transpose)

        # Alloc
        smem = cutlass.memory.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # Tensor memory dealloc barrier init
        tmem = cutlass.memory.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_ptr_sync_bar,
            allocator_warp_id=self.mma_warp_id,
            is_two_cta=True,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr.ptr,
            arch=self.arch,
        )

        load_q_pipeline = self.make_and_init_load_qkv_pipeline(
            storage.load_q_mbar_ptr.data_ptr(),
            cta_layout_vmnk,
            self.load_q_stage,
            self.tma_copy_q_bytes,
        )
        load_k_pipeline = self.make_and_init_load_qkv_pipeline(
            storage.load_k_mbar_ptr.data_ptr(),
            cta_layout_vmnk,
            self.load_k_stage,
            self.tma_copy_kc_bytes,
        )
        load_v_pipeline = self.make_and_init_load_qkv_pipeline(
            storage.load_v_mbar_ptr.data_ptr(),
            cta_layout_vmnk,
            self.load_v_stage,
            self.tma_copy_vc_bytes,
        )
        mma_s_pipeline = self.make_and_init_mma_s_pipeline(
            storage.mma_s_mbar_ptr.data_ptr(), cta_layout_vmnk
        )
        p_mma_pipeline = self.make_and_init_p_mma_pipeline(
            storage.p_mma_mbar_ptr.data_ptr(), cta_layout_vmnk
        )
        mma_o_pipeline = self.make_and_init_mma_o_pipeline(
            storage.mma_o_mbar_ptr.data_ptr(), cta_layout_vmnk
        )

        if cutlass.const_expr(self.unit_table and mAccO is not None):  # 01d: an empty unit exits before the  # PERFBOT item-03
            if perfbot_entry >= PERFBOT_EMPTY:  # cluster arrive, TMEM alloc and griddepcontrol_wait (both CTAs)  # PERFBOT item-03
                nvvm.exit()  # PERFBOT item-03
        # Cluster arrive after cross-CTA barrier init.
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mnk, is_relaxed=True)

        # Generate smem tensor Q/KC/VC/exchange
        # (MMA, MMA_H, MMA_R, PIPE)
        sQ = storage.smem_q_latent.get_tensor(
            q_latent_smem_layout_staged.outer, swizzle=q_latent_smem_layout_staged.inner
        )
        sQ_rope = storage.smem_q_rope.get_tensor(
            q_rope_smem_layout_staged.outer, swizzle=q_rope_smem_layout_staged.inner
        )
        # (MMA, MMA_K, MMA_R, PIPE)
        sKC = storage.smem_kc_latent.get_tensor(
            kc_latent_smem_layout_staged.outer,
            swizzle=kc_latent_smem_layout_staged.inner,
        )
        sKC_rope = storage.smem_kc_rope.get_tensor(
            kc_rope_smem_layout_staged.outer, swizzle=kc_rope_smem_layout_staged.inner
        )
        sKC_for_tma = storage.smem_kc_latent.get_tensor(
            kc_latent_smem_layout_for_tma.outer,
            swizzle=kc_latent_smem_layout_for_tma.inner,
        )
        sKC_rope_for_tma = storage.smem_kc_rope.get_tensor(
            kc_rope_smem_layout_for_tma.outer, swizzle=kc_rope_smem_layout_for_tma.inner
        )
        # (MMA, MMA_D, MMA_K, PIPE)
        sVC = storage.smem_vc.get_tensor(
            vc_smem_layout_staged.outer, swizzle=vc_smem_layout_staged.inner
        )
        sVC_for_tma = storage.smem_vc.get_tensor(
            vc_smem_layout_for_tma.outer, swizzle=vc_smem_layout_for_tma.inner
        )
        # (MMA, MMA_H, MMA_K)
        sP = storage.smem_p.get_tensor(
            p_smem_layout_staged.outer, swizzle=p_smem_layout_staged.inner
        )
        # (compute_threads,)
        softmax_smem_exchange = storage.softmax_smem_exchange.get_tensor(
            cute.make_layout(self.num_compute_warps * self.threads_per_warp)
        )
        epilogue_smem_exchange = storage.epilogue_smem_exchange.get_tensor(
            cute.make_layout(self.num_compute_warps * self.threads_per_warp)
        )
        # E2b: running max hand-off (lo -> hi, hi -> lo); the hi -> lo slot also carries WG0's final max
        handoff_smem = storage.handoff_smem.get_tensor(
            cute.make_layout(2 * self.num_compute_warps * self.threads_per_warp)
        )
        final_sum_smem = storage.final_sum_smem.get_tensor(
            cute.make_layout(self.num_compute_warps * self.threads_per_warp)
        )

        #
        # Cluster wait before tensor memory alloc
        #
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mnk)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Load warps, including page table and data tensors
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.empty_warp_ids[0] and warp_idx <= self.empty_warp_ids[-1]:
            cute.arch.setmaxregister_decrease(self.other_reg_num)

        if warp_idx == self.load_tma_k_warp_id:
            cute.arch.setmaxregister_decrease(self.other_reg_num)
            cute.arch.griddepcontrol_wait()
            load_k_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.load_k_stage
            )
            k_index, k_tile_count, local_split_kv = self.get_k_tile_count(
                split_kv, cache_seqs, blk_coord
            )
            if k_tile_count > 0:
                tma_common_params = SimpleNamespace(
                    blk_coord=blk_coord,
                    local_split_kv=local_split_kv,
                    load_k_pipeline=load_k_pipeline,
                    mPT=mPT,
                )
                tma_k_params = SimpleNamespace(
                    tiled_mma_qk=tiled_mma_qk,
                    tma_atom_c_latent=tma_atom_c_latent,
                    tma_atom_c_rope=tma_atom_c_rope,
                    mCL=mCL,
                    mKR=mKR,
                    sKC=sKC_for_tma,
                    sKC_rope=sKC_rope_for_tma,
                )
                # Load K only
                load_k_producer_state = self.load_tma_k(
                    tma_common_params,
                    tma_k_params,
                    k_index,
                    k_tile_count,
                    load_k_producer_state,
                )
            load_k_pipeline.producer_tail(load_k_producer_state)

        if warp_idx == self.load_tma_v_warp_id:
            cute.arch.setmaxregister_decrease(self.other_reg_num)
            cute.arch.griddepcontrol_wait()
            load_q_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.load_q_stage
            )
            load_v_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.load_v_stage
            )
            k_index, k_tile_count, local_split_kv = self.get_k_tile_count(
                split_kv, cache_seqs, blk_coord
            )
            if k_tile_count > 0:
                # Load Q first (once per work tile)
                tma_q_common_params = SimpleNamespace(
                    blk_coord=blk_coord,
                    load_q_pipeline=load_q_pipeline,
                )
                tma_q_params = SimpleNamespace(
                    tiled_mma_qk=tiled_mma_qk,
                    tma_atom_q_latent=tma_atom_q_latent,
                    tma_atom_q_rope=tma_atom_q_rope,
                    mQL=mQL,
                    mQR=mQR,
                    sQ=sQ,
                    sQ_rope=sQ_rope,
                )
                load_q_producer_state = self.load_tma_q(
                    tma_q_common_params,
                    tma_q_params,
                    load_q_producer_state,
                )
                # Then load V
                tma_v_common_params = SimpleNamespace(
                    blk_coord=blk_coord,
                    local_split_kv=local_split_kv,
                    load_v_pipeline=load_v_pipeline,
                    mPT=mPT,
                )
                tma_pv_params = SimpleNamespace(
                    tiled_mma_pv=tiled_mma_pv,
                    tma_atom_c_latent_transpose=tma_atom_c_latent_transpose,
                    mCLT=mCLT,
                    sVC=sVC_for_tma,
                )
                load_v_producer_state = self.load_tma_v(
                    tma_v_common_params,
                    tma_pv_params,
                    k_index,
                    k_tile_count,
                    load_v_producer_state,
                )
            load_q_pipeline.producer_tail(load_q_producer_state)
            load_v_pipeline.producer_tail(load_v_producer_state)

        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA warp
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.other_reg_num)
            # Fire off async TMEM allocation request early, then overlap
            # the allocation latency with pipeline state creation.
            tmem.allocate(_TMEM_ALLOC_COLS)

            load_q_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.load_q_stage
            )
            load_k_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.load_k_stage
            )
            load_v_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.load_v_stage
            )
            mma_s_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.mma_s_stage
            )
            p_mma_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.p_mma_stage
            )
            mma_o_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.mma_o_stage
            )

            # Now wait for TMEM allocation and broadcast pointer
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            k_index, k_tile_count, local_split_kv = self.get_k_tile_count(
                split_kv, cache_seqs, blk_coord
            )
            if k_tile_count > 0:
                mma_common_params = SimpleNamespace(
                    blk_coord=blk_coord,
                    K=cache_seqs[blk_coord[2]],
                    k_index=k_index,
                    local_split_kv=local_split_kv,
                    load_q_pipeline=load_q_pipeline,
                    load_k_pipeline=load_k_pipeline,
                    load_v_pipeline=load_v_pipeline,
                    tmem_ptr=tmem_ptr,
                    is_leader_cta=is_leader_cta,
                    L=mCL.shape[1],
                )
                mma_qk_params = SimpleNamespace(
                    mma_s_pipeline=mma_s_pipeline,
                    sQ=sQ,
                    sQ_rope=sQ_rope,
                    sKC=sKC,
                    sKC_rope=sKC_rope,
                )
                mma_pv_params = SimpleNamespace(
                    p_mma_pipeline=p_mma_pipeline,
                    mma_o_pipeline=mma_o_pipeline,
                    sP=sP,
                    sVC=sVC,
                )
                (
                    tiled_mma_qk,
                    tiled_mma_pv,
                    load_q_consumer_state,
                    load_k_consumer_state,
                    load_v_consumer_state,
                    mma_s_producer_state,
                    p_mma_consumer_state,
                    mma_o_producer_state,
                ) = self.mma(
                    mma_common_params,
                    mma_qk_params,
                    mma_pv_params,
                    k_tile_count,
                    tiled_mma_qk,
                    tiled_mma_pv,
                    load_q_consumer_state,
                    load_k_consumer_state,
                    load_v_consumer_state,
                    mma_s_producer_state,
                    p_mma_consumer_state,
                    mma_o_producer_state,
                    role=1 if self.split_mma else 0,  # PERFBOT item-03 MLA-LOOP
                )

            mma_s_pipeline.producer_tail(mma_s_producer_state)
            cute.arch.griddepcontrol_launch_dependents()

            self.tmem_free_bar.arrive_and_wait()
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

        if cutlass.const_expr(self.split_mma):  # PERFBOT item-03 MLA-LOOP
            if warp_idx == self.pv_warp_id:  # PERFBOT item-03 MLA-LOOP
                load_q_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.load_q_stage)  # PERFBOT item-03 MLA-LOOP
                load_k_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.load_k_stage)  # PERFBOT item-03 MLA-LOOP
                load_v_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.load_v_stage)  # PERFBOT item-03 MLA-LOOP
                mma_s_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.mma_s_stage)  # PERFBOT item-03 MLA-LOOP
                p_mma_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.p_mma_stage)  # PERFBOT item-03 MLA-LOOP
                mma_o_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.mma_o_stage)  # PERFBOT item-03 MLA-LOOP
                tmem.wait_for_alloc()  # PERFBOT item-03 MLA-LOOP
                tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)  # PERFBOT item-03 MLA-LOOP
                k_index, k_tile_count, local_split_kv = self.get_k_tile_count(split_kv, cache_seqs, blk_coord)  # PERFBOT item-03 MLA-LOOP
                if k_tile_count > 0:  # PERFBOT item-03 MLA-LOOP
                    (tiled_mma_qk, tiled_mma_pv, load_q_consumer_state, load_k_consumer_state, load_v_consumer_state,  # PERFBOT item-03 MLA-LOOP
                     mma_s_producer_state, p_mma_consumer_state, mma_o_producer_state) = self.mma(  # PERFBOT item-03 MLA-LOOP
                        SimpleNamespace(blk_coord=blk_coord, K=cache_seqs[blk_coord[2]], k_index=k_index,  # PERFBOT item-03 MLA-LOOP
                                        local_split_kv=local_split_kv, load_q_pipeline=load_q_pipeline,  # PERFBOT item-03 MLA-LOOP
                                        load_k_pipeline=load_k_pipeline, load_v_pipeline=load_v_pipeline,  # PERFBOT item-03 MLA-LOOP
                                        tmem_ptr=tmem_ptr, is_leader_cta=is_leader_cta, L=mCL.shape[1]),  # PERFBOT item-03 MLA-LOOP
                        SimpleNamespace(mma_s_pipeline=mma_s_pipeline, sQ=sQ, sQ_rope=sQ_rope, sKC=sKC, sKC_rope=sKC_rope),  # PERFBOT item-03 MLA-LOOP
                        SimpleNamespace(p_mma_pipeline=p_mma_pipeline, mma_o_pipeline=mma_o_pipeline, sP=sP, sVC=sVC),  # PERFBOT item-03 MLA-LOOP
                        k_tile_count, tiled_mma_qk, tiled_mma_pv, load_q_consumer_state, load_k_consumer_state,  # PERFBOT item-03 MLA-LOOP
                        load_v_consumer_state, mma_s_producer_state, p_mma_consumer_state, mma_o_producer_state,  # PERFBOT item-03 MLA-LOOP
                        role=2,  # PERFBOT item-03 MLA-LOOP
                    )  # PERFBOT item-03 MLA-LOOP

        # ///////////////////////////////////////////////////////////////////////////////
        #  Softmax WGs (E2b): WG0 = warps 0-3 (even tiles), WG1 = warps 4-7 (odd tiles, then the merge + epilogue)
        # ///////////////////////////////////////////////////////////////////////////////
        # warps 0-3 and 4-7 are the first eight warps
        if warp_idx <= self.correction_warp_ids[-1]:
            if warp_idx <= self.compute_warp_ids[-1]:
                cute.arch.setmaxregister_increase(self.softmax_reg_num)
            else:
                cute.arch.setmaxregister_increase(self.correction_reg_num)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            k_index, k_tile_count, local_split_kv = self.get_k_tile_count(
                split_kv, cache_seqs, blk_coord
            )
            if k_tile_count > 0:
                common = SimpleNamespace(
                    blk_coord=blk_coord,
                    split_kv=split_kv,
                    local_split_kv=local_split_kv,
                    mAccO=mAccO,
                    mO=mO,
                    K=cache_seqs[blk_coord[2]],
                    K_causal=causal_seqs[blk_coord[2]],
                    L=mCL.shape[1],
                    H=cutlass.min(
                        self.mma_qk_tiler[0],
                        self.total_q_rows - blk_coord[1] * self.mma_qk_tiler[0],
                    ),
                    tmem_ptr=tmem_ptr,
                    tidx=tidx,
                    tiled_mma_pv=tiled_mma_pv,
                    mma_o_pipeline=mma_o_pipeline,
                    smem_exchange=softmax_smem_exchange,
                )
                nthr = self.num_compute_warps * self.threads_per_warp
                sm_params = SimpleNamespace(
                    tiled_mma_qk=tiled_mma_qk,
                    sP=sP,
                    mma_s_pipeline=mma_s_pipeline,
                    p_mma_pipeline=p_mma_pipeline,
                    softmax_scale_log2=softmax_scale_log2,
                    handoff_in=cute.make_tensor(handoff_smem.iterator + nthr, cute.make_layout(nthr)),
                    handoff_out=cute.make_tensor(handoff_smem.iterator, cute.make_layout(nthr)),
                )
                tid = tidx % nthr
                # WG1's views (built at trace time, before the warp branch: no rebinding inside staged control flow)
                common_hi = SimpleNamespace(**vars(common))
                common_hi.smem_exchange = epilogue_smem_exchange
                sm_hi = SimpleNamespace(**vars(sm_params))
                sm_hi.handoff_in = sm_params.handoff_out
                sm_hi.handoff_out = sm_params.handoff_in
                if warp_idx <= self.compute_warp_ids[-1]:
                    row_max, row_sum = self.compute_wg(common, sm_params, 0, k_index, k_tile_count)
                    # the final max goes to WG0's hi -> lo hand-off slot: WG0 read it last (same thread), and WG1
                    # writes it again only on a tile that WG0 follows, which does not exist after WG0's last tile.
                    handoff_smem[nthr + tid] = row_max
                    final_sum_smem[tid] = row_sum
                    self.final_bar.arrive()
                else:
                    row_max, row_sum = self.compute_wg(common_hi, sm_hi, 1, k_index, k_tile_count)
                    # merge the two WGs' row sums at the final max (the chain max is monotone, so it is the larger)
                    self.final_bar.arrive_and_wait()
                    max_lo = handoff_smem[nthr + tid]
                    sum_lo = final_sum_smem[tid]
                    max_fin = cute.arch.fmax(max_lo, row_max)
                    sum_fin = sum_lo * cute.math.exp2(
                        (max_lo - max_fin) * softmax_scale_log2, fastmath=True
                    )
                    if k_tile_count > 1:
                        sum_fin = sum_fin + row_sum * cute.math.exp2(
                            (row_max - max_fin) * softmax_scale_log2, fastmath=True
                        )
                    self._ik("epi")
                    self.epilogue(
                        common_hi,
                        SimpleNamespace(
                            output_scale=output_scale,
                            softmax_scale_log2=softmax_scale_log2,
                            mAccLSE=mAccLSE,
                            mLSE=mLSE,
                        ),
                        k_tile_count - 1,
                        sum_fin,
                        max_fin,
                    )
                    self._ik()
            if warp_idx > self.compute_warp_ids[-1]:
                self.tmem_free_bar.arrive()

        return

    @cute.kernel
    def reduction_kernel(
        self,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mAccO: cute.Tensor,
        mAccLSE: cute.Tensor,
        split_kv: cutlass.Int32,
        cache_seqs: cute.Tensor,
        block_split_kvs: Optional[cute.Tensor],  # PERFBOT item-03
        output_scale: cutlass.Float32,
    ):
        """The reduction kernel for Multi-Head Latent Attention (MLA) that combines intermediate results
        from multiple split_kv blocks into final outputs.

        :param mO: Output tensor for storing final results
        :type mO: cute.Tensor
        :param mLSE: Log-sum-exp tensor for storing final LSE values
        :type mLSE: cute.Tensor
        :param mAccO: Accumulated output tensor from split_kv blocks
        :type mAccO: cute.Tensor
        :param mAccLSE: Accumulated LSE tensor from split_kv blocks
        :type mAccLSE: cute.Tensor
        :param split_kv: Number of split_kv blocks
        :type split_kv: cutlass.Int32
        :param cache_seqs: Cache sequence lengths tensor
        :type cache_seqs: cute.Tensor
        :param output_scale: Scale applied after combining normalized partial outputs
        :type output_scale: cutlass.Float32
        """
        bidx, bidy, bidz = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        # Output coordinates retain the original query/head layout for packed Q.
        # Each CTA writes a disjoint D band; only band zero writes the LSE.
        d_tile_idx = bidx % self.reducer_d_tiles
        blk_coord = (bidx // self.reducer_d_tiles, bidy, bidz)
        flat_row = bidy * self.num_heads + blk_coord[0]
        acc_row = flat_row % self.mma_qk_tiler[0]
        acc_tile = flat_row // self.mma_qk_tiler[0]
        # the splits that hold partials (get_k_tile_count gives the rest no tiles)
        k_tile_total = cute.ceil_div(cache_seqs[blk_coord[2]], self.mma_qk_tiler[1])
        if cutlass.const_expr(self.is_var_split_kv):  # PERFBOT item-03
            split_kv = block_split_kvs[blk_coord[2]]  # PERFBOT item-03
        k_tile_per_cta = cute.ceil_div(k_tile_total, split_kv)
        local_split_kv = cute.ceil_div(k_tile_total, max(1, k_tile_per_cta))

        # Alloc shared memory
        smem = cutlass.memory.SmemAllocator()
        # LSE phase: one weight per lane of warp 0
        reducer_scale_count = (
            ceil_div(self.reducer_max_splits, self.threads_per_warp)
            * self.threads_per_warp
        )
        storage = smem.allocate(reducer_scale_count * self.acc_dtype.width // 8, 16)
        lse_scale_ptr = cute.recast_ptr(storage, dtype=self.acc_dtype)
        smem_lse_scale = cute.make_tensor(
            lse_scale_ptr, cute.make_layout(reducer_scale_count)
        )

        # let the next split kernel get scheduled early; its own wait still orders the data
        cute.arch.griddepcontrol_launch_dependents()
        cute.arch.griddepcontrol_wait()

        # issue the first set of partial loads before the LSE phase; later sets
        # only if the row needs them
        reducer_threads = self.threads_per_warp * self.num_compute_warps
        vec = self.reducer_d_tile // reducer_threads
        gAccO = mAccO[acc_row, None, None, acc_tile, blk_coord[2]]
        band_ptr = cute.make_ptr(
            self.acc_dtype,
            (gAccO.iterator + d_tile_idx * self.reducer_d_tile).toint(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        partial_copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.acc_dtype,
            num_bits_per_copy=vec * self.acc_dtype.width,
        )
        partial_thr_copy = cute.make_tiled_copy_tv(
            partial_copy_atom,
            cute.make_ordered_layout((1, reducer_threads), order=(1, 0)),
            cute.make_layout((1, vec)),
        ).get_slice(tidx)
        partial_params = SimpleNamespace(
            band_ptr=band_ptr,
            split_stride=gAccO.stride[0],
            copy_atom=partial_copy_atom,
            thr_copy=partial_thr_copy,
            local_split_kv=local_split_kv,
            smem_lse_scale=smem_lse_scale,
            vec=vec,
        )
        num_sets = ceil_div(self.reducer_max_splits, self.reducer_batch_splits)
        num_pre_sets = 1
        pre_frags = []
        for s in cutlass.range_constexpr(num_pre_sets):
            pre_frags.append(
                self._reducer_issue_set(partial_params, s * self.reducer_batch_splits)
            )

        gLSE = mAccLSE[acc_row, None, acc_tile, blk_coord[2]]
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            # calculate the global lse and exp ^ (local_lse - global_lse)
            lse_per_thread = cute.ceil_div(
                self.reducer_max_splits, self.threads_per_warp
            )

            local_lse = cute.make_rmem_tensor(
                cute.make_layout(lse_per_thread), self.lse_dtype
            )
            lse_max = -self.lse_dtype.inf
            # find the max lse
            for i in cutlass.range_constexpr(lse_per_thread):
                split_kv_idx = tidx + i * self.threads_per_warp
                local_lse[i] = (
                    gLSE[split_kv_idx]
                    if cute.elem_less(split_kv_idx, local_split_kv)
                    else -self.lse_dtype.inf
                )
                # reduce the local lse
                lse_max = cute.arch.fmax(lse_max, local_lse[i])
            lse_max = cute.arch.warp_reduction_max(lse_max)
            lse_max = lse_max if lse_max != -self.lse_dtype.inf else 0.0
            # calculate sum_lse
            sum_lse = 0.0
            for i in cutlass.range_constexpr(lse_per_thread):
                sum_lse += cute.math.exp2(local_lse[i] - lse_max, fastmath=True)
            sum_lse = cute.arch.warp_reduction_sum(sum_lse)
            # calculate the global_lse
            global_lse = (
                lse_max + cute.math.log2(sum_lse, fastmath=True)
                if not sum_lse == self.lse_dtype(0.0)
                or sum_lse != sum_lse  # noqa: SIM201
                else -self.lse_dtype.inf
            )
            if tidx == 0 and d_tile_idx == 0:
                if cutlass.const_expr(not self.skip_lse):
                    mLSE[blk_coord[0], blk_coord[1], blk_coord[2]] = global_lse
            # store the scale to shared memory, including output_scale (0 past local_split_kv)
            for i in cutlass.range_constexpr(lse_per_thread):
                split_kv_idx = tidx + i * self.threads_per_warp
                smem_lse_scale[split_kv_idx] = (
                    cute.math.exp2(local_lse[i] - global_lse, fastmath=True)
                    * output_scale
                )

        pipeline.sync(barrier_id=4)

        rAccO = cute.make_rmem_tensor(cute.make_layout(vec), self.acc_dtype)
        rAccO.fill(0.0)
        for s in cutlass.range_constexpr(num_pre_sets):
            self._reducer_accumulate_set(
                partial_params, pre_frags[s], s * self.reducer_batch_splits, rAccO
            )
        for s in cutlass.range_constexpr(num_pre_sets, num_sets):
            first = s * self.reducer_batch_splits
            if local_split_kv > first:
                frag = self._reducer_issue_set(partial_params, first)
                self._reducer_accumulate_set(partial_params, frag, first, rAccO)
        rO = cute.make_rmem_tensor(cute.make_layout(vec), self.o_dtype)
        rO.store(rAccO.load().to(self.o_dtype))
        gO = mO[blk_coord[0], None, blk_coord[1], blk_coord[2]]
        out_ptr = cute.make_ptr(
            self.o_dtype,
            (gO.iterator + d_tile_idx * self.reducer_d_tile + tidx * vec).toint(),
            cute.AddressSpace.gmem,
            assumed_align=vec * self.o_dtype.width // 8,
        )
        cute.autovec_copy(rO, cute.make_tensor(out_ptr, cute.make_layout(vec)))

    @cute.jit
    def _reducer_issue_set(self, params: SimpleNamespace, first: int) -> cute.Tensor:
        """Issue predicated partial loads for splits [first, first + batch)."""
        batch = self.reducer_batch_splits
        gSet = cute.make_tensor(
            params.band_ptr + first * params.split_stride,
            cute.make_layout(
                (batch, self.reducer_d_tile), stride=(params.split_stride, 1)
            ),
        )
        tPgP = params.thr_copy.partition_S(gSet)
        tPrP = cute.make_fragment_like(tPgP)
        tPpP = cute.make_rmem_tensor(cute.make_layout((1, batch, 1)), cutlass.Boolean)
        for k in cutlass.range(batch, unroll_full=True):
            tPpP[0, k, 0] = cute.elem_less(first + k, params.local_split_kv)
        tPrP.fill(0.0)
        cute.copy(params.copy_atom, tPgP, tPrP, pred=tPpP)
        return tPrP

    @cute.jit
    def _reducer_accumulate_set(
        self,
        params: SimpleNamespace,
        tPrP: cute.Tensor,
        first: int,
        rAccO: cute.Tensor,
    ):
        """Accumulate splits [first, first + batch) into rAccO, skipping empty groups."""
        batch = self.reducer_batch_splits
        group = self.reducer_accumulate_group
        for g in cutlass.range_constexpr(batch // group):
            if params.local_split_kv > first + g * group:
                for k in cutlass.range_constexpr(group):
                    split = g * group + k
                    lse_scale = params.smem_lse_scale[first + split]
                    for v in cutlass.range_constexpr(params.vec):
                        rAccO[v] += tPrP[v, split, 0].to(self.acc_dtype) * lse_scale
        return

    @cute.jit
    def get_k_tile_count(
        self,
        split_kv: cutlass.Int32,
        cache_seqs: cute.Tensor,
        blk_coord: cute.Coord,
    ) -> tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32]:
        """(k_index, k_tile_count, split_kv) of this split: ceil(tiles / split_kv) tiles per split, so the last
        splits of a short request can get none (k_tile_count 0)."""
        K = cache_seqs[blk_coord[2]]
        k_tile_total = cute.ceil_div(K, self.mma_qk_tiler[1])
        k_tile_per_cta = cute.ceil_div(k_tile_total, split_kv)
        k_index = blk_coord[3] * k_tile_per_cta
        k_tile_count = max(0, min(k_tile_total, k_index + k_tile_per_cta) - k_index)
        return k_index, k_tile_count, split_kv

    @cute.jit
    def load_tma_q(
        self,
        common_params: SimpleNamespace,
        q_params: SimpleNamespace,
        load_q_producer_state: pipeline.PipelineState,
    ) -> pipeline.PipelineState:
        """Load Q tensors (latent + rope) once per work tile.

        :param common_params: The common parameters (blk_coord, load_q_pipeline)
        :type common_params: SimpleNamespace
        :param q_params: The q parameters (tiled_mma_qk, tma atoms, mQL, mQR, sQ, sQ_rope)
        :type q_params: SimpleNamespace
        :param load_q_producer_state: The load q producer state
        :type load_q_producer_state: pipeline.PipelineState

        :return: The updated load q producer state
        :rtype: pipeline.PipelineState
        """
        # Flatten divide and partition global tensors for Q TMA load
        mma_qk_tiler_mk = cute.select(self.mma_qk_tiler, mode=[0, 2])
        gQL = cute.flat_divide(q_params.mQL, mma_qk_tiler_mk)
        mma_qk_tiler_mk_rope = cute.select(self.mma_qk_rope_tiler, mode=[0, 2])
        gQR = cute.flat_divide(q_params.mQR, mma_qk_tiler_mk_rope)

        thr_mma_qk = q_params.tiled_mma_qk.get_slice(
            common_params.blk_coord[0] % cute.size(q_params.tiled_mma_qk.thr_id)
        )
        tSgQL = thr_mma_qk.partition_A(gQL)
        tSgQR = thr_mma_qk.partition_A(gQR)

        tQsQ, tQLgQL_mkl = cpasync.tma_partition(
            q_params.tma_atom_q_latent,
            0,
            cute.make_layout(1),
            cute.group_modes(q_params.sQ, 0, 3),
            cute.group_modes(tSgQL, 0, 3),
        )

        tQsQ_rope, tQRgQR_mkl = cpasync.tma_partition(
            q_params.tma_atom_q_rope,
            0,
            cute.make_layout(1),
            cute.group_modes(q_params.sQ_rope, 0, 3),
            cute.group_modes(tSgQR, 0, 3),
        )

        # packed Q: one S group; the work tile's q tile is the M tile
        q_group = 0
        q_m_tile = common_params.blk_coord[1]
        tQLgQL = tQLgQL_mkl[None, None, None, q_group, common_params.blk_coord[2]]
        tQRgQR = tQRgQR_mkl[None, None, None, q_group, common_params.blk_coord[2]]

        # Issue Q TMA copies
        load_q_pipeline = common_params.load_q_pipeline
        tma_bar_ptr = load_q_pipeline.producer_get_barrier(load_q_producer_state)
        load_q_pipeline.producer_acquire(load_q_producer_state)
        for i in cutlass.range_constexpr(self.iterations_qk_latent):
            cute.copy(
                q_params.tma_atom_q_latent,
                tQLgQL[None, q_m_tile, i],
                tQsQ[None, (i, 0)],
                tma_bar_ptr=tma_bar_ptr,
            )
        for i in cutlass.range_constexpr(self.iterations_qk_rope):
            cute.copy(
                q_params.tma_atom_q_rope,
                tQRgQR[None, q_m_tile, i],
                tQsQ_rope[None, i],
                tma_bar_ptr=tma_bar_ptr,
            )
        load_q_producer_state.advance()
        return load_q_producer_state

    @cute.jit
    def load_tma_k(
        self,
        common_params: SimpleNamespace,
        k_params: SimpleNamespace,
        k_index: cutlass.Int32,
        k_tile_count: cutlass.Int32,
        load_k_producer_state: pipeline.PipelineState,
    ) -> pipeline.PipelineState:
        """Load K tensors (latent + rope) for all k tiles.

        :param common_params: The common parameters (blk_coord, load_k_pipeline, mPT)
        :type common_params: SimpleNamespace
        :param k_params: The k parameters (tiled_mma_qk, tma atoms, mCL, mKR, sKC, sKC_rope)
        :type k_params: SimpleNamespace
        :param k_index: The k index
        :type k_index: cutlass.Int32
        :param k_tile_count: The k tile count
        :type k_tile_count: cutlass.Int32
        :param load_k_producer_state: The load k producer state
        :type load_k_producer_state: pipeline.PipelineState

        :return: The updated load k producer state
        :rtype: pipeline.PipelineState
        """
        # page table
        mPT = common_params.mPT[None, common_params.blk_coord[2]]

        cta_m = min(
            k_params.tiled_mma_qk.op.shape_mnk[0] // k_params.tiled_mma_qk.thr_id.shape,
            self.page_size,
        )
        page_tile_size = min(self.page_size, cta_m)
        gCL = cute.tiled_divide(k_params.mCL, (page_tile_size, self.mma_qk_tiler[2]))
        tSgCL = (
            gCL[
                None,
                common_params.blk_coord[0] % k_params.tiled_mma_qk.thr_id.shape,
                None,
                None,
            ]
            if cta_m < self.page_size
            else gCL[None, 0, None, None]
        )
        gKR = cute.tiled_divide(
            k_params.mKR, (page_tile_size, self.mma_qk_rope_tiler[2])
        )
        tSgKR = (
            gKR[
                None,
                common_params.blk_coord[0] % k_params.tiled_mma_qk.thr_id.shape,
                None,
                None,
            ]
            if cta_m < self.page_size
            else gKR[None, 0, None, None]
        )

        tKCsKC, tCLgCL = cpasync.tma_partition(
            k_params.tma_atom_c_latent,
            0,
            cute.make_layout(1),
            k_params.sKC,
            tSgCL,
        )

        tKCsKC_rope, tKRgKR = cpasync.tma_partition(
            k_params.tma_atom_c_rope,
            0,
            cute.make_layout(1),
            k_params.sKC_rope,
            tSgKR,
        )
        # set extra params
        common_params.mPT = mPT
        k_params.tSgCL = tSgCL
        k_params.tSgKR = tSgKR
        k_params.tCLgCL = tCLgCL
        k_params.tKRgKR = tKRgKR
        k_params.tKCsKC = tKCsKC
        k_params.tKCsKC_rope = tKCsKC_rope

        while k_tile_count > 0:
            self._ik("ld_k")
            load_k_producer_state = self.load_tma_k_one_k_tile(
                common_params,
                k_params,
                k_index,
                load_k_producer_state,
            )
            self._ik()
            k_index += 1
            k_tile_count -= 1

        return load_k_producer_state

    @cute.jit
    def load_tma_v(
        self,
        common_params: SimpleNamespace,
        v_params: SimpleNamespace,
        k_index: cutlass.Int32,
        k_tile_count: cutlass.Int32,
        load_v_producer_state: pipeline.PipelineState,
    ) -> pipeline.PipelineState:
        """Load wrap to load V tensors. Updates the load v producer state.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param v_params: The v parameters
        :type v_params: SimpleNamespace
        :param k_index: The k index
        :type k_index: cutlass.Int32
        :param k_tile_count: The k tile count
        :type k_tile_count: cutlass.Int32
        :param load_v_producer_state: The load v producer state
        :type load_v_producer_state: pipeline.PipelineState

        :return: The load v producer state
        :rtype: pipeline.PipelineState
        """
        # page table
        mPT = common_params.mPT[None, common_params.blk_coord[2]]

        # Flatten divide and partition global tensors for V TMA load
        page_tile_size = min(self.page_size, self.mma_pv_tiler[2])
        gCLT = cute.flat_divide(v_params.mCLT, (self.mma_pv_tiler[1], page_tile_size))
        cta_n = self.mma_pv_tiler[1] // v_params.tiled_mma_pv.thr_id.shape
        gCLT = cute.logical_divide(gCLT, (cta_n,))[
            (None, common_params.blk_coord[0]), None, None, None, None
        ]
        tOgCLT = cute.tiled_divide(gCLT, (cta_n, page_tile_size))
        tOgCLT = tOgCLT[None, 0, 0, None, None, None]
        # tma partition for vc
        # smem: ((atom_v, rest_v), STAGE)
        # gmem: ((atom_v, rest_v), RestM, RestK, RestL)
        tVCsVC, tCLTgCLT = cpasync.tma_partition(
            v_params.tma_atom_c_latent_transpose,
            0,
            cute.make_layout(1),
            v_params.sVC,
            tOgCLT,
        )

        # set extra params
        common_params.mPT = mPT
        v_params.tCLTgCLT = tCLTgCLT
        v_params.tVCsVC = tVCsVC

        while k_tile_count > 0:
            self._ik("ld_v")
            load_v_producer_state = self.load_tma_v_one_k_tile(
                common_params,
                v_params,
                k_index,
                load_v_producer_state,
            )
            self._ik()
            k_index += 1
            k_tile_count -= 1
        return load_v_producer_state

    @cute.jit
    def load_tma_k_one_k_tile(
        self,
        common_params: SimpleNamespace,
        k_params: SimpleNamespace,
        k_index: cutlass.Int32,
        load_k_producer_state: pipeline.PipelineState,
    ) -> pipeline.PipelineState:
        """Load one k-tile of K latent/rope tensors.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param k_params: The k parameters
        :type k_params: SimpleNamespace
        :param k_index: The k index
        :type k_index: cutlass.Int32
        :param load_k_producer_state: The load k producer state
        :type load_k_producer_state: pipeline.PipelineState

        :return: The updated load k producer state
        :rtype: pipeline.PipelineState
        """
        page_per_tile = ceil_div(
            self.mma_qk_tiler[1] // self.page_size, k_params.tiled_mma_qk.thr_id.shape
        )
        k_idx = cute.make_rmem_tensor(cute.make_layout(page_per_tile), cutlass.Int32)
        for i in cutlass.range_constexpr(page_per_tile):
            k_idx[i] = (
                common_params.mPT[k_index]
                if self.mma_qk_tiler[1] // self.page_size == 1
                else common_params.mPT[
                    (
                        k_index * k_params.tiled_mma_qk.thr_id.shape
                        + common_params.blk_coord[0]
                    )
                    * page_per_tile
                    + i
                ]
            )
        tma_bar_ptr = common_params.load_k_pipeline.producer_get_barrier(
            load_k_producer_state
        )
        common_params.load_k_pipeline.producer_acquire(load_k_producer_state)
        for i in range(self.iterations_qk_latent):
            for k in range(page_per_tile):
                cute.copy(
                    k_params.tma_atom_c_latent,
                    k_params.tCLgCL[None, i, k_idx[k]],
                    k_params.tKCsKC[None, k, 0, (i, load_k_producer_state.index)],
                    tma_bar_ptr=tma_bar_ptr,
                )

        for i in cutlass.range_constexpr(self.iterations_qk_rope):
            for k in cutlass.range_constexpr(page_per_tile):
                cute.copy(
                    k_params.tma_atom_c_rope,
                    k_params.tKRgKR[None, i, k_idx[k]],
                    k_params.tKCsKC_rope[None, k, 0, load_k_producer_state.index],
                    tma_bar_ptr=tma_bar_ptr,
                )
        load_k_producer_state.advance()

        return load_k_producer_state

    @cute.jit
    def load_tma_v_one_k_tile(
        self,
        common_params: SimpleNamespace,
        v_params: SimpleNamespace,
        k_index: cutlass.Int32,
        load_v_producer_state: pipeline.PipelineState,
    ) -> pipeline.PipelineState:
        """Load one k-tile of compressed latent transpose tensor(v). Updates the load qkv producer state.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param v_params: The load tma v parameters
        :type v_params: SimpleNamespace
        :param k_index: The k index
        :type k_index: cutlass.Int32
        :param load_v_producer_state: The load v producer state
        :type load_v_producer_state: pipeline.PipelineState

        :return: The load qkv producer state
        :rtype: pipeline.PipelineState
        """
        page_per_tile = self.mma_pv_tiler[2] * self.iterations_pv_k // self.page_size
        page_per_subtile = ceil_div(page_per_tile, self.iterations_pv_k)
        k_idx = cute.make_rmem_tensor(cute.make_layout(page_per_tile), cutlass.Int32)
        for i in cutlass.range_constexpr(page_per_tile):
            k_idx[i] = (
                common_params.mPT[k_index]
                if page_per_tile == 1
                else common_params.mPT[k_index * page_per_tile + i]
            )
        tma_bar_ptr = common_params.load_v_pipeline.producer_get_barrier(
            load_v_producer_state
        )
        common_params.load_v_pipeline.producer_acquire(load_v_producer_state)
        for j in cutlass.range_constexpr(self.iterations_pv_n):
            for i in cutlass.range_constexpr(self.iterations_pv_k):
                if cutlass.const_expr(page_per_tile > 1):
                    for k in cutlass.range_constexpr(page_per_subtile):
                        k_idx_i = k_idx[k + i * page_per_subtile]
                        cute.copy(
                            v_params.tma_atom_c_latent_transpose,
                            v_params.tCLTgCLT[None, j, 0, k_idx_i],
                            v_params.tVCsVC[
                                None,
                                0,
                                k,
                                ((j, i), load_v_producer_state.index),
                            ],
                            tma_bar_ptr=tma_bar_ptr,
                        )
                else:
                    cute.copy(
                        v_params.tma_atom_c_latent_transpose,
                        v_params.tCLTgCLT[None, j, i, k_idx[0]],
                        v_params.tVCsVC[
                            None, 0, 0, ((j, i), load_v_producer_state.index)
                        ],
                        tma_bar_ptr=tma_bar_ptr,
                    )
        load_v_producer_state.advance()
        return load_v_producer_state

    @cute.jit
    def mma(
        self,
        common_params: SimpleNamespace,
        qk_params: SimpleNamespace,
        pv_params: SimpleNamespace,
        k_tile_count: cutlass.Int32,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        load_q_consumer_state: pipeline.PipelineState,
        load_k_consumer_state: pipeline.PipelineState,
        load_v_consumer_state: pipeline.PipelineState,
        mma_s_producer_state: pipeline.PipelineState,
        p_mma_consumer_state: pipeline.PipelineState,
        mma_o_producer_state: pipeline.PipelineState,
        role: cutlass.Constexpr = 0,  # 0 = QK + PV (E2b), 1 = QK only, 2 = PV only (split_mma)  # PERFBOT item-03 MLA-LOOP
    ) -> tuple[
        cute.TiledMma,
        cute.TiledMma,
        pipeline.PipelineState,
        pipeline.PipelineState,
        pipeline.PipelineState,
        pipeline.PipelineState,
        pipeline.PipelineState,
        pipeline.PipelineState,
    ]:
        """MMA warp to compute the result of Q*K^T and P*V. Updates the tiled mma and pipeline states.

        :param common_params: The common parameters for mma qk and pv
        :type common_params: SimpleNamespace
        :param qk_params: The mma qk parameters
        :type qk_params: SimpleNamespace
        :param pv_params: The mma pv parameters
        :type pv_params: SimpleNamespace
        :param k_tile_count: The k tile count
        :type k_tile_count: cutlass.Int32
        :param tiled_mma_qk: The tiled mma qk
        :type tiled_mma_qk: cute.TiledMma
        :param tiled_mma_pv: The tiled mma pv
        :type tiled_mma_pv: cute.TiledMma
        :param load_q_consumer_state: The load q consumer state
        :type load_q_consumer_state: pipeline.PipelineState
        :param load_k_consumer_state: The load k consumer state
        :type load_k_consumer_state: pipeline.PipelineState
        :param load_v_consumer_state: The load v consumer state
        :type load_v_consumer_state: pipeline.PipelineState
        :param mma_s_producer_state: The mma s producer state
        :type mma_s_producer_state: pipeline.PipelineState
        :param p_mma_consumer_state: The p mma consumer state
        :type p_mma_consumer_state: pipeline.PipelineState
        :param mma_o_producer_state: The mma o producer state
        :type mma_o_producer_state: pipeline.PipelineState

        :return: The tiled mma qk, the tiled mma pv, the load q consumer state, the load k consumer state, the load v consumer state, the mma s producer state, the p mma consumer state, and the mma o producer state
        :rtype: tuple[cute.TiledMma, cute.TiledMma, pipeline.PipelineState, pipeline.PipelineState, pipeline.PipelineState, pipeline.PipelineState, pipeline.PipelineState, pipeline.PipelineState]
        """

        tSrQ = tiled_mma_qk.make_fragment_A(qk_params.sQ)
        tSrQ_rope = tiled_mma_qk.make_fragment_A(qk_params.sQ_rope)
        tSrKC = tiled_mma_qk.make_fragment_B(qk_params.sKC)
        tSrKC_rope = tiled_mma_qk.make_fragment_B(qk_params.sKC_rope)
        tOrP = tiled_mma_pv.make_fragment_A(pv_params.sP)
        tOrVC = tiled_mma_pv.make_fragment_B(pv_params.sVC)

        tStS_shape = tiled_mma_qk.partition_shape_C(
            cute.select(self.mma_qk_tiler, mode=[0, 1])
        )
        tStS_staged_fake = tiled_mma_qk.make_fragment_C(
            cute.append(tStS_shape, self.mma_s_stage)
        )
        # use real tmem ptr for tStS
        tStS_staged = cute.make_tensor(common_params.tmem_ptr, tStS_staged_fake.layout)
        tOtO_shape = tiled_mma_pv.partition_shape_C(
            cute.select(self.mma_pv_tiler, mode=[0, 1])
        )
        # mma O has 1 stage.
        tOtO = tiled_mma_pv.make_fragment_C(tOtO_shape)
        tOtO_layout = cute.append(
            tOtO.layout,
            cute.make_layout(
                common_params.L // self.mma_pv_tiler[1],
                stride=self.pv_acc_cols,
            ),
        )
        tOtO_staged = cute.make_tensor(
            tStS_staged.iterator + self.tmem_o_offset, tOtO_layout
        )

        # set more parameters
        qk_params.tSrQ = tSrQ
        qk_params.tSrQ_rope = tSrQ_rope
        qk_params.tSrKC = tSrKC
        qk_params.tSrKC_rope = tSrKC_rope
        qk_params.tStS_staged = tStS_staged
        pv_params.tOrP = tOrP
        pv_params.tOrVC = tOrVC
        pv_params.tOtO_staged = tOtO_staged

        # mma O accumulates on K, so the accumulate flag is set to False once before all K blocks.
        tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, False)
        load_q_pipeline = common_params.load_q_pipeline
        pv_k_index = common_params.k_index
        if cutlass.const_expr(role == 1):  # PERFBOT item-03 MLA-LOOP
            if common_params.is_leader_cta:  # PERFBOT item-03 MLA-LOOP
                load_q_release_state = load_q_consumer_state.clone()  # PERFBOT item-03 MLA-LOOP
                (tiled_mma_qk, load_q_consumer_state, load_k_consumer_state, mma_s_producer_state) = self.mma_qk(  # PERFBOT item-03 MLA-LOOP
                    common_params, qk_params, tiled_mma_qk, load_q_consumer_state, load_k_consumer_state,  # PERFBOT item-03 MLA-LOOP
                    mma_s_producer_state, wait_q=True)  # PERFBOT item-03 MLA-LOOP
                qk_tile_count = k_tile_count - 1  # PERFBOT item-03 MLA-LOOP
                while qk_tile_count > 0:  # PERFBOT item-03 MLA-LOOP
                    (tiled_mma_qk, load_q_consumer_state, load_k_consumer_state, mma_s_producer_state) = self.mma_qk(  # PERFBOT item-03 MLA-LOOP
                        common_params, qk_params, tiled_mma_qk, load_q_consumer_state, load_k_consumer_state,  # PERFBOT item-03 MLA-LOOP
                        mma_s_producer_state, wait_q=False)  # PERFBOT item-03 MLA-LOOP
                    qk_tile_count -= 1  # PERFBOT item-03 MLA-LOOP
                load_q_pipeline.consumer_release(load_q_release_state)  # PERFBOT item-03 MLA-LOOP
                load_q_release_state.advance()  # PERFBOT item-03 MLA-LOOP
        if cutlass.const_expr(role == 2):  # PERFBOT item-03 MLA-LOOP
            if common_params.is_leader_cta:  # PERFBOT item-03 MLA-LOOP
                pv_tile_count = k_tile_count  # PERFBOT item-03 MLA-LOOP
                while pv_tile_count > 0:  # PERFBOT item-03 MLA-LOOP
                    (tiled_mma_pv, load_v_consumer_state, p_mma_consumer_state, mma_o_producer_state) = self.mma_pv(  # PERFBOT item-03 MLA-LOOP
                        common_params, pv_params, pv_k_index, tiled_mma_pv, load_v_consumer_state,  # PERFBOT item-03 MLA-LOOP
                        p_mma_consumer_state, mma_o_producer_state)  # PERFBOT item-03 MLA-LOOP
                    pv_k_index = pv_k_index + 1  # PERFBOT item-03 MLA-LOOP
                    pv_tile_count -= 1  # PERFBOT item-03 MLA-LOOP
        if cutlass.const_expr(role == 0):  # PERFBOT item-03 MLA-LOOP
          if common_params.is_leader_cta:  # PERFBOT item-03 MLA-LOOP (indent under role 0)
            load_q_release_state = load_q_consumer_state.clone()
            (
                tiled_mma_qk,
                load_q_consumer_state,
                load_k_consumer_state,
                mma_s_producer_state,
            ) = self.mma_qk(
                common_params,
                qk_params,
                tiled_mma_qk,
                load_q_consumer_state,
                load_k_consumer_state,
                mma_s_producer_state,
                wait_q=True,
            )
            qk_tile_count = k_tile_count - 1
            pv_tile_count = k_tile_count
            # QK leads PV by 2 tiles (2-CTA schedule)
            if qk_tile_count > 0:
                (
                    tiled_mma_qk,
                    load_q_consumer_state,
                    load_k_consumer_state,
                    mma_s_producer_state,
                ) = self.mma_qk(
                    common_params,
                    qk_params,
                    tiled_mma_qk,
                    load_q_consumer_state,
                    load_k_consumer_state,
                    mma_s_producer_state,
                    wait_q=False,
                )
                qk_tile_count -= 1

            while qk_tile_count > 0:
                (
                    tiled_mma_qk,
                    load_q_consumer_state,
                    load_k_consumer_state,
                    mma_s_producer_state,
                ) = self.mma_qk(
                    common_params,
                    qk_params,
                    tiled_mma_qk,
                    load_q_consumer_state,
                    load_k_consumer_state,
                    mma_s_producer_state,
                    wait_q=False,
                )
                qk_tile_count -= 1
                (
                    tiled_mma_pv,
                    load_v_consumer_state,
                    p_mma_consumer_state,
                    mma_o_producer_state,
                ) = self.mma_pv(
                    common_params,
                    pv_params,
                    pv_k_index,
                    tiled_mma_pv,
                    load_v_consumer_state,
                    p_mma_consumer_state,
                    mma_o_producer_state,
                )
                pv_k_index = pv_k_index + 1
                pv_tile_count -= 1
            # release q consumer states, then drain the remaining PV tiles
            load_q_pipeline.consumer_release(load_q_release_state)
            load_q_release_state.advance()
            while pv_tile_count > 0:
                (
                    tiled_mma_pv,
                    load_v_consumer_state,
                    p_mma_consumer_state,
                    mma_o_producer_state,
                ) = self.mma_pv(
                    common_params,
                    pv_params,
                    pv_k_index,
                    tiled_mma_pv,
                    load_v_consumer_state,
                    p_mma_consumer_state,
                    mma_o_producer_state,
                )
                pv_k_index = pv_k_index + 1
                pv_tile_count -= 1

        return (  # type: ignore[return-value]
            tiled_mma_qk,
            tiled_mma_pv,
            load_q_consumer_state,
            load_k_consumer_state,
            load_v_consumer_state,
            mma_s_producer_state,
            p_mma_consumer_state,
            mma_o_producer_state,
        )

    @cute.jit
    def mma_qk(
        self,
        common_params: SimpleNamespace,
        qk_params: SimpleNamespace,
        tiled_mma_qk: cute.TiledMma,
        load_q_consumer_state: pipeline.PipelineState,
        load_k_consumer_state: pipeline.PipelineState,
        mma_s_producer_state: pipeline.PipelineState,
        wait_q: bool,
    ) -> tuple[
        cute.TiledMma,
        pipeline.PipelineState,
        pipeline.PipelineState,
        pipeline.PipelineState,
    ]:
        """Compute one k-tile of mma for Q*K^T. Updates the tiled MMA QK and pipeline states.

        :param qk_params: The qk parameters
        :type qk_params: SimpleNamespace
        :param tiled_mma_qk: The tiled mma qk
        :type tiled_mma_qk: cute.TiledMma
        :param load_q_consumer_state: The load q consumer state
        :type load_q_consumer_state: pipeline.PipelineState
        :param load_k_consumer_state: The load k consumer state
        :type load_k_consumer_state: pipeline.PipelineState
        :param mma_s_producer_state: The mma s producer state
        :type mma_s_producer_state: pipeline.PipelineState

        :return: The tiled mma qk, the load q consumer state, the load k consumer state, and the mma s producer state
        :rtype: tuple[cute.TiledMma, pipeline.PipelineState, pipeline.PipelineState, pipeline.PipelineState]
        """
        tStS = qk_params.tStS_staged[None, None, None, mma_s_producer_state.index]

        self._ik("qk_wS")
        qk_params.mma_s_pipeline.producer_acquire(mma_s_producer_state)
        self._ik()
        tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, False)
        load_q_pipeline = common_params.load_q_pipeline
        load_k_pipeline = common_params.load_k_pipeline
        if cutlass.const_expr(wait_q):
            load_q_pipeline.consumer_wait(load_q_consumer_state)
        self._ik("qk_wK")
        load_k_pipeline.consumer_wait(load_k_consumer_state)
        self._ik()
        self._ik("qk_issue")
        for q_stage in range(self.iterations_qk_latent):
            kc_stage = load_k_consumer_state.index
            for k_block in cutlass.range_constexpr(
                cute.size(qk_params.tSrQ.shape[2])
            ):
                cute.gemm(
                    tiled_mma_qk,
                    tStS,
                    qk_params.tSrQ[None, None, k_block, (q_stage, 0)],
                    qk_params.tSrKC[None, None, k_block, (q_stage, kc_stage)],
                    tStS,
                )
                tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, True)

        for q_stage in range(self.iterations_qk_rope):
            kc_stage = load_k_consumer_state.index
            for k_block in cutlass.range_constexpr(
                self.rope_dim // tiled_mma_qk.shape_mnk[2]
            ):
                cute.gemm(
                    tiled_mma_qk,
                    tStS,
                    qk_params.tSrQ_rope[None, None, k_block, q_stage],
                    qk_params.tSrKC_rope[None, None, k_block, kc_stage],
                    tStS,
                )
                tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, True)
        load_k_pipeline.consumer_release(load_k_consumer_state)
        load_k_consumer_state.advance()
        if cutlass.const_expr(wait_q):
            load_q_consumer_state.advance()

        qk_params.mma_s_pipeline.producer_commit(mma_s_producer_state)
        mma_s_producer_state.advance()
        self._ik()
        return (
            tiled_mma_qk,
            load_q_consumer_state,
            load_k_consumer_state,
            mma_s_producer_state,
        )

    @cute.jit
    def mma_pv(
        self,
        common_params: SimpleNamespace,
        pv_params: SimpleNamespace,
        k_index: cutlass.Int32,
        tiled_mma_pv: cute.TiledMma,
        load_v_consumer_state: pipeline.PipelineState,
        p_mma_consumer_state: pipeline.PipelineState,
        mma_o_producer_state: pipeline.PipelineState,
    ) -> tuple[
        cute.TiledMma,
        pipeline.PipelineState,
        pipeline.PipelineState,
        pipeline.PipelineState,
    ]:
        """Compute one k-tile of mma for P*V. Updates the tiled mma pv and pipeline states.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param pv_params: The pv parameters
        :type pv_params: SimpleNamespace
        :param tiled_mma_pv: The tiled mma pv
        :type tiled_mma_pv: cute.TiledMma
        :param load_v_consumer_state: The load v consumer state
        :type load_v_consumer_state: pipeline.PipelineState
        :param p_mma_consumer_state: The P MMA consumer state
        :type p_mma_consumer_state: pipeline.PipelineState
        :param mma_o_producer_state: The MMA o producer state
        :type mma_o_producer_state: pipeline.PipelineState

        :return: The tiled mma pv, the load v consumer state, the P MMA consumer state, and the MMA o producer state
        :rtype: tuple[cute.TiledMma, pipeline.PipelineState, pipeline.PipelineState, pipeline.PipelineState]
        """

        self._ik("pv_wP")
        pv_params.p_mma_pipeline.consumer_wait(p_mma_consumer_state)
        self._ik()
        load_v_pipeline = common_params.load_v_pipeline
        accumulate_flag = tiled_mma_pv.get(tcgen05.Field.ACCUMULATE)
        mma_o_pipeline = pv_params.mma_o_pipeline

        self._ik("pv_wV")
        load_v_pipeline.consumer_wait(load_v_consumer_state)
        self._ik()
        self._ik("pv_issue")
        vc_stage = load_v_consumer_state.index
        for acc_stage in range(self.iterations_pv_n):
            # E2b: no O-stage acquire; between PVs only a rescale touches O, and PV(t) waits for it through P(t)
            tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, accumulate_flag)
            for p_stage in range(self.iterations_pv_k):
                tOtO = pv_params.tOtO_staged[None, None, None, acc_stage]
                for k_block in cutlass.range_constexpr(pv_params.tOrP.shape[2]):
                    cute.gemm(
                        tiled_mma_pv,
                        tOtO,
                        pv_params.tOrP[
                            None,
                            None,
                            k_block,
                            (p_stage, p_mma_consumer_state.index),
                        ],
                        pv_params.tOrVC[
                            None, None, k_block, ((acc_stage, p_stage), vc_stage)
                        ],
                        tOtO,
                    )
                    tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, True)

            mma_o_pipeline.producer_commit(mma_o_producer_state)
            mma_o_producer_state.advance()
        load_v_pipeline.consumer_release(load_v_consumer_state)
        load_v_consumer_state.advance()
        pv_params.p_mma_pipeline.consumer_release(p_mma_consumer_state)
        p_mma_consumer_state.advance()
        self._ik()

        return (
            tiled_mma_pv,
            load_v_consumer_state,
            p_mma_consumer_state,
            mma_o_producer_state,
        )

    @cute.jit
    def compute_wg(
        self,
        common_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        wg: int,
        k_index: cutlass.Int32,
        k_tile_count: cutlass.Int32,
    ) -> tuple[cutlass.Float32, cutlass.Float32]:
        """Softmax WG ``wg`` (0: warps 0-3, 1: warps 4-7) over tiles t_local = wg, wg + 2, ... of this split.
        Returns this WG's last used row max (-inf if it had no tile) and its row sum scaled to that max."""
        mma_s_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.mma_s_stage
        )
        p_mma_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.p_mma_stage
        )
        if cutlass.const_expr(wg == 1):
            mma_s_consumer_state.advance()
            p_mma_producer_state.advance()

        k_tile_total = cute.ceil_div(common_params.K, self.mma_qk_tiler[1])
        # Number of tiles from the global-K end that may contain causal-masked positions (see _causal_mask).
        tile_n = self.mma_qk_tiler[1]
        mask_tile_count = (self.seq_len_q - 1 + tile_n - 1) // tile_n + 1
        first_mask_tile_idx = k_tile_total - mask_tile_count

        k_first = k_index
        k_end = k_index + k_tile_count
        row_max = -self.acc_dtype.inf
        row_sum = self.acc_dtype(0)
        k = k_index + wg
        # unmasked tiles, then tiles that may hold masked positions (masking a clean tile is a no-op)
        while k < k_end and k < first_mask_tile_idx:
            mma_s_consumer_state, p_mma_producer_state, row_max, row_sum = self.softmax_tile(
                common_params, softmax_params, k, mma_s_consumer_state, p_mma_producer_state,
                row_max, row_sum, False, k == k_first, k == k_end - 1, k < k_first + 2, k - k_first, wg,
            )
            k = k + 2
        while k < k_end:
            mma_s_consumer_state, p_mma_producer_state, row_max, row_sum = self.softmax_tile(
                common_params, softmax_params, k, mma_s_consumer_state, p_mma_producer_state,
                row_max, row_sum, True, k == k_first, k == k_end - 1, k < k_first + 2, k - k_first, wg,
            )
            k = k + 2
        return row_max, row_sum

    @cute.jit
    def _o_full_wait(self, o_full: cute.Pointer, t_local: cutlass.Int32, iter_n: int):
        """Wait for O column half iter_n of PV(t_local): mma_o full barrier 2 * (t % 2) + iter_n, parity
        (t // 2) % 2. The parity wait is exact only when PV(t - 2) is complete and PV(t + 2) is not. The rescale waiter
        of tile t + 1 holds S(t + 1), and QK(t + 1) was issued after PV(t - 2) (tcgen05.commit covers all prior MMAs);
        PV(t + 2) needs the max that this waiter has not yet published."""
        cute.arch.mbarrier_wait(
            o_full + ((t_local % 2) * self.iterations_pv_n + iter_n), (t_local // 2) % 2
        )

    @cute.jit
    def _rescale_o(
        self,
        common_params: SimpleNamespace,
        o_full: cute.Pointer,
        t_prev: cutlass.Int32,
        correction_factor: cutlass.Float32,
        no_correction: cutlass.Int32,
    ):
        """O *= correction_factor for this thread's rows after PV(t_prev) and before PV(t_prev + 1); skipped when
        no row of the warp moved its max (threshold), as the deleted correction WG did."""
        skip_correction = cute.arch.vote_all_sync(no_correction == 1)
        if not skip_correction:
            for iter_n in cutlass.range_constexpr(self.iterations_pv_n):
                self._o_full_wait(o_full, t_prev, iter_n)
                tmem_load_tiled_copy, tAcc, tTR_tAcc, tTR_gO, tTR_cO, tTR_rAcc = (
                    self._tmem_load_partition(
                        common_params, common_params.tiled_mma_pv, iter_n
                    )
                )
                tmem_store_atom = cute.make_copy_atom(
                    tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)),
                    self.acc_dtype,
                )
                tmem_store_tiled_copy = tcgen05.make_tmem_copy(tmem_store_atom, tAcc)
                cute.copy(tmem_load_tiled_copy, tTR_tAcc, tTR_rAcc)
                for i in cutlass.range(
                    cute.size(tTR_rAcc), vectorize=True, unroll_full=True
                ):
                    tTR_rAcc[i] = tTR_rAcc[i] * correction_factor
                cute.copy(tmem_store_tiled_copy, tTR_rAcc, tTR_tAcc)
            cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def _causal_mask(
        self,
        common_params: SimpleNamespace,
        tTR_rAcc: cute.Tensor,
        tTR_tS: cute.Tensor,
        k_index: cutlass.Int32,
    ):
        """Causal mask in global positions. Packed row r of this CTA is query
        q_tok = (q_tile * 128 + cta * 64 + r) // num_heads; its keys stop at the
        global bound K_causal - (S_q - 1) + q_tok, mapped to this DCP rank's local
        bound. Masked scores take the legacy finite sentinel -1e6."""
        masked_score = self.acc_dtype(-1.0e6)
        cta_m_rows = self.mma_qk_tiler[0] // self.cluster_shape_mnk[0]
        for i in cutlass.range(cute.size(tTR_rAcc), unroll_full=True):
            q_tok = (
                common_params.blk_coord[1] * self.mma_qk_tiler[0]
                + common_params.blk_coord[0] * cta_m_rows
                + tTR_tS[i][0]
            ) // self.num_heads
            if cutlass.const_expr(self.cp_world == 1):
                # Non-DCP: causal bound is the full local context length
                k_bound = common_params.K - (self.seq_len_q - 1) + q_tok
            else:
                global_bound = common_params.K_causal - (self.seq_len_q - 1) + q_tok
                k_bound = self._get_dcp_local_bound(global_bound)
            tTR_rAcc[i] = (
                tTR_rAcc[i]
                if cute.elem_less(
                    tTR_tS[i][1] + self.mma_qk_tiler[1] * k_index,
                    k_bound,
                )
                else masked_score
            )

    @cute.jit
    def softmax_tile(
        self,
        common_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        k_index: cutlass.Int32,
        mma_s_consumer_state: pipeline.PipelineState,
        p_mma_producer_state: pipeline.PipelineState,
        row_max_own: cutlass.Float32,
        row_sum: cutlass.Float32,
        apply_mask: bool,
        first: cutlass.Boolean,
        last: cutlass.Boolean,
        own_first: cutlass.Boolean,
        t_local: cutlass.Int32,
        wg: int,
    ) -> tuple[pipeline.PipelineState, pipeline.PipelineState, cutlass.Float32, cutlass.Float32]:
        """Softmax of KV tile t_local of this split, on one of the two softmax WGs (E2b: WG t_local % 2).

        The running row max passes in series between the WGs: this tile waits for the max that tile t_local - 1
        used (smem + named barrier; not for the first tile of the split), and publishes its own (not for the last).
        Each WG keeps its own row sum, scaled to its own last used max (row_max_own); the epilogue WG merges both.
        When the max grows past the skip threshold, this WG rescales O itself: it waits for PV(t_local - 1) (the
        mma_o full barrier of that tile) after its P store and before its P commit, so PV(t_local) sees the
        rescaled O. mma_s / p_mma stages are this WG's parity: the states advance by 2 per tile.

        :return: mma_s consumer state, p_mma producer state, the used row max, this WG's row sum
        """


        self._ik("sm_wS")
        mma_s_try_token = softmax_params.mma_s_pipeline.consumer_try_wait(
            mma_s_consumer_state
        )
        softmax_params.mma_s_pipeline.consumer_wait(
            mma_s_consumer_state, try_wait_token=mma_s_try_token
        )
        self._ik()

        # load S from tmem
        tStS_shape = softmax_params.tiled_mma_qk.partition_shape_C(
            cute.select(self.mma_qk_tiler, mode=[0, 1])
        )
        tStS_staged_fake = softmax_params.tiled_mma_qk.make_fragment_C(
            cute.append(tStS_shape, self.mma_s_stage)
        )
        tStS_staged = cute.make_tensor(common_params.tmem_ptr, tStS_staged_fake.layout)
        tStS = tStS_staged[None, None, None, mma_s_consumer_state.index]

        tAcc = tStS[(None, None), 0, 0]
        cta_qk_tiler = (
            self.mma_qk_tiler[0] // self.cluster_shape_mnk[0],
            self.mma_qk_tiler[1],
            self.mma_qk_tiler[2],
        )
        cS = cute.make_identity_tensor(cute.select(cta_qk_tiler, mode=[0, 1]))

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype
        )
        tmem_tiled_copy = tcgen05.make_tmem_copy(tmem_load_atom, tAcc)

        tidx = common_params.tidx % (self.num_compute_warps * self.threads_per_warp)

        tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)
        tTR_tAcc = tmem_thr_copy.partition_S(tAcc)
        tTR_tS = tmem_thr_copy.partition_D(cS)

        tTR_rAcc = cute.make_fragment_like(tTR_tS, self.acc_dtype)

        row_max_new = -self.acc_dtype.inf
        if cutlass.const_expr(self.arch == "sm_100"):
            self._ik("sm_ld")
            cute.copy(tmem_tiled_copy, tTR_tAcc, tTR_rAcc)
            # Early S release: fence TMEM load and release S buffer immediately
            # so MMA warp can start the next QK computation without waiting for
            # the entire softmax computation to complete.
            cute.arch.fence_view_async_tmem_load()
            softmax_params.mma_s_pipeline.consumer_release(mma_s_consumer_state)
            mma_s_consumer_state.advance()
            mma_s_consumer_state.advance()
            self._ik()
            self._ik("sm_max")
            if apply_mask:
                self._causal_mask(common_params, tTR_rAcc, tTR_tS, k_index)
            # reduction for row_max
            row_max_new = tTR_rAcc.load().reduce(cute.ReductionOp.MAX, row_max_new, 0)
            self._ik()
        else:  # SM103 supports TMEM load with a MAX reduction.
            tmem_load_red_atom = cute.make_copy_atom(
                tcgen05.copy.LdRed32x32bOp(
                    tcgen05.copy.Repetition(64), redOp=tcgen05.TmemLoadRedOp.MAX
                ),
                self.acc_dtype,
            )
            tmem_red_tiled_copy = tcgen05.make_tmem_copy(tmem_load_red_atom, tAcc)
            tmem_red_thr_copy = tmem_red_tiled_copy.get_slice(tidx)
            tTR_tAcc_red = tmem_red_thr_copy.partition_S(tAcc)
            tTR_tS_red = tmem_red_thr_copy.partition_D(cS)
            tTR_rAcc_red = cute.make_fragment_like(tTR_tS_red, self.acc_dtype)
            tTR_rMax = cute.make_rmem_tensor(
                cute.make_layout((1, tTR_tS_red.shape[1], tTR_tS_red.shape[2])),
                self.acc_dtype,
            )
            # only tTR_rMax[0] is read below: the x64 ld.red must give one max per thread (one row x 64 keys)
            assert cute.size(tTR_rMax) == 1, f"ld.red max layout {tTR_rMax.layout} holds more than one value"
            self._ik("sm_ld")
            cute.copy(
                tmem_red_tiled_copy,
                tTR_tAcc_red,
                (tTR_rAcc_red, tTR_rMax),
            )
            # Early S release for the SM103 reduction path
            cute.arch.fence_view_async_tmem_load()
            softmax_params.mma_s_pipeline.consumer_release(mma_s_consumer_state)
            mma_s_consumer_state.advance()
            mma_s_consumer_state.advance()
            self._ik()
            self._ik("sm_max")
            tTR_rAcc = cute.make_tensor(tTR_rAcc_red.iterator, tTR_rAcc.layout)
            if apply_mask:
                self._causal_mask(common_params, tTR_rAcc, tTR_tS, k_index)
                # reduction for row_max after manual masking
                row_max_new = tTR_rAcc.load().reduce(
                    cute.ReductionOp.MAX, row_max_new, 0
                )
            else:
                # The TMEM load reduction is valid here because
                # tTR_rAcc is unmodified (no mask applied to this tile).
                row_max_new = cute.arch.fmax(row_max_new, tTR_rMax[0])
            self._ik()

        # if warps in N is 2, reduce row_max across warps (0, 1) and (2, 3)
        # Exchange pairs: {warp0, warp2} and {warp1, warp3} are independent.
        # Use split barriers so each pair only waits for its partner.
        self._ik("sm_xchg")
        if cutlass.const_expr(self.warps_in_n == 2):
            common_params.smem_exchange[tidx] = row_max_new
            warp_in_group = tidx // self.threads_per_warp
            if warp_in_group % 2 == 0:
                cute.arch.barrier(barrier_id=self.pair_bar_ids[wg][0], number_of_threads=2 * self.threads_per_warp)
            else:
                cute.arch.barrier(barrier_id=self.pair_bar_ids[wg][1], number_of_threads=2 * self.threads_per_warp)
            row_max_new = cute.arch.fmax(
                row_max_new,
                common_params.smem_exchange[
                    (tidx + 64) % (self.num_compute_warps * self.threads_per_warp)
                ],
            )
        self._ik()
        self._ik("sm_wM")
        # the max of the previous tile of this split (the other WG), -inf on the first tile
        row_max = -self.acc_dtype.inf
        handoff_in = softmax_params.handoff_in
        handoff_out = softmax_params.handoff_out
        o_full = common_params.mma_o_pipeline.sync_object_full.get_barrier(0)
        if not first:
            cute.arch.barrier(barrier_id=self.handoff_bar_ids[1 - wg], number_of_threads=self.wg_pair_threads)
            row_max = handoff_in[tidx]
        self._ik()
        self._ik("sm_cf")
        row_max_new = cute.arch.fmax(row_max_new, row_max)
        # find correction factor
        max_delta = row_max - row_max_new
        correction_factor = cute.math.exp2(
            max_delta * softmax_params.softmax_scale_log2, fastmath=True
        )
        # skip the rescale if the row max barely grew; the first tile never skips
        skip = (
            row_max_new - row_max
        ) * softmax_params.softmax_scale_log2 <= self.skip_correction_threshold
        no_correction = cutlass.Int32(1) if skip else cutlass.Int32(0)
        row_max_new = row_max if skip else row_max_new
        correction_factor = cutlass.Float32(1.0) if skip else correction_factor
        # publish the used max to the WG of the next tile
        if not last:
            handoff_out[tidx] = row_max_new
            cute.arch.barrier_arrive(barrier_id=self.handoff_bar_ids[wg], number_of_threads=self.wg_pair_threads)
        self._ik()
        self._ik("sm_exp")
        # softmax
        fma_b = softmax_params.softmax_scale_log2
        fma_c = (0.0 - row_max_new) * softmax_params.softmax_scale_log2

        for i in cutlass.range(cute.size(tTR_rAcc), vectorize=True, unroll_full=True):
            tTR_rAcc[i] = tTR_rAcc[i] * fma_b + fma_c
            tTR_rAcc[i] = cute.math.exp2(tTR_rAcc[i], fastmath=True)

        # Compute row_sum immediately after exp2, before quantize/P-store.
        # This overlaps the FADD reductions with any remaining MUFU pipeline drain
        # and moves row_sum off the critical path between P-commit and p_cor-acquire.
        self._ik()
        self._ik("sm_rsum")
        # this WG's row sum was last scaled to row_max_own (its previous tile)
        sum_scale = cutlass.Float32(1.0)
        if not own_first:
            sum_scale = cute.math.exp2(
                (row_max_own - row_max_new) * softmax_params.softmax_scale_log2, fastmath=True
            )
        row_sum = row_sum * sum_scale
        row_sum_vec = (0.0, 0.0)
        for i in cutlass.range_constexpr(0, cute.size(tTR_rAcc), 2):
            row_sum_vec = cute.arch.add_packed_f32x2(
                row_sum_vec, (tTR_rAcc[i], tTR_rAcc[i + 1])
            )
        row_sum = row_sum_vec[0] + row_sum_vec[1] + row_sum

        self._ik()
        self._ik("sm_cvt")
        tTR_rS = cute.make_fragment_like(tTR_tS, self.q_dtype)

        # quantize
        tTR_rS.store(tTR_rAcc.load().to(self.q_dtype))

        # Acquire close to STS so softmax work can overlap with downstream MMA.
        self._ik()
        self._ik("sm_pacq")
        softmax_params.p_mma_pipeline.producer_acquire(p_mma_producer_state)
        self._ik()
        self._ik("sm_pst")
        sP = softmax_params.sP[None, None, None, (None, p_mma_producer_state.index)]
        sP_mk_view = cute.make_tensor(
            sP.iterator,
            cute.make_layout(
                (
                    (sP.shape[0][0], sP.shape[1]),
                    (sP.shape[0][1], sP.shape[2], sP.shape[3]),
                ),
                stride=(
                    (sP.stride[0][0], sP.stride[1]),
                    (sP.stride[0][1], sP.stride[2], sP.stride[3]),
                ),
            ),
        )
        # change to PISL
        sP_wo_swizzle_iter = cute.recast_ptr(sP.iterator, swizzle_=None)
        swizzle_bits = (
            int(math.log2(self.mma_pv_tiler[2] * self.q_dtype.width // 8 // 32)) + 1
        )
        swizzle_base = 3 if self.q_dtype.width == 16 else 4
        sP_swizzle = cute.make_swizzle(swizzle_bits, swizzle_base, 3)
        sP_mk_view = cute.make_tensor(
            sP_wo_swizzle_iter,
            cute.make_composed_layout(sP_swizzle, 0, sP_mk_view.layout),
        )
        universal_copy_bits = 128
        smem_copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.q_dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        smem_tiled_copy = cute.make_tiled_copy_D(smem_copy_atom, tmem_tiled_copy)
        smem_thr_copy = smem_tiled_copy.get_slice(tidx)
        rP_copy_view = smem_thr_copy.retile(tTR_rS)
        sP_copy_view = smem_thr_copy.partition_D(sP_mk_view)
        cute.copy(smem_tiled_copy, rP_copy_view, sP_copy_view)
        self._ik()
        if not first:
            self._ik("sm_resc")
            self._rescale_o(common_params, o_full, t_local - 1, correction_factor, no_correction)
            self._ik()
        self._ik("sm_pst")

        # fence between smem store and mma o
        cute.arch.fence_view_async_shared()
        softmax_params.p_mma_pipeline.producer_commit(p_mma_producer_state)
        p_mma_producer_state.advance()
        p_mma_producer_state.advance()
        self._ik()

        return (
            mma_s_consumer_state,
            p_mma_producer_state,
            row_max_new,
            row_sum,
        )

    @cute.jit
    def _tmem_load_partition(
        self, common_params: SimpleNamespace, tiled_mma_pv: cute.TiledMma, iter_n: int
    ) -> tuple[
        cute.TiledMma, cute.TiledMma, cute.TiledMma, cute.TiledMma, cute.TiledMma
    ]:
        """Tensor memory load partition for rescale and epilogue.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param tiled_mma_pv: The tiled mma pv
        :type tiled_mma_pv: cute.TiledMma
        :param iter_n: The iteration number
        :type iter_n: int

        :return: The tiled mma pv, the tiled mma pv, the tiled mma pv, the tiled mma pv, the tiled mma pv
        :rtype: tuple[cute.TiledMma, cute.TiledMma, cute.TiledMma, cute.TiledMma, cute.TiledMma]
        """

        tOtO_shape = tiled_mma_pv.partition_shape_C(
            cute.select(self.mma_pv_tiler, mode=[0, 1])
        )
        tOtO = tiled_mma_pv.make_fragment_C(tOtO_shape)
        tOtO_layout = cute.append(
            tOtO.layout,
            cute.make_layout(
                common_params.L // self.mma_pv_tiler[1],
                stride=self.pv_acc_cols,
            ),
        )
        tOtO = cute.make_tensor(
            common_params.tmem_ptr + self.tmem_o_offset, tOtO_layout
        )
        tOtO = tOtO[None, None, None, iter_n]

        tAcc = tOtO[(None, None), 0, 0]

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype
        )
        tmem_load_tiled_copy = tcgen05.make_tmem_copy(tmem_load_atom, tAcc)
        tmem_load_thr_copy = tmem_load_tiled_copy.get_slice(
            common_params.tidx % (self.num_compute_warps * self.threads_per_warp)
        )

        cta_pv_tiler = (
            self.mma_pv_tiler[0] // self.cluster_shape_mnk[0],
            self.mma_pv_tiler[1],
            self.mma_pv_tiler[2],
        )
        # Flatten divide and partition global tensors for O
        cta_pv_tiler_mn = cute.select(cta_pv_tiler, mode=[0, 1])

        gO = None
        if cutlass.const_expr(common_params.mAccO is not None):
            gO = cute.local_tile(
                common_params.mAccO[None, common_params.blk_coord[3], None, None, None],
                cta_pv_tiler_mn,
                (
                    common_params.blk_coord[0],
                    iter_n,
                    common_params.blk_coord[1],
                    common_params.blk_coord[2],
                ),
            )
            cO = cute.local_tile(
                cute.make_identity_tensor(
                    common_params.mAccO[
                        None, common_params.blk_coord[3], None, None, None
                    ].shape
                ),
                cta_pv_tiler_mn,
                (
                    common_params.blk_coord[0],
                    iter_n,
                    common_params.blk_coord[1],
                    common_params.blk_coord[2],
                ),
            )
        else:
            gO = cute.local_tile(
                common_params.mO,
                cta_pv_tiler_mn,
                (
                    common_params.blk_coord[0],
                    iter_n,
                    common_params.blk_coord[1],
                    common_params.blk_coord[2],
                ),
            )
            cO = cute.local_tile(
                cute.make_identity_tensor(common_params.mO.shape),
                cta_pv_tiler_mn,
                (
                    common_params.blk_coord[0],
                    iter_n,
                    common_params.blk_coord[1],
                    common_params.blk_coord[2],
                ),
            )
        tTR_tAcc = tmem_load_thr_copy.partition_S(tAcc)
        tTR_gO = tmem_load_thr_copy.partition_D(gO)
        tTR_cO = tmem_load_thr_copy.partition_D(cO)
        tTR_rAcc = cute.make_fragment_like(tTR_gO, self.acc_dtype)
        return tmem_load_tiled_copy, tAcc, tTR_tAcc, tTR_gO, tTR_cO, tTR_rAcc  # type: ignore[return-value]

    @cute.jit
    def epilogue(
        self,
        common_params: SimpleNamespace,
        epilogue_params: SimpleNamespace,
        t_last: cutlass.Int32,
        row_sum: cutlass.Float32,
        row_max: cutlass.Float32,
    ):
        """Epilogue for one k-tile. Updates the related pipeline state.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param epilogue_params: The epilogue parameters
        :type epilogue_params: SimpleNamespace
        :param mma_o_consumer_state: The mma o consumer state
        :type mma_o_consumer_state: pipeline.PipelineState
        :param row_sum: The row sum
        :type row_sum: cutlass.Float32
        :param row_max: The row max
        :type row_max: cutlass.Float32

        :return: The MMA o consumer state
        :rtype: pipeline.PipelineState
        """

        tidx = common_params.tidx % (self.num_compute_warps * self.threads_per_warp)
        o_full = common_params.mma_o_pipeline.sync_object_full.get_barrier(0)

        # exchange row_sum between warps (0, 1) and (2, 3)
        # Use split barriers so each pair only waits for its partner.
        if cutlass.const_expr(self.warps_in_n == 2):
            common_params.smem_exchange[tidx] = row_sum
            warp_in_group = tidx // self.threads_per_warp
            if warp_in_group % 2 == 0:
                self.epilogue_exchange_sync_bar_pair02.arrive_and_wait()
            else:
                self.epilogue_exchange_sync_bar_pair13.arrive_and_wait()
            # (64, 2)
            row_sum = (
                row_sum
                + common_params.smem_exchange[
                    (tidx + 64) % (self.num_compute_warps * self.threads_per_warp)
                ]
            )
        # Pre-compute the output scale factor once to avoid redundant MUFU.RCP
        # instructions inside the per-element loop (rcp_approx generates MUFU.RCP
        # which has 8-cycle throughput; hoisting it saves up to ~500 cycles).
        # partials skip output_scale; the reduction kernel applies it
        if cutlass.const_expr(common_params.mAccO is None):
            epi_scale = epilogue_params.output_scale * cute.arch.rcp_approx(row_sum)
            store_dtype = self.o_dtype
        else:
            epi_scale = cute.arch.rcp_approx(row_sum)
            store_dtype = self.acc_dtype

        for iter_n in cutlass.range_constexpr(self.iterations_pv_n):
            # O half iter_n of PV(t_last). The parity wait is exact: PV(t_last - 2) is complete, because the WG of
            # tile t_last acquired its P stage (released by PV(t_last - 2)) before final_bar. Draining half 0
            # overlaps the PV of half 1.
            self._ik("epi_wO")
            self._o_full_wait(o_full, t_last, iter_n)
            self._ik()
            self._ik("epi_ld")
            # tmem load tiled copy and partition results (full tile).
            tmem_load_tiled_copy, tAcc, tTR_tAcc, tTR_gO, tTR_cO, tTR_rAcc = (
                self._tmem_load_partition(
                    common_params, common_params.tiled_mma_pv, iter_n
                )
            )

            # Full TMEM load to registers
            cute.copy(tmem_load_tiled_copy, tTR_tAcc, tTR_rAcc)
            self._ik()
            self._ik("epi_st")
            # Subtiled elementwise + STG: elementwise of subtile N+1 overlaps
            # with STG of subtile N via instruction-level parallelism.
            # tTR_rAcc shape: ((V, 1), 1, rest_tiles) where rest_tiles = 4.
            # Split along rest_tiles (mode 2) for subtiling.
            num_epi_subtiles = tTR_rAcc.shape[2]
            vec_size = cute.size(cute.select(tTR_rAcc.shape, mode=[0]))

            # Prepare output fragment for type conversion
            tR2G_rO_src = cute.make_fragment_like(tTR_gO, store_dtype)
            tR2G_rO_dst = tTR_gO

            for d_sub in cutlass.range_constexpr(num_epi_subtiles):
                # Elementwise: scale and normalize (using pre-computed epi_scale)
                for i in cutlass.range(vec_size, vectorize=True, unroll_full=True):
                    tTR_rAcc[i, 0, d_sub] = tTR_rAcc[i, 0, d_sub] * epi_scale

                # Type convert per subtile using vectorized load/to/store
                tR2G_rO_src[None, None, d_sub].store(
                    tTR_rAcc[None, None, d_sub].load().to(store_dtype)
                )

                # Store this subtile to global memory using autovec_copy
                # for vectorized stores (STG.128).
                if cute.elem_less(tTR_cO[0][0], common_params.H):
                    cute.autovec_copy(
                        tR2G_rO_src[None, None, d_sub],
                        tR2G_rO_dst[None, None, d_sub],
                        l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE,
                    )

            # store the lse to global memory
            cta_pv_tiler = (
                self.mma_pv_tiler[0] // self.cluster_shape_mnk[0],
                self.mma_pv_tiler[1],
                self.mma_pv_tiler[2],
            )
            if cutlass.const_expr(epilogue_params.mAccLSE is None):
                if cutlass.const_expr(not self.skip_lse):
                    lse = (
                        cute.math.log2(row_sum, fastmath=True)
                        + epilogue_params.softmax_scale_log2 * row_max
                    )
                    gLSE = cute.local_tile(
                        epilogue_params.mLSE,
                        (cta_pv_tiler[0], 1, 1),
                        (
                            common_params.blk_coord[0],
                            common_params.blk_coord[1],
                            common_params.blk_coord[2],
                        ),
                        (1, 1, 1),
                    )
                    cLSE = cute.local_tile(
                        cute.make_identity_tensor(epilogue_params.mLSE.shape),
                        (cta_pv_tiler[0], 1, 1),
                        (
                            common_params.blk_coord[0],
                            common_params.blk_coord[1],
                            common_params.blk_coord[2],
                        ),
                        (1, 1, 1),
                    )
                    if cute.elem_less(cLSE[tidx][0], common_params.H):
                        gLSE[tidx] = lse
            else:
                lse = (
                    cute.math.log2(row_sum, fastmath=True)
                    + epilogue_params.softmax_scale_log2 * row_max
                )
                gLSE = cute.local_tile(
                    epilogue_params.mAccLSE[
                        None, common_params.blk_coord[3], None, None
                    ],
                    (cta_pv_tiler[0], 1, 1),
                    (
                        common_params.blk_coord[0],
                        common_params.blk_coord[1],
                        common_params.blk_coord[2],
                    ),
                    (1, 1, 1),
                )
                cLSE = cute.local_tile(
                    cute.make_identity_tensor(
                        epilogue_params.mAccLSE[
                            None, common_params.blk_coord[3], None, None
                        ].shape
                    ),
                    (cta_pv_tiler[0], 1, 1),
                    (
                        common_params.blk_coord[0],
                        common_params.blk_coord[1],
                        common_params.blk_coord[2],
                    ),
                    (1, 1, 1),
                )
                if cute.elem_less(cLSE[tidx][0], common_params.H):
                    gLSE[tidx] = lse

            cute.arch.fence_view_async_tmem_load()
            self._ik()

    def make_and_init_load_qkv_pipeline(
        self, load_qkv_mbar_ptr, cta_layout_vmnk, load_stages, tx_count
    ) -> pipeline.PipelineTmaUmma:
        """Create and initialize the tma load qkv pipeline.

        :param load_qkv_mbar_ptr: The load qkv mbar pointer
        :type load_qkv_mbar_ptr: cute.Tensor
        :param cta_layout_vmnk: The cta layout vmnk
        :type cta_layout_vmnk: tuple[int, int, int]
        :param load_stages: The load stages
        :type load_stages: list[int]
        :param tx_count: The tx count
        :type tx_count: int

        :return: The tma load qkv pipeline
        :rtype: pipeline.PipelineTmaUmma
        """
        load_qkv_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.load_tma_k_warp_id])
        )
        load_qkv_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.mma_warp_id])
        )
        return pipeline.PipelineTmaUmma.create(
            barrier_storage=load_qkv_mbar_ptr,
            num_stages=load_stages,
            producer_group=load_qkv_producer_group,
            consumer_group=load_qkv_consumer_group,
            tx_count=tx_count,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )

    def make_and_init_mma_s_pipeline(
        self, mma_s_mbar_ptr, cta_layout_vmnk
    ) -> pipeline.PipelineUmmaAsync:
        """Create and initialize the mma s pipeline.

        :param mma_s_mbar_ptr: The mma s mbar pointer
        :type mma_s_mbar_ptr: cute.Tensor
        :param cta_layout_vmnk: The cta layout vmnk
        :type cta_layout_vmnk: tuple[int, int, int]

        :return: The mma s pipeline
        :rtype: pipeline.PipelineUmmaAsync
        """

        mma_s_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.mma_warp_id])
        )
        consumer_thread_size = (
            self.threads_per_warp
            * len(self.compute_warp_ids)
            * self.cluster_shape_mnk[0]
        )
        mma_s_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            consumer_thread_size,
        )
        return pipeline.PipelineUmmaAsync.create(
            barrier_storage=mma_s_mbar_ptr,
            num_stages=self.mma_s_stage,
            producer_group=mma_s_producer_group,
            consumer_group=mma_s_consumer_group,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )

    def make_and_init_p_mma_pipeline(
        self, p_mma_mbar_ptr, cta_layout_vmnk
    ) -> pipeline.PipelineAsyncUmma:
        """Create and initialize the p mma pipeline.

        :param p_mma_mbar_ptr: The p mma mbar pointer
        :type p_mma_mbar_ptr: cute.Tensor
        :param cta_layout_vmnk: The cta layout vmnk
        :type cta_layout_vmnk: tuple[int, int, int]

        :return: The p mma pipeline
        :rtype: pipeline.PipelineAsyncUmma
        """

        producer_thread_size = (
            self.threads_per_warp
            * len(self.compute_warp_ids)
            * self.cluster_shape_mnk[0]
        )
        p_mma_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            producer_thread_size,
        )
        p_mma_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.mma_warp_id])
        )
        return pipeline.PipelineAsyncUmma.create(
            barrier_storage=p_mma_mbar_ptr,
            num_stages=self.p_mma_stage,
            producer_group=p_mma_producer_group,
            consumer_group=p_mma_consumer_group,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )

    def make_and_init_mma_o_pipeline(
        self, mma_o_mbar_ptr, cta_layout_vmnk
    ) -> pipeline.PipelineUmmaAsync:
        """Create and initialize the mma o pipeline.

        :param mma_o_mbar_ptr: The mma o mbar pointer
        :type mma_o_mbar_ptr: cute.Tensor
        :param cta_layout_vmnk: The cta layout vmnk
        :type cta_layout_vmnk: tuple[int, int, int]

        :return: The mma o pipeline
        :rtype: pipeline.PipelineUmmaAsync
        """

        mma_o_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.mma_warp_id])
        )
        consumer_thread_size = (
            self.threads_per_warp
            * len(self.compute_warp_ids)
            * self.cluster_shape_mnk[0]
        )
        mma_o_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            consumer_thread_size,
        )
        return pipeline.PipelineUmmaAsync.create(
            barrier_storage=mma_o_mbar_ptr,
            num_stages=self.mma_o_stage,
            producer_group=mma_o_producer_group,
            consumer_group=mma_o_consumer_group,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )

    @cute.jit
    def initialize_workspace(
        self,
        H: cutlass.Int32,
        D: cutlass.Int32,
        S: cutlass.Int32,
        B: cutlass.Int32,
        split_kv: cutlass.Int32,
        acc_dtype: Type[cutlass.Numeric],
        workspace: cute.Tensor,
    ) -> tuple[cute.Tensor, cute.Tensor]:
        """Initialize the workspace for the MLA kernel. Construct the intermediate tensors
        acc_o and acc_lse.

        :param H: The height of the output tensor C
        :type H: cutlass.Int32
        :param D: The depth of the output tensor C
        :type D: cutlass.Int32
        :param S: The sequence length of the output tensor C
        :type S: cutlass.Int32
        :param B: The batch size of the output tensor C
        :type B: cutlass.Int32
        :param split_kv: The split key-value of the output tensor C
        :type split_kv: cutlass.Int32
        :param acc_dtype: The data type of the output tensor C
        :type acc_dtype: Type[cutlass.Numeric]
        :param workspace: The workspace tensor
        :type workspace: cute.Tensor

        :return: The output tensor C and the workspace tensor
        :rtype: tuple[cute.Tensor, cute.Tensor]
        """
        acc_o, acc_lse = None, None
        if cutlass.const_expr(workspace is not None):
            align = 256 // self.q_dtype.width
            acc_o_layout = cute.make_layout(
                (H, split_kv, D, S, B),
                stride=(
                    cute.assume(split_kv * D, align),
                    cute.assume(D, align),
                    1,
                    cute.assume(split_kv * H * D, align),
                    cute.assume(H * split_kv * S * D, align),
                ),
            )
            # partials use partial_dtype, LSE partials stay in acc_dtype
            acc_o_iter = cute.recast_ptr(workspace.iterator, dtype=self.acc_dtype)
            acc_o = cute.make_tensor(acc_o_iter, acc_o_layout)
            acc_lse_layout = cute.make_layout(
                (H, split_kv, S, B),
                stride=(split_kv, 1, H * split_kv, H * split_kv * S),
            )
            # the partial block is a multiple of 1 KB; say so or LSE loads drop to 16-bit halves
            # Convert bits to bytes before multiplying the Int32 element count.
            acc_lse_iter = cute.recast_ptr(
                workspace.iterator
                + cute.assume(
                    cute.cosize(acc_o_layout) * (self.acc_dtype.width // 8), 32
                ),
                dtype=acc_dtype,
            )
            acc_lse = cute.make_tensor(acc_lse_iter, acc_lse_layout)
        return acc_o, acc_lse
