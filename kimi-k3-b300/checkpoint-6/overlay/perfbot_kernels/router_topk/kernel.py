# SPDX-License-Identifier: Apache-2.0
"""Router GEMM + top-K routing in one launch (PerfBot roadmap item r3-301). CuTe DSL, SM100 family.

Computes, for x [M, K] bf16 and the router weight W [N, K] bf16:
  logits = x @ W^T                     (fp32 accumulate, fp32 out, written to `logits` [M, N])
  s_e    = sigmoid(logits_e)           (flashinfer sigmoid_accurate under -use_fast_math)
  sel_e  = s_e + bias_e                (fp32 bias; selection key)
  top-K over sel (ties: smaller expert id), rank order
  w_t    = s_t * route_scale / (sum_t s_t + 1e-20)   (ScaledSumNormalize, sum by warp butterfly)
  packed [M, K_TOP] int32 = (expert_id << 16) | bf16(w_t)   (flashinfer PackedScoreIdx<bf16>)
which is what flashinfer 0.6.17 routingIndicesBlockScoresKernel<SigmoidBias, ScaledSumNormalize> writes for
DeepSeekV3 routing with n_group <= 1 (trtllm_fused_moe_runner.cu: routingCustom, mSumEpsilon 1e-20).

Structure (adapted from vLLM 3696c772 model_executor/kernels/linear/cute_dsl/_ll_bf16_splitk.py):
  grid (ceil(M/16), N/64, SPLIT_K=4), cluster (1, 1, SPLIT_K); 4 cp.async DMA warps + 4 mma.sync warps.
  1. main loop: split-K over the cluster, m16n8k16 bf16 MMA, fp32 partials.
  2. intra-CTA reduce, DSMEM send of each rank's partial to cluster rank 0, cluster barrier.
  3. rank 0: sum the SPLIT_K partials in rank order, write fp32 logits, then the 16x64 tile's top-K per row
     (warp tournament, two rows per warp) as a sorted (key, id) list to a global candidate buffer.
  4. CTA barrier + one acq_rel arrival on a per-row-group counter; the last of the N/64 tiles stages the
     row group's lists in smem and merges them (one list per lane, top-K tournament rounds), normalizes,
     writes `packed` and resets its counter (self-cleaning: graph replays need no memset).
Tie rule: keys compare as (orderable fp32 bits of sel, 65535 - id), so equal sel goes to the smaller id.
The exact flashinfer math is emitted as inline PTX: mul.ftz / tanh.approx / fma.rn.ftz (sigmoid),
add.ftz (bias, butterfly sum, epsilon), div.approx.ftz (normalize), cvt.rn.bf16x2 (weight).

Sample inputs (K3 decode, conc 6 x 8 tokens): x [48, 7168] bf16, W [896, 7168] bf16, bias [896] fp32,
route_scale 1.0 -> packed [48, 16] int32, logits [48, 896] fp32.
"""

import math

import cutlass
import cutlass.cute as cute
from cuda.bindings.driver import CUstream
from cutlass import const_expr
from cutlass._mlir import ir as _ir
from cutlass._mlir.dialects import llvm as _llvm
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.pipeline import sm90 as pipeline

try:  # IKET phase ranges; a default compile strips them.
    from cutlass.cute.experimental import iket as _iket
except Exception:  # pragma: no cover
    _iket = None

SUM_EPS = 1e-20


def _f32():
    return _ir.F32Type.get()


def _i32():
    return _ir.IntegerType.get_signless(32)


def _asm(res_type, args, text, constraints, side_effects=False, loc=None, ip=None):
    return _llvm.inline_asm(
        res_type,
        [a.ir_value(loc=loc, ip=ip) for a in args],
        text,
        constraints,
        has_side_effects=side_effects,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def sigmoid_fast(x, *, loc=None, ip=None):
    """0.5f * tanhf(0.5f * x) + 0.5f exactly as nvcc -use_fast_math lowers it."""
    return cutlass.Float32(
        _asm(
            _f32(),
            [x],
            "{ .reg .f32 t; mul.ftz.f32 t, $1, 0f3F000000; tanh.approx.f32 t, t; "
            "fma.rn.ftz.f32 $0, t, 0f3F000000, 0f3F000000; }",
            "=f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def add_ftz(a, b, *, loc=None, ip=None):
    return cutlass.Float32(_asm(_f32(), [a, b], "add.ftz.f32 $0, $1, $2;", "=f,f,f", loc=loc, ip=ip))


@dsl_user_op
def scale_norm(s, scale, total, eps, *, loc=None, ip=None):
    """s * scale / (total + eps): mul.ftz, add.ftz, div.approx.ftz (fast-math lowering)."""
    return cutlass.Float32(
        _asm(
            _f32(),
            [s, scale, total, eps],
            "{ .reg .f32 n, d; mul.ftz.f32 n, $1, $2; add.ftz.f32 d, $3, $4; div.approx.ftz.f32 $0, n, d; }",
            "=f,f,f,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def order_key(x, *, loc=None, ip=None):
    """fp32 -> uint32 whose unsigned order is the float order."""
    return cutlass.Uint32(
        _asm(
            _i32(),
            [x],
            "{ .reg .b32 b, m; mov.b32 b, $1; shr.s32 m, b, 31; or.b32 m, m, 0x80000000; xor.b32 $0, b, m; }",
            "=r,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def f32_bits(x, *, loc=None, ip=None):
    return cutlass.Uint32(_asm(_i32(), [x], "mov.b32 $0, $1;", "=r,f", loc=loc, ip=ip))


@dsl_user_op
def bits_f32(x, *, loc=None, ip=None):
    return cutlass.Float32(_asm(_f32(), [x], "mov.b32 $0, $1;", "=f,r", loc=loc, ip=ip))


@dsl_user_op
def pack_id_weight(idx, w, *, loc=None, ip=None):
    """(idx << 16) | bf16_rn(w) as int32."""
    return cutlass.Int32(
        _asm(
            _i32(),
            [idx, w],
            "{ .reg .b32 h, z; cvt.rn.bf16x2.f32 h, 0f00000000, $2; shl.b32 z, $1, 16; or.b32 $0, z, h; }",
            "=r,r,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def sel_u32(c, a, b, *, loc=None, ip=None):
    """c != 0 ? a : b (u32), branch-free."""
    return cutlass.Uint32(
        _asm(_i32(), [c, a, b], "{ .reg .pred p; setp.ne.u32 p, $1, 0; selp.b32 $0, $2, $3, p; }",
             "=r,r,r,r", loc=loc, ip=ip)
    )


@dsl_user_op
def lowest_lane(mask, *, loc=None, ip=None):
    """Index of the lowest set bit of a non-zero 32-bit mask."""
    return cutlass.Int32(
        _asm(_i32(), [mask], "{ .reg .b32 r; brev.b32 r, $1; clz.b32 $0, r; }", "=r,r", loc=loc, ip=ip)
    )


@dsl_user_op
def set_block_rank(smem_ptr, peer_rank, *, loc=None, ip=None):
    dsmem_ptr = cute.arch.map_dsmem_ptr(smem_ptr, peer_rank, loc=loc, ip=ip)
    return cutlass.Int32(dsmem_ptr.toint(loc=loc, ip=ip))


@dsl_user_op
def st_shared_remote_f32(remote_addr, val, *, loc=None, ip=None):
    _llvm.inline_asm(
        _i32(),
        [remote_addr.ir_value(loc=loc, ip=ip), val.ir_value(loc=loc, ip=ip)],
        "st.shared::cluster.f32 [$0], $1; mov.u32 $2, 0;",
        "r,f,=r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


def _push(name):
    if _iket is not None:
        _iket.range_push(name)


def _pop():
    if _iket is not None:
        _iket.range_pop()


class RouterTopK:
    """One-launch router GEMM + sigmoid-bias top-K + scaled-sum normalize.

    :param n: number of experts (router N), a multiple of tile_n.
    :param k: hidden size (router K), a multiple of 8.
    :param top_k: experts per token (<= 32, <= tile_n).
    """

    def __init__(self, n=896, k=7168, top_k=16, tile_n=64, tile_k=256, num_stages=2, split_k=4,
                 use_pdl=True, iket=False, gemm_only=False):
        assert n % tile_n == 0 and k % 8 == 0 and top_k <= 32 and top_k <= tile_n
        self.n, self.k, self.top_k = n, k, top_k
        self.tile_m, self.tile_n, self.tile_k = 16, tile_n, tile_k
        self.num_stages, self.split_k, self.use_pdl = num_stages, split_k, use_pdl
        self.iket = iket
        self.gemm_only = gemm_only  # timing probe only: stop after the fp32 logits
        self.grid_n = n // tile_n
        self.num_cand = self.grid_n * top_k  # candidates per row
        self.cand_width = self.num_cand * 2  # (key, lo) u32 pairs
        assert self.grid_n <= 32 and tile_n % 32 == 0  # one sorted tile list per lane; whole columns per lane
        self.copy_bits = 128
        self.mma_shape = (16, 8, 16)
        self.num_dma_warps = 4
        self.num_mma_warps = 4
        self.num_dma_threads = 128
        self.num_mma_threads = 128
        self.num_threads = 256
        self.num_warps = 8
        self.tile_elems = self.tile_m * self.tile_n
        self.elems_per_thread = self.tile_elems // self.num_threads
        assert self.tile_elems % self.num_threads == 0
        assert self.tile_m * tile_k * 2 * num_stages >= self.num_mma_warps * self.tile_elems * 4  # red fits in sA
        assert tile_n * tile_k * 2 * num_stages >= self.tile_m * self.cand_width * 4  # merge lists fit in sB
        assert (self.tile_m * self.cand_width) % self.num_threads == 0
        assert self.tile_m % self.num_warps == 0
        self.rows_per_warp = self.tile_m // self.num_warps
        self.cols_per_lane = self.tile_n // 32

    def _smem_layout(self, dtype, smem_tiler):
        major = min(smem_tiler[1], 64)
        swz = min(int(math.log2(major * dtype.width // self.copy_bits)), 3)
        atom = cute.make_composed_layout(
            cute.make_swizzle(swz, 3, 3), 0, cute.make_layout((8, major), stride=(major, 1))
        )
        return cute.tile_to_shape(atom, smem_tiler, (0, 1, 2))

    def _g2s_copy(self, atom, dtype):
        elems = self.copy_bits // dtype.width
        kt = self.tile_k // elems
        thr = cute.make_layout((self.num_dma_threads // kt, kt), stride=(kt, 1))
        return cute.make_tiled_copy_tv(atom, thr, cute.make_layout((1, elems)))

    @cute.jit
    def _fill_pred(self, pred_flat, coord_tensor, k_tile, dim_limit, K_total):
        coord_ktile = coord_tensor[None, None, 0, k_tile]
        for v in cutlass.range_constexpr(pred_flat.shape[0]):
            for j in cutlass.range_constexpr(pred_flat.shape[1]):
                pred_flat[v, j] = cute.elem_less(coord_ktile[(0, v), j], (dim_limit, K_total))

    def _make_pred(self, tXcX):
        num_vec = tXcX.shape[0][1]
        num_mn = cute.size(tXcX, mode=[1])
        pred_flat = cute.make_rmem_tensor(
            cute.make_layout((num_vec, num_mn), stride=(num_mn, 1)), cutlass.Boolean
        )
        pred = cute.make_tensor(
            pred_flat.iterator,
            cute.make_layout((num_vec, num_mn, cute.size(tXcX, mode=[2])), stride=(num_mn, 1, 0)),
        )
        return pred_flat, pred

    @cute.jit
    def __call__(self, mA: cute.Tensor, mB: cute.Tensor, mBias: cute.Tensor, mLogits: cute.Tensor,
                 mPacked: cute.Tensor, mCand: cute.Tensor, mCnt: cute.Tensor,
                 route_scale: cutlass.Float32, stream: CUstream):
        bM, bN, bK = self.tile_m, self.tile_n, self.tile_k
        sA_layout = self._smem_layout(mA.element_type, (bM, bK, self.num_stages))
        sB_layout = self._smem_layout(mB.element_type, (bN, bK, self.num_stages))

        @cute.struct
        class SharedStorage:
            a: cute.struct.Align[cute.struct.MemRange[mA.element_type, cute.cosize(sA_layout)], 128]
            b: cute.struct.Align[cute.struct.MemRange[mB.element_type, cute.cosize(sB_layout)], 128]
            partials: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, self.split_k * self.tile_elems], 16]
            mbar: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, self.num_stages * 2], 8]
            flag: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 4], 16]

        atom_g2s = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL),
            mA.element_type,
            num_bits_per_copy=self.copy_bits,
        )
        tiled_copy_A = self._g2s_copy(atom_g2s, mA.element_type)
        tiled_copy_B = self._g2s_copy(atom_g2s, mB.element_type)
        op = cute.nvgpu.warp.MmaF16BF16Op(mA.element_type, cutlass.Float32, self.mma_shape)
        tiled_mma = cute.make_tiled_mma(
            op, cute.make_layout((1, 1, 1)), permutation_mnk=(16, 8 * (bN // 8), 16)
        )
        grid_m = (mA.shape[0] + bM - 1) // bM
        self.kernel(
            mA, mB, mBias, mLogits, mPacked, mCand, mCnt, route_scale,
            sA_layout, sB_layout, tiled_copy_A, tiled_copy_B, tiled_mma, SharedStorage,
        ).launch(
            grid=[grid_m, self.grid_n, self.split_k],
            block=[self.num_threads, 1, 1],
            cluster=[1, 1, self.split_k],
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.jit
    def _round(self, lane, t, keys, los, rk, rl):
        """One tournament round over per-lane lists sorted descending, heads at index 0.

        The warp's largest (key, lo) head wins; lane t records it; the winner pops its head."""
        n: cutlass.Constexpr = keys.shape[0]
        h = keys[0]
        lo = los[0]
        mh = cute.arch.warp_redux_sync(h, "umax")
        eq = cutlass.Uint32(h == mh)
        ml = cute.arch.warp_redux_sync(sel_u32(eq, lo, cutlass.Uint32(0)), "umax")
        win = eq & cutlass.Uint32(lo == ml)
        is_t = cutlass.Uint32(lane == t)
        rk = sel_u32(is_t, mh, rk)
        rl = sel_u32(is_t, ml, rl)
        for j in cutlass.range_constexpr(n - 1):
            keys[j] = sel_u32(win, keys[j + 1], keys[j])
            los[j] = sel_u32(win, los[j + 1], los[j])
        keys[n - 1] = sel_u32(win, cutlass.Uint32(0), keys[n - 1])
        los[n - 1] = sel_u32(win, cutlass.Uint32(0), los[n - 1])
        return rk, rl

    @cute.jit
    def _round_smem(self, lane, t, lists, base, ptr, ownu, rk, rl):
        """Tournament round over per-lane sorted lists in smem: head = lists[base + 2 * ptr]."""
        top_k: cutlass.Constexpr = self.top_k
        valid = ownu & cutlass.Uint32(ptr < top_k)
        at = base + cutlass.min(ptr, top_k - 1) * 2
        h = sel_u32(valid, lists[at], cutlass.Uint32(0))
        lo = sel_u32(valid, lists[at + 1], cutlass.Uint32(0))
        mh = cute.arch.warp_redux_sync(h, "umax")
        eq = cutlass.Uint32(h == mh)
        ml = cute.arch.warp_redux_sync(sel_u32(eq, lo, cutlass.Uint32(0)), "umax")
        win = eq & cutlass.Uint32(lo == ml)
        is_t = cutlass.Uint32(lane == t)
        rk = sel_u32(is_t, mh, rk)
        rl = sel_u32(is_t, ml, rl)
        return ptr + cutlass.Int32(win), rk, rl

    @cute.kernel
    def kernel(self, mA, mB, mBias, mLogits, mPacked, mCand, mCnt, route_scale: cutlass.Float32,
               sA_layout: cute.ComposedLayout, sB_layout: cute.ComposedLayout,
               tiled_copy_A: cute.TiledCopy, tiled_copy_B: cute.TiledCopy, tiled_mma: cute.TiledMma,
               shared_storage: cutlass.Constexpr):
        bM, bN, bK = self.tile_m, self.tile_n, self.tile_k
        num_stages = self.num_stages
        top_k: cutlass.Constexpr = self.top_k
        tidx, _, _ = cute.arch.thread_idx()
        bid_m, bid_n, bid_z = cute.arch.block_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        lane = cute.arch.lane_idx()
        is_dma_warp = warp_idx < self.num_dma_warps
        mma_warp_idx = warp_idx - self.num_dma_warps
        M = mA.shape[0]

        cta_tiler = (bM, bN, bK)
        coord = (bid_m, bid_n, None)
        gA = cute.local_tile(mA, tiler=cta_tiler, coord=coord, proj=(1, None, 1))
        gB = cute.local_tile(mB, tiler=cta_tiler, coord=coord, proj=(None, 1, 1))
        cA = cute.local_tile(cute.make_identity_tensor(mA.layout.shape), tiler=cta_tiler, coord=coord,
                             proj=(1, None, 1))
        cB = cute.local_tile(cute.make_identity_tensor(mB.layout.shape), tiler=cta_tiler, coord=coord,
                             proj=(None, 1, 1))
        gA = cute.make_tensor(gA.iterator.align(16), gA.layout)
        gB = cute.make_tensor(gB.iterator.align(16), gB.layout)

        smem = cutlass.utils.SmemAllocator()
        storage = shared_storage(smem.allocate(shared_storage.size_in_bytes(), byte_alignment=128))
        sA = storage.a.get_tensor(sA_layout)
        sB = storage.b.get_tensor(sB_layout)
        # partials[rank, e]: rank 0 receives every rank's CTA-reduced partial through DSMEM.
        partials = storage.partials.get_tensor(
            cute.make_layout((self.split_k, self.tile_elems), stride=(self.tile_elems, 1))
        )
        # red[w, e]: per-MMA-warp partials, aliased onto the A stages (free once the main loop is done).
        red = cute.make_tensor(
            cute.recast_ptr(storage.a.data_ptr(), dtype=cutlass.Float32),
            cute.make_layout((self.num_mma_warps, self.tile_elems), stride=(self.tile_elems, 1)),
        )
        flag = storage.flag.get_tensor(cute.make_layout((4,)))

        mainloop_pipeline = pipeline.PipelineCpAsync.create(
            barrier_storage=storage.mbar.data_ptr(),
            num_stages=num_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_dma_threads),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_mma_threads),
        )

        K_total = cute.size(mA, mode=[1])
        k_tile_count = cute.size(gA, mode=[2])
        k_start = bid_z
        num_k_tiles = cute.ceil_div(k_tile_count - k_start, self.split_k)

        if is_dma_warp:
            if const_expr(self.iket):
                _push("fill_ab")
            thr_A = tiled_copy_A.get_slice(tidx)
            thr_B = tiled_copy_B.get_slice(tidx)
            tAgA = thr_A.partition_S(gA)
            tAsA = thr_A.partition_D(sA)
            tBgB = thr_B.partition_S(gB)
            tBsB = thr_B.partition_D(sB)
            tAcA = thr_A.partition_S(cA)
            tBcB = thr_B.partition_S(cB)
            tApA_flat, tApA = self._make_pred(tAcA)
            tBpB_flat, tBpB = self._make_pred(tBcB)
            self._fill_pred(tApA_flat, tAcA, k_start, mA.shape[0], K_total)
            self._fill_pred(tBpB_flat, tBcB, k_start, mB.shape[0], K_total)
            producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, num_stages)
            mainloop_pipeline.producer_acquire(producer_state)
            cute.copy(tiled_copy_B, tBgB[None, None, None, k_start],
                      tBsB[None, None, None, producer_state.index], pred=tBpB)
            if const_expr(self.use_pdl):
                cute.arch.griddepcontrol_wait()
            cute.copy(tiled_copy_A, tAgA[None, None, None, k_start],
                      tAsA[None, None, None, producer_state.index], pred=tApA)
            mainloop_pipeline.producer_commit(producer_state)
            producer_state.advance()
            for k_tile in cutlass.range(k_start + self.split_k, k_tile_count, self.split_k, unroll=1):
                self._fill_pred(tApA_flat, tAcA, k_tile, mA.shape[0], K_total)
                self._fill_pred(tBpB_flat, tBcB, k_tile, mB.shape[0], K_total)
                mainloop_pipeline.producer_acquire(producer_state)
                cute.copy(tiled_copy_A, tAgA[None, None, None, k_tile],
                          tAsA[None, None, None, producer_state.index], pred=tApA)
                cute.copy(tiled_copy_B, tBgB[None, None, None, k_tile],
                          tBsB[None, None, None, producer_state.index], pred=tBpB)
                mainloop_pipeline.producer_commit(producer_state)
                producer_state.advance()
            mainloop_pipeline.producer_tail(producer_state)
            if const_expr(self.iket):
                _pop()
        else:
            if const_expr(self.iket):
                _push("main_k_loop")
            if const_expr(self.use_pdl):
                cute.arch.griddepcontrol_wait()
            num_mma_warps: cutlass.Constexpr = self.num_mma_warps
            thr_mma = tiled_mma.get_slice(lane)
            tCsA = thr_mma.partition_A(sA)
            tCsB = thr_mma.partition_B(sB)
            red_warp = cute.make_tensor(
                cute.domain_offset((mma_warp_idx, 0), red).iterator,
                cute.make_layout((bM, bN), stride=(bN, 1)),
            )
            tCsC = thr_mma.partition_C(red_warp)
            tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
            tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
            tCrC = tiled_mma.make_fragment_C(tCsC)
            tCrC.fill(0.0)
            atom_s2r = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), mA.element_type)
            tiled_s2r_A = cute.make_tiled_copy_A(atom_s2r, tiled_mma)
            tiled_s2r_B = cute.make_tiled_copy_B(atom_s2r, tiled_mma)
            thr_s2r_A = tiled_s2r_A.get_slice(lane)
            thr_s2r_B = tiled_s2r_B.get_slice(lane)
            tCsA_v = thr_s2r_A.partition_S(sA)
            tCrA_v = thr_s2r_A.retile(tCrA)
            tCsB_v = thr_s2r_B.partition_S(sB)
            tCrB_v = thr_s2r_B.retile(tCrB)
            tCsA_w = cute.logical_divide(tCsA_v, (None, None, num_mma_warps, None))
            tCsB_w = cute.logical_divide(tCsB_v, (None, None, num_mma_warps, None))
            num_k_blocks = cute.size(tCrA, mode=[2])
            kb_per_warp: cutlass.Constexpr = num_k_blocks // num_mma_warps
            consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, num_stages)
            for _ in cutlass.range(num_k_tiles, unroll=1):
                mainloop_pipeline.consumer_wait(consumer_state)
                for ki in cutlass.range_constexpr(kb_per_warp):
                    cute.copy(tiled_s2r_A, tCsA_w[None, None, (mma_warp_idx, ki), consumer_state.index],
                              tCrA_v[None, None, 0])
                    cute.copy(tiled_s2r_B, tCsB_w[None, None, (mma_warp_idx, ki), consumer_state.index],
                              tCrB_v[None, None, 0])
                    cute.gemm(tiled_mma, tCrC, tCrA[None, None, 0], tCrB[None, None, 0], tCrC)
                mainloop_pipeline.consumer_release(consumer_state)
                consumer_state.advance()
            # every MMA warp is done reading sA before any overwrites it with partials
            cute.arch.barrier(barrier_id=1, number_of_threads=self.num_mma_threads)
            cute.autovec_copy(tCrC, tCsC)
            if const_expr(self.iket):
                _pop()

        # ---- common epilogue: all 256 threads ----
        if const_expr(self.iket):
            _push("sync_cluster")
        cute.arch.sync_threads()
        cta_rank = cute.arch.block_idx_in_cluster()
        for i in cutlass.range_constexpr(self.elems_per_thread):
            e = tidx + i * self.num_threads
            total = red[0, e] + red[1, e]
            total = total + red[2, e]
            total = total + red[3, e]
            slot = cute.domain_offset((cta_rank, e), partials).iterator
            st_shared_remote_f32(set_block_rank(slot, cutlass.Int32(0)), total)
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        if const_expr(self.iket):
            _pop()

        if cta_rank == 0:
            if const_expr(self.iket):
                _push("edge_logits")
            row0 = bid_m * bM
            col0 = bid_n * bN
            # Reduce split-K partials (rank order) into red[0, :] and the fp32 logits output.
            for i in cutlass.range_constexpr(self.elems_per_thread):
                e = tidx + i * self.num_threads
                acc = partials[0, e]
                for r in cutlass.range_constexpr(1, self.split_k):
                    acc = acc + partials[r, e]
                red[0, e] = acc
                row = row0 + e // bN
                if row < M:
                    mLogits[row, col0 + e % bN] = acc
            cute.arch.sync_threads()
            if const_expr(self.iket):
                _pop()
        if cta_rank == 0 and not const_expr(self.gemm_only):
            if const_expr(self.iket):
                _push("edge_tile_topk")
            row0 = bid_m * bM
            col0 = bid_n * bN
            # Local top-K of this 16 x tile_n tile: one warp, two rows (A, B) interleaved; each lane
            # sorts its cols_per_lane columns, then top_k tournament rounds; lane t writes winner t.
            cpl: cutlass.Constexpr = self.cols_per_lane
            r_a = warp_idx * 2
            kA = cute.make_rmem_tensor(cute.make_layout((cpl,)), cutlass.Uint32)
            lA = cute.make_rmem_tensor(cute.make_layout((cpl,)), cutlass.Uint32)
            kB = cute.make_rmem_tensor(cute.make_layout((cpl,)), cutlass.Uint32)
            lB = cute.make_rmem_tensor(cute.make_layout((cpl,)), cutlass.Uint32)
            for j in cutlass.range_constexpr(cpl):
                c = lane * cpl + j
                b = mBias[col0 + c]
                kA[j] = order_key(add_ftz(sigmoid_fast(red[0, r_a * bN + c]), b))
                kB[j] = order_key(add_ftz(sigmoid_fast(red[0, (r_a + 1) * bN + c]), b))
                lA[j] = cutlass.Uint32(65535 - (col0 + c))
                lB[j] = lA[j]
            # insertion sort, descending by (key, lo): cols_per_lane is small
            for i in cutlass.range_constexpr(1, cpl):
                for j in cutlass.range_constexpr(i, 0, -1):
                    swA = cutlass.Uint32((kA[j] > kA[j - 1]) | ((kA[j] == kA[j - 1]) & (lA[j] > lA[j - 1])))
                    tk = kA[j]
                    tl = lA[j]
                    kA[j] = sel_u32(swA, kA[j - 1], tk)
                    lA[j] = sel_u32(swA, lA[j - 1], tl)
                    kA[j - 1] = sel_u32(swA, tk, kA[j - 1])
                    lA[j - 1] = sel_u32(swA, tl, lA[j - 1])
                    swB = cutlass.Uint32((kB[j] > kB[j - 1]) | ((kB[j] == kB[j - 1]) & (lB[j] > lB[j - 1])))
                    tk = kB[j]
                    tl = lB[j]
                    kB[j] = sel_u32(swB, kB[j - 1], tk)
                    lB[j] = sel_u32(swB, lB[j - 1], tl)
                    kB[j - 1] = sel_u32(swB, tk, kB[j - 1])
                    lB[j - 1] = sel_u32(swB, tl, lB[j - 1])
            rkA = cutlass.Uint32(0)
            rlA = cutlass.Uint32(0)
            rkB = cutlass.Uint32(0)
            rlB = cutlass.Uint32(0)
            for t in cutlass.range_constexpr(top_k):
                rkA, rlA = self._round(lane, t, kA, lA, rkA, rlA)
                rkB, rlB = self._round(lane, t, kB, lB, rkB, rlB)
            if lane < top_k:
                base = (bid_n * top_k + lane) * 2
                if row0 + r_a < M:
                    mCand[row0 + r_a, base + 0] = rkA
                    mCand[row0 + r_a, base + 1] = rlA
                if row0 + r_a + 1 < M:
                    mCand[row0 + r_a + 1, base + 0] = rkB
                    mCand[row0 + r_a + 1, base + 1] = rlB
            if const_expr(self.iket):
                _pop()

            # Publish: CTA barrier, then one acq_rel arrival on this row group's counter (cumulative release
            # of every thread's candidate and logits stores, ordered before it by the barrier).
            if const_expr(self.iket):
                _push("sync_counter")
            cute.arch.sync_threads()
            if tidx == 0:
                old = cute.arch.atomic_add(mCnt.iterator + bid_m, cutlass.Int32(1),
                                           sem="acq_rel", scope="gpu")
                flag[0] = cutlass.Int32(0)
                if old == self.grid_n - 1:
                    flag[0] = cutlass.Int32(1)
            cute.arch.sync_threads()
            is_last = flag[0] == 1
            if const_expr(self.iket):
                _pop()

            if is_last:
                if const_expr(self.iket):
                    _push("main_topk")
                cute.arch.fence_acq_rel_gpu()
                # Stage the 16 rows' candidate lists in smem (the B stages are free), then merge:
                # lane l < grid_n owns tile l's sorted list; top_k tournament rounds per row, rows A and B
                # interleaved; the winner advances its list pointer.
                cw: cutlass.Constexpr = self.cand_width
                lists = cute.make_tensor(
                    cute.recast_ptr(storage.b.data_ptr(), dtype=cutlass.Uint32),
                    cute.make_layout((bM * cw,)),
                )
                for i in cutlass.range_constexpr(bM * cw // self.num_threads):
                    flat = tidx + i * self.num_threads
                    lists[flat] = mCand[cutlass.min(row0 + flat // cw, M - 1), flat % cw]
                cute.arch.sync_threads()
                rowA = row0 + r_a
                rowB = rowA + 1
                ldA = cutlass.min(rowA, M - 1)
                ldB = cutlass.min(rowB, M - 1)
                ownu = cutlass.Uint32(lane < self.grid_n)
                lst = cutlass.min(lane, self.grid_n - 1)
                baseA = r_a * cw + lst * top_k * 2
                baseB = baseA + cw
                pA = cutlass.Int32(0)
                pB = cutlass.Int32(0)
                rkA = cutlass.Uint32(0)
                rlA = cutlass.Uint32(0)
                rkB = cutlass.Uint32(0)
                rlB = cutlass.Uint32(0)
                for t in cutlass.range_constexpr(top_k):
                    pA, rkA, rlA = self._round_smem(lane, t, lists, baseA, pA, ownu, rkA, rlA)
                    pB, rkB, rlB = self._round_smem(lane, t, lists, baseB, pB, ownu, rkB, rlB)
                # ScaledSumNormalize: un-biased sigmoid of each winner (from the published logits), butterfly sum.
                in_k = lane < top_k
                idxA = cutlass.Int32(65535) - cutlass.Int32(rlA)
                idxB = cutlass.Int32(65535) - cutlass.Int32(rlB)
                sA_ = cutlass.Float32(0.0)
                sB_ = cutlass.Float32(0.0)
                if in_k:
                    sA_ = sigmoid_fast(mLogits[ldA, idxA])
                    sB_ = sigmoid_fast(mLogits[ldB, idxB])
                vA = sA_
                vB = sB_
                for off in cutlass.range_constexpr(5):
                    vA = add_ftz(vA, cute.arch.shuffle_sync_bfly(vA, 16 >> off))
                    vB = add_ftz(vB, cute.arch.shuffle_sync_bfly(vB, 16 >> off))
                wA = scale_norm(sA_, route_scale, vA, cutlass.Float32(SUM_EPS))
                wB = scale_norm(sB_, route_scale, vB, cutlass.Float32(SUM_EPS))
                if in_k:
                    if rowA < M:
                        mPacked[rowA, lane] = pack_id_weight(idxA, wA)
                    if rowB < M:
                        mPacked[rowB, lane] = pack_id_weight(idxB, wB)
                if tidx == 0:
                    mCnt[bid_m] = cutlass.Int32(0)
                if const_expr(self.iket):
                    _pop()
