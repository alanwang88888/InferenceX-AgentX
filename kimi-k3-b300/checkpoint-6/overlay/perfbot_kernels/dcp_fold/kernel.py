# SPDX-License-Identifier: Apache-2.0
"""Direct-DCP q-gather and output A2A with the epoch and signal launches folded in (expedition line D). CuTe DSL 4.6.2.

Image source: vllm @65b7662 csrc/libtorch_stable/attention/dcp_utils/ (dcp_direct_q_gather.cu, dcp_direct_a2a_lse_reduce.cu,
dcp_direct_common.cuh). Same symmetric buffers, signal slots, epoch counters and values as the image ops, so image and
candidate calls interleave on one workspace (also rank by rank).

  QGather   image: increment_epoch (grid > 1) + direct_dcp_q_gather_multimem. Here one launch: every CTA multicasts
            its tokens, fence.sc.sys, arrives on the workspace's completion counter; the last CTA resets it, writes
            epoch e+1, multicasts signal e+1 to slot (e+1)&1, and lanes 0..7 of its warp 0 poll the 8 received
            signals in parallel (the image polls them in series from one thread).
  Dispatch  image: increment_epoch + dispatch_output_lse + signal_kernel. Here one launch: every CTA reads e and uses
            e+1, stores its partial output / LSE slice into the destination's symmetric slot (unchanged data path),
            fence.sc.sys, arrives on a local counter; the last CTA resets it, writes epoch e+1, and lanes 0..7 issue
            the 8 st.release.sys signal stores.
  Combine   image: wait_lse_combine (the image op launches it; it cannot be launched alone). Here one warp per
            (token, head): lanes 0..7 poll the 8 signals in parallel, shfl the LSEs, every lane forms the weights with
            the image's operation sequence (ex2.approx.f32 non-ftz, rcp.rn, add/mul/fma .rn, rank order), then 16 B
            loads and fma in rank order, skipping zero weights, cvt.rn.bf16 -> bit-equal output.
Every kernel starts with griddepcontrol.wait when launched with PDL (no memory access before it).
"""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

W = 8  # DCP world size (decline otherwise)
TIMEOUT_NS = 20_000_000_000  # poll bound, then trap (image: 1e8 spins, then trap)
NEG_INF, L2E = 0xFF800000, 0x3FB8AA3B  # fp32 bits: -inf, CUDART_L2E_F


def _asm(ret, args, body, cons, loc, ip, side=True):
    return llvm.inline_asm(ret, args, body, cons, has_side_effects=side, is_align_stack=False,
                           asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


def _v(a, loc, ip):
    return a.ir_value(loc=loc, ip=ip)


def _u4(st, loc, ip):
    return [Uint32(llvm.extractvalue(T.i32(), st, [i], loc=loc, ip=ip)) for i in range(4)]


# ---------------------------------------------------------------- memory


@dsl_user_op
def ld_v4(addr: Int64, *, loc=None, ip=None):
    st = _asm(llvm.StructType.get_literal([T.i32()] * 4), [_v(Int64(addr), loc, ip)],
              "ld.global.v4.u32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,l", loc, ip)
    return _u4(st, loc, ip)


@dsl_user_op
def st_v4(addr: Int64, a: Uint32, b: Uint32, c: Uint32, d: Uint32, *, loc=None, ip=None):
    _asm(None, [_v(Int64(addr), loc, ip)] + [_v(Uint32(x), loc, ip) for x in (a, b, c, d)],
         "st.global.v4.u32 [$0], {$1, $2, $3, $4};", "l,r,r,r,r", loc, ip)


@dsl_user_op
def mc_st_v4(addr: Int64, a: Uint32, b: Uint32, c: Uint32, d: Uint32, *, loc=None, ip=None):
    """Image multimem_store_16."""
    _asm(None, [_v(Int64(addr), loc, ip)] + [_v(Uint32(x), loc, ip) for x in (a, b, c, d)],
         "multimem.st.relaxed.sys.global.v4.f32 [$0], {$1, $2, $3, $4};", "l,r,r,r,r", loc, ip)


@dsl_user_op
def mc_st_release(addr: Int64, v: Uint32, *, loc=None, ip=None):
    """Image multimem_store_release_system."""
    _asm(None, [_v(Int64(addr), loc, ip), _v(Uint32(v), loc, ip)], "multimem.st.release.sys.global.u32 [$0], $1;",
         "l,r", loc, ip)


@dsl_user_op
def st_relaxed_sys(addr: Int64, v: Uint32, *, loc=None, ip=None):
    """Strong sys-scope store; after fence.acq_rel.sys it forms a release pattern (PTX 8.8)."""
    _asm(None, [_v(Int64(addr), loc, ip), _v(Uint32(v), loc, ip)], "st.relaxed.sys.global.u32 [$0], $1;", "l,r",
         loc, ip)


@dsl_user_op
def ld_acquire_sys(addr: Int64, *, loc=None, ip=None) -> Uint32:
    """Image load_acquire_system."""
    return Uint32(_asm(T.i32(), [_v(Int64(addr), loc, ip)], "ld.acquire.sys.global.u32 $0, [$1];", "=r,l", loc, ip))


@dsl_user_op
def ld_u64(addr: Int64, *, loc=None, ip=None) -> Int64:
    return Int64(_asm(T.i64(), [_v(Int64(addr), loc, ip)], "ld.relaxed.gpu.global.u64 $0, [$1];", "=l,l", loc, ip))


@dsl_user_op
def st_u64(addr: Int64, v: Int64, *, loc=None, ip=None):
    _asm(None, [_v(Int64(addr), loc, ip), _v(Int64(v), loc, ip)], "st.relaxed.gpu.global.u64 [$0], $1;", "l,l",
         loc, ip)


@dsl_user_op
def st_u32(addr: Int64, v: Uint32, *, loc=None, ip=None):
    _asm(None, [_v(Int64(addr), loc, ip), _v(Uint32(v), loc, ip)], "st.relaxed.gpu.global.u32 [$0], $1;", "l,r",
         loc, ip)


@dsl_user_op
def ld_s32(addr: Int64, *, loc=None, ip=None) -> Int32:
    return Int32(_asm(T.i32(), [_v(Int64(addr), loc, ip)], "ld.global.s32 $0, [$1];", "=r,l", loc, ip))


@dsl_user_op
def ld_b32(addr: Int64, *, loc=None, ip=None) -> Uint32:
    return Uint32(_asm(T.i32(), [_v(Int64(addr), loc, ip)], "ld.global.b32 $0, [$1];", "=r,l", loc, ip))


@dsl_user_op
def st_b32(addr: Int64, v: Uint32, *, loc=None, ip=None):
    _asm(None, [_v(Int64(addr), loc, ip), _v(Uint32(v), loc, ip)], "st.global.b32 [$0], $1;", "l,r", loc, ip)


@dsl_user_op
def atom_add(addr: Int64, v: Uint32, *, loc=None, ip=None) -> Uint32:
    """Image atomicAdd (relaxed, gpu scope)."""
    return Uint32(_asm(T.i32(), [_v(Int64(addr), loc, ip), _v(Uint32(v), loc, ip)],
                       "atom.add.relaxed.gpu.global.u32 $0, [$1], $2;", "=r,l,r", loc, ip))


@dsl_user_op
def fence_sc_sys(*, loc=None, ip=None):
    """Image __threadfence_system."""
    _asm(None, [], "fence.sc.sys;", "", loc, ip)


@dsl_user_op
def fence_acq_rel_sys(*, loc=None, ip=None):
    _asm(None, [], "fence.acq_rel.sys;", "", loc, ip)


@dsl_user_op
def shfl(v: Uint32, src: Int32, *, loc=None, ip=None) -> Uint32:
    return Uint32(_asm(T.i32(), [_v(Uint32(v), loc, ip), _v(Int32(src), loc, ip)],
                       "shfl.sync.idx.b32 $0, $1, $2, 0x1f, 0xffffffff;", "=r,r,r", loc, ip))


@dsl_user_op
def lo32(x: Int64, *, loc=None, ip=None) -> Uint32:
    return Uint32(_asm(T.i32(), [_v(Int64(x), loc, ip)], "cvt.u32.u64 $0, $1;", "=r,l", loc, ip, side=False))


@dsl_user_op
def globaltimer(*, loc=None, ip=None) -> Int64:
    return Int64(_asm(T.i64(), [], "mov.u64 $0, %globaltimer;", "=l", loc, ip))


@dsl_user_op
def trap(*, loc=None, ip=None):
    _asm(None, [], "trap;", "", loc, ip)


@cute.jit
def poll(addr: Int64, want: Uint32):
    """Spin with ld.acquire.sys until the word equals `want` (image wait_for_epoch); trap after TIMEOUT_NS."""
    t0 = globaltimer()
    got = ld_acquire_sys(addr)
    while got != want:
        if globaltimer() - t0 > Int64(TIMEOUT_NS):
            trap()
        got = ld_acquire_sys(addr)


# ---------------------------------------------------------------- fp32 math as the image's nvcc build emits it


def _f1(body):
    @dsl_user_op
    def op(a: Uint32, *, loc=None, ip=None) -> Uint32:
        return Uint32(_asm(T.i32(), [_v(Uint32(a), loc, ip)], body, "=r,r", loc, ip, side=False))
    return op


def _f2(body):
    @dsl_user_op
    def op(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
        return Uint32(_asm(T.i32(), [_v(Uint32(a), loc, ip), _v(Uint32(b), loc, ip)], body, "=r,r,r", loc, ip,
                           side=False))
    return op


def _fblk(ins):
    return "{ .reg .f32 a, b, c, d; " + ins + " }"


# fp32 values travel as their bit pattern (Uint32) so the DSL never rewrites the arithmetic.
ex2 = _f1(_fblk("mov.b32 a, $1; ex2.approx.f32 d, a; mov.b32 $0, d;"))  # CUDA exp2f (non-ftz)
rcp = _f1(_fblk("mov.b32 a, $1; rcp.rn.f32 d, a; mov.b32 $0, d;"))  # 1.0f / x
add = _f2(_fblk("mov.b32 a, $1; mov.b32 b, $2; add.rn.f32 d, a, b; mov.b32 $0, d;"))
sub = _f2(_fblk("mov.b32 a, $1; mov.b32 b, $2; sub.rn.f32 d, a, b; mov.b32 $0, d;"))
mul = _f2(_fblk("mov.b32 a, $1; mov.b32 b, $2; mul.rn.f32 d, a, b; mov.b32 $0, d;"))
fmax = _f2(_fblk("mov.b32 a, $1; mov.b32 b, $2; max.f32 d, a, b; mov.b32 $0, d;"))
# lse sanitize: NaN or +inf -> -inf (image: isnan(v) || v == CUDART_INF_F)
clean = _f1("{ .reg .f32 a; .reg .pred p, q; mov.b32 a, $1; setp.nan.f32 p, a, a; setp.eq.f32 q, a, 0f7F800000;"
            " or.pred p, p, q; selp.b32 $0, 0xFF800000, $1, p; }")
# x == 0.0f (also -0) / x > 0.0f / x == -inf as 0/1
is_zero = _f1("{ .reg .f32 a; .reg .pred p; mov.b32 a, $1; setp.eq.f32 p, a, 0f00000000; selp.u32 $0, 1, 0, p; }")
is_pos = _f1("{ .reg .f32 a; .reg .pred p; mov.b32 a, $1; setp.gt.f32 p, a, 0f00000000; selp.u32 $0, 1, 0, p; }")


@dsl_user_op
def fma_bf16(acc: Uint32, word: Uint32, hi: int, w: Uint32, *, loc=None, ip=None) -> Uint32:
    """acc + bf16(word half) * w, one rounding (image: accumulator += to_float(x) * weight, contracted)."""
    get = "and.b32 t, $2, 0xFFFF0000;" if hi else "shl.b32 t, $2, 16;"
    return Uint32(_asm(T.i32(), [_v(Uint32(acc), loc, ip), _v(Uint32(word), loc, ip), _v(Uint32(w), loc, ip)],
                       "{ .reg .b32 t; .reg .f32 a, x, y, d; " + get +
                       " mov.b32 a, $1; mov.b32 x, t; mov.b32 y, $3; fma.rn.f32 d, x, y, a; mov.b32 $0, d; }",
                       "=r,r,r,r", loc, ip, side=False))


@dsl_user_op
def pack_bf16x2(lo: Uint32, hi: Uint32, *, loc=None, ip=None) -> Uint32:
    """__float2bfloat16_rn of two fp32 (no -0 sanitize: the image has none)."""
    return Uint32(_asm(T.i32(), [_v(Uint32(lo), loc, ip), _v(Uint32(hi), loc, ip)],
                       "{ .reg .f32 l, h; mov.b32 l, $1; mov.b32 h, $2; cvt.rn.bf16x2.f32 $0, h, l; }", "=r,r,r",
                       loc, ip, side=False))


# ---------------------------------------------------------------- kernels


class _Fold:
    """Shared part of the two publishing kernels: PDL entry, and the grid-wide arrival that replaces the image's
    increment_epoch launch (and, in Dispatch, the kernel boundary before signal_kernel). Every CTA reads epoch e at
    entry and uses e+1; after its stores it publishes (fence.sc.sys + bar.sync, or fence1: bar.sync + one fence by
    thread 0, cumulative), arrives on `ctr`; the last CTA resets `ctr`, takes the acquire side
    (fence.acq_rel.sys) and writes e+1. arrive() returns 1 in warp 0 of the last CTA, else 0. Every signal store
    after it is issued by that same thread 0 (README.md: the memory-model check)."""

    THREADS = 256

    def __init__(self, pdl: bool = True, fence1: bool = False, tpb: int = 1):
        self.pdl, self.fence1, self.tpb = pdl, fence1, tpb

    @cute.jit
    def entry(self):
        if const_expr(self.pdl):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()

    @cute.jit
    def arrive(self, tidx: Int32, ctr: Int64, epoch: Int64, e0: Int64, blocks: Int32) -> Uint32:
        if const_expr(self.fence1):
            cute.arch.barrier()
            if tidx == 0:
                fence_sc_sys()
        else:
            fence_sc_sys()
            cute.arch.barrier()
        last = Uint32(0)
        if tidx < 32:
            if tidx == 0:
                if atom_add(ctr, Uint32(1)) + Uint32(1) == Uint32(blocks):
                    st_u32(ctr, Uint32(0))
                    fence_acq_rel_sys()  # acquire of all arrivals + release fence of this thread's sys stores
                    st_u64(epoch, e0 + Int64(1))
                    last = Uint32(1)
            last = shfl(last, Int32(0))
            cute.arch.sync_warp()
        return last


class QGather(_Fold):
    """Grid `blocks` (image: min(T, W)), 256 threads. Token t of this rank -> head slice `rank` of token t of every
    rank's final query (NVLS multicast); the last CTA multicasts signal e+1 into slot (e+1)&1 and its lanes 0..7 poll
    the 8 received signals. ctr = the workspace's completion counter (the image's)."""

    @cute.jit
    def __call__(self, src: Int64, src_stride: Int64, mc_q: Int64, dst_stride: Int64, mc_sig: Int64, rsig: Int64,
                 epoch: Int64, ctr: Int64, rank: Int32, t: Int32, items: Int32, blocks: Int32, stream):
        self.kernel(src, src_stride, mc_q, dst_stride, mc_sig, rsig, epoch, ctr, rank, t, items, blocks).launch(
            grid=[blocks, 1, 1], block=[self.THREADS, 1, 1], stream=stream, use_pdl=self.pdl)

    @cute.kernel
    def kernel(self, src: Int64, src_stride: Int64, mc_q: Int64, dst_stride: Int64, mc_sig: Int64, rsig: Int64,
               epoch: Int64, ctr: Int64, rank: Int32, t: Int32, items: Int32, blocks: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        self.entry()
        e0 = ld_u64(epoch)
        ep = lo32(e0) + Uint32(1)
        par = Int64(ep & Uint32(1)) * Int64(W)
        head = Int64(rank) * Int64(items) * 16
        for tok in cutlass.range(bx, t, blocks, unroll=1):
            s0 = src + Int64(tok) * src_stride
            d0 = mc_q + Int64(tok) * dst_stride + head
            for it in cutlass.range(tidx, items, self.THREADS, unroll=1):
                a, b, c, d = ld_v4(s0 + Int64(it) * 16)
                mc_st_v4(d0 + Int64(it) * 16, a, b, c, d)
        if self.arrive(tidx, ctr, epoch, e0, blocks) != Uint32(0):
            if tidx == 0:
                mc_st_release(mc_sig + (par + Int64(rank)) * 4, ep)
            if tidx < W:
                poll(rsig + (par + Int64(tidx)) * 4, ep)


class Dispatch(_Fold):
    """Grid W x ceil(T / tpb) (image: W x T, tpb 1), 256 threads. CTA (dest, chunk): for its tokens, this rank's
    partial output heads of `dest` -> slot (parity, rank, token) of dest's symmetric buffer, LSE likewise (-inf and no
    output for an empty KV shard); thread 0 of the last CTA stores the 8 signals. ctr: a local counter.
    out_stride in bytes, lse_stride in elements (fp32 LSE only)."""

    @cute.jit
    def __call__(self, pout: Int64, out_stride: Int64, plse: Int64, lse_stride: Int64, seq: Int64, qsl: Int64,
                 nseq: Int32, peer_out: Int64, peer_lse: Int64, peer_sig: Int64, epoch: Int64, ctr: Int64,
                 rank: Int32, t: Int32, maxt: Int32, hloc: Int32, dh: Int32, blocks: Int32, stream):
        self.kernel(pout, out_stride, plse, lse_stride, seq, qsl, nseq, peer_out, peer_lse, peer_sig, epoch, ctr,
                    rank, t, maxt, hloc, dh, blocks).launch(grid=[blocks, 1, 1], block=[self.THREADS, 1, 1],
                                                            stream=stream, use_pdl=self.pdl)

    @cute.kernel
    def kernel(self, pout: Int64, out_stride: Int64, plse: Int64, lse_stride: Int64, seq: Int64, qsl: Int64,
               nseq: Int32, peer_out: Int64, peer_lse: Int64, peer_sig: Int64, epoch: Int64, ctr: Int64,
               rank: Int32, t: Int32, maxt: Int32, hloc: Int32, dh: Int32, blocks: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        self.entry()
        nchunk = blocks // W
        dest = bx // nchunk
        t_lo = (bx - dest * nchunk) * self.tpb
        t_hi = cutlass.min(t, t_lo + self.tpb)
        e0 = ld_u64(epoch)
        ep = lo32(e0) + Uint32(1)
        par = Int64(ep & Uint32(1)) * Int64(W)
        po = ld_u64(peer_out + Int64(dest) * 8)
        pl = ld_u64(peer_lse + Int64(dest) * 8)
        vecs = hloc * dh // 8
        head = Int64(dest) * Int64(hloc)
        for tok in cutlass.range(t_lo, t_hi, 1, unroll=1):
            empty = Int32(0)
            if seq != Int64(0):  # image find_sequence: the last seq with query_start_loc[seq] <= tok
                lo = Int32(0)
                hi = nseq
                while lo < hi:
                    mid = (lo + hi) // 2
                    if ld_s32(qsl + Int64(mid) * 4) <= tok:
                        lo = mid + 1
                    else:
                        hi = mid
                if ld_s32(seq + Int64(lo - 1) * 4) == 0:
                    empty = Int32(1)
            ditem = ((par + Int64(rank)) * Int64(maxt) + Int64(tok)) * Int64(hloc)
            if empty == 0:
                s0 = pout + Int64(tok) * out_stride + head * Int64(dh) * 2
                d0 = po + ditem * Int64(dh) * 2
                for v in cutlass.range(tidx, vecs, self.THREADS, unroll=1):
                    a, b, c, d = ld_v4(s0 + Int64(v) * 16)
                    st_v4(d0 + Int64(v) * 16, a, b, c, d)
            for h in cutlass.range(tidx, hloc, self.THREADS, unroll=1):
                val = Uint32(NEG_INF)
                if empty == 0:
                    val = ld_b32(plse + (Int64(tok) * lse_stride + head + Int64(h)) * 4)
                st_b32(pl + (ditem + Int64(h)) * 4, val)
        if self.arrive(tidx, ctr, epoch, e0, blocks) != Uint32(0):
            if tidx == 0:  # same thread as the acquire + release fence: 8 strong sys stores, no warp-barrier edge
                for r in cutlass.range_constexpr(W):
                    st_relaxed_sys(ld_u64(peer_sig + Int64(r * 8)) + (par + Int64(rank)) * 4, ep)


class Combine(_Fold):
    """Grid ceil(T * hloc / wpb), 32 * wpb threads; warp = item (token, head), lane = 8 * VEC output channels
    (VEC = dh / 256, channel (v * 32 + lane) * 8 + k). Output [T, hloc, dh] bf16, contiguous."""

    def __init__(self, dh: int = 512, wpb: int = 4, pdl: bool = True):
        assert dh % 256 == 0
        super().__init__(pdl=pdl)
        self.dh, self.vec, self.wpb = dh, dh // 256, wpb

    @cute.jit
    def __call__(self, rout: Int64, rlse: Int64, rsig: Int64, epoch: Int64, out: Int64, maxt: Int32, hloc: Int32,
                 base_e: Int32, items: Int32, blocks: Int32, stream):
        self.kernel(rout, rlse, rsig, epoch, out, maxt, hloc, base_e, items).launch(
            grid=[blocks, 1, 1], block=[32 * self.wpb, 1, 1], stream=stream, use_pdl=self.pdl)

    @cute.kernel
    def kernel(self, rout: Int64, rlse: Int64, rsig: Int64, epoch: Int64, out: Int64, maxt: Int32, hloc: Int32,
               base_e: Int32, items: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        self.entry()
        lane = tidx % 32
        item = bx * self.wpb + tidx // 32
        ok = item < items
        tok = item // hloc
        hd = item - tok * hloc
        ep = lo32(ld_u64(epoch))
        par = Int64(ep & Uint32(1)) * Int64(W)
        lv = Uint32(NEG_INF)
        if ok:
            if lane < W:
                poll(rsig + (par + Int64(lane)) * 4, ep)
                x = clean(ld_b32(rlse + (((par + Int64(lane)) * Int64(maxt) + Int64(tok)) * Int64(hloc) +
                                         Int64(hd)) * 4))
                if base_e != 0:
                    x = mul(x, Uint32(L2E))
                lv = x
        cute.arch.barrier()  # bar.sync: lane r's acquire -> every thread's reads of rank r (PTX 8.9.4)
        if ok:
            wt = cute.make_rmem_tensor(cute.make_layout(W), Uint32)
            for r in cutlass.range_constexpr(W):
                wt[r] = shfl(lv, Int32(r))
            mx = Uint32(NEG_INF)
            for r in cutlass.range_constexpr(W):
                mx = fmax(mx, wt[r])
            if mx == Uint32(NEG_INF):
                mx = Uint32(0)
            sm = Uint32(0)
            for r in cutlass.range_constexpr(W):
                sm = add(sm, ex2(sub(wt[r], mx)))
            inv = Uint32(0)
            if is_pos(sm) != Uint32(0):
                inv = rcp(sm)
            for r in cutlass.range_constexpr(W):
                wt[r] = mul(ex2(sub(wt[r], mx)), inv)
            # all 8 x VEC loads in flight, then fma in rank order (a zero weight skips its fma, as the image)
            words = cute.make_rmem_tensor(cute.make_layout((W, self.vec, 4), stride=(4 * self.vec, 4, 1)), Uint32)
            for r in cutlass.range_constexpr(W):
                row = rout + (((par + Int64(r)) * Int64(maxt) + Int64(tok)) * Int64(hloc) + Int64(hd)) * (self.dh * 2)
                for v in cutlass.range_constexpr(self.vec):
                    a, b, c, d = ld_v4(row + Int64((v * 32 + lane) * 16))
                    words[r, v, 0] = a
                    words[r, v, 1] = b
                    words[r, v, 2] = c
                    words[r, v, 3] = d
            acc = cute.make_rmem_tensor(cute.make_layout(8 * self.vec), Uint32)
            for k in cutlass.range_constexpr(8 * self.vec):
                acc[k] = Uint32(0)
            for r in cutlass.range_constexpr(W):
                if is_zero(wt[r]) == Uint32(0):
                    for v in cutlass.range_constexpr(self.vec):
                        for q in cutlass.range_constexpr(4):
                            acc[v * 8 + 2 * q] = fma_bf16(acc[v * 8 + 2 * q], words[r, v, q], 0, wt[r])
                            acc[v * 8 + 2 * q + 1] = fma_bf16(acc[v * 8 + 2 * q + 1], words[r, v, q], 1, wt[r])
            obase = out + Int64(item) * (self.dh * 2)
            for v in cutlass.range_constexpr(self.vec):
                st_v4(obase + Int64((v * 32 + lane) * 16), pack_bf16x2(acc[v * 8], acc[v * 8 + 1]),
                      pack_bf16x2(acc[v * 8 + 2], acc[v * 8 + 3]), pack_bf16x2(acc[v * 8 + 4], acc[v * 8 + 5]),
                      pack_bf16x2(acc[v * 8 + 6], acc[v * 8 + 7]))
