# SPDX-License-Identifier: Apache-2.0
"""K3 latent-MoE tail in one launch (expedition line B, docs/expedition/PLAN-B-tail.md). CuTe DSL 4.6.2, sm_100 family.

Replaces the image's three launches (vllm/models/kimi_k3/nvidia/ops/cute_dsl/latent_moe_tail: routed all-reduce +
RMSNorm + shared reduce-scatter, fused-add multicast up GEMM, Lamport copy) for the r3-318 `prefilled` routed source:
fc2 already stored this rank's bf16 latent rows [M, 3584] into slot (token, rank) of every rank's routed Lamport
buffer (generation flags[0]). Same buffers, flags, sentinels and generation rotation as the image op, so image and
candidate calls interleave on one op.

  out[M, 7168] = all_gather_r( RMSNorm(sum_r routed_r) @ W_up[r*896:(r+1)*896]^T + sum_q shared_q[:, r*896:...] )

Grid 7 x 14 = 98 CTAs, cluster (7, 1, 1), 256 threads. CTA (j, s): n-tile j = rows [128 j, 128 j + 128) of this
rank's up shard, k-slice s = latent channels [256 s, 256 s + 256).
  0  before griddepcontrol.wait: cp.async the 128 x 256 weight tile (UMMA K-major SW128 layout), the lane's
     gamma words (weights only), mbarrier init, TMEM alloc. After the wait: launch_dependents (all 98 CTAs are resident by then).
  1  one batch of loads: flags of both Lamport buffers, the shared-send fragments.
  2  routed all-reduce, warp = tokens t and t + 56 (both polls in flight), lane = 8 channels of the k-slice: fp32
     sum in rank order (= image), x = bf16(sum), sumsq (bf16 squares, warp tree), y = bf16(x * gamma) -> DSMEM
     row of all 7 CTAs. Cluster barrier, then the shared partial -> owner ranks (unicast).
  3  one thread issues 16 tcgen05.mma 128 x 64 x 16 (TMEM accumulator); tcgen05.ld -> fp32 partial [t][row] and
     sumsq in L2, -0 stored as +0.
  5  epilogue, warp = token, lane = 4 rows: poll the n-tile's 14 partials and sums until no word is fp32 -0 (the
     workspace starts all -0; the reader resets it), sum in k order, * rsqrt(sumsq / 3584 + eps), + shared sum
     (poll 8 slots, fp32 rank order, bf16 like the image), one bf16 rounding, -0 -> +0; the 256 B row segment is
     staged in smem and sent by one cp.async.bulk into every rank's mailbox (NVLS multicast).
  6  clear the previous generation of both Lamport buffers (idle time while peers finish), gather: poll the
     local mailbox [M, 7168] (all of a thread's polls in flight), copy to out, reset sentinels. CTA 0 rotates
     both flag sets after all 98 CTAs arrived.
"""

import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Float32, Int32, Int64, Uint32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import tcgen05
from cutlass.cutlass_dsl import T, dsl_user_op

TP, H, L, SH = 8, 7168, 3584, 896  # ranks, hidden, latent, hidden shard
NT, BN = 7, 128  # n-tiles per shard, rows per n-tile
S, BK = 14, 256  # k-slices, channels per k-slice
THREADS, WARPS = 256, 8
MAXM = 64  # tokens (moe_stage MAX_M: the prefilled path)
CTAS = NT * S
NBUF = 3  # Lamport generations (image NUM_LAMPORT_BUFFERS)
SENT = 0x80000000
SEND_IT = 3  # shared-send fragments per thread: ceil(64 * 7168 / 8 / (98 * 256))
GIT = 3  # gather fragments per thread, same count


def _asm(ret, args, body, cons, loc, ip, side=True):
    return llvm.inline_asm(ret, args, body, cons, has_side_effects=side, is_align_stack=False,
                           asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


def _v(a, loc, ip):
    return a.ir_value(loc=loc, ip=ip)


def _u4(st, loc, ip):
    return [Uint32(llvm.extractvalue(T.i32(), st, [i], loc=loc, ip=ip)) for i in range(4)]


# ---------------------------------------------------------------- global memory


@dsl_user_op
def ld_vol_v4(addr: Int64, *, loc=None, ip=None):
    """Lamport polling load (image primitives.load_global_u32x4(volatile=True))."""
    st = _asm(llvm.StructType.get_literal([T.i32()] * 4), [_v(Int64(addr), loc, ip)],
              "ld.volatile.global.v4.u32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,l", loc, ip)
    return _u4(st, loc, ip)


@dsl_user_op
def ld_vol_v2(addr: Int64, *, loc=None, ip=None):
    st = _asm(llvm.StructType.get_literal([T.i32()] * 2), [_v(Int64(addr), loc, ip)],
              "ld.volatile.global.v2.u32 {$0, $1}, [$2];", "=r,=r,l", loc, ip)
    return [Uint32(llvm.extractvalue(T.i32(), st, [i], loc=loc, ip=ip)) for i in range(2)]


@dsl_user_op
def ld_v4(addr: Int64, *, loc=None, ip=None):
    st = _asm(llvm.StructType.get_literal([T.i32()] * 4), [_v(Int64(addr), loc, ip)],
              "ld.global.v4.u32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,l", loc, ip)
    return _u4(st, loc, ip)


@dsl_user_op
def ld_u32(addr: Int64, *, loc=None, ip=None) -> Uint32:
    return Uint32(_asm(T.i32(), [_v(Int64(addr), loc, ip)], "ld.global.cg.u32 $0, [$1];", "=r,l", loc, ip))


@dsl_user_op
def ld_u64(addr: Int64, *, loc=None, ip=None) -> Int64:
    return Int64(_asm(T.i64(), [_v(Int64(addr), loc, ip)], "ld.global.cg.u64 $0, [$1];", "=l,l", loc, ip))


@dsl_user_op
def st_v4(addr: Int64, a: Uint32, b: Uint32, c: Uint32, d: Uint32, *, loc=None, ip=None):
    """16-byte store to an ordinary, peer (LSA) or NVLS multicast global address."""
    _asm(None, [_v(Int64(addr), loc, ip), _v(Uint32(a), loc, ip), _v(Uint32(b), loc, ip), _v(Uint32(c), loc, ip),
                _v(Uint32(d), loc, ip)], "st.global.v4.u32 [$0], {$1, $2, $3, $4};", "l,r,r,r,r", loc, ip)


@dsl_user_op
def st_u32(addr: Int64, v: Uint32, *, loc=None, ip=None):
    _asm(None, [_v(Int64(addr), loc, ip), _v(Uint32(v), loc, ip)], "st.global.u32 [$0], $1;", "l,r", loc, ip)


@dsl_user_op
def st_i64(addr: Int64, v: Int64, *, loc=None, ip=None):
    _asm(None, [_v(Int64(addr), loc, ip), _v(Int64(v), loc, ip)], "st.global.u64 [$0], $1;", "l,l", loc, ip)


@dsl_user_op
def st_sentinel(addr: Int64, *, loc=None, ip=None):
    """Reset one Lamport fragment: four fp32 -0 words (image store_lamport_sentinel_128)."""
    _asm(None, [_v(Int64(addr), loc, ip)],
         "{ .reg .b32 s; mov.b32 s, 0x80000000; st.global.v4.u32 [$0], {s, s, s, s}; }", "l", loc, ip)


@dsl_user_op
def ld_acquire_u32(addr: Int64, *, loc=None, ip=None) -> Uint32:
    return Uint32(_asm(T.i32(), [_v(Int64(addr), loc, ip)], "ld.acquire.gpu.global.u32 $0, [$1];", "=r,l", loc, ip))


@dsl_user_op
def fence_acq_rel_gpu(*, loc=None, ip=None):
    _asm(None, [], "fence.acq_rel.gpu;", "", loc, ip)


# ---------------------------------------------------------------- shared memory / cluster


@dsl_user_op
def cp_async16(saddr: Int32, gaddr: Int64, *, loc=None, ip=None):
    _asm(None, [_v(Int32(saddr), loc, ip), _v(Int64(gaddr), loc, ip)], "cp.async.cg.shared.global [$0], [$1], 16;",
         "r,l", loc, ip)


@dsl_user_op
def cp_async_commit(*, loc=None, ip=None):
    _asm(None, [], "cp.async.commit_group;", "", loc, ip)


@dsl_user_op
def cp_async_wait_all(*, loc=None, ip=None):
    _asm(None, [], "cp.async.wait_all;", "", loc, ip)


@dsl_user_op
def lds_u32(saddr: Int32, *, loc=None, ip=None) -> Uint32:
    return Uint32(_asm(T.i32(), [_v(Int32(saddr), loc, ip)], "ld.shared.u32 $0, [$1];", "=r,r", loc, ip))


@dsl_user_op
def lds_f32(saddr: Int32, *, loc=None, ip=None) -> Float32:
    return Float32(_asm(T.f32(), [_v(Int32(saddr), loc, ip)], "ld.shared.f32 $0, [$1];", "=f,r", loc, ip))


@dsl_user_op
def stc_v4(saddr: Int32, crank: Int32, a: Uint32, b: Uint32, c: Uint32, d: Uint32, *, loc=None, ip=None):
    """st.shared::cluster.v4 at this CTA's smem address `saddr` mapped to cluster CTA `crank`."""
    _asm(None, [_v(Int32(saddr), loc, ip), _v(Int32(crank), loc, ip), _v(Uint32(a), loc, ip),
                _v(Uint32(b), loc, ip), _v(Uint32(c), loc, ip), _v(Uint32(d), loc, ip)],
         "{ .reg .b32 ra; mapa.shared::cluster.u32 ra, $0, $1; st.shared::cluster.v4.u32 [ra], {$2, $3, $4, $5}; }",
         "r,r,r,r,r,r", loc, ip)


@dsl_user_op
def stc_f32(saddr: Int32, crank: Int32, v: Float32, *, loc=None, ip=None):
    _asm(None, [_v(Int32(saddr), loc, ip), _v(Int32(crank), loc, ip), _v(Float32(v), loc, ip)],
         "{ .reg .b32 ra; mapa.shared::cluster.u32 ra, $0, $1; st.shared::cluster.f32 [ra], $2; }", "r,r,f", loc, ip)


@dsl_user_op
def cluster_arrive_relaxed(*, loc=None, ip=None):
    _asm(None, [], "barrier.cluster.arrive.relaxed.aligned;", "", loc, ip)


@dsl_user_op
def cluster_arrive(*, loc=None, ip=None):
    _asm(None, [], "barrier.cluster.arrive.release.aligned;", "", loc, ip)


@dsl_user_op
def cluster_wait(*, loc=None, ip=None):
    _asm(None, [], "barrier.cluster.wait.acquire.aligned;", "", loc, ip)


# ---------------------------------------------------------------- math


@dsl_user_op
def bf16_lo(w: Uint32, *, loc=None, ip=None) -> Float32:
    return Float32(_asm(T.f32(), [_v(Uint32(w), loc, ip)], "{ .reg .b32 t; shl.b32 t, $1, 16; mov.b32 $0, t; }",
                        "=f,r", loc, ip, side=False))


@dsl_user_op
def bf16_hi(w: Uint32, *, loc=None, ip=None) -> Float32:
    return Float32(_asm(T.f32(), [_v(Uint32(w), loc, ip)],
                        "{ .reg .b32 t; and.b32 t, $1, 0xFFFF0000; mov.b32 $0, t; }", "=f,r", loc, ip, side=False))


@dsl_user_op
def rnd_bf16(x: Float32, *, loc=None, ip=None) -> Float32:
    """f32 -> bf16 (RN) -> f32."""
    return Float32(_asm(T.f32(), [_v(Float32(x), loc, ip)],
                        "{ .reg .b16 h; cvt.rn.bf16.f32 h, $1; cvt.f32.bf16 $0, h; }", "=f,f", loc, ip, side=False))


@dsl_user_op
def pack_bf16x2(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Uint32:
    """RN to bf16 pair; a -0 half becomes +0 (the Lamport sentinel word is fp32 -0)."""
    return Uint32(_asm(T.i32(), [_v(Float32(lo), loc, ip), _v(Float32(hi), loc, ip)],
                       "{ .reg .b32 w, l, h; .reg .pred pl, ph; cvt.rn.bf16x2.f32 w, $2, $1;"
                       " and.b32 l, w, 0xFFFF; and.b32 h, w, 0xFFFF0000;"
                       " setp.eq.u32 pl, l, 0x8000; setp.eq.u32 ph, h, 0x80000000;"
                       " @pl and.b32 w, w, 0xFFFF0000; @ph and.b32 w, w, 0x0000FFFF; mov.b32 $0, w; }",
                       "=r,f,f", loc, ip, side=False))


@dsl_user_op
def no_neg_zero(w: Uint32, *, loc=None, ip=None) -> Uint32:
    """Image sanitize_negative_zero on one word: a bf16 -0 half -> +0."""
    return Uint32(_asm(T.i32(), [_v(Uint32(w), loc, ip)],
                       "{ .reg .b32 w, l, h; .reg .pred pl, ph; mov.b32 w, $1; and.b32 l, w, 0xFFFF;"
                       " and.b32 h, w, 0xFFFF0000; setp.eq.u32 pl, l, 0x8000; setp.eq.u32 ph, h, 0x80000000;"
                       " @pl and.b32 w, w, 0xFFFF0000; @ph and.b32 w, w, 0x0000FFFF; mov.b32 $0, w; }",
                       "=r,r", loc, ip, side=False))


@dsl_user_op
def rsqrt_approx(x: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(_asm(T.f32(), [_v(Float32(x), loc, ip)], "rsqrt.approx.f32 $0, $1;", "=f,f", loc, ip, side=False))


@dsl_user_op
def globaltimer(*, loc=None, ip=None) -> Int64:
    return Int64(_asm(T.i64(), [], "mov.u64 $0, %globaltimer;", "=l", loc, ip))


@cute.jit
def dirty2(w0: Uint32, w1: Uint32):
    s = Uint32(SENT)
    return (w0 == s) | (w1 == s)


@cute.jit
def dirty4(w0: Uint32, w1: Uint32, w2: Uint32, w3: Uint32):
    s = Uint32(SENT)
    return (w0 == s) | (w1 == s) | (w2 == s) | (w3 == s)


@dsl_user_op
def sts_v2(saddr: Int32, a: Uint32, b: Uint32, *, loc=None, ip=None):
    _asm(None, [_v(Int32(saddr), loc, ip), _v(Uint32(a), loc, ip), _v(Uint32(b), loc, ip)],
         "st.shared.v2.u32 [$0], {$1, $2};", "r,r,r", loc, ip)


@dsl_user_op
def fence_proxy_async_smem(*, loc=None, ip=None):
    _asm(None, [], "fence.proxy.async.shared::cta;", "", loc, ip)


@dsl_user_op
def bulk_s2g(gaddr: Int64, saddr: Int32, nbytes: int, *, loc=None, ip=None):
    """cp.async.bulk smem -> global (here the NVLS multicast mailbox), one bulk group per call."""
    _asm(None, [_v(Int64(gaddr), loc, ip), _v(Int32(saddr), loc, ip)],
         f"cp.async.bulk.global.shared::cta.bulk_group [$0], [$1], {nbytes}; cp.async.bulk.commit_group;", "l,r",
         loc, ip)


@dsl_user_op
def bulk_wait_read(*, loc=None, ip=None):
    _asm(None, [], "cp.async.bulk.wait_group.read 0;", "", loc, ip)


@dsl_user_op
def smid(*, loc=None, ip=None) -> Int32:
    return Int32(_asm(T.i32(), [], "mov.u32 $0, %smid;", "=r", loc, ip))


@dsl_user_op
def nanosleep500(*, loc=None, ip=None):
    _asm(None, [], "nanosleep.u32 500;", "", loc, ip)


@dsl_user_op
def red_relaxed_u32(addr: Int64, v: Uint32, *, loc=None, ip=None):
    _asm(None, [_v(Int64(addr), loc, ip), _v(Uint32(v), loc, ip)], "red.relaxed.gpu.global.add.u32 [$0], $1;",
         "l,r", loc, ip)


@dsl_user_op
def fence_proxy_async_cluster(*, loc=None, ip=None):
    _asm(None, [], "fence.proxy.async.shared::cluster;", "", loc, ip)


@dsl_user_op
def tc_fence_after(*, loc=None, ip=None):
    _asm(None, [], "tcgen05.fence::after_thread_sync;", "", loc, ip)


@dsl_user_op
def tc_fence_before(*, loc=None, ip=None):
    _asm(None, [], "tcgen05.fence::before_thread_sync;", "", loc, ip)


@dsl_user_op
def tmem_ld16(taddr: Int32, *, loc=None, ip=None):
    """16 fp32 columns of one TMEM lane per thread (32x32b.x16) and wait::ld in the same asm block, so no copy of
    an output register can be scheduled before the data arrived."""
    st = _asm(llvm.StructType.get_literal([T.f32()] * 16), [_v(Int32(taddr), loc, ip)],
              "tcgen05.ld.sync.aligned.32x32b.x16.b32 {" + ", ".join(f"${i}" for i in range(16)) + "}, [$16]; "
              "tcgen05.wait::ld.sync.aligned;", ",".join(["=f"] * 16) + ",r", loc, ip)
    return [Float32(llvm.extractvalue(T.f32(), st, [i], loc=loc, ip=ip)) for i in range(16)]


@dsl_user_op
def red_relaxed_u64(addr: Int64, v: Int64, *, loc=None, ip=None):
    """Relaxed: the caller's fence.acq_rel.gpu before it makes the pair a release."""
    _asm(None, [_v(Int64(addr), loc, ip), _v(Int64(v), loc, ip)], "red.relaxed.gpu.global.add.u64 [$0], $1;",
         "l,l", loc, ip)


@dsl_user_op
def ld_acquire_u64(addr: Int64, *, loc=None, ip=None) -> Int64:
    return Int64(_asm(T.i64(), [_v(Int64(addr), loc, ip)], "ld.acquire.gpu.global.u64 $0, [$1];", "=l,l", loc, ip))


@dsl_user_op
def ld_vol_u32(addr: Int64, *, loc=None, ip=None) -> Uint32:
    return Uint32(_asm(T.i32(), [_v(Int64(addr), loc, ip)], "ld.volatile.global.u32 $0, [$1];", "=r,l", loc, ip))


@dsl_user_op
def st_f32_nz(addr: Int64, v: Float32, *, loc=None, ip=None):
    """fp32 store, -0 stored as +0 (fp32 -0 is the "not written" word of the self-validating partials)."""
    _asm(None, [_v(Int64(addr), loc, ip), _v(Float32(v), loc, ip)],
         "{ .reg .b32 b; .reg .pred p; mov.b32 b, $1; setp.eq.b32 p, b, 0x80000000; @p mov.b32 b, 0;"
         " st.global.b32 [$0], b; }", "l,f", loc, ip)


@dsl_user_op
def as_f32(w: Uint32, *, loc=None, ip=None) -> Float32:
    return Float32(_asm(T.f32(), [_v(Uint32(w), loc, ip)], "mov.b32 $0, $1;", "=f,r", loc, ip, side=False))


# ---------------------------------------------------------------- kernel


@cute.jit
def a_off(row: Int32, c: Int32):
    """Byte offset of 16 B chunk c (8 channels) of weight row `row` in the UMMA A tile: K-major SWIZZLE_128B, 8-row
    x 128 B atoms stacked along M (1 KB each), the next 64 channels at BN x 128 B (the descriptors of
    make_smem_layout_a step 16384 B per 64 channels; checked in the PTX of B-tail-06)."""
    return (c // 8) * (BN * 128) + row * 128 + (((c % 8) ^ (row % 8)) * 16)


@cute.jit
def b_off(t: Int32, c: Int32):
    """Byte offset of chunk c of token row t in the UMMA B tile (as a_off with MAXM rows: 8192 B per 64 channels)."""
    return (c // 8) * (MAXM * 128) + t * 128 + (((c % 8) ^ (t % 8)) * 16)


class MoeTail:
    """The tail launch. Every name is assigned in one section only (DSL loop-carried analysis), and vector loads
    are unpacked into scalars at once. The n-tile's split-K partials are published by a u64 arrival counter per
    n-tile (14 CTAs; only grows). postfence: thread 0 also fences after its acquire poll (redundant with
    ld.acquire + bar.sync, the CUTLASS GenericBarrier pattern; kept as an A/B knob)."""

    def __init__(self, stamps: bool = False, postfence: bool = False):
        self.stamps, self.postfence = stamps, postfence

    @cute.jit
    def __call__(self, p_w: cute.Pointer, p_gamma: cute.Pointer, p_rws: cute.Pointer, p_rflags: cute.Pointer,
                 p_ssrc: cute.Pointer, p_sws: cute.Pointer, p_sflags: cute.Pointer, p_peers: cute.Pointer,
                 p_mbox: cute.Pointer, mbox_mc: Int64, p_out: cute.Pointer, p_part: cute.Pointer,
                 p_sums: cute.Pointer, p_ctr: cute.Pointer, p_dbg: cute.Pointer, m: Int32, rank: Int32,
                 eps: Float32, stream):
        tiled_mma = sm100_utils.make_trivial_tiled_mma(cutlass.BFloat16, tcgen05.OperandMajorMode.K,
                                                       tcgen05.OperandMajorMode.K, Float32, tcgen05.CtaGroup.ONE,
                                                       (BN, MAXM))
        a_lay = sm100_utils.make_smem_layout_a(tiled_mma, (BN, MAXM, BK), cutlass.BFloat16, 1)
        b_lay = sm100_utils.make_smem_layout_b(tiled_mma, (BN, MAXM, BK), cutlass.BFloat16, 1)
        self.kernel(tiled_mma, a_lay, b_lay, Int64(p_w.toint()), Int64(p_gamma.toint()), Int64(p_rws.toint()),
                    Int64(p_rflags.toint()), Int64(p_ssrc.toint()), Int64(p_sws.toint()), Int64(p_sflags.toint()),
                    Int64(p_peers.toint()), Int64(p_mbox.toint()), mbox_mc, Int64(p_out.toint()),
                    Int64(p_part.toint()), Int64(p_sums.toint()), Int64(p_ctr.toint()), Int64(p_dbg.toint()), m,
                    rank, eps).launch(
            grid=[NT, S, 1], block=[THREADS, 1, 1], cluster=[NT, 1, 1], stream=stream, use_pdl=True)

    @cute.jit
    def stamp(self, dbg: Int64, cta: Int32, tidx: Int32, k: int):
        if const_expr(self.stamps):
            if tidx == 0:
                st_i64(dbg + Int64(cta * 16 + k) * 8, globaltimer())
                if const_expr(k == 0):
                    st_i64(dbg + Int64(cta * 16 + 15) * 8, Int64(smid()))

    @cute.jit
    def ar_finish(self, words: cute.Tensor, gf: cute.Tensor, t: Int32, lane: Int32, b_s: Int32, ss_s: Int32):
        """All 8 rank slots of (token t, 8 channels) valid in `words`: fp32 sum in rank order (= image), x =
        bf16(sum), sumsq (bf16 squares, warp tree), y = bf16(x * gamma) -> B row t of all 7 cluster CTAs."""
        xa = cute.make_rmem_tensor(cute.make_layout(8), Float32)
        for e2 in cutlass.range_constexpr(8):
            xa[e2] = Float32(0.0)
        for r2 in cutlass.range_constexpr(TP):
            for e2 in cutlass.range_constexpr(4):
                xa[2 * e2] = xa[2 * e2] + bf16_lo(words[r2, e2])
                xa[2 * e2 + 1] = xa[2 * e2 + 1] + bf16_hi(words[r2, e2])
        sq = Float32(0.0)
        for e2 in cutlass.range_constexpr(8):
            xa[e2] = rnd_bf16(xa[e2])  # x: bit-equal to the image's norm_input_bf16
            sq = sq + rnd_bf16(xa[e2] * xa[e2])
        for o2 in cutlass.range_constexpr(5):
            sq = sq + cute.arch.shuffle_sync_bfly(sq, 16 >> o2)
        y0 = pack_bf16x2(xa[0] * gf[0], xa[1] * gf[1])
        y1 = pack_bf16x2(xa[2] * gf[2], xa[3] * gf[3])
        y2 = pack_bf16x2(xa[4] * gf[4], xa[5] * gf[5])
        y3 = pack_bf16x2(xa[6] * gf[6], xa[7] * gf[7])
        dst = b_s + b_off(t, lane)
        for c2 in cutlass.range_constexpr(NT):
            stc_v4(dst, Int32(c2), y0, y1, y2, y3)
        if lane == 0:
            for c2 in cutlass.range_constexpr(NT):
                stc_f32(ss_s + t * 4, Int32(c2), sq)

    @cute.kernel
    def kernel(self, tiled_mma: cute.TiledMma, a_lay: cute.ComposedLayout, b_lay: cute.ComposedLayout, w: Int64,
               gamma: Int64, rws: Int64, rflags: Int64, ssrc: Int64, sws: Int64, sflags: Int64, peers: Int64,
               mbox: Int64, mbox_mc: Int64, out: Int64, part: Int64, sums: Int64, ctr: Int64, dbg: Int64,
               m: Int32, rank: Int32, eps: Float32):
        tidx, _, _ = cute.arch.thread_idx()
        j, s, _ = cute.arch.block_idx()  # j = n-tile = rank in the cluster (cluster along x), s = k-slice
        warp = tidx // 32
        lane = tidx % 32
        cta = s * NT + j
        gt = cta * THREADS + tidx
        k0 = s * BK
        self.stamp(dbg, cta, tidx, 0)

        @cute.struct
        class Storage:
            mbar: cute.struct.MemRange[cutlass.Int64, 1]
            thold: cute.struct.MemRange[Int32, 1]
            ss: cute.struct.Align[cute.struct.MemRange[Float32, MAXM], 16]

        smem = cutlass.utils.SmemAllocator()
        st = smem.allocate(Storage)
        ss_s = Int32(st.ss.data_ptr().toint())
        mbar_p = st.mbar.data_ptr()
        thold_p = st.thold.data_ptr()
        sA = smem.allocate_tensor(element_type=cutlass.BFloat16, layout=a_lay.outer, byte_alignment=1024,
                                  swizzle=a_lay.inner)
        sB = smem.allocate_tensor(element_type=cutlass.BFloat16, layout=b_lay.outer, byte_alignment=1024,
                                  swizzle=b_lay.inner)
        a_s = Int32(sA.iterator.toint())
        b_s = Int32(sB.iterator.toint())

        # ---- 0. before the grid dependency: weights, constants, smem/TMEM setup only
        cluster_arrive_relaxed()
        row0 = rank * SH + j * BN
        for q in cutlass.range_constexpr(BN * BK // 8 // THREADS):  # 16 B chunks
            ia = tidx + q * THREADS
            ra_row = ia // (BK // 8)
            ra_col = ia % (BK // 8)
            cp_async16(a_s + a_off(ra_row, ra_col), w + (Int64(row0 + ra_row) * L + Int64(k0 + ra_col * 8)) * 2)
        cp_async_commit()
        gw0, gw1, gw2, gw3 = ld_v4(gamma + Int64(k0 + lane * 8) * 2)
        gf = cute.make_rmem_tensor(cute.make_layout(8), Float32)
        gf[0] = bf16_lo(gw0)
        gf[1] = bf16_hi(gw0)
        gf[2] = bf16_lo(gw1)
        gf[3] = bf16_hi(gw1)
        gf[4] = bf16_lo(gw2)
        gf[5] = bf16_hi(gw2)
        gf[6] = bf16_lo(gw3)
        gf[7] = bf16_hi(gw3)
        if tidx == 0:
            cute.arch.mbarrier_init(mbar_p, 1)
            cute.arch.mbarrier_init_fence()
        if warp == 0:
            cute.arch.alloc_tmem(MAXM, thold_p)
            cute.arch.relinquish_tmem_alloc_permit()

        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()  # all 98 CTAs run (the dependent cannot take their SMs)
        self.stamp(dbg, cta, tidx, 1)

        # ---- 1. one batch of loads: flags of both Lamport buffers, the shared-send fragments
        cur = ld_u32(rflags)
        bpb = ld_u32(rflags + 8)
        rdirty = ld_u32(rflags + 4)
        rstages = ld_u32(rflags + 12)
        rclear = ld_u32(rflags + 16)
        scur = ld_u32(sflags)
        sdirty = ld_u32(sflags + 4)
        sbpb = ld_u32(sflags + 8)
        sstages = ld_u32(sflags + 12)
        sclear = ld_u32(sflags + 16)
        ctr0 = ld_u64(ctr + Int64(j) * 8)  # 14 k + (arrivals of this call so far, < 14)
        sv = cute.make_rmem_tensor(cute.make_layout((SEND_IT, 4), stride=(4, 1)), Uint32)
        spp = cute.make_rmem_tensor(cute.make_layout(SEND_IT), Int64)
        for it in cutlass.range_constexpr(SEND_IT):
            fs = gt + it * (CTAS * THREADS)
            if fs < m * (H // 8):
                v0, v1, v2, v3 = ld_v4(ssrc + Int64(fs) * 16)
                sv[it, 0] = v0
                sv[it, 1] = v1
                sv[it, 2] = v2
                sv[it, 3] = v3
                spp[it] = ld_u64(peers + Int64((fs % (H // 8)) // (SH // 8)) * 8)

        # ---- 2. routed all-reduce of this k-slice, warp = tokens ta and ta + 56 (both polls in flight)
        cluster_wait()  # every cluster CTA has started: DSMEM targets are live
        r_gen = rws + Int64(cur) * Int64(bpb)
        wa = cute.make_rmem_tensor(cute.make_layout((TP, 4), stride=(4, 1)), Uint32)
        wb = cute.make_rmem_tensor(cute.make_layout((TP, 4), stride=(4, 1)), Uint32)
        ta = j * WARPS + warp
        tb = ta + NT * WARPS
        base_a = r_gen + (Int64(ta * TP) * L + Int64(k0 + lane * 8)) * 2
        base_b = r_gen + (Int64(tb * TP) * L + Int64(k0 + lane * 8)) * 2
        if ta < m:
            for r2 in cutlass.range_constexpr(TP):
                p0, p1, p2, p3 = ld_vol_v4(base_a + Int64(r2 * L * 2))
                wa[r2, 0] = p0
                wa[r2, 1] = p1
                wa[r2, 2] = p2
                wa[r2, 3] = p3
            if tb < m:
                for r2 in cutlass.range_constexpr(TP):
                    q0, q1, q2, q3 = ld_vol_v4(base_b + Int64(r2 * L * 2))
                    wb[r2, 0] = q0
                    wb[r2, 1] = q1
                    wb[r2, 2] = q2
                    wb[r2, 3] = q3
            va = True
            for r2 in cutlass.range_constexpr(TP):
                va = va & (not dirty4(wa[r2, 0], wa[r2, 1], wa[r2, 2], wa[r2, 3]))
            while not va:
                va = True
                for r2 in cutlass.range_constexpr(TP):
                    u0, u1, u2, u3 = ld_vol_v4(base_a + Int64(r2 * L * 2))
                    wa[r2, 0] = u0
                    wa[r2, 1] = u1
                    wa[r2, 2] = u2
                    wa[r2, 3] = u3
                    va = va & (not dirty4(u0, u1, u2, u3))
            self.ar_finish(wa, gf, ta, lane, b_s, ss_s)
            if tb < m:
                vb = True
                for r2 in cutlass.range_constexpr(TP):
                    vb = vb & (not dirty4(wb[r2, 0], wb[r2, 1], wb[r2, 2], wb[r2, 3]))
                while not vb:
                    vb = True
                    for r2 in cutlass.range_constexpr(TP):
                        n0, n1, n2, n3 = ld_vol_v4(base_b + Int64(r2 * L * 2))
                        wb[r2, 0] = n0
                        wb[r2, 1] = n1
                        wb[r2, 2] = n2
                        wb[r2, 3] = n3
                        vb = vb & (not dirty4(n0, n1, n2, n3))
                self.ar_finish(wb, gf, tb, lane, b_s, ss_s)
        self.stamp(dbg, cta, tidx, 2)
        fence_proxy_async_cluster()  # this thread's DSMEM rows -> the async proxy (tcgen05.mma reads)
        cluster_arrive()  # .release = MEMBAR.ALL.GPU: no remote store may be in flight here (it would wait)
        cluster_wait()

        # shared partial -> owner ranks (unicast); the epilogue of the owner polls it
        s_off = Int64(scur) * Int64(sbpb)  # generation offset, the same on every rank
        for it in cutlass.range_constexpr(SEND_IT):
            fs = gt + it * (CTAS * THREADS)
            if fs < m * (H // 8):
                st_v4(spp[it] + s_off + (Int64((fs // (H // 8)) * TP + rank) * SH
                                          + Int64(((fs % (H // 8)) % (SH // 8)) * 8)) * 2,
                      no_neg_zero(sv[it, 0]), no_neg_zero(sv[it, 1]), no_neg_zero(sv[it, 2]),
                      no_neg_zero(sv[it, 3]))
        self.stamp(dbg, cta, tidx, 3)
        cp_async_wait_all()
        fence_proxy_async_smem()
        cute.arch.barrier()
        self.stamp(dbg, cta, tidx, 4)

        # ---- 3. MMA over the k-slice: one thread issues 16 tcgen05.mma 128 x 64 x 16 into TMEM; tcgen05.ld ->
        #         fp32 partial [t][row] in L2. A -0 partial is stored as +0: fp32 -0 is the "not written" word.
        pbase = part + Int64((j * S + s) * MAXM) * (BN * 4)
        acc_ptr = cute.arch.retrieve_tmem_ptr(Float32, alignment=16, ptr_to_buffer_holding_addr=thold_p)
        acc_fake = tiled_mma.make_fragment_C(tiled_mma.partition_shape_C((BN, MAXM)))
        t_acc = cute.make_tensor(acc_ptr, acc_fake.layout)
        t_a = tiled_mma.make_fragment_A(sA)
        t_b = tiled_mma.make_fragment_B(sB)
        if warp == 0:
            tc_fence_after()
            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for kb in cutlass.range_constexpr(BK // 16):
                cute.gemm(tiled_mma, t_acc, t_a[(None, None, kb, 0)], t_b[(None, None, kb, 0)], t_acc)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            with cute.arch.elect_one():
                tcgen05.commit(mbar_p)
        cute.arch.mbarrier_wait(mbar_p, 0)
        tc_fence_after()
        # D row = TMEM lane: warp w reads rows 32 (w % 4) .. + 31, token columns 32 (w // 4) .. + 31
        taddr = Int32(lds_u32(Int32(thold_p.toint()))) + (((warp % 4) * 32) << 16)
        prow3 = (warp % 4) * 32 + lane
        dv = cute.make_rmem_tensor(cute.make_layout(32), Float32)
        for h3 in cutlass.range_constexpr(2):
            c3 = (warp // 4) * 2 + h3
            if c3 * 16 < m:
                e = tmem_ld16(taddr + c3 * 16)
                for i3 in cutlass.range_constexpr(16):
                    dv[h3 * 16 + i3] = e[i3]
        for h3 in cutlass.range_constexpr(2):
            for i3 in cutlass.range_constexpr(16):
                tq3 = ((warp // 4) * 2 + h3) * 16 + i3
                if tq3 < m:
                    st_f32_nz(pbase + Int64(tq3 * BN + prow3) * 4, dv[h3 * 16 + i3])
        tc_fence_before()
        if tidx < m:
            st_f32_nz(sums + Int64((j * S + s) * MAXM + tidx) * 4, lds_f32(ss_s + tidx * 4))
        self.stamp(dbg, cta, tidx, 5)
        # ---- 4. n-tile arrival: bar.sync, then thread 0 fences (gpu scope, cumulative over the CTA's stores) and
        #         arrives; it polls with ld.acquire and bar.sync releases the CTA. The same fence orders this CTA's
        #         flag reads (section 1) before its flag-set arrivals (CTA 0 rotates at the end).
        cute.arch.barrier()
        if tidx == 0:
            fence_acq_rel_gpu()
            caddr = ctr + Int64(j) * 8
            red_relaxed_u64(caddr, Int64(1))
            red_relaxed_u32(rflags + 32, Uint32(1))
            red_relaxed_u32(sflags + 32, Uint32(1))
            target = (ctr0 // Int64(S) + Int64(1)) * Int64(S)
            seen = ld_acquire_u64(caddr)
            while seen < target:
                seen = ld_acquire_u64(caddr)
            if const_expr(self.postfence):
                fence_acq_rel_gpu()
        cute.arch.barrier()

        # ---- 5. epilogue, warp = one token of the n-tile (tokens s, s + 14, ...), lane = rows 4 lane .. + 3. Sum
        #         the 14 partials and sums of the n-tile (published by section 4). The 256 B row segment is staged in smem (the A tile, free
        #         once the MMA committed) and sent by one cp.async.bulk to the multicast mailbox.
        s_gen = sws + s_off
        pw = cute.make_rmem_tensor(cute.make_layout((S, 4), stride=(4, 1)), Uint32)
        psq = cute.make_rmem_tensor(cute.make_layout(S), Uint32)
        swords = cute.make_rmem_tensor(cute.make_layout((TP, 2), stride=(2, 1)), Uint32)
        o = cute.make_rmem_tensor(cute.make_layout(4), Float32)
        sh = cute.make_rmem_tensor(cute.make_layout(4), Float32)
        stage = a_s + warp * 256
        for t5 in cutlass.range(s + S * warp, m, S * WARPS, unroll=1):
            sbase = s_gen + (Int64(t5 * TP) * SH + Int64(j * BN + lane * 4)) * 2
            pb5 = part + (Int64(j * S * MAXM + t5) * BN + Int64(lane * 4)) * 4  # + k5 * MAXM * BN * 4
            qb5 = sums + Int64(j * S * MAXM + t5) * 4  # + k5 * MAXM * 4
            for r5 in cutlass.range_constexpr(TP):
                z0, z1 = ld_vol_v2(sbase + Int64(r5 * SH * 2))
                swords[r5, 0] = z0
                swords[r5, 1] = z1
            for k5 in cutlass.range_constexpr(S):
                l0, l1, l2, l3 = ld_vol_v4(pb5 + Int64(k5 * MAXM * BN * 4))
                pw[k5, 0] = l0
                pw[k5, 1] = l1
                pw[k5, 2] = l2
                pw[k5, 3] = l3
                psq[k5] = ld_vol_u32(qb5 + Int64(k5 * MAXM * 4))
            for e5 in cutlass.range_constexpr(4):
                o[e5] = Float32(0.0)
                sh[e5] = Float32(0.0)
            ssum = Float32(0.0)
            for k5 in cutlass.range_constexpr(S):
                for e5 in cutlass.range_constexpr(4):
                    o[e5] = o[e5] + as_f32(pw[k5, e5])
                ssum = ssum + as_f32(psq[k5])
            inv = rsqrt_approx(ssum / Float32(L) + eps)
            self.stamp(dbg, cta, tidx, 6)
            svalid = True
            for r5 in cutlass.range_constexpr(TP):
                svalid = svalid & (not dirty2(swords[r5, 0], swords[r5, 1]))
            while not svalid:
                svalid = True
                for r5 in cutlass.range_constexpr(TP):
                    y0_, y1_ = ld_vol_v2(sbase + Int64(r5 * SH * 2))
                    swords[r5, 0] = y0_
                    swords[r5, 1] = y1_
                    svalid = svalid & (not dirty2(y0_, y1_))
            for r5 in cutlass.range_constexpr(TP):
                for e5 in cutlass.range_constexpr(2):
                    sh[2 * e5] = sh[2 * e5] + bf16_lo(swords[r5, e5])
                    sh[2 * e5 + 1] = sh[2 * e5 + 1] + bf16_hi(swords[r5, e5])
            for e5 in cutlass.range_constexpr(4):
                o[e5] = o[e5] * inv + rnd_bf16(sh[e5])
            bulk_wait_read()  # this warp's previous bulk store has read its stage
            cute.arch.sync_warp()
            sts_v2(stage + lane * 8, pack_bf16x2(o[0], o[1]), pack_bf16x2(o[2], o[3]))
            cute.arch.sync_warp()
            if lane == 0:
                fence_proxy_async_smem()
                bulk_s2g(mbox_mc + (Int64(t5) * H + Int64(rank * SH + j * BN)) * 2, stage, 256)
        if lane == 0:
            bulk_wait_read()  # the smem source is read; the remote writes complete on their own
        self.stamp(dbg, cta, tidx, 7)
        if rstages > Uint32(0):
            rd_gen = rws + Int64(rdirty) * Int64(bpb)
            for f4 in cutlass.range(gt, Int32((rclear + Uint32(15)) // Uint32(16)), CTAS * THREADS, unroll=1):
                st_sentinel(rd_gen + Int64(f4) * 16)
        if sstages > Uint32(0):
            sd_gen = sws + Int64(sdirty) * Int64(sbpb)
            for f5 in cutlass.range(gt, Int32((sclear + Uint32(15)) // Uint32(16)), CTAS * THREADS, unroll=1):
                st_sentinel(sd_gen + Int64(f5) * 16)

        # ---- 6. gather: local mailbox -> out (all of a thread's polls in flight first), then reset; CTA 0 rotates
        #         both flag sets after 98 arrivals each
        gv = cute.make_rmem_tensor(cute.make_layout((GIT, 4), stride=(4, 1)), Uint32)
        for it6 in cutlass.range_constexpr(GIT):
            f6 = gt + it6 * (CTAS * THREADS)
            if f6 < m * (H // 8):
                w0, w1, w2, w3 = ld_vol_v4(mbox + Int64(f6) * 16)
                gv[it6, 0] = w0
                gv[it6, 1] = w1
                gv[it6, 2] = w2
                gv[it6, 3] = w3
        for it6 in cutlass.range_constexpr(GIT):
            f6 = gt + it6 * (CTAS * THREADS)
            if f6 < m * (H // 8):
                while dirty4(gv[it6, 0], gv[it6, 1], gv[it6, 2], gv[it6, 3]):
                    x0, x1, x2, x3 = ld_vol_v4(mbox + Int64(f6) * 16)
                    gv[it6, 0] = x0
                    gv[it6, 1] = x1
                    gv[it6, 2] = x2
                    gv[it6, 3] = x3
                st_v4(out + Int64(f6) * 16, gv[it6, 0], gv[it6, 1], gv[it6, 2], gv[it6, 3])
        self.stamp(dbg, cta, tidx, 8)
        for f7 in cutlass.range(gt, m * (H // 8), CTAS * THREADS, unroll=1):
            st_sentinel(mbox + Int64(f7) * 16)
        self.stamp(dbg, cta, tidx, 9)
        cute.arch.barrier()  # every warp's TMEM loads are done before the dealloc
        if warp == 0:
            tc_fence_after()
            cute.arch.dealloc_tmem(acc_ptr, MAXM)
        if cta == 0:
            if tidx == 0:
                n_r = ld_acquire_u32(rflags + 32)
                while n_r < Uint32(CTAS):
                    n_r = ld_acquire_u32(rflags + 32)
                n_s = ld_acquire_u32(sflags + 32)
                while n_s < Uint32(CTAS):
                    n_s = ld_acquire_u32(sflags + 32)
                st_u32(rflags, (cur + Uint32(1)) % Uint32(NBUF))
                st_u32(rflags + 4, cur)
                st_u32(rflags + 8, bpb)
                st_u32(rflags + 12, Uint32(1))
                st_u32(rflags + 16, Uint32(m) * Uint32(TP * L * 2))
                st_u32(sflags, (scur + Uint32(1)) % Uint32(NBUF))
                st_u32(sflags + 4, scur)
                st_u32(sflags + 8, sbpb)
                st_u32(sflags + 12, Uint32(1))
                st_u32(sflags + 16, Uint32(m) * Uint32(TP * SH * 2))
                for k6 in cutlass.range_constexpr(5, 9):
                    st_u32(rflags + 4 * k6, Uint32(0))
                    st_u32(sflags + 4 * k6, Uint32(0))


class Feed:
    """Self-test PDL predecessor that plays fc2 + the aux shared-down GEMM: griddepcontrol.wait, launch_dependents,
    `sleeps` x nanosleep(500 ns) per thread, then this rank's routed rows [M, 3584] -> slot (token, rank) of
    generation flags[0] on every rank (NVLS multicast, -0 -> +0, as moe_stage fc2), and shared_src -> shared_dst."""

    @cute.jit
    def __call__(self, p_rows: cute.Pointer, mc: Int64, p_rflags: cute.Pointer, p_ssrc: cute.Pointer,
                 p_sdst: cute.Pointer, rank: Int32, m: Int32, sleeps: Int32, stream):
        self.kernel(Int64(p_rows.toint()), mc, Int64(p_rflags.toint()), Int64(p_ssrc.toint()),
                    Int64(p_sdst.toint()), rank, m, sleeps).launch(grid=[64, 1, 1], block=[256, 1, 1],
                                                                   stream=stream, use_pdl=True)

    @cute.kernel
    def kernel(self, rows: Int64, mc: Int64, rflags: Int64, ssrc: Int64, sdst: Int64, rank: Int32, m: Int32,
               sleeps: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        for _i in cutlass.range(sleeps, unroll=1):
            nanosleep500()
        gen = ld_u32(rflags)
        bpb = ld_u32(rflags + 8)
        gt = bx * 256 + tidx
        for f in cutlass.range(gt, m * (H // 8), 64 * 256, unroll=1):
            v0, v1, v2, v3 = ld_v4(ssrc + Int64(f) * 16)
            st_v4(sdst + Int64(f) * 16, v0, v1, v2, v3)
        for f2 in cutlass.range(gt, m * (L // 8), 64 * 256, unroll=1):
            tk = f2 // (L // 8)
            u0, u1, u2, u3 = ld_v4(rows + Int64(f2) * 16)
            st_v4(mc + Int64(gen) * Int64(bpb) + (Int64(tk * TP + rank) * L + Int64((f2 % (L // 8)) * 8)) * 2,
                  no_neg_zero(u0), no_neg_zero(u1), no_neg_zero(u2), no_neg_zero(u3))
