"""r3-318 MoE stage kernels (CuTe DSL 4.6.2, sm_100/sm_103): permute (index build + row gather) and two grouped
block-scaled GEMMs.

Layout facts these kernels rely on (checked on the seat by dev/probe.py and dev/mmaproof2.py, round 3 item 15):
  * w13 [E, 768, 1792] u8 (fp4 pairs): rows 2i = up(w3)_i, 2i+1 = gate(w1)_i, then a 32-row block shuffle
    (physical row p of a block holds source row 4*(p%8) + p//8). s13 [E, 768, 112] ue8m0 in the 128x4 swizzled
    SF atom layout over the same physical rows. w2 [E, 3584, 192] / s2 [E, 3584, 12] likewise (rows = channels).
  * activations: xq [M, 3584] e4m3, xs [M, 112] ue8m0 linear (ceil-ue8m0(amax/448), RNE e4m3).
  * tcgen05 kind::mxf8f6f4 with A = e2m1 weights works when A sits in smem in 8-bit containers (TMA U4 unpack,
    expect-tx = packed global bytes) and B = e4m3; an N < 128 MMA reads SFB rows 0..N-1 of its own 128-row SF
    atom (one atom per N tile: "pad" layout).

Formulation (decode, M <= MAX_M tokens per rank). Routed (token, k) pairs are grouped by local expert (ascending
expert id, tokens ascending inside an expert) into tiles of TN = 16 slots; an expert with c tokens owns ceil(c/16)
tiles. Output row of a pair = rows of all smaller experts + its rank inside its expert (the permute map).
  permute: every CTA rebuilds the expert histogram and both prefix sums (deterministic); CTA t quantizes token t's
           bf16 latent row to MXFP8 (the MXFP8QuantizeLinearKernel formula: e8m0 = round-up(amax * rcp(448)),
           x * rcp(2^e), satfinite RN e4m3) and writes the e4m3 row and SF to the slot of each of its pairs (x_perm,
           sfx_perm in the pad SF layout), the map, the bf16 weights and the row records (token, fp32 weight);
           CTA 0 writes the tile records (expert, slots used, first output row).
  fc1 / fc2: persistent units (tile, 128-row weight m-tile): A = the expert's weights, B = the tile's 16 slots.
  fc1 epilogue: SiTU-GLU (gate = w1, up = w3, alpha 4, beta 25) -> MXFP8 (ceil ue8m0 over 32 intermediate
           columns) -> inter [tile*16 + slot, 384] e4m3 + SF (pad layout) = fc2's B.
  fc2 epilogue: fp32 value x top-K weight -> red.add into acc [token, channel] fp32 (the top-K finalize). After its
           last unit each CTA arrives once on a grid counter; when all CTAs arrived, the whole grid reads acc, zeroes
           it, and stores bf16 (-0 -> +0) straight into the latent tail's Lamport multicast buffer: slot (token,
           rank) of generation flags[0] (the layout of the image's AllReduceRMSNormWithReduceScatterEarlyExit routed
           role). The tail then only polls. Measured (selftest --bench-fc2): an arrival per unit (needed to store
           each 128-channel m-tile as soon as it completes) costs 2-4 us per unit in fence + atomic round trips
           (fc2 at M 48: 250 us tile-major, 432 us m-tile-major, against 64 us without it); one arrival per CTA
           costs one round trip.
"""

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import T, dsl_user_op

E, TOPK, H, I = 896, 16, 3584, 384
SF_VEC = 32
A_DTYPE = cutlass.Float4E2M1FN
B_DTYPE = cutlass.Float8E4M3FN
SF_DTYPE = cutlass.Float8E8M0FNU
PERMUTE_THREADS = 1024

try:  # IKET phase ranges; compiled in only with iket=True (PERFBOT_MOE_STAGE_IKET=1)
    from cutlass.cute.experimental import iket as _iket
except Exception:  # pragma: no cover
    _iket = None


def _push(on, name):
    if cutlass.const_expr(on and _iket is not None):
        _iket.range_push(name)


def _pop(on):
    if cutlass.const_expr(on and _iket is not None):
        _iket.range_pop()


# ------------------------------------------------------------------ scalar helpers (inline PTX)

@dsl_user_op
def ue8m0_of_amax(amax: Float32, *, loc=None, ip=None) -> Uint32:
    """ceil-ue8m0(amax / 448): biased exponent rounded toward +inf, clamped to [0, 254], 0 for amax <= 0."""
    return Uint32(llvm.inline_asm(
        T.i32(), [Float32(amax).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .pred p_zero, p_has_mant, p_exp_zero, p_tiny_sub, p_ovf;
            .reg .u32 bits, exp_biased, mantissa, bump, result;
            .reg .f32 norm;
            div.rn.f32 norm, $1, 0f43E00000;
            setp.le.f32 p_zero, norm, 0f00000000;
            mov.b32 bits, norm;
            shr.b32 exp_biased, bits, 23;
            and.b32 exp_biased, exp_biased, 255;
            and.b32 mantissa, bits, 0x7FFFFF;
            setp.ne.u32 p_has_mant, mantissa, 0;
            selp.u32 bump, 1, 0, p_has_mant;
            setp.eq.u32 p_exp_zero, exp_biased, 0;
            setp.le.u32 p_tiny_sub, mantissa, 0x400000;
            and.pred p_tiny_sub, p_exp_zero, p_tiny_sub;
            @p_tiny_sub mov.u32 bump, 0;
            add.u32 result, exp_biased, bump;
            setp.gt.u32 p_ovf, result, 254;
            selp.u32 result, 254, result, p_ovf;
            selp.u32 $0, 0, result, p_zero;
        }
        """, "=r,f", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip))


@dsl_user_op
def inv_scale_of(e8: Uint32, *, loc=None, ip=None) -> Float32:
    """2^(127 - e8) as an exact float (0 for e8 == 0)."""
    return Float32(llvm.inline_asm(
        T.f32(), [Uint32(e8).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .s32 new_exp;
            .reg .pred p_zero;
            setp.eq.u32 p_zero, $1, 0;
            sub.s32 new_exp, 254, $1;
            max.s32 new_exp, new_exp, 0;
            shl.b32 $0, new_exp, 23;
            @p_zero mov.b32 $0, 0;
        }
        """, "=f,r", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip))


@dsl_user_op
def e4m3_byte(x: Float32, *, loc=None, ip=None) -> Uint32:
    """RN, satfinite e4m3 of x in the low byte."""
    return Uint32(llvm.inline_asm(
        T.i32(), [Float32(x).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 pair;
            cvt.rn.satfinite.e4m3x2.f32 pair, 0f00000000, $1;
            cvt.u32.u16 $0, pair;
        }
        """, "=r,f", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip))


def _asm(ret, args, body, cons, loc, ip):
    """Inline PTX; stores, fences, atomics and loads are side-effecting (no hoisting or CSE across them)."""
    return llvm.inline_asm(ret, args, body, cons, has_side_effects=ret is None or "ld." in body or "atom." in body,
                           is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def f32_from_bits(u: Uint32, *, loc=None, ip=None) -> Float32:
    return Float32(_asm(T.f32(), [Uint32(u).ir_value(loc=loc, ip=ip)], "mov.b32 $0, $1;", "=f,r", loc, ip))


@dsl_user_op
def mx_e8m0(amax: Float32, *, loc=None, ip=None) -> Uint32:
    """MXFP8QuantizeLinearKernel's scale: e8m0 of amax * rcp.approx.ftz(448), rounded toward +inf, satfinite."""
    return Uint32(_asm(T.i32(), [Float32(amax).ir_value(loc=loc, ip=ip)], """
        {
            .reg .pred p_zero, p_has_mant, p_exp_zero, p_tiny_sub, p_ovf;
            .reg .u32 bits, exp_biased, mantissa, bump, result;
            .reg .f32 r448, norm;
            rcp.approx.ftz.f32 r448, 0f43E00000;
            mul.f32 norm, $1, r448;
            setp.le.f32 p_zero, norm, 0f00000000;
            mov.b32 bits, norm;
            shr.b32 exp_biased, bits, 23;
            and.b32 exp_biased, exp_biased, 255;
            and.b32 mantissa, bits, 0x7FFFFF;
            setp.ne.u32 p_has_mant, mantissa, 0;
            selp.u32 bump, 1, 0, p_has_mant;
            setp.eq.u32 p_exp_zero, exp_biased, 0;
            setp.le.u32 p_tiny_sub, mantissa, 0x400000;
            and.pred p_tiny_sub, p_exp_zero, p_tiny_sub;
            @p_tiny_sub mov.u32 bump, 0;
            add.u32 result, exp_biased, bump;
            setp.gt.u32 p_ovf, result, 254;
            selp.u32 result, 254, result, p_ovf;
            selp.u32 $0, 0, result, p_zero;
        }
        """, "=r,f", loc, ip))


@dsl_user_op
def mx_inv(e8: Uint32, amax: Float32, *, loc=None, ip=None) -> Float32:
    """MXFP8QuantizeLinearKernel's output scale: amax != 0 ? rcp.approx.ftz(2^(e8 - 127)) : 0."""
    return Float32(_asm(T.f32(), [Uint32(e8).ir_value(loc=loc, ip=ip), Float32(amax).ir_value(loc=loc, ip=ip)], """
        {
            .reg .b32 sfb;
            .reg .f32 sf, inv;
            .reg .pred pz, pa;
            setp.eq.u32 pz, $1, 0;
            shl.b32 sfb, $1, 23;
            @pz mov.b32 sfb, 0x00400000;
            mov.b32 sf, sfb;
            rcp.approx.ftz.f32 inv, sf;
            setp.eq.f32 pa, $2, 0f00000000;
            selp.f32 $0, 0f00000000, inv, pa;
        }
        """, "=f,r,f", loc, ip))


@dsl_user_op
def e4m3x2_bits(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Uint32:
    """satfinite RN e4m3 of (lo, hi) in bits 7:0 and 15:8."""
    return Uint32(_asm(T.i32(), [Float32(lo).ir_value(loc=loc, ip=ip), Float32(hi).ir_value(loc=loc, ip=ip)], """
        {
            .reg .b16 pair;
            cvt.rn.satfinite.e4m3x2.f32 pair, $2, $1;
            cvt.u32.u16 $0, pair;
        }
        """, "=r,f,f", loc, ip))


@dsl_user_op
def bf16x2_bits(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Uint32:
    """RN bf16 of (lo, hi) in bits 15:0 and 31:16."""
    return Uint32(_asm(T.i32(), [Float32(lo).ir_value(loc=loc, ip=ip), Float32(hi).ir_value(loc=loc, ip=ip)],
                       "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f", loc, ip))


@dsl_user_op
def no_neg_zero(w: Uint32, *, loc=None, ip=None) -> Uint32:
    """bf16 -0 halves -> +0, so no word can equal the Lamport sentinel 0x80000000."""
    return Uint32(_asm(T.i32(), [Uint32(w).ir_value(loc=loc, ip=ip)], """
        {
            .reg .b32 lo, hi, out;
            .reg .pred pl, ph;
            and.b32 lo, $1, 0xFFFF;
            shr.b32 hi, $1, 16;
            setp.eq.u32 pl, lo, 0x8000;
            setp.eq.u32 ph, hi, 0x8000;
            mov.b32 out, $1;
            @pl and.b32 out, out, 0xFFFF0000;
            @ph and.b32 out, out, 0x0000FFFF;
            mov.b32 $0, out;
        }
        """, "=r,r", loc, ip))


@dsl_user_op
def red_add_f32(addr: Int64, v: Float32, *, loc=None, ip=None):
    _asm(None, [Int64(addr).ir_value(loc=loc, ip=ip), Float32(v).ir_value(loc=loc, ip=ip)],
         "red.relaxed.gpu.global.add.f32 [$0], $1;", "l,f", loc, ip)


@dsl_user_op
def fence_acq_rel_gpu(*, loc=None, ip=None):
    _asm(None, [], "fence.acq_rel.gpu;", "", loc, ip)


@dsl_user_op
def atom_add_u64(addr: Int64, v: Int64, *, loc=None, ip=None) -> Int64:
    return Int64(_asm(T.i64(), [Int64(addr).ir_value(loc=loc, ip=ip), Int64(v).ir_value(loc=loc, ip=ip)],
                      "atom.acq_rel.gpu.global.add.u64 $0, [$1], $2;", "=l,l,l", loc, ip))


@dsl_user_op
def ld_acquire_u64(addr: Int64, *, loc=None, ip=None) -> Int64:
    return Int64(_asm(T.i64(), [Int64(addr).ir_value(loc=loc, ip=ip)], "ld.acquire.gpu.global.u64 $0, [$1];",
                      "=l,l", loc, ip))


@dsl_user_op
def ld_cg_f32(addr: Int64, *, loc=None, ip=None) -> Float32:
    return Float32(_asm(T.f32(), [Int64(addr).ir_value(loc=loc, ip=ip)], "ld.global.cg.f32 $0, [$1];", "=f,l",
                        loc, ip))


@dsl_user_op
def st_cg_zero_f32(addr: Int64, *, loc=None, ip=None):
    _asm(None, [Int64(addr).ir_value(loc=loc, ip=ip)], "st.global.cg.f32 [$0], 0f00000000;", "l", loc, ip)


@dsl_user_op
def st_v4_u32(addr: Int64, a: Uint32, b: Uint32, c: Uint32, d: Uint32, *, loc=None, ip=None):
    """16-byte store to an ordinary or NVLS multicast global address."""
    _asm(None, [Int64(addr).ir_value(loc=loc, ip=ip)] + [Uint32(x).ir_value(loc=loc, ip=ip) for x in (a, b, c, d)],
         "st.global.v4.u32 [$0], {$1, $2, $3, $4};", "l,r,r,r,r", loc, ip)


@cute.jit
def sf_offset(row: Int32, kb: Int32, kt: cutlass.Constexpr):
    """Byte offset of SF (row, k-block) in the 128x4 swizzled atom layout (kt = ceil(k-blocks / 4))."""
    return ((row // 128) * (512 * kt) + (kb // 4) * 512 + (row % 32) * 16 + ((row % 128) // 32) * 4
            + kb % 4)


@dsl_user_op
def tmem_ld1(taddr: Int32, *, loc=None, ip=None) -> Float32:
    """One fp32 per thread from TMEM column (taddr & 0xFFFF), lanes from taddr >> 16 (32x32b.x1), then wait::ld."""
    return Float32(llvm.inline_asm(
        T.f32(), [Int32(taddr).ir_value(loc=loc, ip=ip)],
        """
        tcgen05.ld.sync.aligned.32x32b.x1.b32 {$0}, [$1];
        tcgen05.wait::ld.sync.aligned;
        """, "=f,r", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip))


@cute.jit
def low_bit_index(x: Int32) -> Int32:
    """Index of the lowest set bit of x (x != 0)."""
    low = x & (Int32(0) - x)
    return Int32(cute.arch.popc(low - Int32(1)))


MAX_M = 64  # tokens per call served by the stage
TP = 8  # latent tail ranks (K3 TP 8): multicast slot (token, rank)
TN = 16  # slots per tile (MMA N)
TMAX = E + MAX_M * TOPK // TN  # tile capacity: sum over experts of ceil(c/16) <= E + pairs/16
PERM_THREADS = 512


# ------------------------------------------------------------------ permute

QUANT_THREADS = H // 8  # permute: one thread per 8 bf16 of the token row; 4 threads per 32-column MX block


class MoePermute:
    """grid = M CTAs. Input: the routed latent x [M, H] bf16 (read as M*H/2 words). Outputs: tmeta[t] = (expert,
    slots used, first output row, 0) per tile, cnt[0] = tiles; x_perm [TMAX*16, H] e4m3 and sfx_perm (pad SF
    layout, one 128-row atom per tile) per slot; bf16 weights and the permute map per (token, k) (-1 for an expert
    not on this rank); rtok / rw per output row (token, fp32 weight bits) for fc2's finalize."""

    def __init__(self, e_off: int = 0, iket: bool = False):
        self.e_off = e_off
        self.iket = iket

    @cute.jit
    def __call__(self, p_packed: cute.Pointer, p_x: cute.Pointer, p_tmeta: cute.Pointer, p_cnt: cute.Pointer,
                 p_wts: cute.Pointer, p_perm: cute.Pointer, p_xperm: cute.Pointer, p_sfperm: cute.Pointer,
                 p_rtok: cute.Pointer, p_rw: cute.Pointer, m: Int32, stream):
        lay = lambda n: cute.make_layout(n)
        self.kernel(cute.make_tensor(p_packed, lay(MAX_M * TOPK)), cute.make_tensor(p_x, lay(MAX_M * H // 2)),
                    cute.make_tensor(p_tmeta, lay(TMAX * 4)),
                    cute.make_tensor(p_cnt, lay(4)), cute.make_tensor(p_wts, lay(MAX_M * TOPK)),
                    cute.make_tensor(p_perm, lay(MAX_M * TOPK)), cute.make_tensor(p_xperm, lay(TMAX * TN * H // 4)),
                    cute.make_tensor(p_sfperm, lay(TMAX * 128 * (H // SF_VEC))),
                    cute.make_tensor(p_rtok, lay(MAX_M * TOPK)), cute.make_tensor(p_rw, lay(MAX_M * TOPK)), m).launch(
            grid=[m, 1, 1], block=[PERM_THREADS, 1, 1], stream=stream, use_pdl=True)

    @cute.jit
    def block_exclusive_scan(self, x: Int32, lane: Int32, warp: Int32, wsum: cute.Tensor):
        """(exclusive prefix of x over the CTA in thread order, CTA total). Uses wsum[32] and two barriers."""
        inc = x
        for d in cutlass.range_constexpr(5):
            off = 1 << d
            y = cute.arch.shuffle_sync(inc, cutlass.select_(lane >= Int32(off), lane - Int32(off), Int32(0)))
            inc = inc + cutlass.select_(lane >= Int32(off), y, Int32(0))
        if lane == Int32(31):
            wsum[warp] = inc
        cute.arch.barrier()
        base = Int32(0)
        total = Int32(0)
        for w in cutlass.range_constexpr(PERM_THREADS // 32):
            v = wsum[w]
            base = base + cutlass.select_(Int32(w) < warp, v, Int32(0))
            total = total + v
        cute.arch.barrier()
        return base + inc - x, total

    @cute.kernel
    def kernel(self, packed: cute.Tensor, x: cute.Tensor, tmeta: cute.Tensor, cnt: cute.Tensor,
               wts: cute.Tensor, perm: cute.Tensor, xperm: cute.Tensor, sfperm: cute.Tensor, rtok: cute.Tensor,
               rw: cute.Tensor, m: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        t, _, _ = cute.arch.block_idx()
        lane = tidx % 32
        warp = tidx // 32
        EPT = 2  # experts per thread in the scans (E <= 2 * PERM_THREADS)

        @cute.struct
        class Storage:
            pk: cute.struct.MemRange[Int32, MAX_M * TOPK]
            hist: cute.struct.MemRange[Int32, PERM_THREADS * EPT]
            toff: cute.struct.MemRange[Int32, PERM_THREADS * EPT]
            roff: cute.struct.MemRange[Int32, PERM_THREADS * EPT]
            dst: cute.struct.MemRange[Int32, TOPK]
            wsum: cute.struct.MemRange[Int32, 32]

        smem = utils.SmemAllocator()
        st = smem.allocate(Storage)
        mk = lambda r, n: cute.make_tensor(r.data_ptr(), cute.make_layout(n))
        pk, hist, toff, roff = (mk(st.pk, MAX_M * TOPK), mk(st.hist, PERM_THREADS * EPT),
                                mk(st.toff, PERM_THREADS * EPT), mk(st.roff, PERM_THREADS * EPT))
        dst = mk(st.dst, TOPK)
        wsum = mk(st.wsum, 32)

        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        _push(self.iket, "main_index")
        npairs = m * TOPK
        for i in cutlass.range_constexpr(EPT):
            hist[tidx * EPT + i] = Int32(0)
        cute.arch.barrier()
        for i in cutlass.range(tidx, npairs, PERM_THREADS):
            e = (packed[i] >> 16) - Int32(self.e_off)
            if e >= Int32(0):
                if e < Int32(E):
                    cute.arch.atomic_add(hist.iterator + e, Int32(1), scope="cta")
                else:
                    e = Int32(-1)
            else:
                e = Int32(-1)
            pk[i] = e
        cute.arch.barrier()

        # per-thread experts tidx*EPT .. +EPT-1: rows and tiles, then CTA-wide exclusive scans
        rsum = Int32(0)
        tsum = Int32(0)
        for i in cutlass.range_constexpr(EPT):
            c = hist[tidx * EPT + i]
            rsum = rsum + c
            tsum = tsum + (c + Int32(TN - 1)) // Int32(TN)
        rbase, _ = self.block_exclusive_scan(rsum, lane, warp, wsum)
        tbase, n_tiles = self.block_exclusive_scan(tsum, lane, warp, wsum)
        for i in cutlass.range_constexpr(EPT):
            c = hist[tidx * EPT + i]
            roff[tidx * EPT + i] = rbase
            toff[tidx * EPT + i] = tbase
            if t == 0:
                for q in cutlass.range((c + Int32(TN - 1)) // Int32(TN)):
                    tile = tbase + q
                    tmeta[4 * tile] = Int32(tidx * EPT + i)
                    tmeta[4 * tile + 1] = cutlass.select_(c - q * TN < Int32(TN), c - q * TN, Int32(TN))
                    tmeta[4 * tile + 2] = rbase + q * TN
                    tmeta[4 * tile + 3] = Int32(0)
            rbase = rbase + c
            tbase = tbase + (c + Int32(TN - 1)) // Int32(TN)
        if t == 0:
            if tidx == 0:
                cnt[0] = n_tiles
        cute.arch.barrier()

        # pair k of token t is handled by warp k: rank = earlier tokens routed to the same expert
        if warp < TOPK:
            pidx = t * TOPK + warp
            e = pk[pidx]
            r = Int32(0)
            for c0 in cutlass.range(0, t * TOPK, 32):
                idx = c0 + lane
                hit = Int32(0)
                if idx < t * TOPK:
                    if pk[idx] == e:
                        hit = Int32(1)
                r = r + Int32(cute.arch.popc(cute.arch.vote_ballot_sync(hit != Int32(0))))
            if lane == 0:
                wbits = packed[pidx] & Int32(0xFFFF)
                wts[pidx] = cutlass.Int16(wbits)
                if e >= Int32(0):
                    row = roff[e] + r
                    perm[pidx] = row
                    rtok[row] = t
                    rw[row] = wbits << 16  # fp32 bits of the bf16 weight
                    dst[warp] = toff[e] * TN + r  # slot row in x_perm: tile = toff[e] + r // 16, slot = r % 16
                else:
                    perm[pidx] = Int32(-1)
                    dst[warp] = Int32(-1)
        cute.arch.barrier()
        _pop(self.iket)

        # quant + gather: thread i holds bf16 columns 8i..8i+7 of token t; an MX block is 4 neighbouring lanes
        _push(self.iket, "edge_quant_gather")
        if warp < QUANT_THREADS // 32:
            base = t * (H // 2) + 4 * tidx
            f = cute.make_rmem_tensor(cute.make_layout(8), Float32)
            for j in cutlass.range_constexpr(4):
                wd = Uint32(x[base + j])
                f[2 * j] = f32_from_bits(wd << Uint32(16))
                f[2 * j + 1] = f32_from_bits(wd & Uint32(0xFFFF0000))
            a = Float32(0.0)
            for j in cutlass.range_constexpr(8):
                v = f[j]
                a = cute.arch.fmax(a, cutlass.select_(v < Float32(0.0), -v, v))
            a = cute.arch.fmax(a, cute.arch.shuffle_sync_bfly(a, offset=1))
            a = cute.arch.fmax(a, cute.arch.shuffle_sync_bfly(a, offset=2))
            e8 = mx_e8m0(a)
            inv = mx_inv(e8, a)
            q0 = e4m3x2_bits(f[0] * inv, f[1] * inv) | (e4m3x2_bits(f[2] * inv, f[3] * inv) << Uint32(16))
            q1 = e4m3x2_bits(f[4] * inv, f[5] * inv) | (e4m3x2_bits(f[6] * inv, f[7] * inv) << Uint32(16))
            kb = tidx // 4
            words = H // 4
            for k in cutlass.range_constexpr(TOPK):
                d = dst[k]
                if d >= Int32(0):
                    xperm[d * words + 2 * tidx] = Int32(q0)
                    xperm[d * words + 2 * tidx + 1] = Int32(q1)
                    if tidx % 4 == 0:
                        sfperm[sf_offset((d // TN) * 128 + d % TN, kb, (H // SF_VEC) // 4)] = cutlass.Uint8(e8)
        _pop(self.iket)


# ------------------------------------------------------------------ grouped GEMM

class MoeGemm:
    """Persistent warp-specialized grouped GEMM over (tile, 128-row weight m-tile) units, u -> (tile = u // MT,
    mt = u % MT), CTA b takes u = b, b + grid, ... Warps 0-3 epilogue, warp 4 MMA, warp 5 TMA. Tile records are
    staged in smem once per CTA."""

    def __init__(self, fc1: bool, num_ctas: int, alpha: float = 4.0, beta: float = 25.0, k_tile: int = 128,
                 stages: int = 0, iket: bool = False, variant: int = 0):
        self.iket = iket
        self.variant = variant  # bench only: 0 production fc2, 1 red.add without the finalize, 2 neither
        self.fc1 = fc1
        self.K = H if fc1 else I
        self.rows_per_e = 2 * I if fc1 else H
        self.MT = self.rows_per_e // 128
        self.kt = k_tile
        self.k_tiles = self.K // k_tile
        assert self.K % k_tile == 0
        self.num_ctas = num_ctas
        self.alpha = alpha
        self.beta = beta
        stage_bytes = 128 * k_tile + TN * k_tile + 2 * 128 * (k_tile // SF_VEC)
        fit = (227 * 1024 - TMAX * 16 - 4096) // stage_bytes
        self.num_ab_stage = min(stages, fit) if stages else min(12, fit)
        self.num_acc_stage = 4
        self.mma_tiler = (128, TN, k_tile)
        self.mma_tiler_sfb = (128, 128, k_tile)
        self.threads = 192
        self.epi_threads = 128
        self.num_tmem_alloc_cols = 128

    @cute.jit
    def __call__(self, p_a: cute.Pointer, p_sfa: cute.Pointer, p_b: cute.Pointer, p_sfb: cute.Pointer,
                 p_tmeta: cute.Pointer, p_cnt: cute.Pointer, p_out: cute.Pointer, p_osf: cute.Pointer,
                 p_rtok: cute.Pointer, p_rw: cute.Pointer, p_ctr: cute.Pointer, p_flags: cute.Pointer, mc: Int64,
                 rank: Int32, m: Int32, stream):
        """fc1: out = inter (u8), osf = its SF. fc2: out = acc [MAX_M, H] fp32 (zero between calls), osf unused
        (a dummy), rtok / rw = permute's row records, ctr = u64 grid arrival counter (only grows),
        flags / mc / rank = the latent tail's routed Lamport flags, multicast base address and this rank."""
        K = self.K
        a_rows = E * self.rows_per_e
        a_tensor = cute.make_tensor(p_a, cute.make_layout((a_rows, K, 1), stride=(K, 1, a_rows * K)))
        sfa_tensor = cute.make_tensor(p_sfa, blockscaled_utils.tile_atom_to_shape_SF(a_tensor.shape, SF_VEC))
        b_tensor = cute.make_tensor(p_b, cute.make_layout((TMAX * TN, K, 1), stride=(K, 1, TMAX * TN * K)))
        sfb_tensor = cute.make_tensor(p_sfb, blockscaled_utils.tile_atom_to_shape_SF((TMAX * 128, K, 1), SF_VEC))
        if cutlass.const_expr(self.fc1):
            out = cute.make_tensor(p_out, cute.make_layout(TMAX * TN * I))
            osf = cute.make_tensor(p_osf, cute.make_layout(TMAX * 128 * (I // SF_VEC)))
        else:
            out = cute.make_tensor(p_out, cute.make_layout(MAX_M * H))
            osf = cute.make_tensor(p_osf, cute.make_layout(4))
        tmeta = cute.make_tensor(p_tmeta, cute.make_layout(TMAX * 4))
        cnt = cute.make_tensor(p_cnt, cute.make_layout(4))
        rtok = cute.make_tensor(p_rtok, cute.make_layout(MAX_M * TOPK))
        rw = cute.make_tensor(p_rw, cute.make_layout(MAX_M * TOPK))
        ctr = cute.make_tensor(p_ctr, cute.make_layout(4))
        flags = cute.make_tensor(p_flags, cute.make_layout(4))

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            A_DTYPE, B_DTYPE, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K, SF_DTYPE, SF_VEC,
            tcgen05.CtaGroup.ONE, self.mma_tiler[:2])
        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            A_DTYPE, B_DTYPE, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K, SF_DTYPE, SF_VEC,
            tcgen05.CtaGroup.ONE, self.mma_tiler_sfb[:2])
        cluster_vmnk = cute.tiled_divide(cute.make_layout((1, 1, 1)), (tiled_mma.thr_id.shape,))

        a_smem_staged = sm100_utils.make_smem_layout_a(tiled_mma, self.mma_tiler, B_DTYPE, self.num_ab_stage)
        b_smem_staged = sm100_utils.make_smem_layout_b(tiled_mma, self.mma_tiler, B_DTYPE, self.num_ab_stage)
        sfa_smem_staged = blockscaled_utils.make_smem_layout_sfa(tiled_mma, self.mma_tiler, SF_VEC,
                                                                 self.num_ab_stage)
        sfb_smem_staged = blockscaled_utils.make_smem_layout_sfb(tiled_mma_sfb, self.mma_tiler_sfb, SF_VEC,
                                                                 self.num_ab_stage)
        a_smem = cute.slice_(a_smem_staged, (None, None, None, 0))
        b_smem = cute.slice_(b_smem_staged, (None, None, None, 0))
        sfa_smem = cute.slice_(sfa_smem_staged, (None, None, None, 0))
        sfb_smem = cute.slice_(sfb_smem_staged, (None, None, None, 0))
        g2s = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_atom_a, tma_a = cute.nvgpu.make_tiled_tma_atom_A(g2s, a_tensor, a_smem, self.mma_tiler, tiled_mma,
                                                             cluster_vmnk.shape, internal_type=cutlass.Uint8)
        tma_atom_b, tma_b = cute.nvgpu.make_tiled_tma_atom_B(g2s, b_tensor, b_smem, self.mma_tiler, tiled_mma,
                                                             cluster_vmnk.shape)
        tma_atom_sfa, tma_sfa = cute.nvgpu.make_tiled_tma_atom_A(g2s, sfa_tensor, sfa_smem, self.mma_tiler,
                                                                 tiled_mma, cluster_vmnk.shape,
                                                                 internal_type=cutlass.Int16)
        tma_atom_sfb, tma_sfb = cute.nvgpu.make_tiled_tma_atom_B(g2s, sfb_tensor, sfb_smem, self.mma_tiler_sfb,
                                                                 tiled_mma_sfb, cluster_vmnk.shape,
                                                                 internal_type=cutlass.Int16)
        # expect-tx counts global bytes: A is packed fp4 in global memory even though smem holds 8-bit containers
        self.num_tma_load_bytes = (128 * self.kt // 2 + cute.size_in_bytes(B_DTYPE, b_smem)
                                   + cute.size_in_bytes(SF_DTYPE, sfa_smem) + cute.size_in_bytes(SF_DTYPE, sfb_smem))
        self.kernel(tiled_mma, tiled_mma_sfb, tma_atom_a, tma_a, tma_atom_b, tma_b, tma_atom_sfa, tma_sfa,
                    tma_atom_sfb, tma_sfb, a_smem_staged, b_smem_staged, sfa_smem_staged, sfb_smem_staged,
                    tmeta, cnt, out, osf, rtok, rw, ctr, flags, mc, rank, m).launch(
            grid=[self.num_ctas, 1, 1], block=[self.threads, 1, 1], cluster=(1, 1, 1), stream=stream,
            use_pdl=True)

    @cute.jit
    def situ(self, g: Float32, u: Float32) -> Float32:
        left = Float32(self.alpha) * cute.math.tanh(g * Float32(1.0 / self.alpha)) / (
            Float32(1.0) + cute.math.exp(-g))
        right = Float32(self.beta) * cute.math.tanh(u * Float32(1.0 / self.beta))
        return left * right

    @cute.jit
    def unit(self, u: Int32, n_tiles: Int32):
        """(tile, m-tile) of unit u, tile-major."""
        tile = u // Int32(self.MT)
        return tile, u - tile * Int32(self.MT)

    @cute.jit
    def finalize(self, bid: Int32, m: Int32, tidx: Int32, out: cute.Tensor, flags: cute.Tensor, mc: Int64,
                 rank: Int32):
        """After the grid-wide arrival: this CTA's share of acc [m, H] fp32 -> bf16 (-0 -> +0) -> the tail's
        multicast slot (token, rank) of generation flags[0]; acc back to zero. One 16-byte store per 8 channels."""
        gen_base = mc + Int64(Uint32(flags[0])) * Int64(Uint32(flags[2]))
        step = Int32(self.num_ctas * self.epi_threads)
        for i in cutlass.range(bid * Int32(self.epi_threads) + tidx, m * (H // 8), step):
            tok = i // (H // 8)
            ch0 = (i - tok * (H // 8)) * 8
            addr0 = Int64((out.iterator + (tok * H + ch0)).toint())
            vals = cute.make_rmem_tensor(cute.make_layout(8), Float32)
            for j in cutlass.range_constexpr(8):
                vals[j] = ld_cg_f32(addr0 + Int64(4 * j))
            for j in cutlass.range_constexpr(8):
                st_cg_zero_f32(addr0 + Int64(4 * j))
            dst = gen_base + (Int64(tok * TP + rank) * Int64(H) + Int64(ch0)) * Int64(2)
            st_v4_u32(dst, no_neg_zero(bf16x2_bits(vals[0], vals[1])), no_neg_zero(bf16x2_bits(vals[2], vals[3])),
                      no_neg_zero(bf16x2_bits(vals[4], vals[5])), no_neg_zero(bf16x2_bits(vals[6], vals[7])))

    @cute.jit
    def slot(self, s: Int32, j: Int32, taddr: Int32, tile: Int32, mt: Int32, lane: Int32, w: Int32, row0: Int32,
             s_amax: cute.Tensor, epi_barrier, out: cute.Tensor, osf: cute.Tensor, rtok: cute.Tensor,
             rw: cute.Tensor):
        """Epilogue of slot s (accumulator column s) of a tile; j counts slots over the CTA run (fc1's amax double
        buffer)."""
        v = tmem_ld1(taddr + s)
        if cutlass.const_expr(self.fc1):
            # lane L of warp w holds intermediate column 64*mt + 16*w + 2*(L%8) + L//16; up rows have lane bit 3
            # clear, the gate is lane L^8; a 32-column MX block spans warps (2b, 2b+1)
            other = cute.arch.shuffle_sync_bfly(v, offset=8)
            is_up = (lane & 8) == 0
            up = cutlass.select_(is_up, v, other)
            gate = cutlass.select_(is_up, other, v)
            h = self.situ(gate, up)
            a = cutlass.select_(h < Float32(0.0), -h, h)
            a = cute.arch.fmax(a, cute.arch.shuffle_sync_bfly(a, offset=16))
            a = cute.arch.fmax(a, cute.arch.shuffle_sync_bfly(a, offset=4))
            a = cute.arch.fmax(a, cute.arch.shuffle_sync_bfly(a, offset=2))
            a = cute.arch.fmax(a, cute.arch.shuffle_sync_bfly(a, offset=1))
            buf = (j % 2) * 4
            if lane == 0:
                s_amax[buf + w] = a
            epi_barrier.arrive_and_wait()
            amax = cute.arch.fmax(a, s_amax[buf + (w ^ 1)])
            e8 = ue8m0_of_amax(amax)
            q = e4m3_byte(h * inv_scale_of(e8))
            c = mt * 64 + w * 16 + 2 * (lane % 8) + lane // 16
            if is_up:
                out[(tile * TN + s) * I + c] = cutlass.Uint8(q & Uint32(0xFF))
            if lane == 0:
                if (w % 2) == 0:
                    osf[sf_offset(tile * 128 + s, mt * 2 + w // 2, (I // SF_VEC) // 4)] = cutlass.Uint8(e8)
        else:
            # lane L of warp w holds channel 128*mt + 32*w + 4*(L%8) + L//8; the pair's token and top-K weight
            # come from permute's row records
            ch = mt * 128 + w * 32 + 4 * (lane % 8) + lane // 8
            row = row0 + s
            tok = rtok[row]
            wgt = f32_from_bits(Uint32(rw[row]))
            if cutlass.const_expr(self.variant < 2):
                red_add_f32(Int64((out.iterator + (tok * H + ch)).toint()), v * wgt)

    @cute.kernel
    def kernel(self, tiled_mma: cute.TiledMma, tiled_mma_sfb: cute.TiledMma, tma_atom_a: cute.CopyAtom,
               mA: cute.Tensor, tma_atom_b: cute.CopyAtom, mB: cute.Tensor, tma_atom_sfa: cute.CopyAtom,
               mSFA: cute.Tensor, tma_atom_sfb: cute.CopyAtom, mSFB: cute.Tensor,
               a_smem_staged: cute.ComposedLayout, b_smem_staged: cute.ComposedLayout,
               sfa_smem_staged: cute.Layout, sfb_smem_staged: cute.Layout, tmeta: cute.Tensor, cnt: cute.Tensor,
               out: cute.Tensor, osf: cute.Tensor, rtok: cute.Tensor, rw: cute.Tensor, ctr: cute.Tensor,
               flags: cute.Tensor, mc: Int64, rank: Int32, m: Int32):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        nblk = Int32(self.num_ctas)
        MT = self.MT

        if warp_idx == 5:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb)

        @cute.struct
        class SharedStorage:
            ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tmem_holding_buf: Int32
            amax: cute.struct.MemRange[Float32, 8]
            meta: cute.struct.MemRange[Int32, TMAX * 4]

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sA = smem.allocate_tensor(element_type=B_DTYPE, layout=a_smem_staged.outer, byte_alignment=1024,
                                  swizzle=a_smem_staged.inner)
        sB = smem.allocate_tensor(element_type=B_DTYPE, layout=b_smem_staged.outer, byte_alignment=1024,
                                  swizzle=b_smem_staged.inner)
        sSFA = smem.allocate_tensor(element_type=SF_DTYPE, layout=sfa_smem_staged, byte_alignment=128)
        sSFB = smem.allocate_tensor(element_type=SF_DTYPE, layout=sfb_smem_staged, byte_alignment=128)
        s_amax = cute.make_tensor(storage.amax.data_ptr(), cute.make_layout(8))
        s_meta = cute.make_tensor(storage.meta.data_ptr(), cute.make_layout(TMAX * 4))
        s_tmem = cute.make_tensor(storage.tmem_holding_buf.ptr, cute.make_layout(1))

        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_mbar_ptr.data_ptr(), num_stages=self.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.num_tma_load_bytes)
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar_ptr.data_ptr(), num_stages=self.num_acc_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 4))
        cta_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=self.threads)
        epi_barrier = pipeline.NamedBarrier(barrier_id=2, num_threads=self.epi_threads)
        tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=3, num_threads=32 + self.epi_threads)
        tmem = utils.TmemAllocator(storage.tmem_holding_buf.ptr, barrier_for_retrieve=tmem_alloc_barrier,
                                   allocator_warp_id=0)

        gA = cute.local_tile(mA, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
        gB = cute.local_tile(mB, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None))
        gSFA = cute.local_tile(mSFA, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
        gSFB = cute.local_tile(mSFB, cute.slice_(self.mma_tiler_sfb, (0, None, None)), (None, None, None))
        thr_mma = tiled_mma.get_slice(0)
        thr_mma_sfb = tiled_mma_sfb.get_slice(0)
        tCgA = thr_mma.partition_A(gA)
        tCgB = thr_mma.partition_B(gB)
        tCgSFA = thr_mma.partition_A(gSFA)
        tCgSFB = thr_mma_sfb.partition_B(gSFB)
        one = cute.make_layout(1)
        tAsA, tAgA = cpasync.tma_partition(tma_atom_a, 0, one, cute.group_modes(sA, 0, 3),
                                           cute.group_modes(tCgA, 0, 3))
        tBsB, tBgB = cpasync.tma_partition(tma_atom_b, 0, one, cute.group_modes(sB, 0, 3),
                                           cute.group_modes(tCgB, 0, 3))
        tAsSFA, tAgSFA = cpasync.tma_partition(tma_atom_sfa, 0, one, cute.group_modes(sSFA, 0, 3),
                                               cute.group_modes(tCgSFA, 0, 3))
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)
        tBsSFB, tBgSFB = cpasync.tma_partition(tma_atom_sfb, 0, one, cute.group_modes(sSFB, 0, 3),
                                               cute.group_modes(tCgSFB, 0, 3))
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        # PDL: everything above touches only smem / TMEM / descriptors; the previous launch's outputs are read below
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        n_tiles = cnt[0]
        for i in cutlass.range(tidx, n_tiles * 4, self.threads):
            s_meta[i] = tmeta[i]
        cta_barrier.arrive_and_wait()
        total = n_tiles * Int32(MT)

        # ---------------------------------------------------------- TMA warp
        if warp_idx == 5:
            prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_ab_stage)
            u = Int32(bid)
            while u < total:
                tile, mt = self.unit(u, n_tiles)
                m_idx = s_meta[4 * tile] * MT + mt
                _push(self.iket, "main_tma_unit")
                for k in cutlass.range(self.k_tiles, unroll=1):
                    ab_pipeline.producer_acquire(prod)
                    bar = ab_pipeline.producer_get_barrier(prod)
                    cute.copy(tma_atom_a, tAgA[(None, m_idx, k, 0)], tAsA[(None, prod.index)], tma_bar_ptr=bar)
                    cute.copy(tma_atom_b, tBgB[(None, tile, k, 0)], tBsB[(None, prod.index)], tma_bar_ptr=bar)
                    cute.copy(tma_atom_sfa, tAgSFA[(None, m_idx, k, 0)], tAsSFA[(None, prod.index)],
                              tma_bar_ptr=bar)
                    cute.copy(tma_atom_sfb, tBgSFB[(None, tile, k, 0)], tBsSFB[(None, prod.index)],
                              tma_bar_ptr=bar)
                    prod.advance()
                _pop(self.iket)
                u += nblk
            ab_pipeline.producer_tail(prod)

        # ---------------------------------------------------------- MMA warp
        if warp_idx == 4:
            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(Float32)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
            sfa_tmem_ptr = cute.recast_ptr(acc_tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base),
                                           dtype=SF_DTYPE)
            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(tiled_mma, self.mma_tiler, SF_VEC,
                                                                   cute.slice_(sfa_smem_staged,
                                                                               (None, None, None, 0)))
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)
            sfb_tmem_ptr = cute.recast_ptr(acc_tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc_base)
                                           + tcgen05.find_tmem_tensor_col_offset(tCtSFA), dtype=SF_DTYPE)
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(tiled_mma_sfb, self.mma_tiler_sfb, SF_VEC,
                                                                   cute.slice_(sfb_smem_staged,
                                                                               (None, None, None, 0)))
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

            copy_atom_s2t = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE), SF_DTYPE)
            tCtSFA_compact = cute.filter_zeros(tCtSFA)
            tiled_s2t_sfa = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFA_compact)
            thr_s2t_sfa = tiled_s2t_sfa.get_slice(0)
            tCsSFA_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_s2t_sfa,
                                                          thr_s2t_sfa.partition_S(cute.filter_zeros(sSFA)))
            tCtSFA_s2t = thr_s2t_sfa.partition_D(tCtSFA_compact)
            tCtSFB_compact = cute.filter_zeros(tCtSFB)
            tiled_s2t_sfb = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFB_compact)
            thr_s2t_sfb = tiled_s2t_sfb.get_slice(0)
            tCsSFB_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_s2t_sfb,
                                                          thr_s2t_sfb.partition_S(cute.filter_zeros(sSFB)))
            tCtSFB_s2t = thr_s2t_sfb.partition_D(tCtSFB_compact)

            cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_ab_stage)
            acc_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_acc_stage)
            u = Int32(bid)
            while u < total:
                tCtAcc = tCtAcc_base[(None, None, None, acc_prod.index)]
                _push(self.iket, "sync_acc_empty")
                acc_pipeline.producer_acquire(acc_prod)
                _pop(self.iket)
                _push(self.iket, "main_mma_unit")
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for k in cutlass.range(self.k_tiles, unroll=1):
                    ab_pipeline.consumer_wait(cons)
                    s2t_coord = (None, None, None, None, cons.index)
                    cute.copy(tiled_s2t_sfa, tCsSFA_s2t[s2t_coord], tCtSFA_s2t)
                    cute.copy(tiled_s2t_sfb, tCsSFB_s2t[s2t_coord], tCtSFB_s2t)
                    for kb in cutlass.range(cute.size(tCrA, mode=[2]), unroll_full=True):
                        tiled_mma.set(tcgen05.Field.SFA, tCtSFA[(None, None, kb)].iterator)
                        tiled_mma.set(tcgen05.Field.SFB, tCtSFB[(None, None, kb)].iterator)
                        cute.gemm(tiled_mma, tCtAcc, tCrA[(None, None, kb, cons.index)],
                                  tCrB[(None, None, kb, cons.index)], tCtAcc)
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    ab_pipeline.consumer_release(cons)
                    cons.advance()
                acc_pipeline.producer_commit(acc_prod)
                _pop(self.iket)
                acc_prod.advance()
                u += nblk
            acc_pipeline.producer_tail(acc_prod)

        # ---------------------------------------------------------- epilogue warps
        if warp_idx < 4:
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(Float32)
            lane = tidx % 32
            w = tidx // 32
            # TMEM address: lanes in bits 31:16 (warp w reads lanes 32w..32w+31 = weight rows), columns in 15:0
            # (acc stage s, slot n at column s*TN + n; M=128 cta_group::1 accumulator: row = lane)
            tbase = s_tmem[0] + ((w * 32) << 16)
            acc_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_acc_stage)
            j = Int32(0)
            u = Int32(bid)
            while u < total:
                tile, mt = self.unit(u, n_tiles)
                used = s_meta[4 * tile + 1]
                row0 = s_meta[4 * tile + 2]
                taddr = tbase + acc_cons.index * TN
                _push(self.iket, "sync_acc_full")
                acc_pipeline.consumer_wait(acc_cons)
                _pop(self.iket)
                _push(self.iket, "edge_epilogue")
                for s in cutlass.range(used, unroll=1):
                    self.slot(s, j, taddr, tile, mt, lane, w, row0, s_amax, epi_barrier, out, osf, rtok, rw)
                    j = j + 1
                cute.arch.fence_view_async_tmem_load()
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_cons)
                acc_cons.advance()
                _pop(self.iket)
                u += nblk
            if cutlass.const_expr(not self.fc1 and self.variant == 0):
                # grid-wide arrival: every thread's red.add ordered before thread 0's acq_rel arrival (CUTLASS
                # semaphore pattern), then wait for all CTAs of this launch (generation = old // grid), then the
                # whole grid finalizes. The counter only grows; 148 CTAs x 92 layers per step never wraps a u64.
                _push(self.iket, "sync_grid_arrive")
                fence_acq_rel_gpu()
                epi_barrier.arrive_and_wait()
                if tidx == 0:
                    ctr_addr = Int64(ctr.iterator.toint())
                    old = atom_add_u64(ctr_addr, Int64(1))
                    target = (old // Int64(self.num_ctas) + Int64(1)) * Int64(self.num_ctas)
                    seen = ld_acquire_u64(ctr_addr)
                    while seen < target:
                        seen = ld_acquire_u64(ctr_addr)
                epi_barrier.arrive_and_wait()
                fence_acq_rel_gpu()
                _pop(self.iket)
                _push(self.iket, "edge_finalize_store")
                self.finalize(Int32(bid), m, tidx, out, flags, mc, rank)
                _pop(self.iket)
            tmem.relinquish_alloc_permit()
            epi_barrier.arrive_and_wait()
            tmem.free(acc_tmem_ptr)
