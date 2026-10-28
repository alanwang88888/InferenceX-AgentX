# SPDX-License-Identifier: Apache-2.0
"""Draft AR consumer: two-shot Lamport all-reduce (TP8) + residual add + RMSNorm in one launch.

Replaces the served flashinfer fused op (`fused_allreduce_rms_norm`, vllm/models/common/ops/
fused_allreduce_rms_norm.py) at the draft call sites (dspark_mla.py:117, 125, 427). Same outputs:
  residual_out = bf16(bf16(sum_r x_r) + residual)        (the served op writes it into allreduce_in)
  normed       = bf16(residual_out * rstd * gamma)
Protocol = vLLM common/twoshot.py (push 8-column chunks to the column owner's receive slot, owner polls the
TP slots, sums in rank order, rounds once, multicasts the BF16 sum into every rank's gather row, re-arms;
every rank polls its gather row and re-arms it). Arithmetic = the served kernel's, bit for bit
(flashinfer 0.7.2 trtllm_allreduce_fusion.cuh oneshot Lamport, built -use_fast_math; SASS read):
  sum      FADD.FTZ in rank order 0..7, one RN rounding to bf16          (allreduce_sum, cuh:1468-1477)
  residual FADD.FTZ(sum, residual), RN to bf16                           (vec_add, cuh:813-820; 1110)
  sumsq    per thread FFMA.FTZ chain over its 8 elements; warp butterfly; block butterfly over the
           warp sums (zero padded to 32 lanes); cluster sum 0 + b0 + b1 ...  (rms_norm, cuh:1291-1323)
  rstd     MUFU.RSQ(FFMA.FTZ(acc, MUFU.RCP(hidden), eps))              (SASS 0x11e0, 0x41d0, 0x41e0)
  out      FMUL.FTZ(FMUL.FTZ(y, rstd), FADD.FTZ(gamma, 0)), RN to bf16    (cuh:1325-1327)
The served block = WB warps (7 at M <= 37, 14 at M 38..74 on 148 SMs), so the sumsq tree depends on M;
compile one variant per WB. Two changes from twoshot.reduce, both for the served bits: add.ftz (served
FADD.FTZ) instead of add.rn; and a -0 sum half is sent as 0x8001 (-2^-133) instead of +0 (twoshot's "latent"
policy). -0 is the gather row's sentinel, a sum after FADD.FTZ is never a bf16 subnormal, and the consumer's
FADD.FTZ flushes 0x8001 back to -0, so the residual add sees the served -0 (a +0 would give +0 + -0 = +0
where the served op gives -0: measured in the "eps" salt).

Modes (one source):
  owners=8, ranks=1, copies=1   real TP8: recv [TP, cap, 896] (TwoShot.views), peers = owners' recv bases,
                                gather [1, cap, 7168] + multicast address.
  owners=1, ranks=8, copies=8   one-GPU stand-in (twoshot.py `owners = 1`): x [8, M, 7168] holds the 8 ranks'
                                partials, rank 0 owns every column, the 8 slots play the ranks, the gather
                                multicast becomes 8 local copies, and each copy is normed like one rank.
Grid M CTAs x 896 threads; thread t = columns [8t, 8t + 8) of row blockIdx.x.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32, const_expr

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.mailbox import poll, rearm
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.primitives import (
    alloc,
    asm,
    at,
    early_weights,
    launch,
    load_vector,
    multicast_words,
    no_neg_zero,
    pack_pair,
    published,
    store_relaxed,
    store_words,
    unpack_bf16,
    zeros,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.twoshot import push

try:  # IKET phase ranges; compiled in only with iket=True.
    from cutlass.cute.experimental import iket as _iket
except ImportError:  # pragma: no cover
    _iket = None

TP, H, CAP, PARITIES = 8, 7168, 64, 2
THREADS = H // 8  # 896: one 16-byte chunk per thread
WARPS = THREADS // 32  # 28
OWNED = THREADS // TP  # 112 chunks per owner rank
MUTANTS = ("", "order", "norearm", "noresidual")


@asm(Uint32, pure=True)
def pack_sum(low, high):
    """BF16x2 of (low, high) with each -0 half sent as 0x8001 (never the 0x80000000 sentinel word)."""
    code = "{ .reg .b32 bits; .reg .pred p; cvt.rn.bf16x2.f32 $0,$2,$1; "
    code += "and.b32 bits,$0,0xffff; setp.eq.u32 p,bits,0x8000; @p add.u32 $0,$0,1; "
    code += "shr.u32 bits,$0,16; setp.eq.u32 p,bits,0x8000; @p add.u32 $0,$0,0x10000; }"
    return code, "=r,f,f", (Float32(low), Float32(high))


@asm(Float32, pure=True)
def add_ftz(a, b):
    return "add.rn.ftz.f32 $0,$1,$2;", "=f,f,f", (Float32(a), Float32(b))


@asm(Float32, pure=True)
def mul_ftz(a, b):
    return "mul.rn.ftz.f32 $0,$1,$2;", "=f,f,f", (Float32(a), Float32(b))


@asm(Float32, pure=True)
def fma_ftz(a, b, c):
    return "fma.rn.ftz.f32 $0,$1,$2,$3;", "=f,f,f,f", (Float32(a), Float32(b), Float32(c))


@asm(Float32, pure=True)
def rstd_ftz(total, hidden, eps):
    """MUFU.RSQ(FFMA.FTZ(total, MUFU.RCP(hidden), eps)): the served SASS; ``hidden`` is a run-time
    argument so the reciprocal is the hardware's, never a compile-time constant."""
    code = "{ .reg .f32 r,m; rcp.approx.ftz.f32 r,$2; fma.rn.ftz.f32 m,$1,r,$3; "
    code += "rsqrt.approx.ftz.f32 $0,m; }"
    return code, "=f,f,f,f", (Float32(total), Float32(hidden), Float32(eps))


class DraftAR:
    def __init__(self, owners=8, ranks=1, copies=1, wb=7, mutant="", iket=False):
        assert owners in (1, 8) and ranks * owners == TP and wb in (7, 14) and mutant in MUTANTS
        self.owners, self.ranks, self.copies, self.wb = owners, ranks, copies, wb
        self.blocks = WARPS // wb
        self.mutant, self.iket = mutant, iket and _iket is not None

    def _range(self, name):
        if const_expr(self.iket):
            _iket.range_push(name)

    def _pop(self):
        if const_expr(self.iket):
            _iket.range_pop()

    @cute.jit
    def __call__(self, x, res, gamma, normed, rout, recv, peers, gather, gather_mc, rank, eps, hidden, stream):
        """x [ranks, M, H], res [M, H], gamma [H], normed/rout [copies, M, H], recv [TP, cap, H/owners],
        peers [owners] Int64, gather [copies, cap, H]."""
        args = (x, res, gamma, normed, rout, recv, peers, gather, gather_mc, rank, eps, hidden, stream)
        launch(self.kernel, args, (x.shape[1], 1, 1), THREADS)

    @cute.jit
    def reduce(self, recv, local, row):
        """Poll the TP slots of (row, local) until published, sum in rank order (FADD.FTZ), round once;
        returns the 4 BF16x2 words (-0 halves as 0x8001, see pack_sum)."""
        raw = zeros((TP, 4), Uint32)
        pending = Int32((1 << TP) - 1)
        while pending != 0:
            for slot in cutlass.range_constexpr(TP):
                if (pending & (1 << slot)) != 0:
                    words = load_vector(at(recv, slot, row, local), cache="volatile")
                    for j in cutlass.range_constexpr(4):
                        raw[slot, j] = words[j]
                    if published(words):
                        pending &= ~(1 << slot)
        order = tuple(range(TP))[:: -1 if const_expr(self.mutant == "order") else 1]
        out = []
        for j in cutlass.range_constexpr(4):
            low, high = unpack_bf16(raw[order[0], j])
            for i in cutlass.range_constexpr(1, TP):
                a, b = unpack_bf16(raw[order[i], j])
                low, high = add_ftz(low, a), add_ftz(high, b)
            out.append(pack_sum(low, high))
        if const_expr(self.mutant != "norearm"):
            for slot in cutlass.range_constexpr(TP):
                rearm(at(recv, slot, row, local))
        return out

    @cute.kernel
    def kernel(self, args):
        x, res, gamma, normed, rout, recv, peers, gather, gather_mc, rank, eps, hidden = args
        row, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        warp, lane = tid // 32, tid % 32
        col = tid * 8
        smem = cutlass.utils.SmemAllocator()
        wsum = alloc(smem, Float32, (self.blocks, 32))  # warp sums by served block, zero padded
        rstd_box = alloc(smem, Float32, 4)
        if (tid < self.blocks * 32) & (tid % 32 >= self.wb):  # padding only: no race with the sums
            wsum[tid // 32, tid % 32] = Float32(0.0)
        gw = early_weights(at(gamma, col), count=4)  # weights may load before the grid-dependency wait
        gam = cute.make_rmem_tensor(8, Float32)
        for j in cutlass.range_constexpr(4):
            gam[2 * j], gam[2 * j + 1] = unpack_bf16(gw[j])
        cute.arch.griddepcontrol_launch_dependents()
        self._range("sync_griddep")
        cute.arch.griddepcontrol_wait()
        self._pop()
        rw = load_vector(at(res, row, col))

        self._range("main_push")
        xw = [load_vector(at(x, e, row, col)) for e in range(self.ranks)]  # all loads in flight first
        for e in cutlass.range_constexpr(self.ranks):
            slot = rank if const_expr(self.ranks == 1) else Int32(e)
            push(recv, peers, slot, row, col, tuple(no_neg_zero(w) for w in xw[e]))
        self._pop()

        if const_expr(self.owners == 1):
            self._range("sync_reduce")
            words = self.reduce(recv, col, row)
            for c in cutlass.range_constexpr(self.copies):
                store_relaxed(at(gather, c, row, col), words)
            self._pop()
        else:
            if tid < OWNED:
                self._range("sync_reduce")
                owned = rank * (H // TP) + col
                words = self.reduce(recv, col, row)
                offset = cute.crd2idx((0, row, owned), gather.layout)
                multicast_words(gather_mc + Int64(offset) * 2, words)
                self._pop()

        rv = cute.make_rmem_tensor(8, Float32)
        for j in cutlass.range_constexpr(4):
            rv[2 * j], rv[2 * j + 1] = unpack_bf16(rw[j])
        for c in cutlass.range_constexpr(self.copies):
            self._range("sync_gather")
            got = poll(at(gather, c, row, col), 0, 1)
            if const_expr(self.mutant != "norearm"):
                rearm(at(gather, c, row, col))
            self._pop()
            self._range("main_norm")
            y = cute.make_rmem_tensor(8, Float32)
            ywords = []
            for j in cutlass.range_constexpr(4):
                lo, hi = unpack_bf16(got[0, j])
                if const_expr(self.mutant != "noresidual"):
                    lo, hi = add_ftz(lo, rv[2 * j]), add_ftz(hi, rv[2 * j + 1])
                word = pack_pair(lo, hi, policy=None)
                ywords.append(word)
                y[2 * j], y[2 * j + 1] = unpack_bf16(word)
            store_words(at(rout, c, row, col), tuple(ywords))
            acc = Float32(0.0)
            for k in cutlass.range_constexpr(8):
                acc = fma_ftz(y[k], y[k], acc)
            for m in cutlass.range_constexpr(5):
                acc = add_ftz(acc, cute.arch.shuffle_sync_bfly(acc, 16 >> m))
            if lane == 0:
                wsum[warp // self.wb, warp % self.wb] = acc
            cute.arch.sync_threads()
            if warp == 0:
                total = Float32(0.0)
                for blk in cutlass.range_constexpr(self.blocks):
                    v = wsum[blk, lane]
                    for m in cutlass.range_constexpr(5):
                        v = add_ftz(v, cute.arch.shuffle_sync_bfly(v, 16 >> m))
                    total = add_ftz(total, v)
                if lane == 0:
                    rstd_box[0] = rstd_ftz(total, hidden, eps)
            cute.arch.sync_threads()
            s = rstd_box[0]
            owords = []
            for j in cutlass.range_constexpr(4):
                o0 = mul_ftz(mul_ftz(y[2 * j], s), add_ftz(gam[2 * j], 0.0))
                o1 = mul_ftz(mul_ftz(y[2 * j + 1], s), add_ftz(gam[2 * j + 1], 0.0))
                owords.append(pack_pair(o0, o1, policy=None))
            store_words(at(normed, c, row, col), tuple(owords))
            self._pop()
            if const_expr(self.copies > 1):
                cute.arch.sync_threads()  # wsum / rstd_box reuse by the next copy
