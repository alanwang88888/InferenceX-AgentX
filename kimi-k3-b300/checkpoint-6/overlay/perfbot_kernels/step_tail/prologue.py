"""Fused next-step prologue (P-11 lever D, unit E): one CuTe DSL launch for the 17 image launches
#3 is_padding fills, #4 _expand_idx_mapping_kernel, #5 _prepare_pos_seq_lens_kernel, #6 _combine_sampled_and_draft_tokens,
#7 _gather_block_tables_kernel, #8 _compute_slot_mappings_kernel, #9 preprocess_mamba_align_fused_kernel,
#11 _dcp_local_seq_lens_kernel, #12-14 new_ones + gather + copy (num_accepted_tmp),
#15 get_aligned_state_indices_multi_group_kernel, #16-18 _stage_spec_decode_metadata_kernel x3,
#19-20 torch.add (map_to_kernel_block_table) x NKT.
Launches that stay image calls: #1 index_put (only when draft_tokens_prev is given; before the fused kernel) and
#10 precopy_mamba_align_fused_kernel (after the fused kernel; it reads src_col / state_idx that #9 writes).
Bit-equal integer math in the image widths: #5 / #6 / #9 / #11 / #15 / #19-20 in i32, #8 in i64 with the i32
block_number * block_size product, truncating div / rem (div.s32 / div.s64).

Grid (no inter-CTA dependency; every address has one writer; idx_mapping rows must be distinct slots, as in serving):
  CTA r < R       request r: #4, #5 positions, #6, #7 row r, #8 tokens of row r (all G groups), #9 slot idx[r],
                  seq_lens[r], dcp[r], na_tmp[r]; bar.sync; #15 row r (reads the row #7 just wrote), #16-18 row r,
                  #19-20 row r (reads the row #7 just wrote, stale columns included).
  CTA R <= r < Rp padded row: #7 zero row, na_tmp 1, staged rows (NULL, qsl[R], 1); bar.sync; #15, #19-20 row r.
  CTA Rp ..       pad: #8 PAD tail [qsl[R], M) of every group (grid-strided); CTA Rp also #3, seq_lens / dcp [R, N) = 0,
                  staged qsl row Rp.
Declines (runs the image path) on has_prefill and outside the served envelope; see _fusable.
#8 slot math runs in 32-bit when block_size * cp_size fits in i32 (checked per group in the kernel; same bits, see
slot_id32), else in the image's i64 form (slot_id; judged with P11E_PROLOGUE_ABL=32).
B200 (seat 5113311 GPU 6, CUPTI, 256 MB L2 flush, median of 3 repeats): image #3-#9, #11-#20 busy 43.1 us (R=1) /
43.9 us (R=2); fused kernel 7.3 / 8.1 us; span fused start -> #10 end 9.6 / 10.6 us. Host cost per call ~74 us
(_fusable 15, 4 x torch.empty 12, argument list 16, #10 Triton launch 22, TVM-FFI call + rest ~9); image scope ~387 us.
"""

import os
import sys

import torch

from . import ptx as P  # noqa: F401
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass import Int32, Int64

from .prep import ld32, ldx, n32, op32, op64, sel64, st32, st64, st8, w64

I, L = Int32, Int64
NT = 256          # threads per CTA
PAD_PER_T = 8     # PAD stores per thread in the pad CTAs
SB = __import__("perfbot_kernels.k5q", fromlist=["k5q"]).SPEC_BLOCKS  # spec state blocks per request = K + 1  # PERFBOT K5 item 5 (land-8: 8)
FUSED_CALLS = 0
_compiled = {}
# dev-only ablation bits (timing probes and the i64-path judge run; results are wrong for 1-16): 1 no PAD tail,
# 2 no kernel tables, 4 no UVA reads, 8 no num_blocks read, 16 no slot math, 32 force the i64 slot path. 0 in production.
_ABL = int(os.environ.get("P11E_PROLOGUE_ABL", "0"))


def ld8(a):  # bool load -> 0/1
    return P.asm("{ .reg .b16 h; ld.global.u8 h, [$1]; cvt.u32.u16 $0, h; }", I, a, side=True)


def ld64(a):
    return ldx(a, 8)


def sel32(c, a, b):
    return P.asm("{ .reg .pred p; setp.ne.s32 p, $1, 0; selp.b32 $0, $2, $3, p; }", I, c, a, b)


def eq64(a, b):
    return P.asm("{ .reg .pred p; setp.eq.s64 p, $1, $2; selp.b32 $0, 1, 0, p; }", I, a, b)


def and32(a, b):
    return P.asm("and.b32 $0, $1, $2;", I, a, b)


def ldp32(a, p):  # predicated load, 0 when p == 0
    return P.asm("{ .reg .pred q; .reg .b32 t; setp.ne.s32 q, $2, 0; mov.b32 t, 0; @q ld.global.b32 t, [$1];"
                 " mov.b32 $0, t; }", I, a, p, side=True)


def ldp64(a, p):
    return P.asm("{ .reg .pred q; .reg .b64 t; setp.ne.s32 q, $2, 0; mov.b64 t, 0; @q ld.global.b64 t, [$1];"
                 " mov.b64 $0, t; }", L, a, p, side=True)


def stp32(a, v, p):  # predicated store
    P.asm("{ .reg .pred q; setp.ne.s32 q, $2, 0; @q st.global.b32 [$0], $1; }", None, a, v, p, side=True)


def slot_id(pos, real, req, bt, W, bs, en, sh, rank, cps, il):
    """_compute_slot_mappings_kernel (block_table.py:320-353), CP_SIZE != 1, one token; branch-free."""
    bs64 = w64(bs)
    vbs = bs64 * cps
    vbi = op64("div", pos, vbs)
    vbo = op64("rem", pos, vbs)
    isl = eq64(op64("rem", op64("div", vbo, il), cps), rank)
    lo = op64("div", vbo, il * cps) * il + op64("rem", vbo, il)
    lp = sel64(sh, vbi * bs64 + lo, pos)
    isl = sel32(sh, isl, I(1))
    bi = sel64(en, op64("div", lp, bs64), L(0))
    bo = op64("rem", lp, bs64)
    ok = and32(isl, real)
    bn = P.asm("{ .reg .pred p; .reg .b32 t; setp.ne.s32 p, $2, 0; mov.b32 t, 0; @p ld.global.b32 t, [$1];"
               " mov.b32 $0, t; }", I, bt + (req * W + bi) * 4, ok, side=True)
    s = w64(op32("mul.lo", bn, bs)) + bo
    s = sel64(isl, s, L(-1))
    return sel64(and32(en, real), s, L(-1))


def slot_id32(pos, real, req, bt, W, bs, en, sh, rank, cps, il):
    """slot_id in 32-bit: exact when bs * cps and il * cps fit in i32 (then every i64 operand of the image's
    division chain is an i32 value: pos = i32 num_computed + i32 offset, |lp| <= |pos|), with trunc division
    identities trunc(trunc(a / b) / c) = trunc(a / (b c)) and a % b = a - trunc(a / b) b."""
    vbs = bs * cps
    vbi = op32("div", pos, vbs)
    vbo = pos - vbi * vbs
    q1 = op32("div", vbo, il)
    q2 = op32("div", q1, cps)
    isl = P.icmp("eq", q1 - q2 * cps, rank)
    lo = q2 * il + (vbo - q1 * il)
    lp = sel32(sh, vbi * bs + lo, pos)
    isl = sel32(sh, isl, I(1))
    qb = op32("div", lp, bs)
    bi = sel32(en, qb, I(0))
    bo = lp - qb * bs
    ok = and32(isl, real)
    bn = P.asm("{ .reg .pred p; .reg .b32 t; setp.ne.s32 p, $2, 0; mov.b32 t, 0; @p ld.global.b32 t, [$1];"
               " mov.b32 $0, t; }", I, bt + (req * W + w64(bi)) * 4, ok, side=True)
    s = w64(op32("mul.lo", bn, bs)) + w64(bo)
    s = sel64(isl, s, L(-1))
    return sel64(and32(en, real), s, L(-1))


@cute.kernel
def pro_kernel(
    R: I, Rp: I, T: I, Tp: I, M: I, N: I, P1: I, MB: I, KPB: I, alpha: I,
    imap: L, cu: L, qsl: L, nct: L, last: L, last_st: L, pfl: L, drf: L, drf_st: L,
    sb0: L, sb1: L, sb2: L, sb3: L, ib0: L, ib1: L, ib2: L, ib3: L, W0: L, W1: L, W2: L, W3: L,
    nbu: L, nbu_st: L, bsz: L, en: L, shd: L, rank: L, cps: L, il: L,
    o_ids: L, o_pos: L, o_pad: L, o_sl: L, o_sm: L, sm_st: L, o_dcp: L, o_al: L,
    o_eim: L, o_elp: L, o_li: L, o_na: L, s_idx: L, s_col: L, s_off: L, nacc: L,
    ss0: L, ss1: L, ss2: L, sq0: L, sq1: L, sq2: L, sn0: L, sn1: L, sn2: L,
    kt0: L, kt1: L, kts: L, koff: L, drank: I, dsz: I, dil: I,
    BE: cutlass.Constexpr, BC: cutlass.Constexpr, MIDS: cutlass.Constexpr, MLA: cutlass.Constexpr,
    NKT: cutlass.Constexpr, CW: cutlass.Constexpr, ABL: cutlass.Constexpr,
):
    t, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    sbs = [sb0, sb1, sb2, sb3]
    ibs = [ib0, ib1, ib2, ib3]
    Ws = [W0, W1, W2, W3]
    sss = [ss0, ss1, ss2]
    sqs = [sq0, sq1, sq2]
    sns = [sn0, sn1, sn2]
    kts_ = [kt0, kt1]
    sm = cute.arch.alloc_smem(Int32, 64 + NT).toint()  # [0, 64) kernel_block_offsets, [64, 64 + W_mla) MLA row
    if bid < Rp:
        r = bid
        seq = I(0)
        av0 = I(0)  # padded rows: #7 zeroes the row, so #15 reads 0
        av1 = I(0)
        av2 = I(0)
        mv = I(0)
        Wm = n32(Ws[MLA])
        ov = ldp32(koff + w64(t) * 4, P.icmp("lt", t, KPB))
        if bid < R:
            # loads first (independent loads in flight together), stores after
            req = ld64(imap + w64(r) * 8)
            qs = ld32(qsl + w64(r) * 4)
            qe = ld32(qsl + w64(r) * 4 + 4)
            cs = ld32(cu + w64(r) * 4)
            ce = ld32(cu + w64(r) * 4 + 4)
            bss = [ld32(bsz + g * 4) for g in range(4)]
            ens = [ld8(en + g) for g in range(4)]
            shs = [ld8(shd + g) for g in range(4)]
            nbs = [ld32(nbu + (L(g) * nbu_st + req) * 4) for g in range(4)]  # UVA
            pl = ld32(pfl + req * 4)  # UVA
            if cutlass.const_expr(ABL & 4):
                nbs = [n32(Ws[g]) for g in range(4)]
                pl = I(0)
            if cutlass.const_expr(ABL & 8):
                nbs = [n32(Ws[g]) for g in range(4)]
            nc = ld32(nct + req * 4)
            si = ld32(s_idx + req * 4)
            na = ld32(nacc + req * 4)
            nl = ce - cs
            nd = nl - 1
            lastv = ld64(last + req * last_st * 8)
            drv = ldp64(drf + (req * drf_st + w64(t)) * 8, and32(P.icmp("lt", t, nd), P.icmp("lt", t, I(BC))))
            gv = []
            for g in cutlass.range_constexpr(4):
                for i in cutlass.range_constexpr(CW):
                    gv.append(ldp32(sbs[g] + (req * Ws[g] + w64(t + i * NT)) * 4, P.icmp("lt", t + i * NT, n32(Ws[g]))))
            ql = qe - qs
            seq = nc + ql
            lst = qe - nl
            # 15 sources: the value #7 leaves in the gathered row (slot table if col < nb, else the stale value; a
            # stale column has no writer in this launch, so its read does not race)
            first = op32("max", op32("div", seq - 1, MB), I(0))
            avl = []
            for j in cutlass.range_constexpr(3):
                gi = MIDS[j]
                src = ldp32(sbs[gi] + (req * Ws[gi] + w64(first + t)) * 4, P.icmp("lt", t, I(SB)))  # PERFBOT K5 item 5
                old = ldp32(ibs[gi] + (w64(r) * Ws[gi] + w64(first + t)) * 4, P.icmp("lt", t, I(SB)))  # PERFBOT K5 item 5
                avl.append(sel32(P.icmp("lt", first + t, nbs[gi]), src, old))
            av0 = avl[0]
            av1 = avl[1]
            av2 = avl[2]
            mold = ldp32(ibs[MLA] + (w64(r) * Ws[MLA] + w64(t)) * 4, P.icmp("lt", t, n32(Ws[MLA])))
            mv = sel32(P.icmp("lt", t, nbs[MLA]), gv[MLA * CW], mold)
            # 8: slot mappings of the row's tokens (4 groups per token)
            fit = [P.asm("{ .reg .pred p, q; setp.le.s64 p, $1, 2147483647; setp.gt.s32 q, $2, 0; and.pred p, p, q;"
                         " selp.b32 $0, 1, 0, p; }", I, w64(bss[g]) * cps, bss[g]) for g in range(4)]
            if cutlass.const_expr(ABL & 32):
                fit = [I(0) for g in range(4)]
            real = P.asm("{ .reg .pred p; setp.ge.s64 p, $1, 0; selp.b32 $0, 1, 0, p; }", I, req)
            for j in range(t, ql if cutlass.const_expr(ABL & 16 == 0) else 0, NT):
                sl4 = []
                for g in cutlass.range_constexpr(4):
                    sg = L(0)
                    if fit[g] != 0:
                        sg = slot_id32(nc + j, real, req, sbs[g], Ws[g], bss[g], ens[g], shs[g], n32(rank), n32(cps),
                                       n32(il))
                    else:
                        sg = slot_id(w64(nc + j), real, req, sbs[g], Ws[g], bss[g], ens[g], shs[g], rank, cps, il)
                    sl4.append(sg)
                for g in cutlass.range_constexpr(4):
                    st64(o_sm + (L(g) * sm_st + w64(qs + j)) * 8, sl4[g])
                # 5: positions (i32 add, stored as i64)
                st64(o_pos + w64(qs + j) * 8, w64(nc + j))
            # 7: gather stores
            for g in cutlass.range_constexpr(4):
                for i in cutlass.range_constexpr(CW):
                    stp32(ibs[g] + (w64(r) * Ws[g] + w64(t + i * NT)) * 4, gv[g * CW + i],
                          P.icmp("lt", t + i * NT, nbs[g]))
            # 4 + 6 (logits indices)
            if t < nl:
                if t < BE:
                    st64(o_eim + w64(cs + t) * 8, req)
                    st32(o_elp + w64(cs + t) * 4, t)
                if t < BC:
                    st64(o_li + w64(cs + t) * 8, w64(lst + t))
            # 6: sampled + draft tokens
            if seq > pl:
                if t == 0:
                    if seq - nl >= pl:
                        st32(o_ids + w64(lst) * 4, n32(lastv))
                if t < nd:
                    if t < BC:
                        st32(o_ids + w64(qe - nd + t) * 4, n32(drv))
            if t == 0:
                st32(o_sl + w64(r) * 4, seq)
                # 11 (i32)
                rem = op32("rem", seq, dsz * dil)
                rem = op32("min", op32("max", rem - drank * dil, I(0)), dil)
                st32(o_dcp + w64(r) * 4, op32("div", seq, dsz * dil) * dil + rem)
                # 9
                st32(s_col + req * 4, si)
                st32(s_off + req * 4, op32("max", na - 1, I(0)))
                new = op32("div", seq + MB - 1, MB) - 1
                st32(s_idx + req * 4, new)
                if si >= 0:
                    if si != new:
                        st32(nacc + req * 4, I(1))
                        na = I(1)
                # 12-14, 16-18 (qsl, num_accepted)
                st32(o_na + w64(r) * 4, na)
                for j in cutlass.range_constexpr(3):
                    st32(sns[j] + w64(r) * 4, na)
                    st32(sqs[j] + w64(r) * 4, qs)
        else:
            for g in cutlass.range_constexpr(4):
                for c in range(t, n32(Ws[g]), NT):
                    st32(ibs[g] + (w64(r) * Ws[g] + w64(c)) * 4, I(0))
            if t == 0:
                st32(o_na + w64(r) * 4, I(1))
                q = ld32(qsl + w64(R) * 4)
                for j in cutlass.range_constexpr(3):
                    st32(sns[j] + w64(r) * 4, I(1))
                    st32(sqs[j] + w64(r) * 4, q)
        # 15 + 16-18 state indices; stage the MLA row (as #7 leaves it) and the offsets in smem
        if t < Wm:
            P.sts(sm + (64 + t) * 4, mv)
        if t < KPB:
            P.sts(sm + t * 4, ov)
        av = [av0, av1, av2]
        if t < SB:  # PERFBOT K5 item 5
            for j in cutlass.range_constexpr(3):
                st32(o_al + ((L(j) * w64(N) + w64(r)) * SB + w64(t)) * 4, av[j])  # PERFBOT K5 item 5
                st32(sss[j] + (w64(r) * SB + w64(t)) * 4, sel32(P.icmp("lt", r, R), av[j], I(0)))  # PERFBOT K5 item 5
        P.bar()
        # 19-20 kernel block tables (i32)
        for e2 in range(t, Wm * KPB if cutlass.const_expr(ABL & 2 == 0) else 0, NT):
            c = op32("div", e2, KPB)
            v = P.lds(sm + (64 + c) * 4) * alpha + P.lds(sm + (e2 - c * KPB) * 4)
            for a in cutlass.range_constexpr(NKT):
                st32(kts_[a] + (w64(r) * kts + w64(e2)) * 4, v)
    else:
        p = bid - Rp
        q = ld32(qsl + w64(R) * 4)
        nP = M - q
        for i in range(p * NT + t, 4 * nP if cutlass.const_expr(ABL & 1 == 0) else 0, P1 * NT):
            g = op32("div", i, nP)
            o = i - g * nP
            st64(o_sm + (w64(g) * sm_st + w64(q + o)) * 8, L(-1))
        if p == 0:
            for i in range(t, Tp, NT):
                st8(o_pad + w64(i), P.icmp("ge", i, T))
            for i in range(R + t, N, NT):
                st32(o_sl + w64(i) * 4, I(0))
                st32(o_dcp + w64(i) * 4, I(0))
            if t == 0:
                for j in cutlass.range_constexpr(3):
                    st32(sqs[j] + w64(Rp) * 4, q)


@cute.jit
def pro_launch(
    R: I, Rp: I, T: I, Tp: I, M: I, N: I, P1: I, MB: I, KPB: I, alpha: I,
    imap: L, cu: L, qsl: L, nct: L, last: L, last_st: L, pfl: L, drf: L, drf_st: L,
    sb0: L, sb1: L, sb2: L, sb3: L, ib0: L, ib1: L, ib2: L, ib3: L, W0: L, W1: L, W2: L, W3: L,
    nbu: L, nbu_st: L, bsz: L, en: L, shd: L, rank: L, cps: L, il: L,
    o_ids: L, o_pos: L, o_pad: L, o_sl: L, o_sm: L, sm_st: L, o_dcp: L, o_al: L,
    o_eim: L, o_elp: L, o_li: L, o_na: L, s_idx: L, s_col: L, s_off: L, nacc: L,
    ss0: L, ss1: L, ss2: L, sq0: L, sq1: L, sq2: L, sn0: L, sn1: L, sn2: L,
    kt0: L, kt1: L, kts: L, koff: L, drank: I, dsz: I, dil: I, G: I, stream: cuda.CUstream,
    BE: cutlass.Constexpr, BC: cutlass.Constexpr, MIDS: cutlass.Constexpr, MLA: cutlass.Constexpr,
    NKT: cutlass.Constexpr, CW: cutlass.Constexpr, ABL: cutlass.Constexpr,
):
    pro_kernel(R, Rp, T, Tp, M, N, P1, MB, KPB, alpha, imap, cu, qsl, nct, last, last_st, pfl, drf, drf_st,
               sb0, sb1, sb2, sb3, ib0, ib1, ib2, ib3, W0, W1, W2, W3, nbu, nbu_st, bsz, en, shd, rank, cps, il,
               o_ids, o_pos, o_pad, o_sl, o_sm, sm_st, o_dcp, o_al, o_eim, o_elp, o_li, o_na,
               s_idx, s_col, s_off, nacc, ss0, ss1, ss2, sq0, sq1, sq2, sn0, sn1, sn2,
               kt0, kt1, kts, koff, drank, dsz, dil, BE, BC, MIDS, MLA, NKT, CW, ABL).launch(
        grid=[G, 1, 1], block=[NT, 1, 1], stream=stream)


_TYPES = [I] * 10 + [L] * 58 + [I] * 4  # pro_launch argument types, stream excluded


def _npow2(x):
    return 1 << max(0, int(x) - 1).bit_length()


_DT = dict(idx_mapping=torch.int64, cu_num_logits=torch.int32, query_start_loc=torch.int32, num_computed=torch.int32,
           last_sampled=torch.int64, prefill_len=torch.int32, req_draft_tokens=torch.int64, num_blocks=torch.int32,
           block_sizes=torch.int32, slot_mapping_enabled=torch.bool, dcp_sharded=torch.bool, input_ids=torch.int32,
           positions=torch.int64, is_padding=torch.bool, seq_lens=torch.int32, slot_mappings=torch.int64,
           dcp_local_seq_lens=torch.int32, aligned_state_indices=torch.int32, mamba_state_idx=torch.int32,
           mamba_src_col=torch.int32, mamba_src_off=torch.int32, num_accepted_gpu=torch.int32,
           kernel_block_offsets=torch.int32)


def _fusable(a):
    if a["has_prefill"]:
        return False
    R, Rp, T, Tp = a["R"], a["Rp"], a["T"], a["Tp"]
    sbt, ibt, kt = a["slot_block_tables"], a["input_block_tables"], a["kernel_block_tables"]
    mids, mla = list(a["mamba_group_ids"]), a["mla_group_id"]
    N = a["seq_lens"].shape[0]
    M = a["slot_mappings"].shape[1]
    if not (len(sbt) == 4 and len(ibt) == 4 and len(mids) == 3 and sorted(mids + [mla]) == [0, 1, 2, 3]
            and 1 <= len(kt) <= 2 and 1 <= R <= Rp <= N and T <= Tp <= M and a["dcp_size"] > 1
            and a["cp_interleave"] >= 1 and a["idx_mapping"].shape[0] == R
            and a["cp_interleave"] * a["dcp_size"] < 2 ** 31 and 0 <= a["dcp_rank"] < a["dcp_size"]):
        return False
    if not all(a[k].dtype == d and a[k].is_contiguous() for k, d in _DT.items()):
        return False
    if a["slot_mappings"].shape[0] != 4 or a["aligned_state_indices"].shape != (3, N, SB):  # PERFBOT K5 item 5
        return False
    if a["dcp_local_seq_lens"].shape[0] != N or a["last_sampled"].dim() != 2 or a["req_draft_tokens"].dim() != 2:
        return False
    for i in range(4):
        if not (sbt[i].dtype == ibt[i].dtype == torch.int32 and sbt[i].is_contiguous() and ibt[i].is_contiguous()
                and sbt[i].stride(0) == ibt[i].stride(0)):
            return False
    for j in range(3):
        s, q, n = a["kda_spec_state_indices"][j], a["kda_spec_query_start_loc"][j], a["kda_num_accepted"][j]
        if not (s.is_contiguous() and s.shape[1] == SB and s.shape[0] >= Rp and q.shape[0] >= Rp + 1  # PERFBOT K5 item 5
                and n.shape[0] >= Rp and s.dtype == q.dtype == n.dtype == torch.int32):
            return False
    W = ibt[mla].shape[1]
    KPB = a["kernel_block_offsets"].numel()
    if not (W <= NT and KPB <= 64 and a["aligned_state_indices"].shape[2] == SB):  # PERFBOT K5 item 5
        return False
    return all(k.dtype == torch.int32 and k.is_contiguous() and k.shape[0] == 1 and k.shape[1] >= Rp
               and k.shape[2] >= W * KPB for k in kt)


def _precopy(a):
    """#10: the image kernel, launched as mamba_utils.MambaSpecDecodeGPUContext.run_fused_precopy does."""
    from vllm.v1.worker import mamba_utils as mu

    R = a["R"]
    ns = a["mamba_state_base_addrs"].numel()
    mu.precopy_mamba_align_fused_kernel[(R, ns, mu._TEMPORAL_TILES)](
        a["mamba_state_idx"], a["mamba_src_col"], a["mamba_src_off"], a["mamba_block_table_ptrs"],
        a["mamba_block_table_stride"], a["mamba_state_base_addrs"], a["mamba_state_block_strides"],
        a["mamba_state_elem_sizes"], a["mamba_state_inner_sizes"], a["mamba_state_conv_widths"],
        a["mamba_state_group_indices"], a["mamba_state_dim_row_count"], a["mamba_state_dim_row_stride"],
        a["idx_mapping"], R, COPY_BLOCK_SIZE=1024, CONV_STATE_DIM_FIRST=a["conv_state_dim_first"],
        HAS_IDX_MAPPING=True, TEMPORAL_TILES=mu._TEMPORAL_TILES)


def _draft_fold(a):  # #1 (model_runner.py:2273), image op
    if a.get("draft_tokens_prev") is not None:
        a["req_draft_tokens"][a["idx_mapping_prev"]] = a["draft_tokens_prev"]


def _image_path(a):
    """The image functions in image order (model_runner.py:1377-1899, mamba_hybrid.py:180-325, utils.py:380-395)."""
    from types import SimpleNamespace as NS

    from vllm.models.kimi_k3.nvidia.kda_metadata import stage_spec_decode_metadata
    from vllm.v1.worker.gpu.block_table import BlockTables
    from vllm.v1.worker.gpu.cp_utils import prepare_dcp_local_seq_lens
    from vllm.v1.worker.gpu.input_batch import (combine_sampled_and_draft_tokens, expand_idx_mapping,
                                                prepare_pos_seq_lens, prepare_prefill_inputs)
    from vllm.v1.worker import mamba_utils as mu

    R, Rp, T, Tp, L = a["R"], a["Rp"], a["T"], a["Tp"], a["L"]
    im, qsl = a["idx_mapping"], a["query_start_loc"][: Rp + 1]
    _draft_fold(a)
    pad = a["is_padding"][:Tp]
    pad[:T].fill_(False)
    pad[T:].fill_(True)
    eim, elp = expand_idx_mapping(im, L, a["cu_num_logits"], a["decode_query_len"])
    if a["has_prefill"]:
        prepare_prefill_inputs(a["input_ids"], a["next_prefill_tokens"], im, qsl, a["all_token_ids"],
                               a["prefill_len"], a["num_computed"])
    prepare_pos_seq_lens(im, qsl, a["num_computed"], a["positions"], a["seq_lens"])
    li = combine_sampled_and_draft_tokens(a["input_ids"], im, a["last_sampled"], qsl, a["seq_lens"][:Rp],
                                          a["prefill_len"], a["req_draft_tokens"], a["cu_num_logits"], L, 1)
    G = len(a["slot_block_tables"])
    bt = NS(num_kv_cache_groups=G, input_block_tables=a["input_block_tables"],
            input_block_table_ptrs=a["input_block_table_ptrs"], block_table_ptrs=a["block_table_ptrs"],
            block_table_strides=a["block_table_strides"], num_blocks=NS(gpu=a["num_blocks"]),
            slot_mappings=a["slot_mappings"], block_sizes_tensor=a["block_sizes"],
            slot_mapping_enabled=a["slot_mapping_enabled"], dcp_sharded=a["dcp_sharded"],
            cp_rank=a["dcp_rank"], cp_size=a["dcp_size"], cp_interleave=a["cp_interleave"])
    BlockTables.gather_block_tables(bt, im, num_reqs_padded=Rp)
    BlockTables.compute_slot_mappings(bt, im, qsl, a["positions"][:Tp], num_tokens_padded=Tp)
    mu.preprocess_mamba_align_fused_kernel[(-(-R // 256),)](
        im, a["mamba_state_idx"], a["num_computed"], qsl, a["num_accepted_gpu"], a["mamba_src_col"],
        a["mamba_src_off"], R, BLOCK_SIZE=256, MAMBA_BLOCK_SIZE=a["mamba_block_size"])
    _precopy(a)
    prepare_dcp_local_seq_lens(a["dcp_local_seq_lens"], a["seq_lens"][:Rp], R, a["dcp_size"], a["dcp_rank"],
                               a["cp_interleave"], num_reqs_padded=Rp)
    na = a["num_accepted_gpu"].new_ones(Rp)
    na[:R] = a["num_accepted_gpu"][im]
    al = a["aligned_state_indices"]
    ctx = NS(is_initialized=True, aligned_state_indices=al,
             aligned_index_block_table_ptrs=a["mamba_aligned_block_table_ptrs"],
             block_table_stride_req=a["mamba_block_table_stride"], block_size=a["mamba_block_size"],
             num_groups=al.shape[0])
    mu.MambaSpecDecodeGPUContext.compute_aligned_state_indices(ctx, a["seq_lens"][:Rp], Rp)
    if not a["has_prefill"]:
        ns = al.shape[2]
        for j in range(al.shape[0]):
            stage_spec_decode_metadata(
                state_indices=al[j][:Rp][:R, :ns], query_start_loc=qsl[: R + 1], num_accepted_tokens=na[:Rp][:R],
                staged_state_indices=a["kda_spec_state_indices"][j][:Rp],
                staged_query_start_loc=a["kda_spec_query_start_loc"][j][: Rp + 1],
                staged_num_accepted_tokens=a["kda_num_accepted"][j][:Rp], num_spec_decodes=R)
    src = a["input_block_tables"][a["mla_group_id"]][:Rp]
    KPB = a["kernel_block_offsets"].numel()
    rows, cols = src.shape
    for kt in a["kernel_block_tables"]:
        out = kt[0, :rows, : cols * KPB]
        torch.add(a["kernel_block_offsets"], src.unsqueeze(-1), alpha=a["kernel_block_stride"] or KPB,
                  out=out.unflatten(1, (cols, -1)))
    return eim, elp, li, na


_ws = {}    # device -> [slot 0, slot 1]; slot = {"cap": (Lmax, Rmax), "bufs": 4 tensors, "views": {(L, Rp): 4 views}}
_turn = {}  # device -> next slot


def _outputs(a, dev, L, Rp):
    """The 4 returned tensors: [:L] / [:Rp] views of a per-device workspace of two slot sets, alternated per fused call
    (the previous step's tensors may still be read when the next prologue runs). The kernel writes every element of
    the views on every call (one writer per element), so no data of an earlier call leaks. Sizes: Lmax =
    max_num_reqs * decode_query_len, Rmax = max_num_reqs (from the inputs); a larger request grows that slot."""
    slots = _ws.get(dev)
    if slots is None:
        N = a["seq_lens"].shape[0]
        slots = _ws[dev] = [{"cap": (0, 0), "bufs": None, "views": {}} for _ in range(2)]
        for sl in slots:
            _grow(sl, dev, N * a["decode_query_len"], N)
    i = _turn.get(dev, 0)
    _turn[dev] = 1 - i
    sl = slots[i]
    v = sl["views"].get((L, Rp))
    if v is None:
        if L > sl["cap"][0] or Rp > sl["cap"][1]:
            _grow(sl, dev, max(L, sl["cap"][0]), max(Rp, sl["cap"][1]))
        e, p, g, n = sl["bufs"]
        v = sl["views"][(L, Rp)] = (e[:L], p[:L], g[:L], n[:Rp])
    return v


def _grow(sl, dev, Lmax, Rmax):
    sl["cap"] = (Lmax, Rmax)
    sl["bufs"] = (torch.empty(Lmax, dtype=torch.int64, device=dev), torch.empty(Lmax, dtype=torch.int32, device=dev),
                  torch.empty(Lmax, dtype=torch.int64, device=dev), torch.empty(Rmax, dtype=torch.int32, device=dev))
    sl["views"] = {}


def step_prologue(**a):
    """Unit E launches #3-#20 (and #1 when draft_tokens_prev is given). Writes the persistent buffers in place.
    Returns (expanded_idx_mapping i64[L], expanded_local_pos i32[L], logits_indices i64[L], num_accepted_tmp i32[Rp]).
    Keyword arguments: the unit E world (judge/p11e.py build_world, public keys)."""
    global FUSED_CALLS
    if not _fusable(a):
        return _image_path(a)
    FUSED_CALLS += 1
    sys.modules[__package__].STEP_PROLOGUE_FUSED_CALLS += 1
    _draft_fold(a)
    R, Rp, L = a["R"], a["Rp"], a["L"]
    dev = a["idx_mapping"].device
    eim, elp, li, na = _outputs(a, dev, L, Rp)
    key, args = _args(a, eim, elp, li, na)
    if key not in _compiled:  # TVM-FFI: the call takes plain Python ints
        typed = [t(x) for t, x in zip(_TYPES, args[:-1])] + [args[-1]]
        _compiled[key] = cute.compile(pro_launch, *typed, *key, options="--enable-tvm-ffi")
    _compiled[key](*args)
    _precopy(a)
    return eim, elp, li, na


def _args(a, eim, elp, li, na):
    R, Rp, T, Tp = a["R"], a["Rp"], a["T"], a["Tp"]
    sm = a["slot_mappings"]
    N, M = a["seq_lens"].shape[0], sm.shape[1]
    sbt, ibt, kt = a["slot_block_tables"], a["input_block_tables"], a["kernel_block_tables"]
    KPB = a["kernel_block_offsets"].numel()
    P1 = max(1, -(-(4 * (M - T)) // (NT * PAD_PER_T)))
    p = lambda x: x.data_ptr()  # noqa: E731
    kts = list(kt) + [kt[0]] * (2 - len(kt))
    args = [R, Rp, T, Tp, M, N, P1, a["mamba_block_size"], KPB, a["kernel_block_stride"] or KPB,
            p(a["idx_mapping"]), p(a["cu_num_logits"]), p(a["query_start_loc"]), p(a["num_computed"]),
            p(a["last_sampled"]), a["last_sampled"].stride(0), p(a["prefill_len"]), p(a["req_draft_tokens"]),
            a["req_draft_tokens"].stride(0),
            *[p(x) for x in sbt], *[p(x) for x in ibt], *[x.stride(0) for x in ibt],
            p(a["num_blocks"]), a["num_blocks"].stride(0), p(a["block_sizes"]), p(a["slot_mapping_enabled"]),
            p(a["dcp_sharded"]), a["dcp_rank"], a["dcp_size"], a["cp_interleave"],
            p(a["input_ids"]), p(a["positions"]), p(a["is_padding"]), p(a["seq_lens"]), p(sm), sm.stride(0),
            p(a["dcp_local_seq_lens"]), p(a["aligned_state_indices"]),
            p(eim), p(elp), p(li), p(na), p(a["mamba_state_idx"]), p(a["mamba_src_col"]), p(a["mamba_src_off"]),
            p(a["num_accepted_gpu"]),
            *[p(x) for x in a["kda_spec_state_indices"]], *[p(x) for x in a["kda_spec_query_start_loc"]],
            *[p(x) for x in a["kda_num_accepted"]],
            p(kts[0]), p(kts[1]), kt[0].stride(1), p(a["kernel_block_offsets"]),
            a["dcp_rank"], a["dcp_size"], a["cp_interleave"], Rp + P1,
            cuda.CUstream(torch.cuda.current_stream(dev_of(a)).cuda_stream)]
    key = (_npow2(a["decode_query_len"]), _npow2(a["req_draft_tokens"].shape[1] + 1),
           tuple(a["mamba_group_ids"]), a["mla_group_id"], len(kt), -(-max(x.shape[1] for x in ibt) // NT), _ABL)
    assert len(args) == len(_TYPES) + 1, (len(args), len(_TYPES))
    return key, args


def dev_of(a):
    return a["idx_mapping"].device
