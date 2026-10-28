"""One persistent CuTe DSL kernel for the rejection-sampler tail (served config, bit-equal to the image path).

Grid: G CTAs of 128 threads, cooperative launch (all CTAs resident), G = min(2 * SMs, items). CTA c runs items
c, c + G, ... in order. Items: L * NB stats items (row, 8192 block), then R * NB1 resample items (request, 1024 block).
A stats index is always lower than a resample index, so a CTA never waits before its own stats items: no deadlock.

Hand-off (no grid barrier):
  stats item   -> stores block stats (thread 0), atom.add.acq_rel row counter (+1).
  resample item-> thread 0 waits (ld.acquire) until each draft row of its request has NB stats; bar.
                  4 warps compute the per-step verdicts (step w and w + 4; lane = vocab block), smem, bar;
                  every thread runs the serial accept chain of _rejection_kernel from smem;
                  the CTA computes its 1024-block Gumbel argmax, stores it, atom.add.acq_rel request counter;
                  the last arriver of a request is the insert: argmax over NB1 blocks, writes all request outputs.
  exit         -> each CTA adds 1 to the exit counter after its last item; the last CTA zeroes all counters, so
                  the next launch (or graph replay) starts from zero.
CTA 0 also writes draft_sampled [L] and pos [L] (gather_draft_sampled).
"""

import os

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass import Float32, Int32, Int64

from . import blocks as B
from . import ptx as P

F, I = Float32, Int32
NB_MAX, ROWS_MAX, REQ_MAX, NB1_MAX = 32, 128, 16, 256
# workspace (int32 words)
W_ROWCNT = 0                         # [ROWS_MAX] stats blocks done per row
W_REQCNT = W_ROWCNT + ROWS_MAX       # [REQ_MAX] resample blocks done per request
W_EXIT = W_REQCNT + REQ_MAX          # [1] CTAs done
W_TMAX = 256                         # [ROWS_MAX, NB_MAX] f32
W_TSUM = W_TMAX + ROWS_MAX * NB_MAX
W_DMAX = W_TSUM + ROWS_MAX * NB_MAX
W_DSUM = W_DMAX + ROWS_MAX * NB_MAX
W_TARG = W_DSUM + ROWS_MAX * NB_MAX  # [ROWS_MAX, NB_MAX] i32 token
W_RES = W_TARG + ROWS_MAX * NB_MAX   # [REQ_MAX, NB1_MAX] (f32 block max, i32 token) pairs
W_WORDS = W_RES + REQ_MAX * NB1_MAX * 2
# dev trace (PERFBOT_TAIL_TRACE=1 at import): per CTA 16 int64 %globaltimer stamps after W_WORDS
TRACE = os.environ.get("PERFBOT_TAIL_TRACE", "0") == "1"
W_TRACE_CTAS = 512


def poll(addr, n, need):
    """1 when every lane with need != 0 sees counter >= n (ld.acquire.gpu, warp vote)."""
    return P.asm("{ .reg .pred p, q; .reg .b32 c; ld.acquire.gpu.global.b32 c, [$1]; setp.ge.s32 p, c, $2;"
                 " setp.eq.s32 q, $3, 0; or.pred p, p, q; vote.sync.all.pred p, p, -1; selp.b32 $0, 1, 0, p; }",
                 I, addr, n, need, side=True)


def stamp(ws, t, bid, k):
    if TRACE:
        P.asm("{ .reg .pred p; .reg .b64 g; setp.eq.s32 p, $0, 0; mov.u64 g, %globaltimer; @p st.global.b64 [$1], g; }",
              None, t, ws + Int64(W_WORDS) * 4 + (Int64(bid) * 16 + Int64(k)) * 8, side=True)
# smem (bytes): 0..31 reductions, 48 insert flag, 64.. step table [8 steps x 8 words]
S_FLAG, S_STEP = 48, 64
F_ACC, F_VALID, F_TLSE, F_DLSE, F_GTOK, F_DTOK = range(6)


def lo32(addr):
    return P.ld32(addr)


def wsa(ws, base, k):
    return ws + Int64(base) * 4 + Int64(k) * 4


def split64(x):
    return P.asm("{ .reg .b64 h; cvt.u32.u64 $0, $2; shr.b64 h, $2, 32; cvt.u32.u64 $1, h; }", (I, I), x)


def draft_of(row, ids, positions, lidx, eim, elp, pfl):
    """gather_draft_sampled for one row: (-1 if placeholder else input_ids[logits_indices[row]], pos as Int64)."""
    li = Int64(lo32(lidx + Int64(row) * 8))
    tok = P.ld32(ids + li * 4)
    pos = P.ld64(positions + li * 8)
    lp = P.ld32(elp + Int64(row) * 4)
    pf = P.ld32(pfl + Int64(lo32(eim + Int64(row) * 8)) * 4)
    ph = P.asm("{ .reg .pred p, q; .reg .b64 d; setp.gt.s32 p, $2, 0; cvt.s64.s32 d, $2; sub.s64 d, $3, d;"
               " setp.lt.s64 q, d, $4; and.pred p, p, q; selp.b32 $0, -1, $1, p; }",
               I, tok, lp, pos, Int64(pf))
    return ph, pos


@cute.kernel
def tail_kernel(
    L: Int32, R: Int32, V: Int32, NB: Int32, NB1: Int32, K: Int32, G: Int32,
    logits: Int64, lstride: Int64, draft: Int64, ds0: Int64, ds1: Int64,
    ids: Int64, positions: Int64, lidx: Int64, eim: Int64, elp: Int64, pfl: Int64, cu: Int64, imap: Int64,
    seq_lens: Int64, temps: Int64, seeds: Int64, rates: Int64,
    o_sampled: Int64, o_ns: Int64, o_nr: Int64, o_ds: Int64, o_pos: Int64, o_tlse: Int64, o_dlse: Int64,
    ws: Int64, f32: cutlass.Constexpr, synth: cutlass.Constexpr,
):
    t, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    w = t >> 5
    lane = t & 31
    sm = cute.arch.alloc_smem(Int32, 128).toint()
    es = 4 if f32 else 2

    if bid == 0:  # gather_draft_sampled outputs
        for j in range(t, L, 128):
            d, pos = draft_of(j, ids, positions, lidx, eim, elp, pfl)
            P.st32(o_ds + Int64(j) * 4, d)
            P.st64(o_pos + Int64(j) * 8, pos)

    NS = L * NB
    stamp(ws, t, bid, 0)
    if cutlass.const_expr(TRACE):
        P.asm("{ .reg .pred p; .reg .b32 s; .reg .b64 g; setp.eq.s32 p, $0, 0; mov.u32 s, %smid; cvt.u64.u32 g, s;"
              " @p st.global.b64 [$1], g; }", None, t, ws + Int64(W_WORDS) * 4 + (Int64(bid) * 16 + 11) * 8, side=True)
    for it in range(bid, NS + R * NB1, G):
        P.bar()
        jt = P.asm("min.s32 $0, $1, 1;", I, (it - bid) // G) * 4
        stamp(ws, t, bid, 1 + jt)
        if it < NS:
            # ---------------- stats item: _compute_local_logits_stats_kernel (row l, block b) ----------------
            l = it // NB
            b = it - l * NB
            # all loads first (no dependent branch before them); a bonus row (lp >= K) loads and discards
            vals = B.load_block(logits + Int64(l) * lstride + Int64(b * 8192 * es), t, 8, f32)
            lp = P.ld32(elp + Int64(l) * 4)
            slot = Int64(lo32(eim + Int64(l) * 8))
            temp = P.ld32(temps + slot * 4, F)
            lpc = P.asm("min.s32 $0, $1, $2;", I, lp, K - 1)
            dvals = B.load_block(draft + slot * ds0 + Int64(lpc) * ds1 + Int64(b * 8192 * 2), t, 8, False)
            if lp < K:
                o = l * NB_MAX + b
                if temp == Float32(0.0):
                    v, i = B.block_argmax(vals, t, sm, w)
                    if t == 0:
                        P.st32(wsa(ws, W_TARG, o), i + b * 8192)
                        P.st32(wsa(ws, W_TMAX, o), v)
                else:
                    m, s = B.max_sumexp(vals, sm, w, t)
                    dm, dsum = B.max_sumexp([P.fdiv(x, temp) for x in dvals], sm, w, t)
                    if t == 0:
                        P.st32(wsa(ws, W_TMAX, o), m)
                        P.st32(wsa(ws, W_TSUM, o), s)
                        P.st32(wsa(ws, W_DMAX, o), dm)
                        P.st32(wsa(ws, W_DSUM, o), dsum)
                if t == 0:
                    P.atom_add_acqrel(wsa(ws, W_ROWCNT, l), Int32(1))
            stamp(ws, t, bid, 4 + jt)
        else:
            # ---------------- resample item (request r, 1024 block b) ----------------
            q = it - NS
            r = q // NB1
            b = q - r * NB1
            start = P.ld32(cu + Int64(r) * 4)
            end = P.ld32(cu + Int64(r) * 4 + 4)
            nd = end - start - 1
            slot = Int64(lo32(imap + Int64(r) * 8))
            temp = P.ld32(temps + slot * 4, F)
            greedy = 1 - P.fne0(temp)
            sl_r = P.ld32(seq_lens + Int64(r) * 4)  # for the insert, loaded early
            pf_r = P.ld32(pfl + slot * 4)
            s_lo = P.ld32(seeds + slot * 8)
            s_hi = P.ld32(seeds + slot * 8 + 4)
            # before the wait: everything that does not read stats (gather, Philox, logits at the draft token)
            P.asm("prefetch.global.L2 [$0];", None,
                  logits + Int64(end - 1) * lstride + Int64((b * 1024 + t * 8) * es), side=True)
            pre = []
            for k in cutlass.range_constexpr(2):
                i = w + 4 * k
                ic = P.asm("{ .reg .b32 x; max.s32 x, $2, 0; min.s32 $0, $1, x; }", I, i, nd - 1)
                row = start + ic
                d, _ = draft_of(P.asm("min.s32 $0, $1, $2;", I, row + 1, L - 1), ids, positions, lidx, eim, elp,
                                pfl)
                dtok = P.asm("max.s32 $0, $1, 0;", I, d)
                p_lo, p_hi = split64(P.ld64(positions + Int64(lo32(lidx + Int64(row) * 8)) * 8))
                u = P.philox_u(s_lo, s_hi, p_lo, p_hi)
                tl_ = B.load1(logits + Int64(row) * lstride + Int64(dtok) * es, f32)
                dl_ = P.fdiv(B.bf_lo(P.ld_u16(draft + slot * ds0 + Int64(ic) * ds1 + Int64(dtok) * 2)), temp)
                rate = P.ld32(rates + Int64(ic) * 4, F) if synth else Float32(0.0)
                pre.append((i, row, P.icmp("ge", d, Int32(0)), dtok, u, P.logf(u), tl_, dl_, rate))
            stamp(ws, t, bid, 2 + jt)
            if w == 0:  # wait for the stats of every draft row: lane i polls row start + i, one round trip per poll
                caddr = wsa(ws, W_ROWCNT, start + P.asm("{ .reg .b32 x; max.s32 x, $2, 0; min.s32 $0, $1, x; }",
                                                        I, lane, nd - 1))
                need = P.icmp("lt", lane, nd)
                ok = poll(caddr, NB, need)
                while ok == 0:
                    P.nanosleep()
                    ok = poll(caddr, NB, need)
            P.bar()
            # per-step verdicts: warp w handles steps w and w + 4 (lane = vocab block)
            inb = P.icmp("lt", lane, NB)
            lc = P.asm("min.s32 $0, $1, $2;", I, lane, NB - 1)
            for k in cutlass.range_constexpr(2):
                i, row, valid, dtok, u, logu, tl_, dl_, rate = pre[k]
                if i < nd:
                    o = row * NB_MAX + lc
                    acc = Int32(0)
                    tlse = Float32(0.0)
                    dlse = Float32(0.0)
                    gtok = Int32(0)
                    if greedy != 0:
                        m = P.sel(inb, P.nan_to_ninf(P.ldcg(wsa(ws, W_TMAX, o), F)), Float32(float("-inf")))
                        m, lc2 = B.warp_argmax(m, lane)
                        gtok = P.ldcg(wsa(ws, W_TARG, row * NB_MAX + lc2))
                        if cutlass.const_expr(synth):
                            acc = P.fcmp("lt", u, rate) & valid
                        else:
                            acc = P.icmp("eq", gtok, dtok) & valid
                    else:
                        tlse = P.lse32(P.sel(inb, P.ldcg(wsa(ws, W_TMAX, o), F), Float32(float("-inf"))),
                                       P.sel(inb, P.ldcg(wsa(ws, W_TSUM, o), F), Float32(0.0)))
                        dlse = P.lse32(P.sel(inb, P.ldcg(wsa(ws, W_DMAX, o), F), Float32(float("-inf"))),
                                       P.sel(inb, P.ldcg(wsa(ws, W_DSUM, o), F), Float32(0.0)))
                        if cutlass.const_expr(synth):
                            acc = P.fcmp("lt", u, rate)
                        else:
                            acc = P.fcmp("gt", P.sub(tl_, tlse), P.add(P.sub(dl_, dlse), logu))
                    if lane == 0:
                        e = sm + S_STEP + i * 32
                        P.sts(e + F_ACC * 4, acc)
                        P.sts(e + F_VALID * 4, valid)
                        P.sts(e + F_TLSE * 4, tlse)
                        P.sts(e + F_DLSE * 4, dlse)
                        P.sts(e + F_GTOK * 4, gtok)
                        P.sts(e + F_DTOK * 4, dtok)
            P.bar()
            stamp(ws, t, bid, 3 + jt)
            # serial accept chain of _rejection_kernel (every thread, same result)
            verifying = Int32(1)
            acc_len = Int32(0)
            tl_out = Float32(0.0)
            dl_out = Float32(0.0)
            for i in cutlass.range_constexpr(7):
                e = sm + S_STEP + i * 32
                act = P.icmp("lt", Int32(i), nd)
                vld = P.lds(e + F_VALID * 4)
                stp = verifying & act & (greedy | vld)
                upd = stp & (1 - greedy)
                tl_out = P.sel(upd, P.lds(e + F_TLSE * 4, F), tl_out)
                dl_out = P.sel(upd, P.lds(e + F_DLSE * 4, F), dl_out)
                a = stp & P.lds(e + F_ACC * 4)
                acc_len = acc_len + a
                verifying = P.sel(act, a, verifying)
            rrow = start + acc_len
            bonus = P.icmp("eq", acc_len, nd)
            work = bonus | (1 - greedy)
            if work != 0:
                rslot = Int64(lo32(eim + Int64(rrow) * 8))
                rtemp = P.ld32(temps + rslot * 4, F)
                raw = B.load_block(logits + Int64(rrow) * lstride + Int64(b * 1024 * es), t, 1, f32)
                rvalid = P.lds(sm + S_STEP + acc_len * 32 + F_VALID * 4)
                resid = (1 - bonus) & rvalid
                dstep = P.asm("min.s32 $0, $1, $2;", I, acc_len, K - 1)  # in range also for the bonus row
                dv = B.load_block(draft + rslot * ds0 + Int64(dstep) * ds1 + Int64(b * 1024 * 2), t, 1, False)
                raw = [P.sel(resid, P.residual(x, y, tl_out, dl_out, rtemp), x) for x, y in zip(raw, dv)]
                rli = Int64(lo32(lidx + Int64(rrow) * 8))
                rpos = P.ld32(positions + rli * 8)
                mpre = P.murmur_prefix(P.ld32(seeds + rslot * 8), P.ld32(seeds + rslot * 8 + 4), rpos)
                g = P.fne0(rtemp)
                vals = [P.sel(g, P.gumbel_value(x, mpre, b * 1024 + t * 8 + j), x) for j, x in enumerate(raw)]
                v, i = B.block_argmax(vals, t, sm, w)
                if t == 0:
                    P.asm("st.global.v2.b32 [$0], {$1, $2};", None, wsa(ws, W_RES, (r * NB1_MAX + b) * 2), v,
                          i + b * 1024, side=True)
            stamp(ws, t, bid, 4 + jt)
            if t == 0:
                old = P.atom_add_acqrel(wsa(ws, W_REQCNT, r), Int32(1))
                P.sts(sm + S_FLAG, P.icmp("eq", old, NB1 - 1))
            P.bar()
            if P.lds(sm + S_FLAG) != 0:
                # ---------------- insert (_insert_resampled_kernel + get_num_sampled_and_rejected) ----------------
                tok = Int32(0)
                if work != 0:
                    k2 = t + 128
                    # the token is the tie-break index: tokens rise with the block index, pads come after all tokens
                    va, ta = P.asm("ld.global.cg.v2.b32 {$0, $1}, [$2];", (F, I),
                                   wsa(ws, W_RES, (r * NB1_MAX + t) * 2), side=True)
                    vb, tb = P.asm("ld.global.cg.v2.b32 {$0, $1}, [$2];", (F, I),
                                   wsa(ws, W_RES, (r * NB1_MAX + k2) * 2), side=True)
                    inr = P.icmp("lt", k2, NB1)
                    v, i = P.amax(P.nan_to_ninf(va), ta, P.sel(inr, P.nan_to_ninf(vb), Float32(float("-inf"))),
                                  P.sel(inr, tb, k2 + 0x40000000))
                    v, tok = B.cta_argmax(v, i, sm, w, t)
                if t < 8:
                    if t <= acc_len:
                        e = sm + S_STEP + t * 32
                        last = P.sel(work, tok, P.lds(sm + S_STEP + acc_len * 32 + F_GTOK * 4))
                        val = P.sel(P.icmp("lt", t, acc_len), P.lds(e + F_DTOK * 4), last)
                        P.st64(o_sampled + Int64(r * (K + 1) + t) * 8, Int64(val))
                if t == 0:
                    chunked = P.icmp("lt", sl_r, pf_r)
                    ns = acc_len + 1
                    P.st32(o_ns + Int64(r) * 4, P.sel(chunked, Int32(0), ns))
                    P.st32(o_nr + Int64(r) * 4, P.sel(chunked, Int32(0), end - start - ns))
                    P.st32(o_tlse + Int64(r) * 4, tl_out)
                    P.st32(o_dlse + Int64(r) * 4, dl_out)
                stamp(ws, t, bid, 9)

    stamp(ws, t, bid, 10)
    # ---------------- exit: the last CTA resets the counters for the next launch ----------------
    P.bar()
    if t == 0:
        old = P.atom_add_acqrel(wsa(ws, W_EXIT, 0), Int32(1))
        P.sts(sm + S_FLAG, P.icmp("eq", old, G - 1))
    P.bar()
    if P.lds(sm + S_FLAG) != 0:
        for j in range(t, ROWS_MAX + REQ_MAX + 1, 128):
            P.st_relaxed(wsa(ws, 0, j), Int32(0))


@cute.jit
def tail_launch(
    L: Int32, R: Int32, V: Int32, NB: Int32, NB1: Int32, K: Int32, G: Int32,
    logits: Int64, lstride: Int64, draft: Int64, ds0: Int64, ds1: Int64,
    ids: Int64, positions: Int64, lidx: Int64, eim: Int64, elp: Int64, pfl: Int64, cu: Int64, imap: Int64,
    seq_lens: Int64, temps: Int64, seeds: Int64, rates: Int64,
    o_sampled: Int64, o_ns: Int64, o_nr: Int64, o_ds: Int64, o_pos: Int64, o_tlse: Int64, o_dlse: Int64,
    ws: Int64, stream: cuda.CUstream, f32: cutlass.Constexpr, synth: cutlass.Constexpr,
):
    tail_kernel(L, R, V, NB, NB1, K, G, logits, lstride, draft, ds0, ds1, ids, positions, lidx, eim, elp, pfl, cu,
                imap, seq_lens, temps, seeds, rates, o_sampled, o_ns, o_nr, o_ds, o_pos, o_tlse, o_dlse, ws,
                f32, synth).launch(grid=[G, 1, 1], block=[128, 1, 1], stream=stream, cooperative=True)
