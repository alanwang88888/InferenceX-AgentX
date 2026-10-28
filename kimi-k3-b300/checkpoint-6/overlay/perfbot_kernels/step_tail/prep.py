"""Fused DFlash/DSpark draft prep (P-11 lever D, unit C): one CuTe DSL launch for the 4 image launches
  1 _prepare_dflash_inputs_kernel (speculator.py:873), 2 torch.cat(aux, out=aux_staging[:T]) (:410),
  3 context_slot[T:NC].fill_(-1) (:232), 4 _dcp_local_seq_lens_kernel (cp_utils.py:24).
Bit-equal to the image (vLLM 49cb2bbf): integer math in the image's widths (i64 slot math, i32 seq_lens / dcp),
C-style (truncating) div / rem as Triton emits, the same pad writes, the same byte copy for the cat.

Grid (no inter-CTA dependency; every address has one writer):
  CTA r < R          request r: context rows [qsl[r], qsl[r+1]), query rows [Q r, Q r + Q), sample rows, qsl[r],
                     seq_lens[r], dcp_local_seq_lens[r] (from the i32 seq_lens it writes), temperature / seeds copy.
                     CTA 0 also: qsl (R, NR] = Q R, seq_lens [R, NR) = 0, sample pad [k R, k NR), dcp [R, ND) = 0,
                     context_slot [T, NC) = -1.
  CTA R .. R+P1-1    query pad [Q R, M): input_ids 0, positions 0, is_padding True, query slot -1.
  CTA R+P1 ..        cat: aux_staging[t, i*H + c] = aux[i][t, c], 16 B vectors.
Declines (runs the 4 image launches) outside the served envelope; see _fusable. FUSED_CALLS (here) and
perfbot_kernels.step_tail.DRAFT_PREP_FUSED_CALLS count the fused calls. Compiled with TVM-FFI: the call takes plain
Python ints (B200 host cost ~26 us per call: _fusable 10, argument list 8, launch 4). No workspace, no cooperative
launch: graph capture is safe; two calls may run on two streams.
"""

import sys

import torch

from . import ptx as P  # noqa: F401
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass import Int32, Int64

I, L = Int32, Int64
NT = 128          # threads per CTA
NAUX = 5          # aux layers of the served drafter
PER_T = 4         # bulk elements per thread
FUSED_CALLS = 0
_compiled = {}


# ---- PTX. Plain ld.global (no .nc): a cached read-only path is not needed, and aliasing stays safe. ----
def ld32(a):
    return P.asm("ld.global.b32 $0, [$1];", I, a, side=True)


def ldx(a, size):  # integer load, sign-extended to i64 (size 4 or 8 bytes)
    if size == 8:
        return P.asm("ld.global.b64 $0, [$1];", L, a, side=True)
    return P.asm("{ .reg .b32 t; ld.global.b32 t, [$1]; cvt.s64.s32 $0, t; }", L, a, side=True)


def st32(a, v):
    P.asm("st.global.b32 [$0], $1;", None, a, v, side=True)


def st64(a, v):
    P.asm("st.global.b64 [$0], $1;", None, a, v, side=True)


def st8(a, v):  # bool store, v in {0, 1}
    P.asm("{ .reg .b16 h; cvt.u16.u32 h, $1; st.global.u8 [$0], h; }", None, a, v, side=True)


def w64(x):  # i32 -> i64 sign extend
    return P.asm("cvt.s64.s32 $0, $1;", L, x)


def n32(x):  # i64 -> i32 truncate (Triton store of i64 into an i32 pointer)
    return P.asm("cvt.u32.u64 $0, $1;", I, x)


def op64(o, a, b):
    return P.asm(f"{o}.s64 $0, $1, $2;", L, a, b)


def op32(o, a, b):
    return P.asm(f"{o}.s32 $0, $1, $2;", I, a, b)


def sel64(c, a, b):  # c != 0 ? a : b
    return P.asm("{ .reg .pred p; setp.ne.s32 p, $1, 0; selp.b64 $0, $2, $3, p; }", L, c, a, b)


def cp_slot(pos, blk, bs, rank, cps, il):
    """cp_utils.cp_local_slot (CP_SIZE > 1), all i64: local slot or -1."""
    off = op64("rem", pos, bs * cps)
    is_local = P.icmp("eq", n32(op64("rem", op64("div", off, il), cps)), n32(rank))
    local = op64("div", off, il * cps) * il + op64("rem", off, il)
    return sel64(is_local, blk * bs + local, L(-1))


def slot_of(pos, valid, r, bt, bts, bs, rank, cps, il):
    """Row slot of _prepare_dflash_inputs_kernel: -1 unless valid and the block is not the null block 0.
    Branch-free (helpers are not AST-rewritten): the block-table load is predicated on valid, else 0."""
    bn = op64("min", op64("div", pos, bs * cps), bts - 1)
    blk = P.asm("{ .reg .pred p; .reg .b32 t; setp.ne.s32 p, $2, 0; mov.b32 t, 0; @p ld.global.b32 t, [$1];"
                " cvt.s64.s32 $0, t; }", L, bt + (w64(r) * bts + bn) * 4, valid, side=True)
    ok = P.asm("{ .reg .pred p, q; setp.ne.s32 p, $1, 0; setp.ne.s64 q, $2, 0; and.pred p, p, q;"
               " selp.b32 $0, 1, 0, p; }", I, valid, blk)
    return sel64(ok, cp_slot(pos, blk, bs, rank, cps, il), L(-1))


@cute.kernel
def prep_kernel(
    R: I, T: I, NC: I, M: I, NR: I, ND: I, Q: I, K: I, P1: I, P2: I, NV: I, H8: I,
    tpos: L, tqsl: L, imap: L, last: L, nxt: L, nsamp: L, nrej: L, tin: L, sin: L, bt: L, bts: L,
    o_ids: L, o_pos: L, o_pad: L, o_qsl: L, o_sl: L, o_dcp: L, o_qslot: L, o_cpos: L, o_cslot: L,
    o_si: L, o_sp: L, o_sm: L, o_t: L, o_s: L, o_aux: L, aux_rs: L,
    a0: L, a1: L, a2: L, a3: L, a4: L, r0: L, r1: L, r2: L, r3: L, r4: L,
    mask_id: I, bs: L, rank: L, cps: L, il: L, mml: I,
    isz: cutlass.Constexpr, lsz: cutlass.Constexpr, nsz: cutlass.Constexpr, sfa: cutlass.Constexpr,
):
    t, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    if bid < R:
        r = bid
        rsi = ldx(imap + w64(r) * isz, isz)
        cs = ld32(tqsl + w64(r) * 4)
        ce = ld32(tqsl + w64(r) * 4 + 4)
        nctx = ce - cs
        vce = ce - ld32(nrej + w64(r) * 4)
        nvc = vce - cs
        lvp = ldx(tpos + w64(vce - 1) * 8, 8)
        qbase = r * Q
        for j in range(t, nctx + Q, NT):
            # context row j (j < num_ctx): position (0 for rejected rows) and slot
            if j < nctx:
                valid = P.icmp("lt", j, nvc)
                cpos = L(0)
                if valid != 0:
                    cpos = ldx(tpos + w64(cs + j) * 8, 8)
                st64(o_cpos + w64(cs + j) * 8, cpos)
                st64(o_cslot + w64(cs + j) * 8, slot_of(cpos, valid, r, bt, bts, bs, rank, cps, il))
            # query row: j in [num_valid_ctx, num_valid_ctx + Q)
            qoff = j - nvc
            if j >= nvc:
                if qoff < Q:
                    qpos = lvp + 1 + w64(qoff)
                    qidx = qbase + qoff
                    tok = I(mask_id)
                    if qoff == 0:
                        if ld32(nsamp + w64(r) * 4) > 0:
                            tok = n32(ldx(last + rsi * lsz, lsz))
                        else:
                            tok = n32(ldx(nxt + rsi * nsz, nsz))
                    st32(o_ids + w64(qidx) * 4, tok)
                    st8(o_pad + w64(qidx), I(0))
                    st64(o_pos + w64(qidx) * 8, op64("min", qpos, w64(mml - 1)))
                    st64(o_qslot + w64(qidx) * 8, slot_of(qpos, I(1), r, bt, bts, bs, rank, cps, il))
                    soff = 0 if sfa else 1
                    if qoff >= soff:
                        sidx = w64(r * K + (qoff - soff))
                        st64(o_si + sidx * 8, w64(qidx))
                        st64(o_sp + sidx * 8, qpos + 1 if sfa else qpos)
                        st64(o_sm + sidx * 8, rsi)
        if t == 0:
            st32(o_qsl + w64(r) * 4, qbase)
            sl = n32(op64("min", lvp + 1 + w64(Q), w64(mml)))
            st32(o_sl + w64(r) * 4, sl)
            st32(o_t + rsi * 4, ld32(tin + rsi * 4))
            st64(o_s + rsi * 8, ldx(sin + rsi * 8, 8))
            # _dcp_local_seq_lens_kernel on the i32 seq_lens just written (i32 math)
            ci, di, ri = n32(il), n32(cps), n32(rank)
            rem = op32("rem", sl, di * ci)
            rem = op32("min", op32("max", rem - ri * ci, I(0)), ci)
            st32(o_dcp + w64(r) * 4, op32("div", sl, di * ci) * ci + rem)
        if r == 0:
            for i in range(R + t, NR + 1, NT):
                st32(o_qsl + w64(i) * 4, R * Q)
            for i in range(R + t, NR, NT):
                st32(o_sl + w64(i) * 4, I(0))
            for i in range(R + t, ND, NT):
                st32(o_dcp + w64(i) * 4, I(0))
            for i in range(R * K + t, NR * K, NT):
                st64(o_si + w64(i) * 8, L(0))
                st64(o_sp + w64(i) * 8, L(0))
                st64(o_sm + w64(i) * 8, L(-1))
            for i in range(T + t, NC, NT):
                st64(o_cslot + w64(i) * 8, L(-1))
    elif bid < R + P1:
        # query pad [Q R, M): thread-consecutive, CTA-strided
        for i in range(R * Q + (bid - R) * NT + t, M, P1 * NT):
            st32(o_ids + w64(i) * 4, I(0))
            st64(o_pos + w64(i) * 8, L(0))
            st8(o_pad + w64(i), I(1))
            st64(o_qslot + w64(i) * 8, L(-1))
    else:
        # cat: vector v -> (aux i, row, 16 B column c); NV = NAUX * T * H8
        tH = T * H8
        for v in range(((bid - R - P1) * NT + t), NV, P2 * NT):
            i = v // tH
            rem = v - i * tH
            row = rem // H8
            c = rem - row * H8
            src = sel64(P.icmp("eq", i, I(1)), a1, a0)
            src = sel64(P.icmp("eq", i, I(2)), a2, src)
            src = sel64(P.icmp("eq", i, I(3)), a3, src)
            src = sel64(P.icmp("eq", i, I(4)), a4, src)
            rs = sel64(P.icmp("eq", i, I(1)), r1, r0)
            rs = sel64(P.icmp("eq", i, I(2)), r2, rs)
            rs = sel64(P.icmp("eq", i, I(3)), r3, rs)
            rs = sel64(P.icmp("eq", i, I(4)), r4, rs)
            x0, x1, x2, x3 = P.asm("ld.global.v4.b32 {$0, $1, $2, $3}, [$4];", (I, I, I, I),
                                   src + w64(row) * rs + w64(c) * 16, side=True)
            P.asm("st.global.v4.b32 [$0], {$1, $2, $3, $4};", None,
                  o_aux + w64(row) * aux_rs + w64(i * H8 + c) * 16, x0, x1, x2, x3, side=True)


@cute.jit
def prep_launch(
    R: I, T: I, NC: I, M: I, NR: I, ND: I, Q: I, K: I, P1: I, P2: I, NV: I, H8: I,
    tpos: L, tqsl: L, imap: L, last: L, nxt: L, nsamp: L, nrej: L, tin: L, sin: L, bt: L, bts: L,
    o_ids: L, o_pos: L, o_pad: L, o_qsl: L, o_sl: L, o_dcp: L, o_qslot: L, o_cpos: L, o_cslot: L,
    o_si: L, o_sp: L, o_sm: L, o_t: L, o_s: L, o_aux: L, aux_rs: L,
    a0: L, a1: L, a2: L, a3: L, a4: L, r0: L, r1: L, r2: L, r3: L, r4: L,
    mask_id: I, bs: L, rank: L, cps: L, il: L, mml: I, G: I, stream: cuda.CUstream,
    isz: cutlass.Constexpr, lsz: cutlass.Constexpr, nsz: cutlass.Constexpr, sfa: cutlass.Constexpr,
):
    prep_kernel(R, T, NC, M, NR, ND, Q, K, P1, P2, NV, H8, tpos, tqsl, imap, last, nxt, nsamp, nrej, tin, sin, bt,
                bts,
                o_ids, o_pos, o_pad, o_qsl, o_sl, o_dcp, o_qslot, o_cpos, o_cslot, o_si, o_sp, o_sm, o_t, o_s,
                o_aux, aux_rs, a0, a1, a2, a3, a4, r0, r1, r2, r3, r4, mask_id, bs, rank, cps, il, mml,
                isz, lsz, nsz, sfa).launch(grid=[G, 1, 1], block=[NT, 1, 1], stream=stream)


_OUT_DT = dict(input_ids=torch.int32, positions=torch.int64, is_padding=torch.bool, query_start_loc=torch.int32,
               seq_lens=torch.int32, dcp_local_seq_lens=torch.int32, query_slot=torch.int64,
               context_positions=torch.int64, context_slot_mappings=torch.int64, sample_indices=torch.int64,
               sample_pos=torch.int64, sample_idx_mapping=torch.int64, temperature=torch.float32,
               seeds=torch.int64, target_positions=torch.int64, target_qsl=torch.int32, num_sampled=torch.int32,
               num_rejected=torch.int32, temperature_in=torch.float32, seeds_in=torch.int64, block_table=torch.int32)
_ANY_INT = ("idx_mapping", "last_sampled", "next_prefill_tokens")


def _aux_ok(h, T):
    return (h.dtype == torch.bfloat16 and h.dim() == 2 and h.stride(1) == 1 and h.shape[0] >= T
            and h.data_ptr() % 16 == 0 and (h.stride(0) * 2) % 16 == 0)


def _fusable(a):
    if a["context_slot_mappings"].dim() != 2 or a["context_slot_mappings"].shape[0] != 1:  # one draft KV group
        return False
    R, T, NC = a["num_reqs"], a["num_tokens"], a["num_context"]
    Q, K = a["num_query_per_req"], a["num_speculative_steps"]
    if not (1 <= R <= a["max_num_reqs"] and 1 <= T <= NC <= a["max_num_tokens"] and R * Q <= a["max_num_tokens"]
            and a["cp_size"] > 1 and 0 <= a["cp_rank"] < a["cp_size"] and a["cp_interleave"] >= 1
            and a["block_size"] >= 1 and Q == (K if a["sample_from_anchor"] else K + 1)):
        return False
    if not all(a[n].dtype == d and a[n].is_cuda for n, d in _OUT_DT.items()):
        return False
    if not all(a[n].dtype in (torch.int32, torch.int64) for n in _ANY_INT):
        return False
    bt, aux, st = a["block_table"], a["aux_hidden"], a["aux_staging"]
    if bt.dim() != 2 or bt.stride(1) != 1:
        return False
    if aux is None:
        return False
    H = aux[0].shape[1]
    return (len(aux) == NAUX and H % 8 == 0 and all(_aux_ok(h, T) and h.shape[1] == H for h in aux)
            and _aux_ok(st, T) and st.shape[1] == NAUX * H)


def _image_path(a):
    """The 4 image launches, in served order (speculator.py:512, :410, :232, cp_utils.py:24)."""
    from types import SimpleNamespace as NS

    import numpy as np
    from vllm.v1.worker.gpu.cp_utils import prepare_dcp_local_seq_lens
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import prepare_dflash_inputs

    R, T, NC = a["num_reqs"], a["num_tokens"], a["num_context"]
    nst = a.get("num_scheduled_tokens")
    if nst is None:
        nst = a["target_qsl"][:R + 1].diff().cpu().numpy()
    ib = NS(input_ids=a["input_ids"], positions=a["positions"], is_padding=a["is_padding"],
            query_start_loc=a["query_start_loc"], seq_lens=a["seq_lens"])
    batch = NS(num_reqs=R, num_scheduled_tokens=np.asarray(nst), positions=a["target_positions"],
               query_start_loc=a["target_qsl"], idx_mapping=a["idx_mapping"])
    prepare_dflash_inputs(ib, a["query_slot"], a["context_positions"], a["context_slot_mappings"][0],
                          a["sample_indices"],
                          a["sample_pos"], a["sample_idx_mapping"], a["temperature"], a["seeds"], batch,
                          a["num_sampled"], a["num_rejected"], a["last_sampled"], a["next_prefill_tokens"],
                          a["temperature_in"], a["seeds_in"], a["block_table"], a["block_size"], a["cp_rank"],
                          a["cp_size"], a["cp_interleave"], a["mask_token_id"], a["num_query_per_req"],
                          a["num_speculative_steps"], a["max_num_reqs"], a["max_num_tokens"], a["max_model_len"],
                          a["sample_from_anchor"])
    aux = a["aux_hidden"]
    if aux is not None and a["aux_staging"] is not None:
        start = min(T, NC)
        torch.cat([h[:start] for h in aux], dim=-1, out=a["aux_staging"][:start])
    if T <= NC:
        a["context_slot_mappings"][:, T:NC].fill_(-1)
    if a["cp_size"] > 1:
        prepare_dcp_local_seq_lens(a["dcp_local_seq_lens"], a["seq_lens"], R, a["cp_size"], a["cp_rank"],
                                   a["cp_interleave"])


def draft_prep(**a):
    """Launches 1-4 of the served draft prep. Writes the persistent output buffers in place; returns None.

    Keyword arguments (tensors on one CUDA device; outputs are the persistent buffers the graph reads):
      inputs : target_positions i64 [>=T], target_qsl i32 [>=R+1], idx_mapping int [R], num_sampled i32 [R],
               num_rejected i32 [R], last_sampled int [NR(,1)], next_prefill_tokens int (row 0 used),
               temperature_in f32 [NR], seeds_in i64 [NR], block_table i32 [NR, nblk] (stride(1) == 1),
               aux_hidden list of 5 bf16 [>=T, H] or None
      outputs: input_ids i32 [M], positions i64 [M], is_padding bool [M], query_start_loc i32 [NR+1],
               seq_lens i32 [NR], dcp_local_seq_lens i32 [ND], query_slot i64 [M], context_positions i64 [M],
               context_slot_mappings i64 [1, M], sample_indices / sample_pos /
               sample_idx_mapping i64 [NR*k], temperature f32 [NR], seeds i64 [NR], aux_staging bf16 [>=NC, 5H]
      scalars: num_tokens T, num_context NC, mask_token_id, block_size, cp_rank, cp_size, cp_interleave,
               num_query_per_req Q, num_speculative_steps k, max_num_reqs NR, max_num_tokens M, max_model_len,
               sample_from_anchor (default True, DSpark served), num_scheduled_tokens (host array; the decline
               path reads it), num_graph_reqs (not read). R = idx_mapping.shape[0].
    """
    global FUSED_CALLS
    a = dict(a, num_reqs=a["idx_mapping"].shape[0])
    a.setdefault("sample_from_anchor", True)
    if not _fusable(a):
        return _image_path(a)
    FUSED_CALLS += 1
    sys.modules[__package__].DRAFT_PREP_FUSED_CALLS += 1
    key, args = _args(a)
    if key not in _compiled:  # TVM-FFI: the call takes plain Python ints (host cost ~4 us instead of ~90 us)
        typed = [t(x) for t, x in zip(_TYPES, args[:-1])] + [args[-1]]
        _compiled[key] = cute.compile(prep_launch, *typed, *key, options="--enable-tvm-ffi")
    _compiled[key](*args)


_TYPES = [I] * 12 + [L] * 37 + [I] + [L] * 4 + [I, I]  # prep_launch argument types, stream excluded


def _args(a):
    """(compile key, launch arguments) of the fused kernel."""
    R, T, NC, M, NR, Q, K = (a[n] for n in ("num_reqs", "num_tokens", "num_context", "max_num_tokens",
                                            "max_num_reqs", "num_query_per_req", "num_speculative_steps"))
    aux, st, bt = a["aux_hidden"], a["aux_staging"], a["block_table"]
    H8 = aux[0].shape[1] // 8
    NV = NAUX * T * H8
    P1 = max(1, -(-(M - R * Q) // (NT * PER_T)))
    P2 = max(1, -(-NV // (NT * PER_T)))
    p = lambda x: x.data_ptr()  # noqa: E731
    dev = st.device
    args = [R, T, NC, M, NR, a["dcp_local_seq_lens"].shape[0], Q, K, P1, P2, NV, H8,
            p(a["target_positions"]), p(a["target_qsl"]), p(a["idx_mapping"]), p(a["last_sampled"]),
            p(a["next_prefill_tokens"]), p(a["num_sampled"]), p(a["num_rejected"]), p(a["temperature_in"]),
            p(a["seeds_in"]), p(bt), bt.stride(0),
            p(a["input_ids"]), p(a["positions"]), p(a["is_padding"]), p(a["query_start_loc"]), p(a["seq_lens"]),
            p(a["dcp_local_seq_lens"]), p(a["query_slot"]), p(a["context_positions"]), p(a["context_slot_mappings"]),
            p(a["sample_indices"]), p(a["sample_pos"]), p(a["sample_idx_mapping"]), p(a["temperature"]),
            p(a["seeds"]), p(st), st.stride(0) * 2,
            *[p(h) for h in aux], *[h.stride(0) * 2 for h in aux],
            a["mask_token_id"], a["block_size"], a["cp_rank"], a["cp_size"],
            a["cp_interleave"], a["max_model_len"], R + P1 + P2,
            cuda.CUstream(torch.cuda.current_stream(dev).cuda_stream)]
    key = (a["idx_mapping"].element_size(), a["last_sampled"].element_size(),
           a["next_prefill_tokens"].element_size(), bool(a["sample_from_anchor"]))
    assert len(args) == len(_TYPES) + 1
    return key, args
