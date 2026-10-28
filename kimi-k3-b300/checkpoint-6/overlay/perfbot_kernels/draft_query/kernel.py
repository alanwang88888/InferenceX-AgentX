# SPDX-License-Identifier: Apache-2.0
"""DraftQuery: the K3 DSpark draft layer's query side in one PDL launch (item 15 F-B, kernel 2).

Replaces the served launches between the fused_qkv_a GEMM output and the DCP attention (mla.py:509-528, 662-711):
  fused_q_kv_rmsnorm (Triton, fused_qk_rmsnorm.py:35-90)   q_a / kv_a RMSNorm
  q_b_proj (nvjet)                                         [M, 1536] x [1536, 8 x 192]
  torch.bmm(q_nope, W_UK_T)                                absorb 128 -> 512 per head
  fused_kimi_k3_mla_decode_q_concat_kv_cache_fp8_insert    RoPE on q_pe / k_pe, FP8 query, FP8 latent cache insert
  increment_epoch + direct_dcp_q_gather_multimem           DCP8 query publish (epoch protocol, dcp_direct_q_gather.cu)

Structure = MlaQuery.tiled (specialized_decode/cutedsl/mla/query.py, 49cb2bbf): cluster of 8 CTAs per head, CTA
`part` owns Q_B rows 24 part .. +23 and latent columns 64 part .. +63; T = 16/32/64 requests per CTA; split-K BF16
MMAs for Q_B (weights in registers), nope rows exchanged by DSMEM bulk copies, W_UK MMAs. Changes:
  * heads 8 (TILED_FROM has no 8), no gate, RoPE on the rope rows and on k_pe (cos/sin fp32 table);
  * the q_a and kv_a norms reproduce the served Triton kernel bit for bit (PTX + SASS read, probe1/2):
      thread t (8 elements 8t..8t+7):  s = (fma(x0,x0,x2*x2) + fma(x4,x4,x6*x6)) + (fma(x1,x1,x3*x3) + fma(x5,x5,x7*x7))
      warp butterfly xor 16..1 (add.f32), then the 8 warp sums by butterfly xor 4, 2, 1 (zero warps exact)
      rstd = MUFU.RSQ(FFMA(MUFU.RCP(size), total, eps))   (div.full.f32 + add.f32 as ptxas emits them, SASS 0x05c0)
      y = bf16(RN((rstd * x) * gamma))                     (mul.f32x2 twice, cvt.rn.bf16x2)
    all adds / muls / fmas IEEE RN, no FTZ (the Triton PTX has no .ftz on them);
  * RoPE and FP8 = the served concat kernel (fused_kimi_k3_mla_key_concat_kv_cache_kernel.cu:122-176, SASS 0x0a70):
      out0 = fma(x, cos, -RN(y * sin)), out1 = fma(x, sin, RN(y * cos))    (pair (2i, 2i+1), cos at i, sin at i + 32)
      fp8 = cvt.rn.satfinite.e4m3(min(max(v * scale_inv, -448), 448))       (FMNMX: a NaN becomes -448)
  * publish = the served epoch protocol (not the target DcpBox Lamport mailbox: that mailbox needs its reader, DcpValue,
    to re-arm it, and the draft chain keeps the served LSE combine until F-B kernel 3). Every CTA reads epoch E0 after
    the grid-dependency wait and writes its query bytes to head block `rank` of every rank's final buffer; a cluster
    barrier (release / acquire) hands the 8 CTAs' stores to CTA 0, which counts the cluster in `completion` with
    atom.acq_rel.gpu (v2: no per-CTA fence.sys, which cost ~6 us); the last count resets the counter, stores
    epoch = E0 + 1, st.release.sys-es E = E0 + 1 into signal[E & 1][rank] of every rank (the one sys fence), then
    waits until every rank's signal[E & 1][s] == E (acquire).

Modes (one source): copies = 0 real (multimem.st to the multicast addresses); copies = 8 one-GPU stand-in: `ranks`
emulated ranks in one launch (grid z = rank x tile), each multicast store becomes `copies` plain stores, and rank e
waits on replica e. world = the number of signalling ranks.
"""

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint32, const_expr

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.primitives import (
    alloc,
    asm,
    at,
    copy_evict,
    copy_g2s,
    copy_s2cluster,
    early_weights,
    launch,
    ldmatrix,
    load_words,
    mma_into,
    prefetch_l2,
    published,
    smem_words,
    store_shared_words,
    unpack_bf16,
    view,
    wait,
    zeros,
)

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.mla.dcp import Q_SENT  # Lamport q mailbox sentinel

try:  # IKET phase ranges; compiled in only with iket=True.
    from cutlass.cute.experimental import iket as _iket
except ImportError:  # pragma: no cover
    _iket = None

HEADS, KP, CLUSTER, R, L = 8, 1536 + 32, 8, 24, 64
SENDERS = (128 + R - 1) // R  # 6 CTAs hold nope rows
MUTANTS = ("", "ropehalf", "noepoch", "slot", "nopoll")  # nopoll: Lamport form without the gathered poll
# timing-only cuts and variants (wrong bytes or no ordering; never gated): selftest.gpu --ablate / --bimodal
ABLATIONS = ("x_empty", "x_nonorm", "x_nopub", "x_nosig", "x_nospin", "x_nobar", "x_norel", "x_v1fence")


# ------------------------------------------------------------------ exact IEEE ops (no FTZ, no contraction)


@asm(Float32, pure=True)
def add_rn(a, b):
    return "add.rn.f32 $0,$1,$2;", "=f,f,f", (Float32(a), Float32(b))


@asm(Float32, pure=True)
def mul_rn(a, b):
    return "mul.rn.f32 $0,$1,$2;", "=f,f,f", (Float32(a), Float32(b))


@asm(Float32, pure=True)
def fma_rn(a, b, c):
    return "fma.rn.f32 $0,$1,$2,$3;", "=f,f,f,f", (Float32(a), Float32(b), Float32(c))


@asm(Float32, pure=True)
def norm_rstd(total, size, eps):
    """MUFU.RSQ(FFMA(MUFU.RCP(size), total, eps)); ``size`` is a run-time value so ptxas cannot fold the RCP."""
    code = "{ .reg .f32 r,m; rcp.approx.ftz.f32 r,$2; fma.rn.f32 m,r,$1,$3; rsqrt.approx.ftz.f32 $0,m; }"
    return code, "=f,f,f,f", (Float32(total), Float32(size), Float32(eps))


@asm(Float32, pure=True)
def square_sum_words(w01, w23, w45, w67):
    """thread_square_sum of 8 bf16 (4 words) with packed f32x2 ops: (a, b) = fma2((x0, x1)^2, (x2, x3)^2),
    (c, d) = fma2((x4, x5)^2, (x6, x7)^2), (a + c, b + d), sum; each lane IEEE RN, so the bits equal the scalar form."""
    code = "{ .reg .b32 l0,h0,l1,h1,l2,h2,l3,h3; .reg .b64 p0,p1,p2,p3,q,ab,cd; .reg .f32 s0,s1; "
    for i in range(4):
        code += f"shl.b32 l{i},${i + 1},16; and.b32 h{i},${i + 1},0xffff0000; mov.b64 p{i},{{l{i},h{i}}}; "
    code += "mul.rn.f32x2 q,p1,p1; fma.rn.f32x2 ab,p0,p0,q; mul.rn.f32x2 q,p3,p3; fma.rn.f32x2 cd,p2,p2,q; "
    code += "add.rn.f32x2 q,ab,cd; mov.b64 {s0,s1},q; add.rn.f32 $0,s0,s1; }"
    return code, "=f,r,r,r,r", tuple(Uint32(w) for w in (w01, w23, w45, w67))


@asm(Uint32, pure=True)
def norm_word(rstd, xw, gw):
    """bf16x2 of (rstd * x) * gamma for the two bf16 of ``xw`` / ``gw``, packed f32x2 muls (IEEE RN per lane)."""
    code = "{ .reg .b32 xl,xh,gl,gh; .reg .b64 x2,g2,r2,t; .reg .f32 a,b; "
    code += "shl.b32 xl,$2,16; and.b32 xh,$2,0xffff0000; shl.b32 gl,$3,16; and.b32 gh,$3,0xffff0000; "
    code += "mov.b64 x2,{xl,xh}; mov.b64 g2,{gl,gh}; mov.b64 r2,{$1,$1}; "
    code += "mul.rn.f32x2 t,r2,x2; mul.rn.f32x2 t,t,g2; mov.b64 {a,b},t; cvt.rn.bf16x2.f32 $0,b,a; }"
    return code, "=r,f,r,r", (Float32(rstd), Uint32(xw), Uint32(gw))


@asm((Float32, Float32), pure=True)
def rope_pair(x, y, cos, sin):
    code = "{ .reg .f32 t,u; mul.rn.f32 t,$3,$5; neg.f32 t,t; fma.rn.f32 $0,$2,$4,t; "
    code += "mul.rn.f32 u,$3,$4; fma.rn.f32 $1,$2,$5,u; }"
    return code, "=f,=f,f,f,f,f", (Float32(x), Float32(y), Float32(cos), Float32(sin))


@asm(Uint32, pure=True)
def fp8_pair(lo, hi, scale):
    """Low 16 bits: E4M3 (lo * scale, hi * scale), each clamped to [-448, 448] first (the served fmaxf / fminf)."""
    code = "{ .reg .f32 a,b; .reg .b16 h; mul.rn.f32 a,$1,$3; mul.rn.f32 b,$2,$3; "
    code += "max.f32 a,a,0fC3E00000; min.f32 a,a,0f43E00000; max.f32 b,b,0fC3E00000; min.f32 b,b,0f43E00000; "
    code += "cvt.rn.satfinite.e4m3x2.f32 h,b,a; cvt.u32.u16 $0,h; }"
    return code, "=r,f,f,f", (Float32(lo), Float32(hi), Float32(scale))


@asm()
def store_shared_b16(pointer, word):
    return "st.shared.b16 [$0],$1;", "r,h,~{memory}", (pointer, cutlass.Uint16(word))


@asm()
def store_global_words(pointer, words):
    n = len(words)
    regs = "{" + ",".join(f"${i + 1}" for i in range(n)) + "}" if n > 1 else "$1"
    vec = f".v{n}" if n > 1 else ""
    return f"st.global{vec}.b32 [$0],{regs};", "l" + ",r" * n + ",~{memory}", (pointer, *[Uint32(w) for w in words])


@asm()
def multimem_words(address, words):
    n = len(words)
    regs = "{" + ",".join(f"${i + 1}" for i in range(n)) + "}" if n > 1 else "$1"
    vec = f".v{n}" if n > 1 else ""
    code = f"multimem.st.relaxed.sys.global{vec}.f32 [$0],{regs};"
    return code, "l" + ",r" * n + ",~{memory}", (Int64(address), *[Uint32(w) for w in words])


@asm()
def fence_sys():  # timing-only (x_v1fence): v1's per-thread fence before the count
    return "fence.acq_rel.sys;", "~{memory}", ()


@asm(Uint32)
def atomic_add_acq_rel_gpu(pointer, value):
    return "atom.add.acq_rel.gpu.global.u32 $0,[$1],$2;", "=r,l,r,~{memory}", (pointer, Uint32(value))


@asm()
def store_u32_relaxed(pointer, value):
    return "st.relaxed.gpu.global.u32 [$0],$1;", "l,r,~{memory}", (pointer, Uint32(value))


@asm()
def store_u64_relaxed(pointer, value):
    return "st.relaxed.gpu.global.u64 [$0],$1;", "l,l,~{memory}", (pointer, Int64(value))


@asm()
def store_release_sys(pointer, value):
    return "st.release.sys.global.u32 [$0],$1;", "l,r,~{memory}", (pointer, Uint32(value))


@asm()
def multimem_release_sys(address, value):
    return "multimem.st.release.sys.global.u32 [$0],$1;", "l,r,~{memory}", (Int64(address), Uint32(value))


@asm(Uint32)
def load_acquire_sys(pointer):
    return "ld.acquire.sys.global.u32 $0,[$1];", "=r,l,~{memory}", (pointer,)


@asm(Int64)
def load_epoch(pointer):
    return "ld.relaxed.gpu.global.s64 $0,[$1];", "=l,l,~{memory}", (pointer,)


@asm(Float32)
def load_f32(pointer):
    return "ld.global.nc.f32 $0,[$1];", "=f,l,~{memory}", (pointer,)


@cute.jit
def butterfly(v):
    for shift in cutlass.range_constexpr(4, -1, -1):
        v = add_rn(v, cute.arch.shuffle_sync_bfly(v, 1 << shift))
    return v


def thread_square_sum(x):
    """The served Triton per-thread partial of 8 values (fused_qk_rmsnorm PTX lines 131-142)."""
    a = fma_rn(x[0], x[0], mul_rn(x[2], x[2]))
    b = fma_rn(x[1], x[1], mul_rn(x[3], x[3]))
    c = fma_rn(x[4], x[4], mul_rn(x[6], x[6]))
    d = fma_rn(x[5], x[5], mul_rn(x[7], x[7]))
    return add_rn(add_rn(a, c), add_rn(b, d))


class DraftQuery:
    def __init__(self, batch, ranks=1, copies=0, world=8, waiters=None, mutant="", iket=False, lamport=False):
        assert 1 <= batch <= 64 and mutant in MUTANTS + ABLATIONS
        self.batch, self.ranks, self.copies, self.world = batch, ranks, copies, world
        # epoch: signals waited on; Lamport: head blocks polled (timing solo mode: 1, own block)
        self.waiters = world if waiters is None else waiters
        self.lamport = lamport
        self.tokens = 16 if batch <= 16 else 32 if batch <= 32 else 64
        self.tiles = (batch + self.tokens - 1) // self.tokens
        self.grid = (CLUSTER, HEADS, ranks * self.tiles)
        self.ctas = CLUSTER * HEADS * self.tiles  # per emulated rank
        self.mutant, self.iket = mutant, iket and _iket is not None

    def _push(self, name):
        if const_expr(self.iket):
            _iket.range_push(name)

    def _pop(self):
        if const_expr(self.iket):
            _iket.range_pop()

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, self.grid, 256, cluster=CLUSTER)

    @cute.jit
    def live(self, token):
        return token < self.batch

    @cute.kernel
    def kernel(self, args):
        if const_expr(self.mutant == "x_empty"):
            cute.arch.griddepcontrol_launch_dependents()
            cute.arch.griddepcontrol_wait()
        else:
            self.body(args)

    @cute.jit
    def body(self, args):
        qkv, gq, gkv, w_qb, w_uk, cos_sin, positions, cache, slots = args[0:9]
        q_scale_inv, k_scale_inv, final, signal, epoch, completion = args[9:15]
        mc_query, mc_signal, rank_arg, eps, size_q, size_kv = args[15:21]
        tid, _, _ = cute.arch.thread_idx()
        part, head, z = cute.arch.block_idx()
        e = z // self.tiles  # emulated rank (0 when real)
        tok0 = (z % self.tiles) * self.tokens
        rank = rank_arg if const_expr(self.copies == 0) else e
        gh = e * HEADS + head  # head row of the (stand-in: stacked) weights
        T = self.tokens
        MT = T // 16
        lane, warp = tid % 32, tid // 32
        group, quarter = lane // 4, lane % 4
        cute.arch.griddepcontrol_launch_dependents()
        smem = cutlass.utils.SmemAllocator()
        xq = alloc(smem, BFloat16, (T, KP), 128)
        uk = alloc(smem, BFloat16, (128, L + 8), 128)  # 144 B rows for ldmatrix
        qa = alloc(smem, BFloat16, 1536)
        cs = alloc(smem, Float32, (T, 12, 2))  # this part's rope (cos, sin) per request and pair
        ss = alloc(smem, Float32, (8, 2))
        barq = smem.allocate(cutlass.Uint64, 8)
        red = view(xq.iterator, 0, Float32, (8, T * R // 32, 32), (T * R, 32, 1))
        qblk = view(xq.iterator, 8 * T * R * 4, BFloat16, (T, R), (R, 1))
        qn = view(xq.iterator, 9 * T * R * 4, BFloat16, (SENDERS, T, R), (T * R, R, 1))
        stage = view(xq.iterator, 9 * T * R * 4 + SENDERS * T * R * 2, cutlass.Uint8, (T, 96), (96, 1))
        if tid == 0:
            cute.arch.mbarrier_init(barq, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.mbarrier_arrive_and_expect_tx(barq, SENDERS * T * R * 2)
        # ---- static weights and step metadata before the grid-dependency wait
        self._push("fill_weights")
        rows = gh * 192 + part * R + group
        qbr = [
            [early_weights(at(w_qb, rows + 8 * n, (6 * warp + cc) * 32 + 8 * quarter), count=4) for n in range(3)]
            for cc in range(6)
        ]
        for c in range(tid, 128 * L // 8, 256):
            p, j = c // (L // 8), c % (L // 8)
            copy_evict(uk.iterator + p * (L + 8) + j * 8, at(w_uk, gh, p, part * L + j * 8))
        if tid < 192:
            copy_evict(qa.iterator + tid * 8, gq.iterator + tid * 8)
        cute.arch.cp_async_commit_group()
        kfirst, npairs = Int32(0), Int32(0)  # first rope dim of this part; rope pairs: 8 / 12 / 12 at parts 5 / 6 / 7
        if R * part + R > 128:
            if R * part > 128:
                kfirst = R * part - 128
            npairs = (R * part + R - 128 - kfirst) // 2
        if part >= 5:
            for i in range(tid, T * 12, 256):
                t, k2 = i // 12, i % 12
                if (k2 < npairs) & self.live(tok0 + t):
                    pos = positions[tok0 + t]
                    pair = kfirst // 2 + k2
                    if const_expr(self.mutant == "ropehalf"):
                        cs[t, k2, 0] = cos_sin[pos, pair + 32]
                        cs[t, k2, 1] = cos_sin[pos, pair]
                    else:
                        cs[t, k2, 0] = cos_sin[pos, pair]
                        cs[t, k2, 1] = cos_sin[pos, pair + 32]
        # KV insert inputs: CTA j of the tile owns request j
        j = head * CLUSTER + part
        ins_valid = (j < T) & self.live(tok0 + j)
        slot = Int64(-1)
        if ins_valid:
            slot = slots[e, tok0 + j]
        gk = cute.make_rmem_tensor(8, Float32)
        kcs = cute.make_rmem_tensor(8, Float32)
        if slot >= 0:
            if tid < 64:
                gw = early_weights(gkv.iterator + tid * 8, count=4)
                for i in cutlass.range_constexpr(4):
                    gk[2 * i], gk[2 * i + 1] = unpack_bf16(gw[i])
            elif tid < 72:
                kpos = positions[tok0 + j]
                for i in cutlass.range_constexpr(4):
                    kpair = 4 * (tid - 64) + i
                    if const_expr(self.mutant == "ropehalf"):
                        kcs[2 * i], kcs[2 * i + 1] = cos_sin[kpos, kpair + 32], cos_sin[kpos, kpair]
                    else:
                        kcs[2 * i], kcs[2 * i + 1] = cos_sin[kpos, kpair], cos_sin[kpos, kpair + 32]
        qsi = q_scale_inv[0]
        ksi = k_scale_inv[0]
        self._pop()
        self._push("sync_griddep")
        cute.arch.griddepcontrol_wait()
        self._pop()
        e0 = load_epoch(epoch.iterator + e)
        for c in range(tid, T * 192, 256):
            t, jj = c // 192, c % 192
            if self.live(tok0 + t):
                copy_g2s(xq.iterator + t * KP + jj * 8, at(qkv, tok0 + t, jj * 8))
        cute.arch.cp_async_commit_group()
        # ---- KV row: Triton-exact kv_a norm, RoPE on k_pe, FP8 insert (overlaps the staging of x)
        self._push("main_insert")
        if slot >= 0:
            xx = cute.make_rmem_tensor(8, Float32)
            total = Float32(0)
            if tid < 64:
                kw = load_words(at(qkv, tok0 + j, 1536 + tid * 8), count=4, mutable=True)
                for i in cutlass.range_constexpr(4):
                    xx[2 * i], xx[2 * i + 1] = unpack_bf16(kw[i])
                total = thread_square_sum(xx)
            total = butterfly(total)
            if (lane == 0) & (warp < 2):
                ss[warp, 0] = total
            cute.arch.sync_threads()
            dst = slot
            if const_expr(self.mutant == "slot"):
                dst = slot + 1
            page, offset = dst // cache.shape[2], dst % cache.shape[2]
            if tid < 64:
                rr = norm_rstd(add_rn(ss[0, 0], ss[1, 0]), size_kv, eps)
                kq = []
                for i in cutlass.range_constexpr(4):
                    lo = Float32(BFloat16(mul_rn(mul_rn(rr, xx[2 * i]), gk[2 * i])))
                    hi = Float32(BFloat16(mul_rn(mul_rn(rr, xx[2 * i + 1]), gk[2 * i + 1])))
                    kq.append(fp8_pair(lo, hi, ksi))
                store_global_words(at(cache, e, page, offset, tid * 8), (kq[0] | (kq[1] << 16), kq[2] | (kq[3] << 16)))
            elif tid < 72:
                ck = tid - 64
                pw = load_words(at(qkv, tok0 + j, 2048 + ck * 8), count=4, mutable=True)
                kr = []
                for i in cutlass.range_constexpr(4):
                    xa, ya = unpack_bf16(pw[i])
                    r0, r1 = rope_pair(xa, ya, kcs[2 * i], kcs[2 * i + 1])
                    kr.append(fp8_pair(r0, r1, ksi))
                rope_out = (kr[0] | (kr[1] << 16), kr[2] | (kr[3] << 16))
                store_global_words(at(cache, e, page, offset, 512 + ck * 8), rope_out)
        self._pop()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        # ---- q_a norm, Triton-exact: lane `lane` of chunk i is Triton thread 32 i + lane
        self._push("main_qnorm")
        for rr0 in cutlass.range_constexpr(0 if const_expr(self.mutant == "x_nonorm") else T // 8):
            row = warp + 8 * rr0
            if self.live(tok0 + row):  # dead rows stay raw: their MMA rows are never published
                xrow = xq.iterator + row * KP
                words = [smem_words(xrow + 256 * i + 8 * lane) for i in range(6)]
                wsum = [butterfly(square_sum_words(*words[i])) for i in range(6)]
                total = add_rn(add_rn(add_rn(wsum[0], wsum[4]), wsum[2]), add_rn(add_rn(wsum[1], wsum[5]), wsum[3]))
                rr = norm_rstd(total, size_q, eps)
                for i in cutlass.range_constexpr(6):
                    gw = smem_words(qa.iterator + 256 * i + 8 * lane)
                    normed = tuple(norm_word(rr, words[i][k], gw[k]) for k in range(4))
                    store_shared_words(xrow + 256 * i + 8 * lane, normed)
        cute.arch.sync_threads()
        self._pop()
        # ---- Q_B: split-K BF16 MMAs (warp w owns k = 192 w .. 192 w + 191)
        self._push("main_qb")
        acc = zeros((MT * 3, 4), Float32)
        for cc in cutlass.range_constexpr(6):
            k = (6 * warp + cc) * 32 + 8 * quarter
            zq = qbr[cc]
            for m in cutlass.range_constexpr(MT):
                x, y = (
                    load_words(xq.iterator + r * KP + k, count=4, shared=True)
                    for r in (16 * m + group, 16 * m + group + 8)
                )
                for n in cutlass.range_constexpr(3):
                    for s in cutlass.range_constexpr(2):
                        a = (x[2 * s], y[2 * s], x[2 * s + 1], y[2 * s + 1])
                        b = (zq[n][2 * s], zq[n][2 * s + 1])
                        mma_into(acc[m * 3 + n, None], a, b, bf16=True)
        cute.arch.sync_threads()
        cute.arch.cluster_arrive_relaxed()  # x is consumed: peers may fill qn
        for i in cutlass.range_constexpr(MT * 12):
            red[warp, i, lane] = acc[i // 4, i % 4]
        cute.arch.sync_threads()
        for h2 in range(tid, T * R // 2, 256):
            ip, ln = h2 // 32, h2 % 32
            i0 = 2 * ip
            v0, v1 = Float32(0), Float32(0)
            for w in cutlass.range_constexpr(8):
                v0 += red[w, i0, ln]
                v1 += red[w, i0 + 1, ln]
            v0, v1 = Float32(BFloat16(v0)), Float32(BFloat16(v1))
            tile, c = i0 // 4, i0 % 4
            t = 16 * (tile // 3) + ln // 4 + 8 * (c // 2)
            r = 8 * (tile % 3) + 2 * (ln % 4)
            p = R * part + r
            if p < 128:
                qblk[t, r] = BFloat16(v0)
                qblk[t, r + 1] = BFloat16(v1)
            elif self.live(tok0 + t):
                k2 = (p - 128 - kfirst) // 2
                o0, o1 = rope_pair(v0, v1, cs[t, k2, 0], cs[t, k2, 1])
                store_shared_b16(stage.iterator + t * 96 + 64 + 2 * k2, fp8_pair(o0, o1, qsi))
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_threads()
        cute.arch.cluster_wait()
        if (part < SENDERS) & (tid < CLUSTER):
            copy_s2cluster(qn.iterator + part * T * R, qblk.iterator, barq, tid, size=T * R * 2)
        self._pop()
        self._push("sync_dsmem")
        wait(barq)
        self._pop()
        cute.arch.cluster_arrive_relaxed()  # peers may exit once qblk is delivered
        # ---- W_UK: warp w owns latent columns 64 part + 8 w .. + 7
        self._push("main_wuk")
        b_frags = [ldmatrix(uk.iterator + (32 * kp + lane) * (L + 8) + 8 * warp, trans=True) for kp in range(4)]
        for m in cutlass.range_constexpr(MT):
            out = zeros(4, Float32)
            t = 16 * m + lane % 8 + 8 * ((lane // 8) % 2)
            for ks in cutlass.range_constexpr(8):
                p = 16 * ks + 8 * (lane // 16)
                a = ldmatrix(qn.iterator + ((p // R) * T + t) * R + p % R)
                b = b_frags[ks // 2][2 * (ks % 2) : 2 * (ks % 2) + 2]
                mma_into(out, a, b, bf16=True)
            for h in cutlass.range_constexpr(2):  # rows group, group + 8: 2 bytes
                row = 16 * m + group + 8 * h
                lo, hi = (Float32(BFloat16(out[2 * h + q])) for q in range(2))
                store_shared_b16(stage.iterator + row * 96 + 8 * warp + 2 * quarter, fp8_pair(lo, hi, qsi))
        cute.arch.sync_threads()
        self._pop()
        # ---- publish: 16 B latent chunks + 8 B rope words of this CTA into head block `rank` of every rank
        self._push("main_publish")
        items = 4 + npairs // 4  # 16 B latent chunks + 8 B rope words (2 at part 5, 3 at parts 6 / 7)
        for i in range(tid, 0 if const_expr(self.mutant == "x_nopub") else T * items, 256):
            t, w = i // items, i % items
            if self.live(tok0 + t):
                row = (tok0 + t) * (HEADS * self.world) + rank * HEADS + head
                if w < 4:
                    chunk = smem_words(stage.iterator + t * 96 + w * 16)
                    self.put(final, mc_query, row, part * L + w * 16, chunk)
                else:
                    rope_words = load_words(stage.iterator + t * 96 + 64 + (w - 4) * 8, count=2, shared=True,
                                            mutable=True)
                    self.put(final, mc_query, row, 512 + kfirst + (w - 4) * 8, rope_words)
        self._pop()
        self._push("sync_publish")
        cute.arch.cluster_wait()  # closes the phase opened after the DSMEM delivery (peers' bulk copies are done)
        if const_expr(self.lamport):
            # Lamport form (mla.dcp.DcpBox q mailbox, MlaQuery.gathered): no epoch, no signal; this CTA polls its share
            # of the gathered rows until no 16 B chunk holds the sentinel. The reader (kernel 3) re-arms the rows.
            # One emulated rank per launch only: 8 ranks in one launch would spin on CTAs that are not resident.
            if const_expr(self.ranks == 1 and self.mutant != "nopoll"):
                self.gathered(final, rank, tid)
        else:
            # Release chain, no fence.sys (lever c): barrier.cluster arrive.release / wait.acquire puts the 8 CTAs' query
            # stores before CTA 0's atom.acq_rel.gpu count; the last count acquires every cluster's release, and its
            # st.release.sys signal releases them all to the peers (PTX causality order is transitive across scopes).
            if const_expr(self.mutant == "x_v1fence"):
                fence_sys()
            if const_expr(self.mutant != "x_nobar"):
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()
            if (part == 0) & (tid == (0 if const_expr(self.mutant != "x_nosig") else 1 << 20)):
                new = Uint32(e0 + 1)
                slot_sig = Int32(new & 1)
                done = atomic_add_acq_rel_gpu(completion.iterator + e, 1)
                if done + 1 == Uint32(self.ctas // CLUSTER):
                    store_u32_relaxed(completion.iterator + e, 0)
                    if const_expr(self.mutant != "noepoch"):
                        store_u64_relaxed(epoch.iterator + e, e0 + 1)
                    if const_expr(self.copies == 0):
                        multimem_release_sys(mc_signal + Int64(slot_sig * self.world + rank) * 4, new)
                    else:
                        for cpy in cutlass.range_constexpr(self.copies):
                            if const_expr(self.mutant == "x_norel"):
                                store_u32_relaxed(at(signal, cpy, slot_sig, rank), new)
                            else:
                                store_release_sys(at(signal, cpy, slot_sig, rank), new)
                    mine = 0 if const_expr(self.copies == 0) else e
                    for s in cutlass.range_constexpr(0 if const_expr(self.mutant == "x_nospin") else self.waiters):
                        got = load_acquire_sys(at(signal, mine, slot_sig, s))
                        while got != new:
                            got = load_acquire_sys(at(signal, mine, slot_sig, s))
        self._pop()

    @cute.jit
    def gathered(self, final, rank, tid):
        """Poll ``waiters`` head blocks (8: all ranks; 1: block ``rank``) of rows 0..M-1 in replica 0 (the local
        mailbox), 8 volatile 16 B loads in flight per thread, until no chunk is the sentinel."""
        part, head, _ = cute.arch.block_idx()
        threads: cutlass.Constexpr = self.ctas * 256
        per_row: cutlass.Constexpr = self.waiters * HEADS * 36
        chunks: cutlass.Constexpr = self.batch * per_row
        rounds: cutlass.Constexpr = -(-chunks // threads)
        first = (head * CLUSTER + part) * 256 + tid
        for i0 in cutlass.range_constexpr(0, rounds, 8):
            n: cutlass.Constexpr = min(8, rounds - i0)
            sources = []
            for i in cutlass.range_constexpr(n):
                c = min(first + (i0 + i) * threads, chunks - 1)  # threads past the end re-poll the last chunk
                t, rem = c // per_row, c % per_row
                block = rem // (HEADS * 36)
                if const_expr(self.waiters == 1):
                    block = rank
                row = t * (HEADS * self.world) + block * HEADS + (rem % (HEADS * 36)) // 36
                sources.append(at(final, 0, Int64(row) * 576 + (rem % 36) * 16))
            loads = [load_words(src, count=4, volatile=True) for src in sources]
            for i in cutlass.range_constexpr(n):
                words = loads[i]
                while not published(words, sentinel=Q_SENT):
                    words = load_words(sources[i], count=4, volatile=True)

    @cute.jit
    def put(self, final, mc_query, row, byte, words):
        """One 16 B / 8 B piece of query row ``row`` (token x gathered head) at ``byte``: multimem (real) or a plain
        store into every replica (stand-in)."""
        if const_expr(self.copies == 0):
            multimem_words(mc_query + Int64(row) * 576 + byte, words)
        else:
            for cpy in cutlass.range_constexpr(self.copies):
                store_global_words(at(final, cpy, Int64(row) * 576 + byte), words)


@asm(Uint32, pure=True)
def _bf16x2(lo, hi):
    return "cvt.rn.bf16x2.f32 $0,$2,$1;", "=r,f,f", (Float32(lo), Float32(hi))
