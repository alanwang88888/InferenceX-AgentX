"""KDA stage phase B ("core"): launches 2-5 of the image's spec-verify path in ONE CuTe DSL launch.

  core(lw, st, proj, out)   lw: common.Layer, st: common.State, proj [T, 6288] bf16 (the in_proj rows), out
                            [1, T, 12, 128] bf16. Updates st.conv and st.ssm in place, as the image does.

Replaces (vllm/models/kimi_k3/nvidia/kda.py @3696c772, spec path): f_b_proj (cuBLAS) -> causal_conv1d_update (spec)
-> fused_recurrent_kda (spec, a state stored after every token) -> o_norm (FusedRMSNormGated, sigmoid).
Port of the cluster layout of TensorRT-LLM k3_kda_verify (Apache-2.0, trtllm-k3-ll
tensorrt_llm/_torch/cute_dsl_kernels/k3_kda_verify/k3_kda_verify_kernel.py): a cluster of VS CTAs per (request n,
head h), CTA s owns V rows [s D/VS, (s + 1) D/VS); the norm partials go to every peer by st.async. CFG = (VS, G, R):
G lanes share a row's keys, each thread holds R rows. Default (4, 32, 4): the fastest of CFGS at N 6 in
A-core-03; 256 threads, 80 registers. The state and conv semantics are vLLM's, not TRT's:
  start state  ssm[spec_idx[n, a - 1]] (a = accepted[n]); the state after token t is stored to ssm[spec_idx[n, t]]
               when that index is > 0; start index <= 0: no state is read or written and the 8 output rows are 0
               (the image writes 0 to row 0 and leaves rows 1..7 to normalize whatever the buffer held).
  conv window  slot c = spec_idx[n, 0] (c <= 0: zero history, slot untouched); raw inputs [old[a-1], old[a], old[a+1],
               x0..x7]; new slot = [old[a], old[a+1], x0..x7].
Arithmetic: g1 = f_a @ w_fb[h]^T on mma.sync (bf16 in, fp32 accumulate, kept fp32); conv4 + SiLU, q/k L2 norm, gate,
beta, the recurrence and the gated RMSNorm in fp32. The image rounds g1, q/k/v and the raw output to bf16; this kernel
does not. Its errors vs fp64 on its bf16 proj input: ssm ~1e-7, core ~1e-4 (bf16 output); the image's ssm ~1e-3.
PDL: before griddepcontrol.wait the kernel reads only w_fb, w_conv, a_log, dt_bias, w_norm; proj, spec_idx,
accepted, conv and ssm are read after it. spec_idx may be row-strided (the server's block_table[:N, :8]); accepted
is contiguous int32. The work split is (VS, H, N): it does not depend on N.
Prologue (A-iket-01: 3.2 us exposed behind cuBLAS in_proj, which does not trigger PDL early): the weight loads are
issued and not waited for; thread 0 inits the norm mailbox, fences it, and every thread arrives on the cluster barrier
relaxed; the cluster wait sits just before the first st.async. After griddepcontrol.wait the proj loads go first,
then the metadata, then the old conv window and the start state. minctasm 3 holds 80 registers (3 CTAs per SM).
Compiles once per (device, lower_bound, cfg); the first call must be eager.
"""

import threading

import torch

H, D, NT, WIDTH = 12, 128, 8, 4
P = H * D
COLS = 6288
C_OG, C_FA, C_B = 3 * P, 4 * P, 4 * P + D  # proj columns: q | k | v | og | f_a | b | pad
EPS, SCALE = 1e-5, D ** -0.5
# (cluster size VS, lanes per row group G, rows per thread R); VS sets the V split, G and R the CTA's thread layout
CFGS = ((8, 32, 2), (8, 16, 2), (4, 16, 2), (4, 16, 4), (4, 32, 4))
CFG = CFGS[4]
_LOCK = threading.Lock()
_COMPILED: dict = {}


def _build(lower_bound: float, cfg: tuple, variant: str = "full"):
    STORE = "nostore" not in variant
    PDL = "nopdl" not in variant
    LANDED = "landed" in variant  # the A-core-04 prologue and load order (sha ffa19496), for same-run timing
    OLD_ORDER = LANDED or "oldorder" in variant  # diagnostic: the relaxed cluster arrive with the landed load order
    IKET = variant.startswith("iket")  # phase ranges for iket_core.py (run-iket); the served variant "full" has none
    # minctasm 3 keeps 3 CTAs per SM (80 registers): the proj-first load order would take 102, the iket ranges 104.
    LAUNCH = {} if LANDED else {"min_blocks_per_mp": 3}
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    from cutlass import Float32, Int32, Int64
    from cutlass._mlir.dialects import llvm
    from cutlass.cutlass_dsl import T, dsl_user_op

    def asm(ret, args, body, cons, loc, ip):
        return llvm.inline_asm(ret, args, body, cons, has_side_effects=True, is_align_stack=False,
                               asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)

    def v(x, loc, ip):
        return x.ir_value(loc=loc, ip=ip)

    def unpack(st, n, loc, ip):
        return [Float32(llvm.extractvalue(T.f32(), st, [i], loc=loc, ip=ip)) for i in range(n)]

    @dsl_user_op
    def ldg_b32(a: Int64, *, loc=None, ip=None) -> Int32:
        return Int32(asm(T.i32(), [v(Int64(a), loc, ip)], "ld.global.b32 $0, [$1];", "=r,l", loc, ip))

    @dsl_user_op
    def ldg_u16(a: Int64, *, loc=None, ip=None) -> Int32:
        return Int32(asm(T.i32(), [v(Int64(a), loc, ip)], "ld.global.u16 $0, [$1];", "=r,l", loc, ip))

    @dsl_user_op
    def stg_u16(a: Int64, x: Int32, *, loc=None, ip=None):
        asm(None, [v(Int64(a), loc, ip), v(Int32(x), loc, ip)], "st.global.u16 [$0], $1;", "l,r", loc, ip)

    @dsl_user_op
    def bf(bits: Int32, *, loc=None, ip=None) -> Float32:
        """fp32 of the bf16 in the low 16 bits."""
        return Float32(asm(T.f32(), [v(Int32(bits), loc, ip)], "{\n.reg .b32 t;\nshl.b32 t, $1, 16;\nmov.b32 $0, t;\n}",
                           "=f,r", loc, ip))

    @dsl_user_op
    def ldg_f32(a: Int64, *, loc=None, ip=None) -> Float32:
        return Float32(asm(T.f32(), [v(Int64(a), loc, ip)], "ld.global.f32 $0, [$1];", "=f,l", loc, ip))

    @dsl_user_op
    def ldg_v4(a: Int64, *, loc=None, ip=None):
        st = asm(llvm.StructType.get_literal([T.f32()] * 4), [v(Int64(a), loc, ip)],
                 "ld.global.v4.f32 {$0, $1, $2, $3}, [$4];", "=f,=f,=f,=f,l", loc, ip)
        return unpack(st, 4, loc, ip)

    @dsl_user_op
    def stg_v4(a: Int64, x0: Float32, x1: Float32, x2: Float32, x3: Float32, *, loc=None, ip=None):
        asm(None, [v(Int64(a), loc, ip)] + [v(Float32(x), loc, ip) for x in (x0, x1, x2, x3)],
            "st.global.v4.f32 [$0], {$1, $2, $3, $4};", "l,f,f,f,f", loc, ip)

    @dsl_user_op
    def stg_bf16(a: Int64, x: Float32, *, loc=None, ip=None):
        asm(None, [v(Int64(a), loc, ip), v(Float32(x), loc, ip)],
            "{\n.reg .b16 t;\ncvt.rn.bf16.f32 t, $1;\nst.global.b16 [$0], t;\n}", "l,f", loc, ip)

    @dsl_user_op
    def bf16_round(x: Float32, *, loc=None, ip=None) -> Float32:
        return Float32(asm(T.f32(), [v(Float32(x), loc, ip)],
                           "{\n.reg .b16 t;\n.reg .b16 z;\nmov.b16 z, 0;\ncvt.rn.bf16.f32 t, $1;\nmov.b32 $0, {z, t};\n}",
                           "=f,f", loc, ip))

    @dsl_user_op
    def lds(a: Int32, *, loc=None, ip=None) -> Float32:
        return Float32(asm(T.f32(), [v(Int32(a), loc, ip)], "ld.shared.f32 $0, [$1];", "=f,r", loc, ip))

    @dsl_user_op
    def lds_v4(a: Int32, *, loc=None, ip=None):
        st = asm(llvm.StructType.get_literal([T.f32()] * 4), [v(Int32(a), loc, ip)],
                 "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];", "=f,=f,=f,=f,r", loc, ip)
        return unpack(st, 4, loc, ip)

    @dsl_user_op
    def sts(a: Int32, x: Float32, *, loc=None, ip=None):
        asm(None, [v(Int32(a), loc, ip), v(Float32(x), loc, ip)], "st.shared.f32 [$0], $1;", "r,f", loc, ip)

    @dsl_user_op
    def sts_i(a: Int32, x: Int32, *, loc=None, ip=None):
        asm(None, [v(Int32(a), loc, ip), v(Int32(x), loc, ip)], "st.shared.b32 [$0], $1;", "r,r", loc, ip)

    @dsl_user_op
    def lds_i(a: Int32, *, loc=None, ip=None) -> Int32:
        return Int32(asm(T.i32(), [v(Int32(a), loc, ip)], "ld.shared.b32 $0, [$1];", "=r,r", loc, ip))

    @dsl_user_op
    def sts_v4(a: Int32, x0: Float32, x1: Float32, x2: Float32, x3: Float32, *, loc=None, ip=None):
        asm(None, [v(Int32(a), loc, ip)] + [v(Float32(x), loc, ip) for x in (x0, x1, x2, x3)],
            "st.shared.v4.f32 [$0], {$1, $2, $3, $4};", "r,f,f,f,f", loc, ip)

    @dsl_user_op
    def mma_bf16(a0: Int32, a1: Int32, a2: Int32, a3: Int32, b0: Int32, b1: Int32, *, loc=None, ip=None):
        """D = A B, m16n8k16 bf16 -> fp32, C = 0."""
        st = asm(llvm.StructType.get_literal([T.f32()] * 4), [v(Int32(x), loc, ip) for x in (a0, a1, a2, a3, b0, b1)],
                 "{\n.reg .f32 z;\nmov.f32 z, 0f00000000;\n"
                 "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, "
                 "{z, z, z, z};\n}", "=f,=f,=f,=f,r,r,r,r,r,r", loc, ip)
        return unpack(st, 4, loc, ip)

    @dsl_user_op
    def mma_acc(c: list, a0: Int32, a1: Int32, a2: Int32, a3: Int32, b0: Int32, b1: Int32, *, loc=None, ip=None):
        st = asm(llvm.StructType.get_literal([T.f32()] * 4),
                 [v(Int32(x), loc, ip) for x in (a0, a1, a2, a3, b0, b1)] + [v(Float32(x), loc, ip) for x in c],
                 "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, "
                 "{$10, $11, $12, $13};", "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f", loc, ip)
        return unpack(st, 4, loc, ip)

    @dsl_user_op
    def mbar_init(mb: Int32, nbytes: Int32, *, loc=None, ip=None):
        asm(None, [v(Int32(mb), loc, ip), v(Int32(nbytes), loc, ip)],
            "mbarrier.init.shared::cta.b64 [$0], 1;\nmbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], $1;\n"
            "fence.mbarrier_init.release.cluster;", "r,r", loc, ip)

    @dsl_user_op
    def cluster_sync(*, loc=None, ip=None):
        asm(None, [], "barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;", "", loc, ip)

    @dsl_user_op
    def cluster_arrive_relaxed(*, loc=None, ip=None):
        """Arrive without release: only the mbarrier init must reach the peers, and fence.mbarrier_init (thread 0,
        before its arrive) releases it at cluster scope (CUTLASS fence_barrier_init + cluster_arrive_relaxed)."""
        asm(None, [], "barrier.cluster.arrive.relaxed.aligned;", "", loc, ip)

    @dsl_user_op
    def cluster_wait(*, loc=None, ip=None):
        asm(None, [], "barrier.cluster.wait.acquire.aligned;", "", loc, ip)

    @dsl_user_op
    def st_async_peer(local: Int32, x: Float32, mb: Int32, peer: Int32, *, loc=None, ip=None):
        """st.async of x to the same smem offset in cluster CTA `peer`, completing peer's mbarrier (TRT's pattern)."""
        asm(None, [v(Int32(local), loc, ip), v(Float32(x), loc, ip), v(Int32(mb), loc, ip), v(Int32(peer), loc, ip)],
            "{\n.reg .b32 ra, rb;\nmapa.shared::cluster.u32 ra, $0, $3;\nmapa.shared::cluster.u32 rb, $2, $3;\n"
            "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [ra], $1, [rb];\n}", "r,f,r,r", loc, ip)

    @dsl_user_op
    def mbar_try_wait_cluster(mb: Int32, *, loc=None, ip=None):
        """mbarrier.try_wait.parity(0).acquire.cluster: the peers' st.async complete_tx release at cluster scope."""
        return Int32(asm(T.i32(), [v(Int32(mb), loc, ip)],
                         "{\n.reg .pred p;\nmbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [$1], 0;\n"
                         "selp.u32 $0, 1, 0, p;\n}", "=r,r", loc, ip)) != Int32(0)

    if IKET:
        from cutlass.cute.experimental import iket

    def push(name):  # warp-uniform phase range, compiled in only for the iket variants
        if cutlass.const_expr(IKET):
            iket.range_push(name)

    def pop():
        if cutlass.const_expr(IKET):
            iket.range_pop()

    def bfly(x, offs=(16, 8, 4, 2, 1)):
        for o in offs:
            x = x + cute.arch.shuffle_sync_bfly(x, offset=o)
        return x

    def sigmoid(x):
        return Float32(1.0) / (Float32(1.0) + cute.math.exp(-x, fastmath=True))

    VS, G, R = cfg
    VR = D // VS  # V rows per CTA (= the cluster rank's slice)
    THREADS = VR // R * G  # a group of G lanes owns R rows; lane q of a group keys 4 q + 4 G j + e
    NJ = D // (4 * G)  # 16-byte key chunks per lane
    W = THREADS // 32
    CPT = 2 * D // THREADS  # q/k conv channels per thread
    NO = (NT * VR + THREADS - 1) // THREADS  # norm entries (token, row) per thread
    assert VR <= 32 and 32 % VR == 0 and THREADS % 32 == 0 and NT % W == 0 and (D // 16) % W == 0
    S_Q, S_K, S_G, S_DEC, S_KD, S_BK = (i * NT * D for i in range(6))
    S_V = 6 * NT * D
    S_OG, S_O, S_BETA = S_V + NT * VR, S_V + 2 * NT * VR, S_V + 3 * NT * VR
    S_SS = S_BETA + 16
    S_CW = S_SS + VS * NT  # the new q/k conv window, raw bf16 bits [10][256]
    SMEM_WORDS = S_CW + (WIDTH - 1 + NT - 1) * 2 * D

    @cute.kernel
    def kernel(proj: Int64, out: Int64, w_fb: Int64, w_conv: Int64, a_log: Int64, dt_bias: Int64, w_norm: Int64,
               conv: Int64, conv_stride: Int64, ssm: Int64, ssm_stride: Int64, spec_idx: Int64, idx_stride: Int64,
               accepted: Int64):
        push("cta")
        push("prologue")
        tidx, _, _ = cute.arch.thread_idx()
        vs, h, n = cute.arch.block_idx()
        lane = tidx % 32
        warp = tidx // 32
        grp = lane // 4
        tig = lane % 4
        alloc = utils.SmemAllocator()
        sb = alloc.allocate(SMEM_WORDS * 4, byte_alignment=128).toint()
        mb = alloc.allocate(8, byte_alignment=8).toint()
        sa = lambda off, i: sb + Int32(4) * (Int32(off) + i)  # noqa: E731
        row0 = Int64(n) * NT

        # ---------------- prologue: weights and constants only (nothing a predecessor writes)
        if tidx == 0:
            mbar_init(mb, Int32(VS * NT * 4))
        cq = [tidx + i * THREADS for i in range(CPT)]  # q (c < 128) or k (c >= 128) channels of this thread
        ch = [cutlass.select_(c < D, h * D + c, P + h * D + c - D) for c in cq]
        wc = [ldg_v4(w_conv + Int64(x) * (WIDTH * 4)) for x in ch]
        chv = 2 * P + h * D + vs * VR + tidx % VR  # v channel (used by tidx < VR)
        wv = ldg_v4(w_conv + Int64(chv) * (WIDTH * 4))
        dtb = ldg_v4(dt_bias + Int64(h * D + lane * 4) * 4)
        a_log_h = ldg_f32(a_log + Int64(h) * 4)
        if cutlass.const_expr(LANDED):
            exp_a = cute.math.exp(a_log_h, fastmath=True)
        ne = [tidx + i * THREADS for i in range(NO)]  # norm entries e = token * VR + row
        wn = [ldg_f32(w_norm + Int64(vs * VR + e % VR) * 4) for e in ne]
        UNITS = D // 16 // W  # f_b m-tiles per warp; with one, its w_fb fragments load here (32 registers)

        def a_frags(mt):
            wrow = w_fb + (Int64(h * D + mt * 16 + grp) * D + tig * 2) * 2
            return [[ldg_b32(wrow + (r8 * D + ks * 16 + c8) * 2) for r8, c8 in ((0, 0), (8, 0), (0, 8), (8, 8))]
                    for ks in range(D // 16)]
        afr0 = a_frags(warp) if UNITS == 1 else None
        pop()
        push("cluster_sync")
        if cutlass.const_expr(LANDED):
            cluster_sync()  # every peer's mailbox is initialized before any st.async
        else:
            cluster_arrive_relaxed()  # its wait comes before the first st.async (norm_send)
        pop()

        if cutlass.const_expr(PDL):
            push("gdc_wait")
            cute.arch.griddepcontrol_wait()
            pop()
            cute.arch.griddepcontrol_launch_dependents()
        push("meta_fb")

        # ---------------- loads. Landed: metadata first, then everything else. Default: the proj loads (no metadata
        # needed) first, then the metadata, then its dependents (old conv window, start state), so the proj and
        # metadata latencies overlap and f_b runs while the dependent loads are in flight.
        def proj_loads():
            xs_ = [[ldg_u16(proj + ((row0 + t) * COLS + Int64(x)) * 2) for t in range(NT)] for x in ch]
            xv_ = [ldg_u16(proj + ((row0 + t) * COLS + Int64(chv)) * 2) for t in range(NT)]
            og_ = [bf(ldg_u16(proj + ((row0 + Int64(e // VR % NT)) * COLS + C_OG + h * D + vs * VR + e % VR) * 2))
                   for e in ne]
            braw_ = bf(ldg_u16(proj + ((row0 + Int64(tidx % NT)) * COLS + C_B + h) * 2))
            return xs_, xv_, og_, braw_

        if cutlass.const_expr(not OLD_ORDER):
            xs, xv, og, braw = proj_loads()
        acc_n = Int32(ldg_b32(accepted + Int64(n) * 4))
        irow = spec_idx + Int64(Int32(n) * Int32(idx_stride)) * 4  # 32-bit row offset: 80 registers, not 102
        idx_t = [Int32(ldg_b32(irow + Int64(t * 4))) for t in range(NT)]
        if cutlass.const_expr(not LANDED):
            exp_a = cute.math.exp(a_log_h, fastmath=True)
        o = acc_n - 1
        idx_c = idx_t[0]
        idx_s = idx_t[0]
        for t in cutlass.range_constexpr(1, NT):
            idx_s = cutlass.select_(o == t, idx_t[t], idx_s)
        gq = tidx % G
        rbase = vs * VR + (tidx // G) * R
        soff = [[(Int64(h * D + rbase + i) * D + gq * 4 + 4 * G * j) * 4 for j in range(NJ)] for i in range(R)]
        sbase = ssm + Int64(cutlass.select_(idx_s > 0, idx_s, Int32(0))) * ssm_stride * 4
        def old_window(x):
            return [cutlass.select_(idx_c > 0, ldg_u16(cbase + (Int64(o + j) * (3 * P) + Int64(x)) * 2), Int32(0))
                    for j in range(WIDTH - 1)]

        if cutlass.const_expr(OLD_ORDER):
            st0 = [[ldg_v4(sbase + soff[i][j]) for j in range(NJ)] for i in range(R)]
            cbase = conv + Int64(cutlass.select_(idx_c > 0, idx_c, Int32(0))) * conv_stride * 2  # slot 0: null block
            raw = [old_window(x) + [ldg_u16(proj + ((row0 + t) * COLS + Int64(x)) * 2) for t in range(NT)]
                   for x in ch]
            rawv = old_window(chv) + [ldg_u16(proj + ((row0 + t) * COLS + Int64(chv)) * 2) for t in range(NT)]
            og = [bf(ldg_u16(proj + ((row0 + Int64(e // VR % NT)) * COLS + C_OG + h * D + vs * VR + e % VR) * 2))
                  for e in ne]
            braw = bf(ldg_u16(proj + ((row0 + Int64(tidx % NT)) * COLS + C_B + h) * 2))
        else:
            cbase = conv + Int64(cutlass.select_(idx_c > 0, idx_c, Int32(0))) * conv_stride * 2
            raw = [old_window(x) + xs[i] for i, x in enumerate(ch)]  # the conv needs these first
            rawv = old_window(chv) + xv
            st0 = [[ldg_v4(sbase + soff[i][j]) for j in range(NJ)] for i in range(R)]  # needed last

        # ---------------- f_b on mma.sync, fp32: warp w computes channels 16 m .. 16 m + 15, m = w + W u
        prow = proj + (row0 + Int64(grp)) * (COLS * 2) + (C_FA + tig * 2) * 2
        bfr = [(ldg_b32(prow + ks * 32), ldg_b32(prow + ks * 32 + 16)) for ks in range(D // 16)]
        for u in cutlass.range_constexpr(UNITS):
            mt = warp + W * u
            afr = afr0 if UNITS == 1 else a_frags(mt)
            g = [Float32(0.0)] * 4
            for ks in cutlass.range_constexpr(D // 16):
                a_ = afr[ks]
                if cutlass.const_expr(ks == 0):
                    g = mma_bf16(a_[0], a_[1], a_[2], a_[3], bfr[ks][0], bfr[ks][1])
                else:
                    g = mma_acc(g, a_[0], a_[1], a_[2], a_[3], bfr[ks][0], bfr[ks][1])
            for j in cutlass.range_constexpr(4):
                sts(sa(S_G, (tig * 2 + j % 2) * D + mt * 16 + grp + (j // 2) * 8), g[j])

        pop()
        push("conv")
        # ---------------- conv4 + SiLU: thread = q/k channels over the 8 tokens; tidx < VR: one v channel
        for i in cutlass.range_constexpr(CPT):
            s_qk = cutlass.select_(cq[i] < D, Int32(S_Q), Int32(S_K)) + cq[i] % D
            for t in cutlass.range_constexpr(NT):
                y = bf(raw[i][t]) * wc[i][0]
                for j in cutlass.range_constexpr(1, WIDTH):
                    y = y + bf(raw[i][t + j]) * wc[i][j]
                sts(sb + Int32(4) * (s_qk + t * D), y * sigmoid(y))
            for j in cutlass.range_constexpr(WIDTH - 1 + NT - 1):  # the new q/k conv window, for CTA 0's rewrite
                sts_i(sa(S_CW, j * 2 * D + cq[i]), raw[i][j + 1])
        if tidx < VR:
            for t in cutlass.range_constexpr(NT):
                y = bf(rawv[t]) * wv[0]
                for j in cutlass.range_constexpr(1, WIDTH):
                    y = y + bf(rawv[t + j]) * wv[j]
                sts(sa(S_V, t * VR + tidx), y * sigmoid(y))
            if idx_c > 0:  # this CTA alone reads and writes its v channels' conv state
                for j in cutlass.range_constexpr(WIDTH - 1 + NT - 1):
                    stg_u16(cbase + (Int64(j) * (3 * P) + Int64(chv)) * 2, rawv[j + 1])
        for i in cutlass.range_constexpr(NO):
            if ne[i] < NT * VR:
                sts(sa(S_OG, ne[i]), sigmoid(og[i]))
        if tidx < NT:
            sts(sa(S_BETA, tidx), sigmoid(braw))
        cute.arch.barrier()
        pop()
        push("qk_norm")

        # ---------------- warp w, tokens w + W u: q, k L2 norm; decay = exp(gate), decay * k, beta * k
        for u in cutlass.range_constexpr(NT // W):
            tk = warp + W * u
            q4 = lds_v4(sa(S_Q, tk * D + lane * 4))
            k4 = lds_v4(sa(S_K, tk * D + lane * 4))
            g4 = lds_v4(sa(S_G, tk * D + lane * 4))
            beta = lds(sa(S_BETA, tk))
            sq = q4[0] * q4[0] + q4[1] * q4[1] + q4[2] * q4[2] + q4[3] * q4[3]
            sk = k4[0] * k4[0] + k4[1] * k4[1] + k4[2] * k4[2] + k4[3] * k4[3]
            rq = cute.math.rsqrt(bfly(sq) + Float32(1e-6), fastmath=True) * Float32(SCALE)
            rk = cute.math.rsqrt(bfly(sk) + Float32(1e-6), fastmath=True)
            dec = [cute.math.exp(Float32(lower_bound) * sigmoid(exp_a * (g4[j] + dtb[j])), fastmath=True)
                   for j in range(4)]
            kn = [k4[j] * rk for j in range(4)]
            sts_v4(sa(S_Q, tk * D + lane * 4), q4[0] * rq, q4[1] * rq, q4[2] * rq, q4[3] * rq)
            sts_v4(sa(S_DEC, tk * D + lane * 4), dec[0], dec[1], dec[2], dec[3])
            sts_v4(sa(S_KD, tk * D + lane * 4), dec[0] * kn[0], dec[1] * kn[1], dec[2] * kn[2], dec[3] * kn[3])
            sts_v4(sa(S_BK, tk * D + lane * 4), beta * kn[0], beta * kn[1], beta * kn[2], beta * kn[3])
        cute.arch.barrier()
        pop()

        # ---------------- recurrence: R rows x 4 NJ keys per lane; a row's dot products reduce over its G lanes
        offs = tuple(G >> s for s in range(1, G.bit_length()))
        S = [[[cutlass.select_(idx_s > 0, x, Float32(0.0)) for x in st0[i][j]] for j in range(NJ)] for i in range(R)]
        if cutlass.const_expr(IKET):  # consume the state registers here, so their remaining load latency shows
            push("state_wait")
            acc0 = Float32(0.0)
            for i in cutlass.range_constexpr(R):
                for j in cutlass.range_constexpr(NJ):
                    for e in cutlass.range_constexpr(4):
                        acc0 = acc0 + S[i][j][e]
            sts(sa(S_BETA + 8, 0), acc0)  # scratch word (S_BETA holds 8 betas in 16 words)
            pop()
        for t in cutlass.range_constexpr(NT):
            if cutlass.const_expr(t == 0):
                push("rec_t0")
            if cutlass.const_expr(t == 1):
                push("rec_t1_7")
            ko = [t * D + gq * 4 + 4 * G * j for j in range(NJ)]
            kd = [lds_v4(sa(S_KD, x)) for x in ko]
            dc = [lds_v4(sa(S_DEC, x)) for x in ko]
            bk = [lds_v4(sa(S_BK, x)) for x in ko]
            qq = [lds_v4(sa(S_Q, x)) for x in ko]
            parts = []
            for i in cutlass.range_constexpr(R):
                dot = Float32(0.0)
                for j in cutlass.range_constexpr(NJ):
                    for e in cutlass.range_constexpr(4):
                        dot = dot + S[i][j][e] * kd[j][e]
                vn = lds(sa(S_V, t * VR + rbase - vs * VR + i)) - bfly(dot, offs)
                S[i] = [[dc[j][e] * S[i][j][e] + vn * bk[j][e] for e in range(4)] for j in range(NJ)]
                od = Float32(0.0)
                for j in cutlass.range_constexpr(NJ):
                    for e in cutlass.range_constexpr(4):
                        od = od + S[i][j][e] * qq[j][e]
                parts.append(od)
                if cutlass.const_expr(STORE):
                    if idx_s > 0:
                        if idx_t[t] > 0:
                            dst = ssm + Int64(idx_t[t]) * ssm_stride * 4
                            for j in cutlass.range_constexpr(NJ):
                                stg_v4(dst + soff[i][j], S[i][j][0], S[i][j][1], S[i][j][2], S[i][j][3])
            parts = [bfly(x, offs) for x in parts]
            if gq == 0:
                for i in cutlass.range_constexpr(R):
                    sts(sa(S_O, t * VR + rbase - vs * VR + i), parts[i])
            if cutlass.const_expr(t == 0 or t == NT - 1):
                pop()
        push("rec_barrier")
        cute.arch.barrier()
        pop()
        if cutlass.const_expr(not LANDED):
            push("cluster_wait")
            cluster_wait()  # the peers' mailboxes are initialized before this CTA's first st.async
            pop()
        push("norm_send")

        # ---------------- gated RMSNorm over V: per-token partial sums of squares to every peer's mailbox
        xo = [lds(sa(S_O, e % (NT * VR))) for e in ne]
        for i in cutlass.range_constexpr(NO):
            ss = bfly(xo[i] * xo[i], tuple(VR >> s for s in range(1, VR.bit_length())))
            if ne[i] < NT * VR:
                if ne[i] % VR == 0:
                    for r in cutlass.range_constexpr(VS):
                        st_async_peer(sa(S_SS, vs * NT + ne[i] // VR), ss, mb, Int32(r))
        pop()
        push("mbar_wait")
        while not mbar_try_wait_cluster(mb):
            pass
        pop()
        push("out_store")
        for i in cutlass.range_constexpr(NO):
            if ne[i] < NT * VR:
                t_o = ne[i] // VR
                tot = Float32(0.0)
                for r in cutlass.range_constexpr(VS):
                    tot = tot + lds(sa(S_SS, r * NT + t_o))
                y = xo[i] * cute.math.rsqrt(tot * Float32(1.0 / D) + Float32(EPS), fastmath=True) * wn[i]
                y = cutlass.select_(idx_s > 0, y * lds(sa(S_OG, ne[i])), Float32(0.0))
                stg_bf16(out + ((row0 + Int64(t_o)) * P + h * D + vs * VR + ne[i] % VR) * 2, y)

        # ---------------- q/k conv state: CTA 0 of the cluster, after every peer's mailbox send (all read it)
        if vs == 0:
            if idx_c > 0:
                for i in cutlass.range_constexpr(CPT):
                    for j in cutlass.range_constexpr(WIDTH - 1 + NT - 1):
                        stg_u16(cbase + (Int64(j) * (3 * P) + Int64(ch[i])) * 2, lds_i(sa(S_CW, j * 2 * D + cq[i])))
        pop()
        pop()

    @cute.jit
    def launch(proj: Int64, out: Int64, w_fb: Int64, w_conv: Int64, a_log: Int64, dt_bias: Int64, w_norm: Int64,
               conv: Int64, conv_stride: Int64, ssm: Int64, ssm_stride: Int64, spec_idx: Int64, idx_stride: Int64,
               accepted: Int64, n: Int32, stream):
        kernel(proj, out, w_fb, w_conv, a_log, dt_bias, w_norm, conv, conv_stride, ssm, ssm_stride, spec_idx,
               idx_stride, accepted).launch(grid=[VS, H, n], block=[THREADS, 1, 1], cluster=[VS, 1, 1],
                                            stream=stream, use_pdl=PDL, **LAUNCH)

    return launch


def compiled(device: torch.device, lower_bound: float, cfg: tuple | None = None, variant: str = "full"):
    cfg = tuple(cfg or CFG)
    key = (device.index, float(lower_bound), tuple(cfg), variant)
    fn = _COMPILED.get(key)
    if fn is not None:
        return fn
    with _LOCK:
        if key not in _COMPILED:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("kda_stage.core: first call inside CUDA-graph capture; call it eagerly first")
            import cutlass
            import cutlass.cute as cute
            from cuda.bindings.driver import CUstream

            i64 = cutlass.Int64(0)
            stream = CUstream(torch.cuda.current_stream(device).cuda_stream)
            _COMPILED[key] = cute.compile(_build(float(lower_bound), tuple(cfg), variant), *([i64] * 14),
                                          cutlass.Int32(1), stream)
    return _COMPILED[key]


def check(lw, st, proj, out):
    """None when core() serves these tensors; else the reason it does not."""
    t = st.t
    if st.n * NT != t or not (1 <= st.n <= 64):
        return f"N={st.n} T={t}"
    if proj.dtype != torch.bfloat16 or tuple(proj.shape) != (t, COLS) or not proj.is_contiguous():
        return f"proj {proj.dtype} {tuple(proj.shape)}"
    if out.dtype != torch.bfloat16 or out.numel() != t * P or not out.is_contiguous():
        return f"out {out.dtype} {tuple(out.shape)}"
    if st.conv.dtype != torch.bfloat16 or st.conv.stride()[1:] != (3 * P, 1) or st.conv.shape[1:] != (10, 3 * P):
        return f"conv {st.conv.dtype} {tuple(st.conv.shape)} {st.conv.stride()}"
    if st.ssm.dtype != torch.float32 or st.ssm.stride()[1:] != (D * D, D, 1) or st.ssm.shape[1:] != (H, D, D):
        return f"ssm {st.ssm.dtype} {tuple(st.ssm.shape)} {st.ssm.stride()}"
    si = st.spec_idx
    if si.dtype != torch.int32 or tuple(si.shape) != (st.n, NT) or si.stride(1) != 1 or si.stride(0) < NT:
        return f"spec_idx {si.dtype} {tuple(si.shape)} {si.stride()}"
    ac = st.accepted
    if ac.dtype != torch.int32 or tuple(ac.shape) != (st.n,) or ac.stride(0) != 1:
        return f"accepted {ac.dtype} {tuple(ac.shape)} {ac.stride()}"
    return None


def core(lw, st, proj, out, cfg: tuple | None = None, variant: str = "full"):
    """One launch of the fused core for st.n requests of 8 tokens (see the module docstring). cfg (None = CFG) and
    variant "nostore" (no state stores) / "nopdl" (no PDL) exist only for the selftest's sweep and ablation."""
    import cutlass
    from cuda.bindings.driver import CUstream

    dev = proj.device
    fn = compiled(dev, lw.lower_bound, cfg, variant)
    a = lambda x: cutlass.Int64(x.data_ptr())  # noqa: E731
    fn(a(proj), a(out), a(lw.w_fb), a(lw.w_conv), a(lw.a_log), a(lw.dt_bias), a(lw.w_norm), a(st.conv),
       cutlass.Int64(st.conv.stride(0)), a(st.ssm), cutlass.Int64(st.ssm.stride(0)), a(st.spec_idx),
       cutlass.Int64(st.spec_idx.stride(0)), a(st.accepted),
       cutlass.Int32(st.n), CUstream(torch.cuda.current_stream(dev).cuda_stream))
    return out
