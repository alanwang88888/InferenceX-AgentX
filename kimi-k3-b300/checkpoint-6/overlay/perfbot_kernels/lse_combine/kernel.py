# SPDX-License-Identifier: Apache-2.0
"""DraftValue: the K3 DSpark draft layer's DCP tail in one PDL launch (item 15 F-B, kernel 3).

Replaces the served launches between the tokenspeed reduce and o_proj (mla.py:713-727; the served op is
dcp_direct_a2a_lse_reduce.cu): increment_epoch, dispatch_output_lse, signal, wait_lse_combine and the W_UV bmm.

Served arithmetic (read in wait_lse_combine_kernel<bf16>, SASS of _C_stable_libtorch sm_100 cubin 195):
  per (token, head), sources s = 0..7 in rank order:
    v_s = NaN or +inf -> -inf; m = max_s v_s (FMNMX); m = -inf -> 0
    e_s = ex2.approx.f32(v_s - m)          (non-ftz exp2f: x < -126 -> EX2(x / 2)^2)
    sum = (((0 + e_0) + e_1) + ...) + e_7  (FADD, rank order)
    inv = sum > 0 ? rcp.rn(sum) : 0         (MUFU.RCP + Newton + slow path = IEEE RN)
    w_s = inv * e_s                         (FMUL)
    acc = 0; for s: if w_s != 0: acc = fma(w_s, x_s, acc)   (FFMA, rank order, zero weights skipped)
    latent = bf16 RN(acc)
  W_UV: torch.bmm(latent [8, M, 512], W_UV [8, 512, 128]) -> [M, 8 x 128] bf16 (cuBLAS, FP32 accumulation).
The merge below keeps that order bit for bit. W_UV here is BF16 MMA with FP32 accumulation, split over 4 K-quarters
(one per cluster CTA) and summed in quarter order over DSMEM: a different order than cuBLAS (judge: fp64 envelope).

Grid (4 K-quarters, 8 heads, ranks x pushers), cluster 4: CTA (q, h, z) with z % pushers == 0 is a reader. Every CTA
pushes a share of this rank's tokenspeed rows [M, 64, 512] + LSE [M, 64] to the head owner's mailbox slot
[parity, rank, token, head % 8]. A reader then merges head h, latent columns 128 q .. +127 of every token from the
8 sources into shared memory, multiplies by W_UV rows 128 q .. +127 (all 128 outputs) and the cluster sums the four
FP32 partials.

Exchange (constexpr ``lamport``):
  False  the served epoch protocol on the served DirectDCPA2AWorkspace: E = epoch + 1 after the grid-dependency wait;
         parity E & 1; empty shards send LSE -inf and no payload; every CTA counts itself in ``completion``
         (bar.sync, then atom.acq_rel.gpu); the last count resets it, stores epoch = E and st.release.sys-es E into
         signal[E & 1][rank] of every rank (the one sys fence, kernel 2's release chain); readers wait
         ld.acquire.sys signal[E & 1][s] == E for every source s.
  True   Lamport mailboxes (mla.dcp.DcpBox form): relaxed.sys stores, payload of empty shards = zeros, LSE NaN / +inf
         / empty -> -inf at the source (no word is a sentinel); the reader polls each 16 B / LSE word until it is not
         the sentinel and re-arms what it read; this kernel also re-arms its parity's q mailbox rows (kernel 2's
         Lamport publish: tokenspeed has read them). ``parity`` alternates by layer.
"""

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint32, const_expr

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.primitives import (
    add_rn,
    alloc,
    asm,
    at,
    bf16x2,
    bits_float,
    early_weights,
    float_bits,
    fma,
    launch,
    load_volatile,
    load_words,
    mma_into,
    no_neg_zero,
    published,
    store_relaxed,
    store_shared_words,
    store_words,
    unpack_bf16,
    zeros,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.mla.dcp import LSE_SENT, PAIR_SENT, Q_SENT

from perfbot_kernels.draft_query.kernel import (
    _iket,
    atomic_add_acq_rel_gpu,
    load_acquire_sys,
    load_epoch,
    mul_rn,
    store_release_sys,
    store_u32_relaxed,
    store_u64_relaxed,
)

HEADS, WORLD, LATENT, VDIM, QUARTERS = 8, 8, 512, 128, 4
KQ = LATENT // QUARTERS  # latent columns per CTA
LS, PS = KQ + 32, VDIM + 8  # smem row pitch (elements): latent bf16 (conflict-free 16 B rows), partial fp32
NEG_INF = -float("inf")
MUTANTS = ("", "rankorder", "nomax", "skipzero", "norearm", "noqrearm", "noepoch")
ABLATIONS = ("x_empty", "x_nopush", "x_nomerge")  # timing-only cuts (wrong bytes; never gated)


@asm(Float32, pure=True)
def ex2(x):
    """exp2f without fast math (the served SASS: EX2 with the x < -126 half-and-square path)."""
    return "ex2.approx.f32 $0,$1;", "=f,f", (Float32(x),)


@asm(Float32, pure=True)
def rcp_rn(x):
    return "rcp.rn.f32 $0,$1;", "=f,f", (Float32(x),)


@asm(Float32, pure=True)
def sub_rn(a, b):
    return "sub.rn.f32 $0,$1,$2;", "=f,f,f", (Float32(a), Float32(b))


@asm((Float32,) * 4)
def load_cluster_f32x4(pointer, peer):
    """16 B of CTA ``peer``'s shared memory at this CTA's offset ``pointer`` (DSMEM)."""
    code = "{ .reg .b32 a; mapa.shared::cluster.u32 a,$4,$5; ld.shared::cluster.v4.f32 {$0,$1,$2,$3},[a]; }"
    return code, "=f,=f,=f,=f,r,r,~{memory}", (pointer, Int32(peer))


class DraftValue:
    def __init__(self, batch, ranks=1, copies=1, lamport=False, pushers=4, mutant="", dump=False, iket=False):
        assert 1 <= batch <= 64 and mutant in MUTANTS + ABLATIONS
        self.batch, self.ranks, self.copies, self.lamport = batch, ranks, copies, lamport
        self.T = 16 if batch <= 16 else 32 if batch <= 32 else 64
        self.MT = self.T // 16
        self.pushers = pushers
        self.grid = (QUARTERS, HEADS, ranks * pushers)
        self.ctas = QUARTERS * HEADS * pushers  # per rank
        self.mutant, self.dump, self.iket = mutant, dump, iket and _iket is not None

    def _push(self, name):
        if const_expr(self.iket):
            _iket.range_push(name)

    def _pop(self):
        if const_expr(self.iket):
            _iket.range_pop()

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, self.grid, 256, cluster=QUARTERS)

    @cute.kernel
    def kernel(self, args):
        if const_expr(self.mutant == "x_empty"):
            cute.arch.griddepcontrol_launch_dependents()
            cute.arch.griddepcontrol_wait()
        else:
            self.body(args)

    @cute.jit
    def body(self, args):
        seq_lens, qsl, w_uv, epoch = args[2], args[3], args[4], args[13]
        rank_arg, parity_arg = args[16], args[17]
        tid, _, _ = cute.arch.thread_idx()
        q4, head, z = cute.arch.block_idx()
        e = z // self.pushers  # emulated rank (0 when real)
        pusher = z % self.pushers
        rank = rank_arg + e
        lane, warp = tid % 32, tid // 32
        group, quarter = lane // 4, lane % 4
        cute.arch.griddepcontrol_launch_dependents()
        smem = cutlass.utils.SmemAllocator()
        lat = alloc(smem, BFloat16, (self.T, LS), 128)
        part = alloc(smem, Float32, (self.T, PS), 16)
        scales = alloc(smem, Float32, (self.T, WORLD))
        empty = alloc(smem, Int32, 64)
        self._push("fill_weights")
        # step metadata before the grid-dependency wait (written at step start, as kernel 2's positions / slots)
        if tid < self.batch:
            lo, hi = Int32(0), Int32(qsl.shape[0] - 1)
            while lo < hi:  # the served find_sequence (dcp_direct_a2a_lse_reduce.cu:25-39)
                mid = (lo + hi) // 2
                if qsl[mid] <= tid:
                    lo = mid + 1
                else:
                    hi = mid
            flag = Int32(0)
            if seq_lens[e, lo - 1] == 0:
                flag = Int32(1)
            empty[tid] = flag
        if pusher == 0:
            # W_UV B fragments of K-quarter q4: warp w owns outputs 16 w .. +15 (two n-tiles), 8 latent values per
            # 32-wide chunk (A uses the same permutation of k)
            wb = [
                [early_weights(at(w_uv, e * HEADS + head, KQ * q4 + 32 * c + 8 * quarter, 16 * warp + 8 * j + group),
                               count=4) for c in range(4)]
                for j in range(2)
            ]
            self._pop()
            self._push("sync_griddep")
            cute.arch.griddepcontrol_wait()
            self._pop()
            e0 = load_epoch(epoch.iterator + e)
            parity = parity_arg if const_expr(self.lamport) else Int32((e0 + 1) & 1)
            cute.arch.sync_threads()  # empty[] is visible
            self.push(args, empty, e, rank, parity, e0, tid)
            self.merge(args, lat, part, scales, wb, e, rank, parity, e0, tid)
        else:
            self._pop()
            self._push("sync_griddep")
            cute.arch.griddepcontrol_wait()
            self._pop()
            e0 = load_epoch(epoch.iterator + e)
            parity = parity_arg if const_expr(self.lamport) else Int32((e0 + 1) & 1)
            cute.arch.sync_threads()
            self.push(args, empty, e, rank, parity, e0, tid)

    @cute.jit
    def push(self, args, empty, e, rank, parity, e0, tid):
        """This CTA's share of the rank's rows / LSE -> the owner's slot; Lamport: q re-arm; epoch: count + signal."""
        rows, rows_lse = args[0], args[1]
        peers, lpeers, speers, epoch, completion, qbox = args[9], args[10], args[12], args[13], args[14], args[15]
        q4, head, z = cute.arch.block_idx()
        mine = e if const_expr(self.copies > 1) else 0
        cap = args[7].shape[3]
        threads: cutlass.Constexpr = self.ctas * 256
        me = (((z % self.pushers) * HEADS + head) * QUARTERS + q4) * 256 + tid
        self._push("main_push")
        # 16 B chunks i = (token b, gathered head gh, chunk k); up to 8 loads in flight per thread. Loads past the end
        # re-read the last chunk; their stores are skipped (a duplicate Lamport store could land after the re-arm).
        total: cutlass.Constexpr = 0 if self.mutant == "x_nopush" else self.batch * 64 * 64
        rounds: cutlass.Constexpr = -(-total // threads)
        for i0 in cutlass.range_constexpr(0, rounds, 8):
            n: cutlass.Constexpr = min(8, rounds - i0)
            index = [me + (i0 + j) * threads for j in range(n)]
            chunk = [min(i, total - 1) for i in index]
            loads = [load_words(at(rows, e, c // 4096, (c % 4096) // 64, 8 * (c % 64)), count=4, mutable=True)
                     for c in chunk]
            for j in cutlass.range_constexpr(n):
                if index[j] < total:
                    c = chunk[j]
                    b, r = c // 4096, c % 4096
                    gh, k = r // 64, r % 64
                    live = empty[b] == 0
                    slot = ((parity * WORLD + rank) * cap + b) * HEADS + gh % HEADS
                    address = peers[e, gh // HEADS] + (Int64(slot) * LATENT + 8 * k) * 2
                    if const_expr(self.lamport):
                        words = tuple(no_neg_zero(w) for w in loads[j])
                        if empty[b] != 0:
                            words = (Uint32(0),) * 4
                        store_relaxed(address, words)
                    elif live:
                        store_words(address, loads[j])
        for i in range(me, self.batch * 64, threads):
            b, gh = i // 64, i % 64
            value = rows_lse[e, b, gh]
            if empty[b] != 0:
                value = Float32(NEG_INF)
            if const_expr(self.lamport):  # the reader's NaN / +inf rule applied at the source: never the sentinel
                if (value != value) | (value == Float32(float("inf"))):
                    value = Float32(NEG_INF)
            slot = ((parity * WORLD + rank) * cap + b) * HEADS + gh % HEADS
            address = lpeers[e, gh // HEADS] + Int64(slot) * 4
            if const_expr(self.lamport):
                store_relaxed(address, (float_bits(value),))
            else:
                store_words(address, (float_bits(value),))
        if const_expr(self.lamport and self.mutant != "noqrearm"):
            for c in range(me, self.batch * 64 * 36, threads):  # q mailbox rows of this parity: tokenspeed read them
                store_words(at(qbox, mine, parity, c * 16), (Uint32(Q_SENT),) * 4)
        self._pop()
        if const_expr(not self.lamport):
            self._push("sync_signal")
            cute.arch.sync_threads()  # every thread's payload stores precede thread 0's count (CTA-scope bar)
            if tid == 0:
                new = Uint32(e0 + 1)
                done = atomic_add_acq_rel_gpu(completion.iterator + e, 1)
                if done + 1 == Uint32(self.ctas):
                    store_u32_relaxed(completion.iterator + e, 0)
                    if const_expr(self.mutant != "noepoch"):
                        store_u64_relaxed(epoch.iterator + e, e0 + 1)
                    for d in cutlass.range_constexpr(WORLD):
                        store_release_sys(speers[e, d] + Int64(((new & 1) * WORLD + rank) * 4), new)
            self._pop()

    @cute.jit
    def accumulate(self, acc, w, words):
        """acc = fma(w, x, acc) over the 8 bf16 of ``words``; skipped when w == 0 (served), unless mutant skipzero."""
        if const_expr(self.mutant == "skipzero"):
            for k in cutlass.range_constexpr(4):
                lo, hi = unpack_bf16(words[k])
                acc[2 * k] = fma(w, lo, acc[2 * k])
                acc[2 * k + 1] = fma(w, hi, acc[2 * k + 1])
        elif w != Float32(0):
            for k in cutlass.range_constexpr(4):
                lo, hi = unpack_bf16(words[k])
                acc[2 * k] = fma(w, lo, acc[2 * k])
                acc[2 * k + 1] = fma(w, hi, acc[2 * k + 1])

    @cute.jit
    def merge(self, args, lat, part, scales, wb, e, rank, parity, e0, tid):
        out, dump, box, lbox, signal = args[5], args[6], args[7], args[8], args[11]
        q4, head, _ = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
        group, quarter = lane // 4, lane % 4
        mine = e if const_expr(self.copies > 1) else 0
        B, T, MT = self.batch, self.T, self.MT
        if const_expr(not self.lamport):
            self._push("sync_exchange")
            if tid < WORLD:
                new = Uint32(e0 + 1)
                source = at(signal, mine, Int32(new & 1), tid)
                got = load_acquire_sys(source)
                while got != new:
                    got = load_acquire_sys(source)
            cute.arch.sync_threads()
            self._pop()
        # ---- weights: warp w owns tokens w, w + 8, ..; lane s < 8 holds source s
        self._push("main_weights")
        for t in range(warp, T, 8):
            scale = Float32(0)
            if t < B:
                v = Float32(NEG_INF)
                if lane < WORLD:
                    source = at(lbox, mine, parity, lane, t, head)
                    if const_expr(self.lamport):
                        word = load_volatile(source)[0]
                        while word == Uint32(LSE_SENT):
                            word = load_volatile(source)[0]
                    else:
                        word = load_words(source, count=1, mutable=True)[0]
                    v = bits_float(word)
                    if (v != v) | (v == Float32(float("inf"))):
                        v = Float32(NEG_INF)
                top = v
                for shift in cutlass.range_constexpr(4, -1, -1):
                    top = max(top, cute.arch.shuffle_sync_bfly(top, 1 << shift))
                if top == Float32(NEG_INF):
                    top = Float32(0)
                if const_expr(self.mutant == "nomax"):
                    top = Float32(0)
                x = ex2(sub_rn(v, top))
                total = Float32(0)
                for s in cutlass.range_constexpr(WORLD):
                    total = add_rn(total, cute.arch.shuffle_sync(x, s))
                inv = Float32(0)
                if total > Float32(0):
                    inv = rcp_rn(total)
                scale = mul_rn(inv, x)
            if lane < WORLD:
                scales[t, lane] = scale
        cute.arch.sync_threads()
        self._pop()
        # ---- merge: thread -> (token r, 16 B chunk c) of this quarter; 8 sources in rank order
        self._push("main_merge")
        order = tuple(range(WORLD - 1, -1, -1)) if self.mutant == "rankorder" else tuple(range(WORLD))
        for it in cutlass.range_constexpr(T // 16):
            i = tid + 256 * it
            r, c = i // 16, i % 16
            col = KQ * q4 + 8 * c
            acc = zeros(8, Float32)
            if const_expr(self.mutant != "x_nomerge"):
                if r < B:
                    w = [scales[r, s] for s in range(WORLD)]
                    if const_expr(self.lamport):
                        sources = [at(box, mine, parity, s, r, head, col) for s in range(WORLD)]
                        loads = [load_volatile(src, count=4) for src in sources]
                        got = []
                        for s in cutlass.range_constexpr(WORLD):
                            words = loads[s]
                            while not published(words):
                                words = load_volatile(sources[s], count=4)
                            if const_expr(self.mutant != "norearm"):
                                store_words(sources[s], (Uint32(PAIR_SENT),) * 4)
                            got.append(words)
                        for si in cutlass.range_constexpr(WORLD):
                            self.accumulate(acc, w[order[si]], got[order[si]])
                    else:
                        for si in cutlass.range_constexpr(WORLD):
                            s = order[si]
                            if const_expr(self.mutant == "skipzero"):
                                self.accumulate(acc, w[s], load_words(at(box, mine, parity, s, r, head, col), count=4,
                                                                     mutable=True))
                            elif w[s] != Float32(0):  # zero-weight sources are never read (stale payloads)
                                self.accumulate(acc, w[s], load_words(at(box, mine, parity, s, r, head, col), count=4,
                                                                     mutable=True))
            pairs = tuple(bf16x2(acc[2 * k], acc[2 * k + 1]) for k in range(4))
            store_shared_words(lat.iterator + r * LS + 8 * c, pairs)
            if const_expr(self.dump):
                if r < B:
                    store_words(at(dump, e, r, head, col), pairs)
        cute.arch.sync_threads()
        self._pop()
        # ---- W_UV on this K-quarter: BF16 MMA, FP32 accumulation
        self._push("main_wuv")
        acc_o = zeros((MT * 2, 4), Float32)
        for c in cutlass.range_constexpr(4):
            for m in cutlass.range_constexpr(MT):
                x, y = (
                    load_words(lat.iterator + rr * LS + 32 * c + 8 * quarter, count=4, shared=True)
                    for rr in (16 * m + group, 16 * m + group + 8)
                )
                for j in cutlass.range_constexpr(2):
                    for s in cutlass.range_constexpr(2):
                        a = (x[2 * s], y[2 * s], x[2 * s + 1], y[2 * s + 1])
                        bb = (wb[j][c][2 * s], wb[j][c][2 * s + 1])
                        mma_into(acc_o[m * 2 + j, None], a, bb, bf16=True)
        for m in cutlass.range_constexpr(MT):
            for j in cutlass.range_constexpr(2):
                for h in cutlass.range_constexpr(2):
                    row = 16 * m + group + 8 * h
                    cc = 16 * warp + 8 * j + 2 * quarter
                    part[row, cc] = acc_o[m * 2 + j, 2 * h]
                    part[row, cc + 1] = acc_o[m * 2 + j, 2 * h + 1]
        self._pop()
        self._push("sync_cluster")
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        self._pop()
        # ---- quarter sum over DSMEM (quarter order 0..3), outputs 32 q4 .. +31 of this head
        self._push("main_store")
        for i in range(tid, B * 8, 256):
            t, c4 = i // 8, i % 8
            cc = 32 * q4 + 4 * c4
            pointer = part.iterator + t * PS + cc
            total = list(load_cluster_f32x4(pointer, 0))
            for p in cutlass.range_constexpr(1, QUARTERS):
                more = load_cluster_f32x4(pointer, p)
                total = [add_rn(total[k], more[k]) for k in range(4)]
            store_words(at(out, e, t, head * VDIM + cc), (bf16x2(total[0], total[1]), bf16x2(total[2], total[3])))
        cute.arch.cluster_arrive()  # peers keep their shared memory until every quarter has read it
        cute.arch.cluster_wait()
        if const_expr(self.lamport and self.mutant != "norearm"):
            if q4 == 0:  # every quarter has read the LSE words
                for i in range(tid, B * WORLD, 256):
                    store_words(at(lbox, mine, parity, i % WORLD, i // WORLD, head), (Uint32(LSE_SENT),))
        self._pop()
