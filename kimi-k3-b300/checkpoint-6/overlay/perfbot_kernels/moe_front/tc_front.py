"""Item 17 R2: the MoE front GEMM on tcgen05 (swap-AB), same publish as the served WsGemm.

y[r, 560] = x[r, 7168] W[560, 7168]^T as D[560, n] = W x^T: M = weight rows in 9 M64 tiles (rows >= 560 are
TMA zero fill, never published; not the card's 5 x M128: 640 rows pad 80, 576 pad 16, and M64 is the served
SharedUmma's proven shape), N = n = rows padded to 8, K split ``ks`` ways over a cluster of ``ks`` CTAs.
Adapted from the served SharedUmma (moe/shared.py: TMA W boxes, M=64 kind::f16 ``mma_bf16_block``, DSMEM bulk
push to owners, the front ``publish``). Prior art: CUTLASS Blackwell tcgen05 GEMM
(github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/blackwell/dense_gemm.py), DeepGEMM swap-AB small-M
(github.com/deepseek-ai/DeepGEMM), CUTLASS split-K reduction (github.com/NVIDIA/cutlass, include/cutlass/
reduction/).

Per CTA (160 threads; warps 0-3 load / reduce / publish, warp 4 MMA):
  before griddepcontrol_wait: all of its W K blocks (64 rows x 896 K = 112 KB) by TMA into SMEM;
  after: its 14 x K blocks by TMA (n rows; rows >= r are zero fill), MMAs as blocks land, K blocks 2j, 2j+1
  into TMEM accumulator j (7), a pairwise FP32 sum of the 7 per (row, token) (``tree``), one DSMEM bulk copy
  of FP32 partials to each owner (rank o owns tile rows 8 o .. 8 o + 7), the owner's rank-order sum in FP64
  rounded once (``owner64``), then the served publish: FP32 words (router rows < valid) / BF16 (latent rows
  >= split), -0 made +0.
Measured variants (item scratch r2/explore_*, r2/{tree,kahan}): 4 or 14 K blocks per accumulator fail the error
bound (tensor-core accumulation); FP64 readout (``wide``) costs 0.6-2.3 us (F2F converts); Kahan FP32 readout
(``kahan``) passes every judge case but costs the gain; ``tree`` + ``owner64`` keeps the gain and misses the
bound in 1 of 56 cases; lane-parallel x issue (``spread``) and M128 x ks16 (``tile``) gain nothing."""
import functools

import cuda.bindings.driver as cuda
import cutlass as c
import cutlass.cute as cute
import torch
from cutlass import Constexpr, Float32, Float64, Int32, Int64, Uint32
from cutlass.cute.runtime import make_fake_stream

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import fake, tensor_maps
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.primitives import (
    acquire_tensormap,
    alloc,
    commit_asm,
    copy_s2cluster,
    fence_tmem,
    launch,
    load_tma2,
    load_tmem,
    mma_bf16_block,
    named_barrier,
    release_tmem,
    smem_desc,
    thread,
    view,
    wait,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm import ws_gemm as G

from .stamps import stamp_gt

K, ROWS = 7168, 560
# stamps (max over CTAs of %globaltimer, red.max into sb[slot]): R2's critical path after c
TC_SLOTS = ("wait", "w_in", "x_in", "mma", "sent", "recv", "pub")


def front_maps(packed, x, n, tile=64):
    """TMA maps: W [560, 7168] in [64, 64] boxes, x [r, 7168] in [n, 64] boxes (rows >= r zero), 128B swizzle."""
    dtype = cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
    swizzle = cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B
    l2 = cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B
    specs = [(packed.data_ptr(), dtype, (K, packed.shape[0]), (2 * packed.stride(0),), (64, tile), swizzle, l2),
             (x.data_ptr(), dtype, (K, x.shape[0]), (2 * x.stride(0),), (64, n), swizzle, l2)]
    return tensor_maps(specs, packed.device)


class TcFront:
    def __init__(self, batch, split, valid, ks=8, pairs=2, mutant=None, stamps=False, wide=False, spread=False,
                 kahan=False, owner64=True, tree=True, tile=64):
        self.stamps, self.wide, self.spread, self.kahan = stamps, wide, spread, kahan
        self.tree = tree and not kahan and not wide
        self.owner64 = owner64  # FP32 partials, the owner's rank-order sum in FP64 (rounded once)
        self.dt = Float64 if wide else Float32  # partial sums: FP64 (rounded once) or FP32
        self.batch, self.split, self.valid, self.ks = batch, split, valid, ks
        self.n = (batch + 7) // 8 * 8
        self.cs = max(32, 1 << (self.n - 1).bit_length())  # TMEM columns per accumulator
        self.tile = tile  # M: 64 (lanes 0-15 of each warp hold rows) or 128 (every lane)
        self.tiles = -(-ROWS // tile)
        self.grid = self.tiles * ks
        self.kb = K // 64 // ks  # K blocks per CTA
        self.pairs = pairs  # K blocks per accumulator
        self.accs = -(-self.kb // pairs)
        self.columns = max(32, 1 << (self.accs * self.cs - 1).bit_length())
        self.own = tile // ks  # rows per owner
        self.wbytes = tile * 128  # one W K block
        self.share = self.own * self.n * (8 if wide or kahan else 4)  # bytes one rank sends one owner
        self.xbytes = self.n * 128
        # judge teeth (never in a real build): "drop_k" (rank 1's K partial ignored), "skip_n" (the last 8-token
        # N slice published as 0), "column" (router rows published at row ^ 8), "silent" (no publish: with an
        # un-re-armed mailbox the route reads the last step's values)
        self.mutant = mutant

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, self.grid, 160, self.ks, min_blocks_per_mp=1)

    @cute.kernel
    def kernel(self, args):
        maps = args[0]
        tid, warp, lane, cta = thread()
        rank, m = cta % self.ks, cta // self.ks
        smem = c.utils.SmemAllocator()
        wring = alloc(smem, c.Uint8, self.kb * self.wbytes, 1024).iterator
        xring = alloc(smem, c.Uint8, self.kb * self.xbytes, 1024).iterator
        received = alloc(smem, c.Uint8, self.ks * self.share, 16).iterator
        rows = alloc(smem, Float32, self.n * (self.own + 1), 16)
        bars = alloc(smem, c.Uint64, self.kb + 3, 8).iterator  # x[kb], W, done, received
        slot = alloc(smem, Uint32, 1)
        wbar, done, rbar = bars + self.kb, bars + self.kb + 1, bars + self.kb + 2
        if tid == 0:
            for i in c.range_constexpr(self.kb + 3):
                cute.arch.mbarrier_init(bars + i, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.mbarrier_arrive_and_expect_tx(rbar, self.ks * self.share)
            acquire_tensormap(maps.iterator)
            acquire_tensormap(maps.iterator + 128)
            # W does not depend on the predecessor: all of this CTA's blocks before the wait.
            cute.arch.mbarrier_arrive_and_expect_tx(wbar, self.kb * self.wbytes)
            for j in c.range_constexpr(self.kb):
                load_tma2(wring + j * self.wbytes, maps.iterator, wbar, 64 * (rank * self.kb + j), self.tile * m)
        if warp == 4:
            cute.arch.alloc_tmem(self.columns, slot.iterator)
            cute.arch.relinquish_tmem_alloc_permit()
        fence_tmem(before=True)
        cute.arch.sync_threads()
        fence_tmem()
        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()  # every owner's receive barrier is armed
        tmem = slot[0]
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        self.mark(args, "wait", tid == 0)
        if c.const_expr(self.spread):  # lane j of warp 0 issues x block j (the issues overlap)
            if tid < self.kb:
                cute.arch.mbarrier_arrive_and_expect_tx(bars + tid, self.xbytes)
                load_tma2(xring + tid * self.xbytes, maps.iterator + 128, bars + tid, 64 * (rank * self.kb + tid), 0)
        elif tid == 0:
            for j in c.range_constexpr(self.kb):
                cute.arch.mbarrier_arrive_and_expect_tx(bars + j, self.xbytes)
                load_tma2(xring + j * self.xbytes, maps.iterator + 128, bars + j, 64 * (rank * self.kb + j), 0)
        if warp == 4:
            self.multiply(args, wring, xring, bars, tmem)
        else:
            self.reduce(args, wring, received, rows, bars, tmem, rank, m)
        release_tmem(tmem, self.columns, warp, 4)
        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()

    @cute.jit
    def mark(self, args, name, when):
        if c.const_expr(self.stamps):
            if when:
                stamp_gt(args[5], TC_SLOTS.index(name))

    @cute.jit
    def multiply(self, args, wring, xring, bars, tmem):
        """Warp 4: K block j into accumulator j // pairs as its x lands; then commit ``done``."""
        instruction: Constexpr = 1 << 4 | 1 << 7 | 1 << 10 | self.n >> 3 << 17 | self.tile >> 4 << 24
        wait(bars + self.kb, 0, scope="cta")
        self.mark(args, "w_in", cute.arch.lane_idx() == 0)
        for j in c.range_constexpr(self.kb):
            wait(bars + j, 0, scope="cta")
            if c.const_expr(j == self.kb - 1):
                self.mark(args, "x_in", cute.arch.lane_idx() == 0)
            cute.arch.fence_proxy("async.shared", space="cta")
            fence_tmem()
            a = smem_desc(wring + j * self.wbytes, True)
            b = smem_desc(xring + j * self.xbytes, True)
            d = tmem + Uint32(j // self.pairs * self.cs)
            mma_bf16_block(d, a, b, instruction, j % self.pairs, bars, 0)
        with cute.arch.elect_one():
            commit_asm(bars + self.kb + 1)

    @cute.jit
    def reduce(self, args, wring, received, rows, bars, tmem, rank, m):
        """Warps 0-3: the sum of the accumulators, the DSMEM push, the owner's sum and the publish."""
        tid, warp, lane, _ = thread()
        n, own = self.n, self.own
        wait(bars + self.kb + 1, 0, scope="cta")
        fence_tmem()
        self.mark(args, "mma", tid == 0)
        dt, kahan, cell = self.dt, self.kahan, own * n  # cell: one (rank, owner) block of values
        total = cute.make_rmem_tensor(self.cs, dt)
        carry = cute.make_rmem_tensor(self.cs, Float32)  # Kahan: minus the low part of total
        for t in c.range_constexpr(n):
            total[t], carry[t] = dt(0), Float32(0)
        if c.const_expr(self.tree):  # pairwise: ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + a6), 3 roundings deep
            self.tree_sum(tmem, warp, total)
        for j in c.range_constexpr(0 if self.tree else self.accs):
            values = load_tmem(tmem + Uint32(32 * warp << 16) + Uint32(j * self.cs), self.cs)
            for t in c.range_constexpr(n):
                if c.const_expr(kahan):
                    y = values[t] - carry[t]
                    z = total[t] + y
                    carry[t] = (z - total[t]) - y
                    total[t] = z
                else:
                    total[t] = total[t] + dt(values[t])
        # stage [owner][token][row % own] in the drained W ring (8 lanes of an owner: one 32 B run per token;
        # Kahan: the high block, then the low block), then one bulk copy per owner
        halves = 2 if kahan else 1
        staging = view(wring, 0, dt, (self.ks * halves * cell,))
        if lane < self.tile // 4:
            row = self.tile // 4 * warp + lane
            base = row // own * halves * cell + row % own
            for t in c.range_constexpr(n):
                staging[base + t * own] = total[t]
                if c.const_expr(kahan):
                    staging[base + cell + t * own] = -carry[t]
        cute.arch.fence_proxy("async.shared", space="cta")
        named_barrier(1, 128)
        if tid < self.ks:
            copy_s2cluster(received + rank * self.share, wring + tid * self.share, bars + self.kb + 2, tid,
                           size=self.share)
        self.mark(args, "sent", tid == 0)
        wait(bars + self.kb + 2, 0)
        self.mark(args, "recv", tid == 0)
        parts = view(received, 0, dt, (self.ks * halves * cell,))
        for i in c.range_constexpr((cell + 127) // 128):
            item = tid + 128 * i
            if item < cell:
                acc = dt(0)
                if c.const_expr(kahan or self.owner64):
                    acc = Float64(0)
                for p in c.range_constexpr(self.ks):
                    if c.const_expr(not (self.mutant == "drop_k" and p == 1)):
                        if c.const_expr(kahan):  # rank order, FP64, rounded once
                            acc = acc + Float64(parts[p * 2 * cell + item]) + Float64(parts[p * 2 * cell + cell + item])
                        elif c.const_expr(self.owner64):
                            acc = acc + Float64(parts[p * cell + item])
                        else:
                            acc = acc + parts[p * cell + item]
                j, t = item % own, item // own
                if c.const_expr(self.mutant == "skip_n"):
                    if t >= n - 8:
                        acc = acc * 0
                rows[t * (own + 1) + j] = Float32(acc)
        named_barrier(1, 128)
        self.publish(args, rows, m, rank)
        self.mark(args, "pub", tid < self.batch)

    def tree_sum(self, tmem, warp, total):
        """total[t] = pairwise sum of the accumulators (FP32): blocks of 2, then 4, ... merged like a binary
        counter, so each value passes through ceil(log2(accs)) additions. Plain Python (no AST rewrite): the
        loops unroll at trace time."""
        n = self.n
        stack = []  # (level, partial) pairs, like a binary counter
        for j in range(self.accs):
            values = load_tmem(tmem + Uint32(32 * warp << 16) + Uint32(j * self.cs), self.cs)
            part = cute.make_rmem_tensor(n, Float32)
            for t in range(n):
                part[t] = values[t]
            level = 0
            while stack and stack[-1][0] == level:
                _, prev = stack.pop()
                for t in range(n):
                    prev[t] = prev[t] + part[t]
                part, level = prev, level + 1
            stack.append((level, part))
        part = stack.pop()[1]
        while stack:
            _, prev = stack.pop()
            for t in range(n):
                prev[t] = prev[t] + part[t]
            part = prev
        for t in range(n):
            total[t] = part[t]

    @cute.jit
    def publish(self, args, rows, m, rank):
        """The served front publish (SharedUmma.publish, width 8): token t's 8 owned rows."""
        out, mc, slot, base = args[1:5]
        tid, _, _, _ = thread()
        own, pitch = self.own, out.shape[2]
        shard: Constexpr = 448
        if tid < self.batch:
            t = tid
            row = self.tile * m + own * rank
            values = [rows[t * (own + 1) + v] for v in range(own)]
            if c.const_expr(self.mutant != "silent"):
                if row < self.valid:
                    for v in c.range_constexpr(own // 4):
                        words = tuple(G.positive_zero32(values[4 * v + i]) for i in range(4))
                        at_row = row
                        if c.const_expr(self.mutant == "column"):
                            at_row = row ^ 8
                        offset = t * pitch + 3584 + 2 * (slot + at_row + 4 * v)
                        G.multicast_words(mc + Int64(offset) * 2, words, kind="bf16x2")
                elif (row >= self.split) & (row - self.split < shard):
                    words = tuple(G.positive_zero(values[i], values[i + 1]) for i in range(0, own, 2))
                    offset = t * pitch + base + row - self.split
                    G.multicast_words(mc + Int64(offset) * 2, words, kind="bf16x2")


@functools.cache
def compile_tc(batch, split, valid, device, ks=8, pairs=2, mutant=None, stamps=False, wide=False, spread=False,
               kahan=False, owner64=True, tree=True, tile=64):
    """Arguments: front_maps (256 B), mailbox box [1, 64, WIDTH], multicast address, logit slot, latent base."""
    kernel = TcFront(batch, split, valid, ks, pairs, mutant, stamps, wide, spread, kahan, owner64, tree, tile)
    args = (fake((256,), None, c.Uint8, 64), fake((1, 64, 5376)), Int64(0), Int32(0), Int32(0))
    args += (Int64(0),) * stamps
    with torch.accelerator.device_index(device):
        return cute.compile(kernel, *args, make_fake_stream())
