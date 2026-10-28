"""Item 17 R0 fork of the served WsGemm (vLLM 49cb2bbf gemm/ws_gemm.py) with optional %globaltimer stamps.

The kernel body and ``settle`` are copies of the served code; every other method (``__init__``, ``tiles``,
``stage_x``, ``gather``, ``epilogue``) is inherited, so the math and the publish are the served ones.
``stamps`` (const_expr): off = the served kernel; on = one extra Int64 argument (the stamp buffer address,
after ``logits``) and the stamps of ``WS_SLOTS``: %clock64 at ``buf[cta * WS_NS + slot]`` plus one
(globaltimer, clock64) edge at ``buf[cta * WS_NS + WS_EDGE]`` (stamps.py)."""
from typing import Any

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint32

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.primitives import (
    alloc,
    at,
    bits_float,
    copy_g2s,
    load_words,
    mma_into,
    push_float,
    smem_words,
    wait,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.ws_gemm import WsGemm as ServedWsGemm

from .stamps import edge, pin, stamp

WS_GROUPS = 3  # at most 3 token groups of 16 at <= 40 rows (tg 16)
# Slot names: entry, wait = griddepcontrol_wait return, cw = cluster_wait return (owners armed, group 0);
# per group g: x<g> = x slice landed, k<g> = MMA + push issued, w<g> = owner: all K parts landed (reduce
# wait), p<g> = owner: reduce + publish issued; exit = last warp done.
# pre = tid 0 just before griddepcontrol_wait (R5; with ``ready``: after the prologue's loads have landed).
WS_SLOTS = ("entry", "pre", "wait", "cw") + tuple(f"{n}{g}" for g in range(WS_GROUPS) for n in "xkwp") + ("exit",)
WS_EDGE = len(WS_SLOTS)  # (globaltimer, clock64) of one edge at the CTA's entry
WS_NS = WS_EDGE + 2
_S = {n: i for i, n in enumerate(WS_SLOTS)}


class WsGemm(ServedWsGemm):
    def __init__(self, *a, stamps=False, ready=False, only=None, **kw):
        super().__init__(*a, **kw)
        # only (R5 bisect): the stamp names (and "edge") compiled in; None = all
        self.only = None if only is None else frozenset(only)
        # ready (R5): drain cp.async and consume the register weights before the wait. Served SASS has ACQBULK
        # (griddepcontrol.wait) BEFORE the weight LDGs: ptxas sinks them (scratch/r5/SASS.txt). Same math.
        self.stamps, self.ready = stamps, ready
        if stamps and (not self.grouped or self.groups > WS_GROUPS):
            raise ValueError("WsGemm stamps: token-group path with <= 3 groups only")

    @cute.jit
    def mark(self, sb, cta, name: cutlass.Constexpr, when):
        if cutlass.const_expr(self.stamps and (self.only is None or name.rstrip("012") in self.only)):
            if when:
                stamp(sb, cta * WS_NS + _S[name])

    @cute.kernel
    def kernel(self, args):
        x, w = args[0:2]
        sb = args[7] if self.stamps else None
        smem = cutlass.utils.SmemAllocator()
        block, _, _ = cute.arch.block_idx()
        cta, kpart = block // self.ks, block % self.ks
        tid, _, _ = cute.arch.thread_idx()
        lane, warp = tid % 32, tid // 32
        self.mark(sb, block, "entry", tid == 0)
        if cutlass.const_expr(self.stamps and (self.only is None or "edge" in self.only)):
            if tid == 0:
                edge(sb, block * WS_NS + WS_EDGE)
        group, part = lane // 4, lane % 4
        tile, quarter = warp // self.kq, warp % self.kq
        row = (cta * self.rt + tile) * 16 + group
        kofs = (kpart * self.kq + quarter) * self.kb * 32 + part * 8
        shape = (self.nt, self.rt, self.kq, 8, 17)
        red = smem.allocate_tensor(Float32, cute.make_layout(shape), 16)
        stage = alloc(smem, Uint32, (self.rt * self.kq, max(self.staged, 1), 2, 32, 4))
        xs = recv = None
        bars: Any = None
        owned: Any = None
        if cutlass.const_expr(self.grouped):
            xs = alloc(smem, Uint32, (self.nt * 8, self.pitch))
            bars = alloc(smem, cutlass.Uint64, (self.groups,), 8)
            recv = alloc(smem, Uint32, (self.slots, self.parts, 32, 4))
            owned = alloc(smem, cutlass.Uint64, (self.slots,), 8)
            if tid == 0:
                for g in cutlass.range_constexpr(self.groups):
                    cute.arch.mbarrier_init(bars.iterator + g, 1)
                for s in cutlass.range_constexpr(self.slots):
                    if s * self.ks + kpart < self.blocks:
                        cute.arch.mbarrier_init(owned.iterator + s, 1)
                        nbytes = self.parts * 512
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            owned.iterator + s, nbytes
                        )
                cute.arch.mbarrier_init_fence()
            cute.arch.sync_threads()
            cute.arch.cluster_arrive_relaxed()
        wp = w.iterator + row * self.k + kofs
        for j in cutlass.range_constexpr(self.staged):
            for h in cutlass.range_constexpr(2):
                dst = at(stage, warp, j, h, lane, 0)
                copy_g2s(dst, wp + h * 8 * self.k + (self.rb + j) * 32)
        cute.arch.cp_async_commit_group()
        wr = {}
        for j in cutlass.range_constexpr(self.rb):
            for h in cutlass.range_constexpr(2):
                src = wp + h * 8 * self.k + j * 32
                wr[j, h] = load_words(src, count=4, evict_first=True)
        if cutlass.const_expr(self.ready):
            cute.arch.cp_async_wait_group(0)
            mix = Uint32(0)
            for j in cutlass.range_constexpr(self.rb):
                for h in cutlass.range_constexpr(2):
                    for i in cutlass.range_constexpr(4):
                        mix = mix ^ Uint32(wr[j, h][i])
            pin(w.iterator, mix, args[4])  # never stores (slot != 0x7FFFFFFF); keeps the loads above the wait
        self.mark(sb, block, "pre", tid == 0)
        cute.arch.griddepcontrol_wait()
        self.mark(sb, block, "wait", tid == 0)
        cute.arch.griddepcontrol_launch_dependents()
        if cutlass.const_expr(self.grouped):
            if warp == 0:
                self.stage_x(x, xs, bars, kpart, lane)
            cute.arch.cp_async_wait_group(0)
        else:
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
        for g in cutlass.range_constexpr(self.groups):
            acc = cute.make_rmem_tensor((self.ng, self.chains, 4), Float32)
            acc.fill(0)
            if cutlass.const_expr(self.grouped):
                wait(bars.iterator + g)
                self.mark(sb, block, f"x{g}", tid == 0)
            for j in cutlass.range_constexpr(self.kb):
                if cutlass.const_expr(j < self.rb):
                    wa, wb = wr[j, 0], wr[j, 1]
                else:
                    sa = at(stage, warp, j - self.rb, 0, lane, 0)
                    wa, wb = smem_words(sa), smem_words(sa + 128)
                for t in cutlass.range_constexpr(self.tiles(g)):
                    token = (g * self.ng + t) * 8 + group
                    xv = (Uint32(0), Uint32(0), Uint32(0), Uint32(0))
                    if cutlass.const_expr(self.grouped):
                        col = (quarter * self.kb + j) * 16 + part * 4
                        xv = smem_words(at(xs, token, col))
                    else:
                        xg = x.iterator + token * x.stride[0] + kofs + j * 32
                        if (g * self.ng + t) * 8 + 7 < self.m or token < self.m:
                            xv = load_words(xg, count=4, mutable=True)
                    for h in cutlass.range_constexpr(2):
                        c = (j * 2 + h) % self.chains
                        a = (wa[2 * h], wb[2 * h], wa[2 * h + 1], wb[2 * h + 1])
                        b = (xv[2 * h], xv[2 * h + 1])
                        mma_into(acc[t, c, None], a, b, bf16=True)
            if cutlass.const_expr(self.grouped and g == 0):
                cute.arch.cluster_wait()  # owners' barriers are armed
                self.mark(sb, block, "cw", tid == 0)
            for t in cutlass.range_constexpr(self.tiles(g)):
                values = []
                for v in cutlass.range_constexpr(4):
                    value = acc[t, 0, v]
                    for c in cutlass.range_constexpr(1, self.chains):
                        value = value + acc[t, c, v]
                    tt, col = g * self.ng + t, group + (v // 2) * 8
                    if cutlass.const_expr(self.grouped):
                        values.append(value)
                    else:
                        red[tt, tile, quarter, 2 * part + v % 2, col] = value
                if cutlass.const_expr(self.grouped):
                    blk = (g * self.ng + t) * self.rt + tile
                    s, peer = blk // self.ks, blk % self.ks
                    dst = at(recv, s, kpart * self.kq + quarter, lane, 0)
                    push_float(dst, tuple(values), owned.iterator + s, peer)
            if cutlass.const_expr(self.grouped):
                self.mark(sb, block, f"k{g}", tid == 0)
            if cutlass.const_expr(self.grouped and g > 0):
                self.settle(args, red, recv, owned, cta, kpart, warp, lane, g - 1, sb, block)
        if cutlass.const_expr(self.grouped):
            self.settle(args, red, recv, owned, cta, kpart, warp, lane, self.groups - 1, sb, block)
        else:
            cute.arch.sync_threads()
            if cutlass.const_expr(self.ks > 1):
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()
            self.gather(args, red, cta, kpart, tid, lane)
            if cutlass.const_expr(self.ks > 1):
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()
        self.mark(sb, block, "exit", lane == 0)

    @cute.jit
    def settle(
        self, args, red, recv, owned, cta, kpart, warp, lane, g: cutlass.Constexpr, sb, block
    ):
        """Served ``settle`` plus the w<g> / p<g> stamps of the owner warps."""
        for t in cutlass.range_constexpr(self.tiles(g)):
            for tile in cutlass.range_constexpr(self.rt):
                blk = (g * self.ng + t) * self.rt + tile
                s = blk // self.ks
                if (kpart == blk % self.ks) & (warp == s % (self.rt * self.kq)):
                    wait(owned.iterator + s)
                    self.mark(sb, block, f"w{g}", lane == 0)
                    total = cute.make_rmem_tensor(4, Float32)
                    for p in cutlass.range_constexpr(self.parts):
                        words = smem_words(at(recv, s, p, lane, 0))
                        for v in cutlass.range_constexpr(4):
                            value = bits_float(words[v])
                            total[v] = value if p == 0 else total[v] + value
                    tt = g * self.ng + t
                    for v in cutlass.range_constexpr(4):
                        col = lane // 4 + v // 2 * 8
                        red[tt, tile, 0, 2 * (lane % 4) + v % 2, col] = total[v]
                    cute.arch.sync_warp()
                    token = tt * 8 + lane // 2
                    if (lane < 16) & (token < self.m):
                        half = lane % 2 * 8
                        values = [
                            red[tt, tile, 0, lane // 2, half + j] for j in range(8)
                        ]
                        col = (cta * self.rt + tile) * 16 + half
                        self.epilogue(args, token, col, values)
                    self.mark(sb, block, f"p{g}", lane == 0)
