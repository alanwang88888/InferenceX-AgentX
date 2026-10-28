"""Item 17: fork of the served MoE front (WsGemm + RouteSitu, vLLM 49cb2bbf) for the R0 phase map.

compile_publish / compile_route mirror the served moe/front.py functions of the same name, plus ``stamps``:
False compiles the served code path (no extra argument), True adds one Int64 argument (stamp buffer address)
before the stream. StandIn is the item-08 harness stand-in (bundle/tools/corun.py) plus a stamp of its end (c)."""
import functools

import cutlass as c
import cutlass.cute as cute
import torch
from cutlass import Int32, Uint64
from cutlass.cute.runtime import make_fake_stream

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import fake
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.primitives import asm, launch
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe import front as F

from .route import RS_EDGE, RS_NS, RS_SLOTS, RouteSitu
from .stamps import edge, gt_clk, stamp, stamp_gt, store2
from .ws_gemm import WS_EDGE, WS_NS, WS_SLOTS, WsGemm

__all__ = ["WF_NS", "WF_SLOTS", "WakeFloor", "RS_EDGE", "WS_EDGE", "STANDIN_NS", "RS_NS", "RS_SLOTS", "WS_NS", "WS_SLOTS", "RouteSitu", "StandIn", "WsGemm", "compile_publish",
           "compile_route"]


@functools.cache
def compile_route(batch, shared, renorm, roles, device, polled=False, plan=False, stamps=False):
    kernel = RouteSitu(batch, shared, renorm, batch <= 2, roles, polled, plan, stamps=stamps)
    f32 = c.Float32
    args: tuple = (fake((batch, 896), None, f32), fake((896,), None, f32))
    args += (fake((batch, 16), None, c.Int32), fake((batch, 16)), fake((1, 64, F.WIDTH)))
    args += (fake((batch, 3584)), fake((batch, 2 * shared), None, f32))
    args += (fake((batch, shared)), *(f32(v) for v in (1, 4, 25)))
    args += (fake((F.PLAN_BYTES,), None, c.Uint8),) * plan
    args += (c.Int64(0),) * stamps + (make_fake_stream(),)
    with torch.accelerator.device_index(device):
        return cute.compile(kernel, *args)


@functools.cache
def compile_publish(n, split, valid, batch, device, ks, geometry=None, stamps=False, ready=False, only=None):
    rt, kq, staged, tg, ch = geometry or F.front_geometry(batch, n, ks)
    kernel = WsGemm(n, 7168, batch, rt, kq, staged, split, valid, ks, tg, ch, stamps=stamps, ready=ready, only=only)
    args: tuple = (fake((batch, 7168)), fake((n, 7168)), fake((1, 64, F.WIDTH)))
    args += (c.Int64(0), c.Int32(0), c.Int32(0), fake((batch, 896), None, c.Float32))
    args += (c.Int64(0),) * stamps
    with torch.accelerator.device_index(device):
        return cute.compile(kernel, *args, make_fake_stream())


@asm(Uint64)
def globaltimer():
    return "mov.u64 $0,%globaltimer;", "=l", ()


STANDIN_NS = 15  # per CTA (3): start edge (gt, clk), end read (gt, clk); then [12 + cta]: last warp's last gt (R5)


class StandIn:
    """O-proj consumer stand-in (item 08 corun.StandIn): 3 CTAs release their dependents at entry, then spin
    ``ns`` from a globaltimer edge. Thread 0 stores the start edge and the exit read (the first read past the
    target, so also an edge) at sb[4 cta : 4 cta + 4]: c = max over CTAs of the exit globaltimer; clock64 per
    ns = (clk_end - clk_start) / (gt_end - gt_start), 6 us apart, edge to edge."""

    @cute.jit
    def __call__(self, ns: Int32, sb: c.Int64, s):
        launch(self.kernel, (ns, sb, s), 3, 128, pdl=False)

    @cute.kernel
    def kernel(self, args):
        ns, sb = args[0], args[1]
        cute.arch.griddepcontrol_launch_dependents()
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        g0, c0 = gt_clk()
        t0, k0 = gt_clk()
        while t0 == g0:
            t0, k0 = gt_clk()
        g, k = gt_clk()
        while g - t0 < Uint64(ns):
            g, k = gt_clk()
        if tid == 0:
            store2(sb, cta * 4, t0, k0)
            store2(sb, cta * 4 + 2, g, k)
        if tid % 32 == 0:  # R5: the CTA's last instruction (last warp), globaltimer ns
            stamp_gt(sb, 12 + cta)


WF_SLOTS = ("entry", "pre", "wait")
WF_NS = len(WF_SLOTS) + 2  # + edge (gt, clk)


class WakeFloor:
    """R5 control: WsGemm's launch shape (``grid`` CTAs x ``threads``, ``cluster``, PDL) doing only
    entry stamp, edge, pre stamp, griddepcontrol_wait, wait stamp (tid 0 of each CTA)."""

    def __init__(self, grid=112, threads=320, cluster=16):
        self.grid, self.threads, self.cluster = grid, threads, cluster

    @cute.jit
    def __call__(self, sb: c.Int64, s):
        launch(self.kernel, (sb, s), self.grid, self.threads, self.cluster if self.cluster > 1 else None,
               min_blocks_per_mp=1)

    @cute.kernel
    def kernel(self, args):
        sb = args[0]
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        if tid == 0:
            stamp(sb, cta * WF_NS + 0)
            edge(sb, cta * WF_NS + 3)
            stamp(sb, cta * WF_NS + 1)
        cute.arch.griddepcontrol_wait()
        if tid == 0:
            stamp(sb, cta * WF_NS + 2)
        cute.arch.griddepcontrol_launch_dependents()
