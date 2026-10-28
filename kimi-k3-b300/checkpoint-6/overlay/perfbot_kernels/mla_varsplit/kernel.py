"""The device greedy of P-14 L6: one thread computes var_splits (the host reference in __init__.py) over a batch's
local lengths inside the captured step, so the vector needs no host hook (specialized_decode runs Python only at
capture). Integer cross-multiplication compares len_i / split_i exactly; the float compare of the host reference
gives the same order at these sizes (lengths < 2^31, splits <= 15: distinct ratios differ by >= 1 / 225)."""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64

EMPTY = 1 << 16  # __init__.EMPTY: an empty unit's table entry flag


class VarSplit:
    """Request b of ``reqs`` gets floor (0-length: 1), then each of up to ``units`` total units goes to the
    non-empty request below ``cap`` with the largest len_b / split_b, the lowest index on a tie. ``order`` (01d):
    then out[reqs:] = unit_table (the host reference in __init__.py)."""

    def __init__(self, reqs, units, cap, floor, gate=False, wave=__import__("perfbot_kernels.k5q", fromlist=["k5q"]).UNITS, order=False):  # PERFBOT K5 item 3 (land-8: wave=12)
        self.reqs, self.units, self.cap, self.floor = reqs, units, cap, floor
        self.gate, self.wave = gate, wave  # gate: units above one wave only if two_wave_pays (host reference)
        self.order = order

    @cute.jit
    def __call__(self, lens, out, stream):
        # No PDL: tokenspeed (PDL-launched after this kernel) then starts only after this kernel's warp exits.
        self.kernel(lens, out).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, lens, out):
        tid, _, _ = cute.arch.thread_idx()
        if tid == 0:
            n = cute.make_rmem_tensor(self.reqs, Int32)
            sp = cute.make_rmem_tensor(self.reqs, Int32)
            total = Int32(0)
            for b in cutlass.range_constexpr(self.reqs):
                n[b] = lens[b]
                s = Int32(1)
                if n[b] > 0:
                    s = Int32(self.floor)
                sp[b] = s
                total += s
            units = Int32(self.units)
            if cutlass.const_expr(self.gate):
                top, lmax = Int32(0), n[0]
                for b in cutlass.range_constexpr(1, self.reqs):
                    if n[b] > lmax:
                        top, lmax = Int32(b), n[b]
                low = Int32(0)
                seen = Int32(0)
                for b in cutlass.range_constexpr(self.reqs):
                    if (top != b) & (n[b] > 0):
                        if (seen == 0) | (n[b] < low):
                            low = n[b]
                        seen = Int32(1)
                if Int64(2) * Int64(low) >= Int64(lmax):
                    units = Int32(min(self.units, self.wave))
            for _ in cutlass.range_constexpr(self.units - self.reqs):
                best, bn, bs = Int32(-1), Int64(0), Int64(1)
                for b in cutlass.range_constexpr(self.reqs):
                    nb, sb = Int64(n[b]), Int64(sp[b])
                    if (sp[b] < self.cap) & (nb > 0) & ((best < 0) | (nb * bs > bn * sb)):
                        best, bn, bs = Int32(b), nb, sb
                if (best >= 0) & (total < units):
                    for b in cutlass.range_constexpr(self.reqs):
                        if best == b:
                            sp[b] = sp[b] + 1
                    total += 1
            for b in cutlass.range_constexpr(self.reqs):
                out[b] = sp[b]
            if cutlass.const_expr(self.order):
                self.table(n, sp, out)
            # tokenspeed reads block_split_kvs before its griddepcontrol_wait (mla_decode_fp8.py:1572/1668/1738) and
            # launches on this warp's exit (pic-c: evidence/picc-prewait): the fence orders the stores before the exit.
            cute.arch.fence_acq_rel_gpu()

    @cute.jit
    def table(self, n, sp, out):
        """out[reqs + u] for u in order: the live units (two waves: each request's units, longest len / split first,
        a selection over <= 5 requests; one wave: z-major), then the empty units in (z, b) order, flagged EMPTY."""
        B = self.reqs
        done = cute.make_rmem_tensor(B, Int32)
        live = Int32(0)
        for b in cutlass.range_constexpr(B):
            done[b] = Int32(0)
            live += sp[b]
        u = Int32(B)
        if live <= self.wave:  # one wave: the live units in z-major order (unit_table)
            for z in cutlass.range_constexpr(self.cap):
                for b in cutlass.range_constexpr(B):
                    if z < sp[b]:
                        out[u] = Int32(b + B * z)
                        u += 1
                        done[b] = Int32(1)
        for _ in cutlass.range_constexpr(B):
            best, bn, bs = Int32(-1), Int64(0), Int64(1)
            for b in cutlass.range_constexpr(B):
                nb, sb = Int64(n[b]), Int64(sp[b])
                if (done[b] == 0) & ((best < 0) | (nb * bs > bn * sb)):
                    best, bn, bs = Int32(b), nb, sb
            for b in cutlass.range_constexpr(B):
                if best == b:
                    done[b] = Int32(1)
                    for z in cutlass.range_constexpr(self.cap):
                        if z < sp[b]:
                            out[u] = Int32(b + B * z)
                            u += 1
        for z in cutlass.range_constexpr(self.cap):
            for b in cutlass.range_constexpr(B):
                if z >= sp[b]:
                    out[u] = Int32(b + B * z + EMPTY)
                    u += 1


@cache
def compiled(reqs, units, cap, floor, gate, device, order=False):
    from cutlass.cute.runtime import make_fake_stream

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import fake

    lens, vec = fake((reqs,), None, torch.int32, 4), fake((reqs * (1 + cap) if order else reqs,), None, torch.int32, 4)
    with torch.accelerator.device_index(device):
        return cute.compile(VarSplit(reqs, units, cap, floor, gate, order=order), lens, vec, make_fake_stream())


def produce(lengths, out, rule, floor, gate=False):
    """out[:reqs] = the vector of ``lengths`` (Int32 [reqs], device) under ``rule`` = (units, cap), on the current
    stream; when out has reqs x (1 + cap) entries, out[reqs:] = the 01d unit table."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import stream, to_cute

    reqs, device = lengths.shape[0], out.device
    order = out.shape[0] > reqs
    assert out.shape[0] == (reqs * (1 + rule[1]) if order else reqs), (out.shape, reqs, rule)
    kernel = compiled(reqs, *rule, floor, gate, device.index, order)
    kernel(to_cute(lengths.contiguous(), 4), to_cute(out, 4), stream(device))
