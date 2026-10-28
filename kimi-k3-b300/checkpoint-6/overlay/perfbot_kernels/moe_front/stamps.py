"""Item 17 R0: per-CTA %clock64 stamps, mapped to %globaltimer at one edge per CTA.

Why clock64: %globaltimer moves in 32 ns steps on B200 (scratch/r0/tick.json) and a read is a few cycles; %clock64
is cycle-exact within a CTA (one SM). (The 0.256 us steps in scratch/r0/v1_globaltimer_r1.jsonl were a host bug:
ns since epoch cast to float64 has a 256 ns ulp; r0.py now subtracts c in int64.) A stamp is ``red.max.u64`` of
%clock64 (one SM per
CTA, so it is monotone in the CTA; max = the CTA's last writer). Each CTA also records one globaltimer EDGE
(the first read after the value changes) with its %clock64: t_ns = edge_gt + (clk - edge_clk) / f, with f
(clock64 per ns) from the stand-in's two edges 6 us apart. The host zeroes the buffer; 0 = not reached.
No memory clobber: the asm has side effects, so it keeps its order with the kernels' other asm."""
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint32, Uint64

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.primitives import asm


@asm()
def stamp(base, index):
    code = "{ .reg .b64 t,a; mov.u64 t,%clock64; mul.wide.u32 a,$1,8; add.s64 a,$0,a; "
    code += "red.relaxed.gpu.global.max.u64 [a],t; }"
    return code, "l,r", (Int64(base), Int32(index))


@asm((Uint64, Uint64))
def gt_clk():
    return "mov.u64 $0,%globaltimer; mov.u64 $1,%clock64;", "=l,=l", ()


@asm()
def store2(base, index, a, b):
    code = "{ .reg .b64 p; mul.wide.u32 p,$1,8; add.s64 p,$0,p; st.global.u64 [p],$2; st.global.u64 [p+8],$3; }"
    return code, "l,r,l,l", (Int64(base), Int32(index), Uint64(a), Uint64(b))


@cute.jit
def edge(base, index):
    """Spin to the next %globaltimer change; store (gt, clock64) of that read at base[index:index+2]."""
    g0, c0 = gt_clk()
    g, clk = gt_clk()
    while g == g0:
        g, clk = gt_clk()
    store2(base, index, g, clk)


@asm()
def stamp_gt(base, index):
    """red.max of %globaltimer (ns) at base[index]."""
    code = "{ .reg .b64 t,a; mov.u64 t,%globaltimer; mul.wide.u32 a,$1,8; add.s64 a,$0,a; "
    code += "red.relaxed.gpu.global.max.u64 [a],t; }"
    return code, "l,r", (Int64(base), Int32(index))


@asm()
def pin(pointer, value, key):
    """Predicated-off global red of ``value``: ``key`` is a runtime kernel argument (the mailbox slot, never
    0x7FFFFFFF), so ptxas cannot prove the predicate false (a tid bound would let it drop the red, as it did)."""
    code = "{ .reg .pred p; setp.eq.u32 p,$2,0x7FFFFFFF; @p red.relaxed.gpu.global.add.u32 [$0],$1; }"
    return code, "l,r,r", (pointer, Uint32(value), Int32(key))
