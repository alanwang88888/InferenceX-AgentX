"""PTX building blocks for the fused rejection tail (CuTe DSL inline asm).

Every float op is the PTX instruction that the image's Triton kernels emit (vLLM 49cb2bbf, Triton 3.8.0, B200):
the PTX in results/cache/triton (YDRH/J2TZ stats, 3K73 rejection, OEMA/2DQQ resample) is the source of each form.
Explicit .rn / .full / .approx forms cannot be contracted by ptxas, and the Triton PTX has no mul -> add pair either.
"""

import os

# Overlay copy (item-02): the lane Q dev copy set CUTE_DSL_CACHE_DIR here; in the server that would move every
# CuTe DSL compile cache, so the server's own setting stands.

from cutlass import Float32, Int32, Int64  # noqa: E402
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

_C = {Float32: ("f", "f32"), Int32: ("r", "i32"), Int64: ("l", "i64")}


@dsl_user_op
def asm(ptx, out, *args, side=False, loc=None, ip=None):
    """One inline PTX block. out: Float32 | Int32 | Int64 | tuple of them | None. $0.. are outputs, then args."""
    outs = () if out is None else out if isinstance(out, tuple) else (out,)
    cons = ["=" + _C[o][0] for o in outs] + [_C[type(a)][0] for a in args] + (["~{memory}"] if side else [])
    if not outs:
        rt = None
    elif len(outs) == 1:
        rt = outs[0].mlir_type
    else:
        rt = ir.Type.parse("!llvm.struct<(" + ",".join(_C[o][1] for o in outs) + ")>")
    v = llvm.inline_asm(rt, [a.ir_value(loc=loc, ip=ip) for a in args], ptx, ",".join(cons),
                        has_side_effects=side, asm_dialect=0, loc=loc, ip=ip)
    if not outs:
        return None
    if len(outs) == 1:
        return outs[0](v)
    return tuple(o(llvm.extractvalue(o.mlir_type, v, [i], loc=loc, ip=ip)) for i, o in enumerate(outs))


F, I, L = Float32, Int32, Int64


# ---- float ops (Triton forms) ----
def add(a, b):
    return asm("add.rn.f32 $0, $1, $2;", F, a, b)


def sub(a, b):
    return asm("sub.rn.f32 $0, $1, $2;", F, a, b)


def mul(a, b):
    return asm("mul.rn.f32 $0, $1, $2;", F, a, b)


def fma(a, b, c):
    return asm("fma.rn.f32 $0, $1, $2, $3;", F, a, b, c)


def fdiv(a, b):  # Triton `x / temp` on fp32: div.full.f32
    return asm("div.full.f32 $0, $1, $2;", F, a, b)


def fmax(a, b):  # max.f32: a NaN operand is ignored
    return asm("max.f32 $0, $1, $2;", F, a, b)


def exp_sub(x, m):  # tl.exp(x - m) = ex2.approx((x - m) * log2e)
    return asm("{ .reg .b32 t; sub.rn.f32 t, $1, $2; mul.rn.f32 t, t, 0f3FB8AA3B; ex2.approx.f32 $0, t; }", F, x, m)


def bf16_f32(h):  # bf16 bits in the low 16 bits of an Int32 -> f32 (exact)
    return asm("shl.b32 $0, $1, 16;", F, h)


def i2f(x):
    return asm("mov.b32 $0, $1;", F, x)


def f2i(x):
    return asm("mov.b32 $0, $1;", I, x)


def i32_f32(x):  # cvt.rn.f32.s32
    return asm("cvt.rn.f32.s32 $0, $1;", F, x)


# libdevice __nv_logf (ftz form) as Triton emits it; covers subnormal, 0, inf, NaN and negative inputs.
LOGF = """{ .reg .pred p, q; .reg .b32 x, xs, e0, t, m, e, f, r, s;
mov.b32 x, $1;
setp.lt.f32 p, x, 0f00800000;
mul.rn.f32 xs, x, 0f4B000000;
selp.b32 xs, xs, x, p;
selp.b32 e0, 0fC1B80000, 0f00000000, p;
add.s32 t, xs, -1059760811;
and.b32 t, t, -8388608;
sub.s32 m, xs, t;
cvt.rn.f32.s32 e, t;
fma.rn.ftz.f32 e, e, 0f34000000, e0;
add.rn.f32 f, m, 0fBF800000;
fma.rn.ftz.f32 r, 0fBE055027, f, 0f3E1039F6;
fma.rn.ftz.f32 r, r, f, 0fBDF8CDCC;
fma.rn.ftz.f32 r, r, f, 0f3E0F2955;
fma.rn.ftz.f32 r, r, f, 0fBE2AD8B9;
fma.rn.ftz.f32 r, r, f, 0f3E4CED0B;
fma.rn.ftz.f32 r, r, f, 0fBE7FFF22;
fma.rn.ftz.f32 r, r, f, 0f3EAAAA78;
fma.rn.ftz.f32 r, r, f, 0fBF000000;
mul.rn.f32 s, f, r;
fma.rn.ftz.f32 s, s, f, f;
fma.rn.ftz.f32 r, e, 0f3F317218, s;
setp.lt.u32 q, xs, 2139095040;
@!q fma.rn.ftz.f32 r, xs, 0f7F800000, 0f7F800000;
setp.eq.f32 q, xs, 0f00000000;
selp.b32 $0, 0fFF800000, r, q;
}"""


def logf(x):
    return asm(LOGF, F, x)


# libdevice __nv_log1pf (ftz form) applied to -ratio, as in the resample residual (OEMA).
LOG1P_NEG = """{ .reg .pred p, q; .reg .b32 rt, x, a, t, i, k, f, sc, r, s;
mov.b32 rt, $1;
neg.f32 x, rt;
sub.rz.ftz.f32 a, 0f3F800000, rt;
add.s32 t, a, -1061158912;
and.b32 t, t, -8388608;
sub.s32 i, x, t;
sub.s32 k, 1082130432, t;
fma.rn.ftz.f32 f, 0f3E800000, k, 0fBF800000;
add.rn.f32 f, f, i;
cvt.rn.f32.s32 sc, t;
mul.rn.f32 sc, sc, 0f34000000;
fma.rn.ftz.f32 r, 0fBD39BF78, f, 0f3DD80012;
fma.rn.ftz.f32 r, r, f, 0fBE0778E0;
fma.rn.ftz.f32 r, r, f, 0f3E146475;
fma.rn.ftz.f32 r, r, f, 0fBE2A68DD;
fma.rn.ftz.f32 r, r, f, 0f3E4CAF9E;
fma.rn.ftz.f32 r, r, f, 0fBE800042;
fma.rn.ftz.f32 r, r, f, 0f3EAAAAE6;
fma.rn.ftz.f32 r, r, f, 0fBF000000;
mul.rn.f32 s, f, r;
fma.rn.ftz.f32 s, s, f, f;
fma.rn.ftz.f32 r, sc, 0f3F317218, s;
setp.lt.u32 p, x, 2139095040;
setp.lt.s32 q, x, -1082130431;
or.pred q, q, p;
@!q fma.rn.ftz.f32 r, x, 0f7F800000, 0f7F800000;
setp.eq.f32 q, rt, 0f00000000;
@!p selp.b32 r, 0f80000000, r, q;
mov.b32 $0, r;
}"""


def residual(t, d, tlse, dlse, temp):
    """Resample residual logit (OEMA): tlp = t - tlse; dlp = d / temp - dlse; ratio = exp(dlp - tlp);
    ratio < 1 ? tlp + log1p(-ratio) : -inf."""
    tlp = sub(t, tlse)
    dlp = sub(fdiv(d, temp), dlse)
    ratio = exp_sub(dlp, tlp)
    l1p = asm(LOG1P_NEG, F, ratio)
    return asm("{ .reg .pred p; .reg .b32 s; setp.lt.f32 p, $2, 0f3F800000; add.rn.f32 s, $1, $3;"
               " selp.b32 $0, s, 0fFF800000, p; }", F, tlp, ratio, l1p)


# ---- Gumbel noise (gumbel.py murmur3_uniform32 + _log1p_neg_stable, fp32) ----
def _mix(h, k):
    return asm("{ .reg .b32 k; mul.lo.s32 k, $2, -862048943; shf.l.wrap.b32 k, k, k, 15; mul.lo.s32 k, k, 461845907;"
               " xor.b32 k, $1, k; shf.l.wrap.b32 k, k, k, 13; mad.lo.s32 $0, k, 5, -430675100; }", I, h, k)


def murmur_prefix(seed_lo, seed_hi, pos_lo):
    return _mix(_mix(_mix(Int32(0), seed_lo), seed_hi), pos_lo)


def gumbel_value(logit, prefix, token):
    """logit + Gumbel noise = logit - log(-log1p_neg_stable(u)), u = murmur3_uniform32(seed, pos, token)."""
    h = _mix(prefix, token)
    h = asm("{ .reg .b32 h, t; xor.b32 h, $1, 16; shr.u32 t, h, 16; xor.b32 h, h, t; mul.lo.s32 h, h, -2048144789;"
            " shr.u32 t, h, 13; xor.b32 h, h, t; mul.lo.s32 h, h, -1028477387; shr.u32 t, h, 16; xor.b32 $0, h, t; }",
            I, h)
    u = asm("{ .reg .b32 hi, lo, a, b; shr.u32 hi, $1, 16; and.b32 lo, $1, 65535; cvt.rn.f32.u32 a, hi;"
            " cvt.rn.f32.u32 b, lo; add.rn.f32 b, b, 0f3F000000; mul.rn.f32 b, b, 0f2F800000;"
            " fma.rn.f32 $0, a, 0f37800000, b; }", F, h)
    p = asm("{ .reg .b32 p; fma.rn.f32 p, $1, 0f3E000000, 0f3E124925; fma.rn.f32 p, $1, p, 0f3E2AAAAB;"
            " fma.rn.f32 p, $1, p, 0f3E4CCCCD; fma.rn.f32 p, $1, p, 0f3E800000; fma.rn.f32 p, $1, p, 0f3EAAAAAB;"
            " fma.rn.f32 p, $1, p, 0f3F000000; fma.rn.f32 p, $1, p, 0f3F800000; neg.f32 $0, $1;"
            " mul.rn.f32 $0, p, $0; }", F, u)
    direct = logf(asm("{ .reg .b32 t; sub.rn.f32 t, 0f3F800000, $1; max.f32 $0, t, 0f33800000; }", F, u))
    sel = asm("{ .reg .pred p; setp.lt.f32 p, $1, 0f3E800000; selp.b32 $0, $2, $3, p; }", F, u, p, direct)
    return sub(logit, logf(asm("neg.f32 $0, $1;", F, sel)))


# ---- Philox 4x32-10 (tl.rand(seed, offset) with an int64 offset), then tl_rand32(includes_zero=False) ----
def philox_u(seed_lo, seed_hi, off_lo, off_hi):
    c0, c1, c2, c3, k0, k1 = off_lo, off_hi, Int32(0), Int32(0), seed_lo, seed_hi
    for _ in range(10):
        c0, c1, c2, c3 = asm("{ .reg .b32 a, b, c, d; mul.hi.u32 a, $5, -845247145; xor.b32 a, a, $4; xor.b32 a, a, $8;"
                             " mul.hi.u32 c, $6, -766435501; xor.b32 c, c, $7; xor.b32 c, c, $9;"
                             " mul.lo.s32 b, $5, -845247145; mul.lo.s32 d, $6, -766435501;"
                             " mov.b32 $0, a; mov.b32 $1, b; mov.b32 $2, c; mov.b32 $3, d; }",
                             (I, I, I, I), c1, c2, c0, c3, k0, k1)
        k0, k1 = asm("{ .reg .b32 a, b; add.s32 a, $2, -1640531527; add.s32 b, $3, -1150833019;"
                     " mov.b32 $0, a; mov.b32 $1, b; }", (I, I), k0, k1)
    return asm("{ .reg .b32 t, s; shr.s32 s, $1, 31; xor.b32 t, s, $1; cvt.rn.f32.s32 t, t; mul.rn.f32 t, t, 0f2FFFFFFF;"
               " max.f32 $0, t, 0f2FFFFFFF; }", F, c0)


# ---- argmax combine (Triton _argmax_combine_tie_break_left): a wins iff a > b or (a == b and ia < ib) ----
def amax(va, ia, vb, ib):
    return asm("{ .reg .pred p, q, e; setp.gt.f32 p, $2, $4; setp.eq.f32 e, $2, $4; setp.lt.s32 q, $3, $5;"
               " and.pred e, e, q; or.pred p, p, e; .reg .b32 v, i; selp.b32 v, $2, $4, p; selp.b32 i, $3, $5, p;"
               " mov.b32 $0, v; mov.b32 $1, i; }",
               (F, I), va, ia, vb, ib)


# ---- warp / memory ----
def shfl(x, k):
    t = type(x)
    return asm(f"shfl.sync.bfly.b32 $0, $1, {k}, 31, -1;", t, x)


def redux_max(x):
    return asm("redux.sync.max.f32 $0, $1, -1;", F, x)


def ld_v4(addr):  # 16 B global load (read-only inputs)
    return asm("ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$4];", (I, I, I, I), addr)


def ld32(addr, t=I):  # read-only input
    return asm("ld.global.nc.b32 $0, [$1];", t, addr)


def ld64(addr):
    return asm("ld.global.nc.b64 $0, [$1];", L, addr)


def ld_u16(addr):
    return asm("ld.global.nc.u16 $0, [$1];", I, addr)


def ldcg(addr, t=I):  # workspace written in this launch: L2, never a stale L1 line
    return asm("ld.global.cg.b32 $0, [$1];", t, addr, side=True)


def st32(addr, v):
    asm("st.global.b32 [$0], $1;", None, addr, v, side=True)


def st64(addr, v):
    asm("st.global.b64 [$0], $1;", None, addr, v, side=True)


def sts(addr, v):
    asm("st.shared.b32 [$0], $1;", None, addr, v, side=True)


def lds(addr, t=I):
    return asm("ld.shared.b32 $0, [$1];", t, addr, side=True)


def bar():
    asm("bar.sync 0;", None, side=True)


def atom_add_acqrel(addr, v):
    return asm("atom.add.acq_rel.gpu.global.s32 $0, [$1], $2;", I, addr, v, side=True)


def ld_acquire(addr):
    return asm("ld.acquire.gpu.global.b32 $0, [$1];", I, addr, side=True)


def st_relaxed(addr, v):
    asm("st.relaxed.gpu.global.b32 [$0], $1;", None, addr, v, side=True)


def nanosleep():
    asm("nanosleep.u32 32;", None, side=True)


def sel(flag, a, b):  # flag != 0 ? a : b (bitwise select, any 32-bit type)
    return asm("{ .reg .pred p; setp.ne.s32 p, $1, 0; selp.b32 $0, $2, $3, p; }", type(a), flag, a, b)


def fne0(x):  # Triton `temp != 0.0` as Int32 (unordered compare is not used: temp is never NaN here)
    return asm("{ .reg .pred p; setp.ne.f32 p, $1, 0f00000000; selp.b32 $0, 1, 0, p; }", Int32, x)


def icmp(op, a, b):  # signed compare as Int32 0/1; op in lt le gt ge eq ne
    return asm(f"{{ .reg .pred p; setp.{op}.s32 p, $1, $2; selp.b32 $0, 1, 0, p; }}", Int32, a, b)


def fcmp(op, a, b):  # ordered float compare as Int32 0/1 (Triton `<` / `>`: false on NaN)
    return asm(f"{{ .reg .pred p; setp.{op}.f32 p, $1, $2; selp.b32 $0, 1, 0, p; }}", Int32, a, b)


def nan_to_ninf(x):  # tl.where(x != x, -inf, x)
    return asm("{ .reg .pred p; setp.nan.f32 p, $1, $1; selp.b32 $0, 0fFF800000, $1, p; }", Float32, x)


def lse32(m, s):
    """_compute_global_logsumexp on one warp, lane l = block l (masked lanes: m = -inf, s = 0), as in 3K73:
    gmax = redux max; p = exp(m - gmax) * s; level 16 is fma(e, s, shfl(p)); levels 8..1 add; gmax + logf(sum)."""
    g = redux_max(m)
    e = exp_sub(m, g)
    x = fma(e, s, shfl(mul(e, s), 16))
    for k in (8, 4, 2, 1):
        x = add(x, shfl(x, k))
    return add(g, logf(x))
