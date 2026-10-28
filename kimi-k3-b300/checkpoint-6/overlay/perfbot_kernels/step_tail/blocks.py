"""CTA-level reductions of the image's Triton kernels, rebuilt in the same order (128 threads, 4 warps).

Layout (Triton blocked, sizePerThread 8): thread t holds elements c*1024 + t*8 + j, register k = 8c + j.
Tree shapes are read from the image PTX (results/cache/triton YDRH, J2TZ, OEMA):
  sum:    E = adjacent tree over even registers, O = same over odd registers, s = E + O; warp bfly 16..1 (own + other);
          all lanes store smem[warp]; bar; lane reads smem[t & 3]; bfly 2, 1.
  max:    any order (max.f32 drops NaN); redux.sync.max; smem as above; bfly 2, 1.
  argmax: adjacent tree over registers, combine(a, b) = a if a > b or (a == b and ia < ib) else b;
          warp bfly 16..1 combine(own, other); all lanes store value, bar, read; store index (+16 B), bar, read;
          bfly 2, 1. With NaN the lanes can disagree: the all-lane smem store is kept as in the image.
These are plain Python helpers (straight-line code); they are inlined into the @cute.kernel at trace time.
"""

from cutlass import Float32, Int32, Int64

from . import ptx as P

F = Float32


def bf_lo(x):
    return P.asm("shl.b32 $0, $1, 16;", F, x)


def bf_hi(x):
    return P.asm("and.b32 $0, $1, -65536;", F, x)


def load8(addr, f32):
    """8 consecutive logits at addr (16 B aligned) as Float32, element order j = 0..7."""
    if f32:
        return [P.i2f(w) for w in P.ld_v4(addr) + P.ld_v4(addr + 16)]
    out = []
    for w in P.ld_v4(addr):
        out += [bf_lo(w), bf_hi(w)]
    return out


def load_block(row, t, nchunk, f32):
    """Elements c*1024 + t*8 + j of the block starting at address row; returns 8*nchunk values (register order)."""
    es = 4 if f32 else 2
    vals = []
    for c in range(nchunk):
        vals += load8(row + Int64((c * 1024 + t * 8) * es), f32)
    return vals


def tree(xs, op):
    while len(xs) > 1:
        xs = [op(xs[i], xs[i + 1]) for i in range(0, len(xs), 2)]
    return xs[0]


def tree2(vs, ix):
    while len(vs) > 1:
        nv, ni = [], []
        for i in range(0, len(vs), 2):
            v, k = P.amax(vs[i], ix[i], vs[i + 1], ix[i + 1])
            nv.append(v)
            ni.append(k)
        vs, ix = nv, ni
    return vs[0], ix[0]


def cta_max(x, sm, w, t):
    m = P.redux_max(x)
    P.bar()
    P.sts(sm + w * 4, m)
    P.bar()
    v = P.lds(sm + (t & 3) * 4, F)
    v = P.fmax(v, P.shfl(v, 2))
    return P.fmax(v, P.shfl(v, 1))


def cta_sum(x, sm, w, t):
    for k in (16, 8, 4, 2, 1):
        x = P.add(x, P.shfl(x, k))
    P.bar()
    P.sts(sm + w * 4, x)
    P.bar()
    v = P.lds(sm + (t & 3) * 4, F)
    v = P.add(v, P.shfl(v, 2))
    return P.add(v, P.shfl(v, 1))


def cta_argmax(v, i, sm, w, t):
    for k in (16, 8, 4, 2, 1):
        v, i = P.amax(v, i, P.shfl(v, k), P.shfl(i, k))
    P.bar()
    P.sts(sm + w * 4, v)
    P.bar()
    v = P.lds(sm + (t & 3) * 4, F)
    P.sts(sm + 16 + w * 4, i)
    P.bar()
    i = P.lds(sm + 16 + (t & 3) * 4)
    v, i = P.amax(v, i, P.shfl(v, 2), P.shfl(i, 2))
    return P.amax(v, i, P.shfl(v, 1), P.shfl(i, 1))


def max_sumexp(vals, sm, w, t):
    """_compute_max_and_sumexp: (max, max > -inf ? sum(exp(x - max)) : 0)."""
    m = cta_max(tree(vals, P.fmax), sm, w, t)
    e = [P.exp_sub(v, m) for v in vals]
    s = cta_sum(P.add(tree(e[0::2], P.add), tree(e[1::2], P.add)), sm, w, t)
    return m, P.asm("{ .reg .pred p; setp.ne.f32 p, $1, 0fFF800000; selp.b32 $0, $2, 0f00000000, p; }", F, m, s)


def block_argmax(vals, t, sm, w):
    """tl.max(vals, return_indices=True) over a block; index is block-local (c*1024 + t*8 + j)."""
    idx = [Int32((k // 8) * 1024 + k % 8) + t * 8 for k in range(len(vals))]
    v, i = tree2(vals, idx)
    return cta_argmax(v, i, sm, w, t)


def load1(addr, f32):
    """One logit as Float32 (bf16 or fp32 source)."""
    return P.ld32(addr, F) if f32 else bf_lo(P.ld_u16(addr))


def warp_argmax(m, idx):
    """_compute_global_target_argmax: warp bfly 16..1 with combine(own, other)."""
    for k in (16, 8, 4, 2, 1):
        m, idx = P.amax(m, idx, P.shfl(m, k), P.shfl(idx, k))
    return m, idx
