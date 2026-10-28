# SPDX-License-Identifier: Apache-2.0
"""Pre-registered judge of DraftQuery. It does not import the kernel.

emulate_norm / emulate_cache are the served arithmetic written in torch from the served source + the read PTX/SASS:
  fused_qk_rmsnorm.py:57-90 (Triton 3.8, num_warps 8, PTX lines 131-212 + SASS 0x0310-0x07d0 of the compiled kernel)
  fused_kimi_k3_mla_key_concat_kv_cache_kernel.cu:122-176 + SASS 0x0a70-0x0df0 (decode fp8 insert, RoPE, split 3)
  G1  cache bytes bit-equal to the served chain (the served Triton norm + served concat op, run on the GPU)
  G2  published query: every byte inside the fp64 envelope of the served math (q_b / W_UK accumulation order may
      differ); reported: bytes equal to the served chain, max |code distance| in E4M3 ordinal steps
  G3  salts (M, positions 0 / near max, large rows), graph replay, epoch / parity over 3 calls
MUFU results are hardware values on GPU (probe kernel); CPU tests use torch's rsqrt and 1 / size.
"""

import torch

FP8_MAX = 448.0


def fma(a, b, c):
    """Single-rounding fp32 a * b + c (IEEE, no FTZ): the fp64 product is exact, TwoSum keeps the add's error, and
    an fp64 sum on an exact fp32 midpoint is resolved by that error's sign."""
    a, b, c = a.double(), b.double(), c.double()
    p = a * b
    s = p + c
    bv = s - p
    err = (p - (s - bv)) + (c - bv)
    r = s.float()
    rd = r.double()
    toward = torch.where(s > rd, torch.full_like(r, float("inf")), torch.full_like(r, -float("inf")))
    n = torch.nextafter(r, toward)
    mid = (s != rd) & ((s - rd).abs() == (n.double() - s).abs())
    go_n = mid & (err != 0) & ((err > 0) == (n.double() > rd))
    out = torch.where(go_n, n, r)
    return torch.where(torch.isfinite(s), out, s.float())


def butterfly(v):
    """v += shfl_xor(v, mask), mask 16..1, over the last dim (32 lanes)."""
    lanes = torch.arange(32, device=v.device)
    for mask in (16, 8, 4, 2, 1):
        v = v + v[..., lanes ^ mask]
    return v


def emulate_norm(x, gamma, eps, rsqrt=torch.rsqrt, rcp=None):
    """Served Triton RMSNorm of rows x [M, S] bf16 (S = 1536 or 512), gamma [S] bf16 -> bf16 [M, S]."""
    m, size = x.shape
    xf = torch.zeros(m, 2048, dtype=torch.float32, device=x.device)
    xf[:, :size] = x.float()
    t = xf.view(m, 256, 8)
    sq = lambda i: t[..., i] * t[..., i]  # exact: bf16 squares fit fp32 (or overflow alike)
    a = fma(t[..., 0], t[..., 0], sq(2))
    b = fma(t[..., 1], t[..., 1], sq(3))
    c = fma(t[..., 4], t[..., 4], sq(6))
    d = fma(t[..., 5], t[..., 5], sq(7))
    s = (a + c) + (b + d)  # [M, 256]
    w = butterfly(s.view(m, 8, 32))[..., 0]  # warp sums [M, 8]
    lanes = torch.arange(8, device=x.device)
    v = w
    for mask in (4, 2, 1):
        v = v + v[:, lanes ^ mask]
    total = v[:, 0]
    if rcp is None:
        rcp = torch.tensor(1.0 / size, dtype=torch.float32)
    rcp = torch.as_tensor(rcp, dtype=torch.float32, device=x.device).expand_as(total)
    mean = fma(rcp, total, torch.full_like(total, float(torch.tensor(eps, dtype=torch.float32))))
    rr = rsqrt(mean)
    y = (rr[:, None] * x.float()) * gamma.float()[None, :]
    return y.to(torch.bfloat16)


def rope(x, y, cos, sin):
    """Served pair rotation: (fma(x, cos, -RN(y sin)), fma(x, sin, RN(y cos)))."""
    return fma(x, cos, -(y * sin)), fma(x, sin, y * cos)


def rope_rows(v, cs):
    """v [..., 64] fp32 interleaved pairs, cs [..., 64] fp32 (cos 0..31 | sin 32..63) -> rotated [..., 64]."""
    x, y = v[..., 0::2], v[..., 1::2]
    cos, sin = cs[..., :32], cs[..., 32:]
    o0, o1 = rope(x, y, cos, sin)
    return torch.stack([o0, o1], -1).flatten(-2)


def to_fp8(v, scale_inv):
    """Served quant: fminf(fmaxf(v * scale_inv, -448), 448) -> E4M3 RN satfinite; returns uint8 codes."""
    s = v.float() * torch.as_tensor(scale_inv, dtype=torch.float32, device=v.device)
    s = torch.where(torch.isnan(s), torch.full_like(s, -FP8_MAX), s)
    s = s.clamp(-FP8_MAX, FP8_MAX)
    return s.to(torch.float8_e4m3fn).view(torch.uint8)


def emulate_cache_rows(qkv, gkv, cos_sin, positions, k_scale_inv, eps, **mufu):
    """[M, 576] uint8 cache entries of the served path for every row (the insert writes those with slot >= 0)."""
    kvn = emulate_norm(qkv[:, 1536:2048], gkv, eps, **mufu)
    kpe = rope_rows(qkv[:, 2048:2112].float(), cos_sin[positions.long()])
    return torch.cat([to_fp8(kvn.float(), k_scale_inv), to_fp8(kpe, k_scale_inv)], -1)


def insert(cache, rows, slots):
    """cache [blocks, bs, 576] uint8 with rows [M, 576] written at slots (>= 0)."""
    out = cache.clone()
    bs = cache.shape[1]
    for t, s in enumerate(slots.tolist()):
        if s >= 0:
            out[s // bs, s % bs] = rows[t]
    return out


# ------------------------------------------------------------------ G2 envelope


def fp8_value(codes):
    return codes.view(torch.float8_e4m3fn).double()


def fp8_half_ulp(v):
    """Half the E4M3 spacing at |v| (normals: 2^(e-4); subnormals: 2^-10)."""
    a = v.abs().clamp(min=2.0 ** -6)
    e = torch.floor(torch.log2(a))
    return torch.exp2(e - 4)


def ordinal(codes):
    """E4M3 codes on one ordered integer line (sign-magnitude -> two's-complement order)."""
    c = codes.int()
    mag = c & 0x7F
    return torch.where((c & 0x80) != 0, -mag, mag)


def query_reference(qn, w_qb, w_uk, qpe_cos_sin, q_scale_inv):
    """fp64 reference of one rank's query [M, 8, 576] (pre-quant, scaled) and its error bound.
    qn [M, 1536] bf16 (the served norm output), w_qb [1536, 1536], w_uk [8, 128, 512], cos_sin rows [M, 64]."""
    u32, u16 = 2.0 ** -24, 2.0 ** -8  # fp32 / bf16 unit roundoff (bf16: 8-bit significand)
    x = qn.double()
    w = w_qb.double()
    q = (x @ w.t()).view(-1, 8, 192)
    qabs = (x.abs() @ w.abs().t()).view(-1, 8, 192)
    eq = 1536 * u32 * qabs * 1.01 + u16 * q.abs() * 1.01  # fp32 accumulation + bf16 rounding of q (|q| bound)
    eq = eq + u16 * eq
    wuk = w_uk.double()
    qn_ = q[..., :128].transpose(0, 1)  # [8, M, 128]
    ql = torch.bmm(qn_, wuk).transpose(0, 1)  # [M, 8, 512]
    ql_abs = torch.bmm((q[..., :128].abs() + eq[..., :128]).transpose(0, 1), wuk.abs()).transpose(0, 1)
    eql = torch.bmm(eq[..., :128].transpose(0, 1), wuk.abs()).transpose(0, 1) + 128 * u32 * ql_abs * 1.01
    eql = eql + u16 * (ql.abs() + eql) * 1.01
    cs = qpe_cos_sin.double()[:, None, :]
    pe = q[..., 128:]
    x0, y0 = pe[..., 0::2], pe[..., 1::2]
    ex, ey = eq[..., 128:][..., 0::2], eq[..., 128:][..., 1::2]
    cos, sin = cs[..., :32], cs[..., 32:]
    o0, o1 = x0 * cos - y0 * sin, x0 * sin + y0 * cos
    e0 = ex * cos.abs() + ey * sin.abs() + 2 * u32 * ((x0.abs() + ex) * cos.abs() + (y0.abs() + ey) * sin.abs())
    e1 = ex * sin.abs() + ey * cos.abs() + 2 * u32 * ((x0.abs() + ex) * sin.abs() + (y0.abs() + ey) * cos.abs())
    rope_v = torch.stack([o0, o1], -1).flatten(-2)
    rope_e = torch.stack([e0, e1], -1).flatten(-2)
    ref = torch.cat([ql, rope_v], -1) * float(q_scale_inv)
    err = torch.cat([eql, rope_e], -1) * float(q_scale_inv) * (1 + 2 * u32)
    return ref, err


def envelope(name, codes, ref, err):
    """Every E4M3 byte within err + half an E4M3 ulp (at the far edge) of ref, saturating at +-448."""
    got = fp8_value(codes)
    edge = ref.abs() + err
    tol = err + fp8_half_ulp(edge) * 1.0001 + 2.0 ** -10
    target = ref.clamp(-FP8_MAX, FP8_MAX)
    bad = (got - target).abs() > tol
    bad |= torch.isnan(got)
    n = int(bad.sum())
    first = None
    if n:
        idx = tuple(int(i) for i in bad.nonzero()[0])
        first = {"index": idx, "got": float(got[idx]), "ref": float(ref[idx]), "tol": float(tol[idx])}
    return {"name": name, "ok": n == 0, "mismatches": n, "first": first}


def compare_bytes(name, got, want):
    diff = got.view(torch.uint8) != want.view(torch.uint8)
    n = int(diff.sum())
    first = None
    if n:
        idx = tuple(int(i) for i in diff.nonzero()[0])
        first = {"index": idx, "got": int(got.view(torch.uint8)[idx]), "want": int(want.view(torch.uint8)[idx])}
    return {"name": name, "ok": n == 0, "mismatches": n, "first": first}


def code_distance(got, want):
    """(bytes differing, max |ordinal distance|) between two E4M3 byte tensors."""
    g, w = got.view(torch.uint8), want.view(torch.uint8)
    d = (ordinal(g) - ordinal(w)).abs()
    return int((g != w).sum()), int(d.max()) if d.numel() else 0


def verdict(checks):
    return all(c["ok"] for c in checks)
