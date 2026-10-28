# SPDX-License-Identifier: Apache-2.0
"""Pre-registered judge of the draft AR consumer: class "AR same rank-order bits".

emulate_served() is the served flashinfer kernel's arithmetic written in torch from its source
(trtllm_allreduce_fusion.cuh, flashinfer 0.7.2, identical to the image copy) and its SASS (built -use_fast_math:
FADD/FMUL/FFMA .FTZ, MUFU.RCP + FFMA for mean + eps, MUFU.RSQ). It does not import the kernel.

  G1  normed and residual_out bit-equal to emulate_served (every emulated rank)
  G2  every rank's bits equal (stand-in: the 8 gather copies and their outputs)
  G3  salts (M, residual 0, large rows, eps edge, re-arm over consecutive calls, graph replay) pass G1 + G2,
      and every receive slot / gather row the call used is back to the sentinel afterwards.
MUFU results are hardware values: on GPU the caller passes rsqrt = the measured MUFU.RSQ (probe kernel) and
rcp_hidden = the measured MUFU.RCP(7168); CPU tests use torch's.
"""

import torch

TINY = 2.0 ** -126
SENTINEL = -0x80000000


def ftz(t):
    """Flush fp32 denormals to signed zero."""
    return torch.where(t.abs() < TINY, t * 0.0, t)


def add(a, b):
    return ftz(ftz(a) + ftz(b))


def mul(a, b):
    return ftz(ftz(a) * ftz(b))


def fma(a, b, c):
    """Single-rounding fp32 a * b + c (FTZ): the product is exact in fp64, TwoSum keeps the add's error, and an
    fp64 sum that sits exactly on an fp32 midpoint is resolved by that error's sign."""
    a, b, c = ftz(a).double(), ftz(b).double(), ftz(c).double()
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
    return ftz(torch.where(go_n, n, r))


def bf16(t):
    return t.to(torch.bfloat16)


def butterfly(v):
    """warpReduceSumV2 over the last dim (32 lanes): v += shfl_xor(v, mask), mask 16..1."""
    lanes = torch.arange(32, device=v.device)
    for mask in (16, 8, 4, 2, 1):
        v = add(v, v[..., lanes ^ mask])
    return v


def emulate_served(xs, residual, gamma, eps, wb, rsqrt=torch.rsqrt, rcp_hidden=None, order=None):
    """xs [TP, M, H] bf16 partials (rank order = dim 0), residual [M, H], gamma [H]; wb = warps per served block.
    Returns (normed, residual_out) bf16 [M, H]."""
    tp, m, h = xs.shape
    order = list(range(tp)) if order is None else order
    x = xs.float()
    x = torch.where(x == 0, torch.zeros_like(x), x)  # remove_neg_zero before the push (cuh:1544)
    acc = x[order[0]]
    for r in order[1:]:  # allreduce_sum Fp32Acc (cuh:1468): FADD.FTZ in rank order, one rounding
        acc = add(acc, x[r])
    total_bf = bf16(acc)
    rout = bf16(add(total_bf.float(), residual.float()))  # vec_add (cuh:813), FADD.FTZ
    y = rout.float()
    t = y.view(m, h // 8, 8)
    s = torch.zeros(m, h // 8, dtype=torch.float32, device=y.device)
    for i in range(8):  # acc += v * v -> FFMA.FTZ chain (SASS 0x29e0..0x2ba0)
        s = fma(t[..., i], t[..., i], s)
    warps = butterfly(s.view(m, h // 256, 32))[..., 0]  # [M, 28]
    blocks = warps.view(m, -1, wb)
    pad = torch.zeros(m, blocks.shape[1], 32 - wb, dtype=torch.float32, device=y.device)
    bsum = butterfly(torch.cat([blocks, pad], -1))[..., 0]  # blockReduceSumV2 second stage
    total = torch.zeros(m, dtype=torch.float32, device=y.device)
    for b in range(bsum.shape[1]):  # cluster sum by thread 0 (cuh:1307-1310)
        total = add(total, bsum[:, b])
    if rcp_hidden is None:
        rcp_hidden = torch.tensor(1.0 / h, dtype=torch.float32)
    rcp_hidden = torch.as_tensor(rcp_hidden, dtype=torch.float32, device=y.device)
    mean = fma(total, rcp_hidden.expand_as(total), torch.full_like(total, float(torch.tensor(eps, dtype=torch.float32))))
    rstd = rsqrt(mean)
    g = add(gamma.float(), torch.zeros_like(gamma.float()))  # weight_bias (0) + gamma, FADD.FTZ
    normed = bf16(mul(mul(y, rstd[:, None]), g[None, :]))
    return normed, rout


def bits(t):
    return t.contiguous().view(torch.int16)


def compare(name, got, want):
    """{name, ok, mismatches, first} on bit patterns."""
    diff = bits(got) != bits(want)
    n = int(diff.sum())
    first = None
    if n:
        idx = tuple(int(i) for i in diff.nonzero()[0])
        first = {"index": idx, "got": float(got[idx]), "want": float(want[idx])}
    return {"name": name, "ok": n == 0, "mismatches": n, "first": first}


def rearmed(name, buffer):
    """Every 32-bit word of ``buffer`` (a used region) is the Lamport sentinel."""
    words = buffer.contiguous().view(torch.int32)
    bad = int((words != SENTINEL).sum())
    return {"name": name, "ok": bad == 0, "mismatches": bad, "first": None}


def judge_call(tag, normed, rout, want_normed, want_rout):
    """G1 + G2 over [copies, M, H] outputs against one emulation."""
    checks = []
    for c in range(normed.shape[0]):
        checks.append(compare(f"{tag}/G1/normed[{c}]", normed[c], want_normed))
        checks.append(compare(f"{tag}/G1/rout[{c}]", rout[c], want_rout))
    for c in range(1, normed.shape[0]):
        checks.append(compare(f"{tag}/G2/normed[{c}]==[0]", normed[c], normed[0]))
        checks.append(compare(f"{tag}/G2/rout[{c}]==[0]", rout[c], rout[0]))
    return checks


def verdict(checks):
    return all(c["ok"] for c in checks)
