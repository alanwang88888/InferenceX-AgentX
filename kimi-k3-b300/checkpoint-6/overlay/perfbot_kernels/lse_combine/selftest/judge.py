# SPDX-License-Identifier: Apache-2.0
"""Pre-registered judge of DraftValue (kernel 3). It does not import the kernel.

emulate_combine is the served wait_lse_combine_kernel<bf16> written in torch from the served source + its SASS
(dcp_direct_a2a_lse_reduce.cu:142-216; sm_100 cubin 195, evidence/lse_combine/served_combine_sass.txt):
  v = NaN or +inf -> -inf; m = max v (-inf -> 0); e = EX2(v - m); sum = rank-order FADD from 0;
  inv = sum > 0 ? RN(1 / sum) : 0; w = RN(inv * e); acc = rank-order FFMA(w, x, acc) from +0, w == 0 skipped;
  latent = bf16 RN(acc).  ``ex2`` is the hardware value (GPU probe) or torch.exp2 (CPU tests).
  G0  emulation == served op (bits), per owner
  G1  the kernel's merged latent == served op (bits)
  G2  output [M, 8 x 128] inside the fp64 envelope of PyTorch-op math (softmax over 2^lse, einsum, W_UV); the served
      latent + bmm must sit inside the same envelope (a check of the envelope itself)
  G3  salts, parities, graph replay, protocol state, late writers, mutants
"""

import torch

from perfbot_kernels.draft_query.selftest.judge import compare_bytes, fma, verdict  # noqa: F401 (re-exported)

U32, U16 = 2.0 ** -24, 2.0 ** -8  # fp32 / bf16 unit roundoff
W_REL = 2.0 ** -19  # weight relative error: EX2 (2 ulp) + 8 adds + RCP + MUL, with margin
W_ABS = 2.0 ** -125  # a weight below the fp32 normal range may round to 0 / a subnormal


def reader_lse(v):
    """The served reader's rule: NaN or +inf -> -inf."""
    return torch.where(torch.isnan(v) | (v == float("inf")), torch.full_like(v, -float("inf")), v)


def owner_inputs(rows, lse, empty, owner, solo=False):
    """Owner ``owner``'s 8 sources: x [8, M, 8, 512] bf16, v [8, M, 8] fp32 (empty shard -> -inf, reader rule).
    rows [E, M, 64, 512], lse [E, M, 64], empty [E, M] bool. solo: E = 1 and source s = head block s of rank 0
    (the shifted peer table)."""
    xs, vs = [], []
    for s in range(8):
        e, block = (0, s) if solo else (s, owner)
        x = rows[e][:, 8 * block : 8 * block + 8]
        v = lse[e][:, 8 * block : 8 * block + 8].float()
        v = torch.where(empty[e][:, None], torch.full_like(v, -float("inf")), v)
        xs.append(x)
        vs.append(reader_lse(v))
    return torch.stack(xs), torch.stack(vs)


def emulate_combine(x, v, ex2=torch.exp2):
    """Served combine of x [8, ..., 512] bf16 with v [8, ...] fp32 (reader rule applied) -> bf16 [..., 512]."""
    m = v.max(0).values
    m = torch.where(m == -float("inf"), torch.zeros_like(m), m)
    e = ex2(v - m)  # fp32 RN subtraction, then EX2
    total = torch.zeros_like(m)
    for s in range(v.shape[0]):
        total = total + e[s]
    inv = torch.where(total > 0, 1.0 / torch.where(total > 0, total, torch.ones_like(total)), torch.zeros_like(total))
    w = inv * e  # [8, ...]
    xf = x.float()
    acc = torch.zeros_like(xf[0])
    for s in range(x.shape[0]):
        ws = w[s][..., None].expand_as(acc)
        acc = torch.where(ws != 0, fma(ws, xf[s], acc), acc)
    return acc.to(torch.bfloat16)


def weights64(v):
    """Exact softmax weights over base-2 LSEs (fp64; all -inf -> 0)."""
    vd = v.double()
    m = vd.max(0).values
    m = torch.where(m == -float("inf"), torch.zeros_like(m), m)
    e = torch.exp2(vd - m)
    total = e.sum(0)
    return torch.where(total > 0, e / torch.where(total > 0, total, torch.ones_like(total)), torch.zeros_like(e))


def reference(x, v, w_uv):
    """fp64 reference of the latent [M, 8, 512] and output [M, 8 x 128] and their error bounds.
    x [8, M, 8, 512] bf16, v [8, M, 8], w_uv [8, 512, 128] (this owner's heads)."""
    w = weights64(v)[..., None]  # [8, M, 8, 1]
    xd = x.double()
    xd = torch.where(w > 0, xd, torch.zeros_like(xd))  # zero-weight sources never contribute (NaN payloads)
    lat = (w * xd).sum(0)
    mag = (w * xd.abs()).sum(0)
    elat = W_REL * mag + W_ABS * xd.abs().sum(0) + 8 * U32 * mag * 1.01
    elat = elat + U16 * (lat.abs() + elat)  # bf16 rounding of the latent
    wd = w_uv.double()
    out = torch.einsum("mhk,hkv->mhv", lat, wd)
    lat_hi = lat.abs() + elat
    eout = torch.einsum("mhk,hkv->mhv", elat, wd.abs()) + 512 * U32 * torch.einsum("mhk,hkv->mhv", lat_hi, wd.abs())
    eout = eout * 1.01 + U16 * (out.abs() + eout)
    m = out.shape[0]
    return lat, elat, out.reshape(m, -1), eout.reshape(m, -1)


def envelope(name, got, ref, err):
    """Every bf16 value within err (+ 2^-120) of ref; NaN fails."""
    g = got.double()
    bad = ((g - ref).abs() > err + 2.0 ** -120) | torch.isnan(g)
    n = int(bad.sum())
    first = None
    if n:
        idx = tuple(int(i) for i in bad.nonzero()[0])
        first = {"index": idx, "got": float(g[idx]), "ref": float(ref[idx]), "tol": float(err[idx])}
    return {"name": name, "ok": n == 0, "mismatches": n, "first": first}


def cancel_rows(x, scale=2.0 ** 20):
    """The rank-order salt: x [.., 8 sources, ...] bf16 with source 1 = -source 0 at ``scale`` (sources 2..7 stay
    O(1)). In rank order the large pair cancels exactly before the small terms; any other order loses them."""
    x = x.clone()
    x[:, 0] = (x[:, 0].float() * scale).to(torch.bfloat16)
    x[:, 1] = -x[:, 0]
    return x


def lamport_payload(x, empty_rows):
    """What a Lamport writer stores: rows with -0 halves made +0, zeros for an empty shard. x [M, 8, 512] bf16."""
    bits = x.contiguous().view(torch.int16).clone()
    bits[bits == -32768] = 0
    bits[empty_rows] = 0
    return bits.view(torch.bfloat16)


def lamport_lse(v, empty_rows):
    """What a Lamport writer stores as LSE: the reader rule at the source; empty -> -inf. v [M, 8]."""
    v = torch.where(empty_rows[:, None], torch.full_like(v, -float("inf")), v.float())
    return reader_lse(v)
