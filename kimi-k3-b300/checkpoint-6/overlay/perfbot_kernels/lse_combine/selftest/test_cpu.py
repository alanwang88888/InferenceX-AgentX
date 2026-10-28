# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the DraftValue judge (stdlib unittest + torch CPU): the served-combine emulation against a scalar
transcription, the envelope, the source mapping and the mutants' signatures. Each test fails if the helper it checks
is wrong.

    python3 -m unittest perfbot_kernels.lse_combine.selftest.test_cpu -v      (PYTHONPATH = overlay)
"""

import struct
import unittest
from fractions import Fraction

import torch

from perfbot_kernels.draft_query.selftest.test_cpu import f32_round
from perfbot_kernels.lse_combine.selftest import judge

NINF = float("-inf")


def f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def bf16(x):
    return float(torch.tensor(x, dtype=torch.float32).to(torch.bfloat16))


def ex2_cpu(t):
    return torch.exp2(t)


def served_scalar(xs, vs):
    """Scalar transcription of wait_lse_combine_kernel (one token, one head): xs [8][D] floats (bf16 values),
    vs [8] floats. Independent of judge.emulate_combine's tensor form."""
    vv = [NINF if (v != v or v == float("inf")) else f32(v) for v in vs]
    m = NINF
    for v in vv:
        m = max(m, v)
    if m == NINF:
        m = 0.0
    e = [float(ex2_cpu(torch.tensor(f32(v - m), dtype=torch.float32))) for v in vv]
    total = 0.0
    for x in e:
        total = f32(total + x)
    inv = f32_round(Fraction(1) / Fraction(total)) if total > 0 else 0.0
    w = [f32(inv * x) for x in e]
    out = []
    for d in range(len(xs[0])):
        acc = 0.0
        for s in range(8):
            if w[s] == 0.0:
                continue
            acc = f32_round(Fraction(w[s]) * Fraction(xs[s][d]) + Fraction(acc))  # one rounding (FFMA)
        out.append(bf16(acc))
    return out


def case(seed, spread=4.0, d=16, neg=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(8, 1, 1, d, generator=g).to(torch.bfloat16)
    v = torch.randn(8, 1, 1, generator=g) * spread
    for s in range(neg):
        v[s] = NINF
    return x, v


class Combine(unittest.TestCase):
    def test_emulation_matches_scalar(self):
        for seed in range(40):
            x, v = case(seed, spread=[0.5, 4.0, 40.0, 300.0][seed % 4], neg=seed % 3)
            got = judge.emulate_combine(x, v, ex2_cpu)[0, 0].float().tolist()
            want = served_scalar(x[:, 0, 0].float().tolist(), v[:, 0, 0].tolist())
            self.assertEqual(got, want, f"seed {seed}")

    def test_all_neg_inf_gives_zero(self):
        x, v = case(3)
        v[:] = NINF
        x[0, 0, 0, 0] = float("nan")
        got = judge.emulate_combine(x, v, ex2_cpu)
        self.assertTrue(bool((got == 0).all()))

    def test_nan_and_plus_inf_lse_are_neg_inf(self):
        x, v = case(4)
        v2 = v.clone()
        v2[2] = float("nan")
        v2[5] = float("inf")
        v3 = v.clone()
        v3[2] = NINF
        v3[5] = NINF
        a = judge.emulate_combine(x, judge.reader_lse(v2), ex2_cpu)
        b = judge.emulate_combine(x, v3, ex2_cpu)
        self.assertTrue(torch.equal(a.view(torch.int16), b.view(torch.int16)))

    def test_rank_order_needs_cancellation(self):
        """Mutant rankorder: on random rows a reversed FFMA order almost never moves a bf16 bit (so random salts
        cannot catch it); with source 1 = -source 0 at 2^20 (the gate's "cancel" salt) it moves most of them."""
        rand, cancel, n = 0, 0, 0
        for seed in range(10):
            x, v = case(seed, spread=1.0, d=512)
            a = judge.emulate_combine(x, v, ex2_cpu)
            b = judge.emulate_combine(x.flip(0), v.flip(0), ex2_cpu)
            rand += int((a.view(torch.int16) != b.view(torch.int16)).sum())
            x = judge.cancel_rows(x.transpose(0, 1)).transpose(0, 1)
            v[:] = 0.5
            a = judge.emulate_combine(x, v, ex2_cpu)
            b = judge.emulate_combine(x.flip(0), v.flip(0), ex2_cpu)
            cancel += int((a.view(torch.int16) != b.view(torch.int16)).sum())
            n += a.numel()
        self.assertLess(rand, n // 100)
        self.assertGreater(cancel, n // 4)

    def test_no_max_shift_breaks_large_lse(self):
        """Mutant nomax: EX2 of an LSE near 200 overflows; the fp64 envelope must reject the result."""
        x, v = case(5, d=64)
        v = v + 200.0
        good = judge.emulate_combine(x, v, ex2_cpu)
        e = torch.exp2(v)
        total = e.sum(0)
        w = (1.0 / total) * e
        bad = (w[..., None] * x.float()).sum(0).to(torch.bfloat16)
        lat, elat, _, _ = judge.reference(x, v, torch.zeros(1, 64, 1, dtype=torch.bfloat16))
        self.assertTrue(judge.envelope("good", good, lat, elat)["ok"])
        self.assertFalse(judge.envelope("bad", bad, lat, elat)["ok"])

    def test_zero_weight_skip(self):
        """Mutant skipzero: a NaN payload in a zero-weight source poisons the sum unless skipped."""
        x, v = case(6, d=8)
        v[3] = NINF
        x[3] = float("nan")
        good = judge.emulate_combine(x, v, ex2_cpu)
        self.assertFalse(bool(torch.isnan(good.float()).any()))
        w = judge.weights64(v)
        self.assertEqual(float(w[3].max()), 0.0)
        self.assertTrue(bool(torch.isnan((w[..., None].float() * x.float()).sum(0)).any()))


class Envelope(unittest.TestCase):
    def test_contains_emulated_path(self):
        g = torch.Generator().manual_seed(7)
        for spread in (0.3, 5.0, 60.0):
            x = torch.randn(8, 3, 8, 512, generator=g).to(torch.bfloat16)
            v = torch.randn(8, 3, 8, generator=g) * spread
            w_uv = (torch.randn(8, 512, 128, generator=g) / 22).to(torch.bfloat16)
            lat_bf = judge.emulate_combine(x, v, ex2_cpu)
            out = torch.einsum("mhk,hkv->mhv", lat_bf.float(), w_uv.float()).reshape(3, -1).to(torch.bfloat16)
            lat, elat, ref, err = judge.reference(x, v, w_uv)
            self.assertTrue(judge.envelope("lat", lat_bf, lat, elat)["ok"])
            self.assertTrue(judge.envelope("out", out, ref, err)["ok"])

    def test_rejects_four_ulp_shift(self):
        g = torch.Generator().manual_seed(8)
        x = torch.randn(8, 2, 8, 512, generator=g).to(torch.bfloat16)
        v = torch.randn(8, 2, 8, generator=g)
        w_uv = (torch.randn(8, 512, 128, generator=g) / 22).to(torch.bfloat16)
        _, _, ref, err = judge.reference(x, v, w_uv)
        out = ref.to(torch.bfloat16)
        i = int(ref.abs().argmax())
        bits = out.view(torch.int16).flatten().clone()
        bits[i] += 4
        self.assertFalse(judge.envelope("shift", bits.view(torch.bfloat16).view_as(out), ref, err)["ok"])


class Mapping(unittest.TestCase):
    def test_owner_and_solo_sources(self):
        rows = torch.arange(8 * 2 * 64 * 512, dtype=torch.float32).view(8, 2, 64, 512).to(torch.bfloat16)
        lse = torch.arange(8 * 2 * 64, dtype=torch.float32).view(8, 2, 64)
        empty = torch.zeros(8, 2, dtype=torch.bool)
        empty[3, 1] = True
        x, v = judge.owner_inputs(rows, lse, empty, 5)
        self.assertTrue(torch.equal(x[2], rows[2][:, 40:48]))
        self.assertEqual(float(v[3, 1, 0]), NINF)
        self.assertEqual(float(v[3, 0, 0]), float(lse[3, 0, 40]))
        xs, vs = judge.owner_inputs(rows[:1], lse[:1], empty[:1], 0, solo=True)
        self.assertTrue(torch.equal(xs[6], rows[0][:, 48:56]))

    def test_lamport_payload(self):
        x = torch.tensor([[-0.0, 1.0, -2.0, 0.0]], dtype=torch.bfloat16).view(1, 1, 4)
        p = judge.lamport_payload(x, torch.tensor([False]))
        self.assertEqual(p.view(torch.int16).tolist(), [[[0, 0x3F80, -16384, 0]]])
        self.assertEqual(int(judge.lamport_payload(x, torch.tensor([True])).view(torch.int16).abs().sum()), 0)
        v = judge.lamport_lse(torch.tensor([[float("nan"), float("inf"), 1.0]]), torch.tensor([False]))
        self.assertEqual(v.tolist(), [[NINF, NINF, 1.0]])


if __name__ == "__main__":
    unittest.main()
