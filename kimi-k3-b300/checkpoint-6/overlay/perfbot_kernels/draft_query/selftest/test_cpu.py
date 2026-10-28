# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the DraftQuery judge (stdlib unittest + torch CPU): exact helpers, served-norm emulation, RoPE, FP8,
insert, envelope and the three mutants' signatures. Each test fails if the helper it checks is wrong.

    python3 -m unittest perfbot_kernels.draft_query.selftest.test_cpu -v      (PYTHONPATH = overlay)
"""

import random
import struct
import unittest
from fractions import Fraction

import torch

from perfbot_kernels.draft_query.selftest import judge


def f32_round(q: Fraction) -> float:
    """Round-to-nearest-even of an exact rational to fp32 (normal range)."""
    if q == 0:
        return 0.0
    sign = -1 if q < 0 else 1
    q = abs(q)
    e = q.numerator.bit_length() - q.denominator.bit_length()
    while Fraction(2) ** e > q:
        e -= 1
    while Fraction(2) ** (e + 1) <= q:
        e += 1
    ulp = Fraction(2) ** (e - 23)
    k = q / ulp
    lo = k.numerator // k.denominator
    rem = k - lo
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and lo % 2 == 1):
        lo += 1
    return sign * float(lo * ulp)


def f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def triton_norm_row(x, gamma, eps):
    """Scalar transcription of the served Triton tree for one row (independent of judge.emulate_norm's tensor
    form): thread t holds 8t..8t+7, warp butterfly 16..1, warps butterfly 4, 2, 1."""
    size = len(x)
    xs = [float(v) for v in x] + [0.0] * (2048 - size)
    part = []
    for t in range(256):
        v = xs[8 * t : 8 * t + 8]
        fm = lambda a, b, c: f32_round(Fraction(a) * Fraction(b) + Fraction(c))
        a = fm(v[0], v[0], f32(v[2] * v[2]))
        b = fm(v[1], v[1], f32(v[3] * v[3]))
        c = fm(v[4], v[4], f32(v[6] * v[6]))
        d = fm(v[5], v[5], f32(v[7] * v[7]))
        part.append(f32(f32(a + c) + f32(b + d)))
    warps = []
    for w in range(8):
        lanes = part[32 * w : 32 * w + 32]
        for mask in (16, 8, 4, 2, 1):
            lanes = [f32(lanes[i] + lanes[i ^ mask]) for i in range(32)]
        warps.append(lanes[0])
    for mask in (4, 2, 1):
        warps = [f32(warps[i] + warps[i ^ mask]) for i in range(8)]
    total = warps[0]
    rcp = f32(1.0 / size)
    mean = f32_round(Fraction(rcp) * Fraction(total) + Fraction(f32(eps)))
    rr = float(torch.rsqrt(torch.tensor(mean, dtype=torch.float32)))
    out = [f32(f32(rr * float(xv)) * float(g)) for xv, g in zip(x, gamma)]
    return torch.tensor(out, dtype=torch.float32).to(torch.bfloat16)


class Helpers(unittest.TestCase):
    def test_fma_single_rounding(self):
        rng = random.Random(1)
        a, b, c = [], [], []
        for _ in range(3000):
            x = f32(rng.uniform(-4, 4))
            y = f32(rng.uniform(-4, 4))
            a.append(x)
            b.append(y)
            c.append(f32(-x * y * (1 + rng.choice([0, 2.0 ** -24, -(2.0 ** -24), 2.0 ** -30]))))
        # exact midpoint constructions: 1 + 2^-24 (+/- 2^-60) lands on / off a tie
        a += [1.0, 1.0, 1.0]
        b += [1.0 + 2.0 ** -23, 1.0 + 2.0 ** -23, 1.0 + 2.0 ** -23]
        c += [f32(2.0 ** -24), f32(2.0 ** -24 + 2.0 ** -46), f32(2.0 ** -24 - 2.0 ** -47)]
        ta, tb, tc = (torch.tensor(v, dtype=torch.float32) for v in (a, b, c))
        got = judge.fma(ta, tb, tc).tolist()
        want = [f32_round(Fraction(x) * Fraction(y) + Fraction(z)) for x, y, z in zip(a, b, c)]
        self.assertEqual(got, want)
        # a non-fused mul + add differs somewhere (the test would pass a wrong fma otherwise)
        self.assertNotEqual((ta * tb + tc).tolist(), want)

    def test_butterfly_is_lane_uniform_sum(self):
        v = torch.randn(4, 32)
        out = judge.butterfly(v)
        self.assertTrue(torch.equal(out, out[:, :1].expand_as(out)))
        self.assertTrue(torch.allclose(out[:, 0], v.sum(-1), atol=1e-5))

    def test_ordinal_and_distance(self):
        codes = torch.tensor([0x00, 0x01, 0x7E, 0x80, 0x81, 0xFE], dtype=torch.uint8)
        self.assertEqual(judge.ordinal(codes).tolist(), [0, 1, 126, 0, -1, -126])
        self.assertEqual(judge.code_distance(codes, codes), (0, 0))
        self.assertEqual(judge.code_distance(torch.tensor([0x01], dtype=torch.uint8),
                                             torch.tensor([0x81], dtype=torch.uint8)), (1, 2))


class ServedArithmetic(unittest.TestCase):
    def test_norm_matches_scalar_transcription(self):
        g = torch.Generator().manual_seed(3)
        for size, scale in ((1536, 1.0), (512, 1.0), (1536, 300.0), (512, 1e-3)):
            x = (torch.randn(2, size, generator=g) * scale).to(torch.bfloat16)
            gamma = (1 + 0.1 * torch.randn(size, generator=g)).to(torch.bfloat16)
            got = judge.emulate_norm(x, gamma, 1e-5)
            for r in range(2):
                want = triton_norm_row(x[r].float().tolist(), gamma.float().tolist(), 1e-5)
                self.assertTrue(torch.equal(got[r].view(torch.int16), want.view(torch.int16)), (size, scale, r))

    def test_norm_close_to_fp64(self):
        x = torch.randn(8, 1536).to(torch.bfloat16)
        gamma = torch.ones(1536).to(torch.bfloat16)
        got = judge.emulate_norm(x, gamma, 1e-5).double()
        xd = x.double()
        ref = xd * torch.rsqrt(xd.pow(2).mean(-1, keepdim=True) + 1e-5)
        self.assertLess(float(((got - ref).abs() / ref.abs().clamp(min=1e-3)).max()), 2.0 ** -7)

    def test_rope_pairs_and_half_order(self):
        v = torch.randn(5, 64)
        ang = torch.rand(5, 32) * 6.28
        cs = torch.cat([ang.cos(), ang.sin()], -1)
        got = judge.rope_rows(v, cs).double()
        x, y = v[:, 0::2].double(), v[:, 1::2].double()
        c, s = cs[:, :32].double(), cs[:, 32:].double()
        ref = torch.stack([x * c - y * s, x * s + y * c], -1).flatten(-2)
        self.assertLess(float((got - ref).abs().max()), 1e-5)
        swapped = judge.rope_rows(v, torch.cat([cs[:, 32:], cs[:, :32]], -1)).double()  # mutant "ropehalf"
        self.assertGreater(float((swapped - ref).abs().max()), 0.1)

    def test_fp8_nan_clamp_rn(self):
        v = torch.tensor([float("nan"), 1000.0, -1000.0, 1.0, 1.0625 + 2.0 ** -10, 0.0])
        codes = judge.to_fp8(v, 1.0).tolist()
        self.assertEqual(codes[:4], [0xFE, 0x7E, 0xFE, 0x38])  # NaN -> -448 (FMNMX), +-448 clamp, 1.0
        self.assertEqual(codes[4], 0x39)  # 1.0625 + tiny rounds up to 1.125 (RN, not truncation)
        self.assertEqual(judge.to_fp8(torch.tensor([2.0]), 0.5).tolist(), [0x38])  # scale_inv is applied

    def test_insert_slots(self):
        cache = torch.zeros(3, 4, 576, dtype=torch.uint8)
        rows = torch.arange(1, 4, dtype=torch.uint8)[:, None].expand(3, 576).contiguous()
        out = judge.insert(cache, rows, torch.tensor([5, -1, 0]))
        self.assertEqual(int(out[1, 1, 0]), 1)
        self.assertEqual(int(out[0, 0, 0]), 3)
        self.assertEqual(int(out.sum()), (1 + 3) * 576)  # slot -1 is skipped
        wrong = judge.insert(cache, rows, torch.tensor([6, -1, 1]))  # mutant "slot": slot + 1
        self.assertFalse(judge.compare_bytes("slot", wrong, out)["ok"])


class Envelope(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(5)
        self.qn = torch.randn(6, 1536, generator=g).to(torch.bfloat16)
        self.w_qb = (torch.randn(1536, 1536, generator=g) / 40).to(torch.bfloat16)
        self.w_uk = (torch.randn(8, 128, 512, generator=g) / 12).to(torch.bfloat16)
        ang = torch.rand(6, 32, generator=g) * 6.28
        self.cs = torch.cat([ang.cos(), ang.sin()], -1)
        self.ref, self.err = judge.query_reference(self.qn, self.w_qb, self.w_uk, self.cs, 1.0)

    def test_rounded_reference_passes(self):
        codes = judge.to_fp8(self.ref.float(), 1.0)
        self.assertTrue(judge.envelope("ref", codes, self.ref, self.err)["ok"])

    def test_two_step_shift_fails(self):
        codes = judge.to_fp8(self.ref.float(), 1.0)
        big = self.ref.abs() > 4
        shifted = codes.clone()
        shifted[big] = (codes[big].int() + 2).to(torch.uint8)  # +2 magnitude codes on large values
        self.assertFalse(judge.envelope("shift", shifted, self.ref, self.err)["ok"])

    def test_ropehalf_reference_fails(self):
        cs_swapped = torch.cat([self.cs[:, 32:], self.cs[:, :32]], -1)
        bad, _ = judge.query_reference(self.qn, self.w_qb, self.w_uk, cs_swapped, 1.0)
        codes = judge.to_fp8(bad.float(), 1.0)
        self.assertFalse(judge.envelope("ropehalf", codes, self.ref, self.err)["ok"])


if __name__ == "__main__":
    unittest.main()
