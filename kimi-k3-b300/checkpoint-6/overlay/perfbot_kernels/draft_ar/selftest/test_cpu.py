# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the judge (stdlib unittest + torch CPU): helpers, emulation, layout rule, mutants.

    python3 -m unittest perfbot_kernels.draft_ar.selftest.test_cpu -v      (PYTHONPATH = overlay)
"""

import random
import unittest
from fractions import Fraction

import torch

from perfbot_kernels.draft_ar import served_warps_per_block
from perfbot_kernels.draft_ar.selftest import judge

H, TP = 7168, 8


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


def inputs(m, seed, scale=1.0, zero_res=False, wide=False):
    """wide: each element scaled by 2^U(-24, 24); fp32 sums of 8 bf16 values are exact (order-free) unless the
    values span more than ~2^16, so only wide rows can tell rank orders apart."""
    g = torch.Generator().manual_seed(seed)
    xs = torch.randn(TP, m, H, generator=g) * scale
    if wide:
        xs = xs * torch.exp2(torch.randint(-24, 25, xs.shape, generator=g).float())
    xs = xs.to(torch.bfloat16)
    res = torch.zeros(m, H, dtype=torch.bfloat16) if zero_res else torch.randn(m, H, generator=g).to(torch.bfloat16)
    gamma = (1 + 0.1 * torch.randn(H, generator=g)).to(torch.bfloat16)
    return xs, res, gamma


class Helpers(unittest.TestCase):
    def test_fma_single_rounding(self):
        rng = random.Random(1)
        a, b, c = [], [], []
        for _ in range(4000):
            x = rng.uniform(-4, 4)
            a.append(x), b.append(rng.uniform(-4, 4)), c.append(rng.uniform(-4, 4) * 2.0 ** rng.randint(-30, 30))
        # ties of the fp64 sum: p exact, c tiny so p + c lands next to an fp32 midpoint
        for k in range(1, 200):
            p = 1.0 + (2 * k + 1) * 2.0 ** -24  # an fp32 midpoint (needs 25 bits)
            a.append(p), b.append(1.0), c.append(2.0 ** -80 * (1 if k % 2 else -1))
        ta, tb, tc = (torch.tensor(v, dtype=torch.float32) for v in (a, b, c))
        got = judge.fma(ta, tb, tc)
        for i in range(len(a)):
            want = f32_round(Fraction(float(ta[i])) * Fraction(float(tb[i])) + Fraction(float(tc[i])))
            self.assertEqual(float(got[i]), want, (float(ta[i]), float(tb[i]), float(tc[i])))

    def test_ftz(self):
        t = torch.tensor([2.0 ** -127, -(2.0 ** -130), 2.0 ** -126, 1.0])
        out = judge.ftz(t)
        self.assertEqual(out.tolist(), [0.0, -0.0, 2.0 ** -126, 1.0])
        self.assertTrue(torch.signbit(out[1]))
        self.assertEqual(float(judge.add(torch.tensor(2.0 ** -127), torch.tensor(2.0 ** -127))), 0.0)

    def test_layout_rule(self):
        self.assertEqual([served_warps_per_block(m) for m in (1, 7, 14, 37, 38, 42, 56, 64)],
                         [7, 7, 7, 7, 14, 14, 14, 14])

    def test_compare_and_rearm(self):
        a = torch.randn(4, 16).to(torch.bfloat16)
        b = a.clone()
        self.assertTrue(judge.compare("x", a, b)["ok"])
        b.view(torch.int16)[1, 3] ^= 1
        r = judge.compare("x", a, b)
        self.assertFalse(r["ok"])
        self.assertEqual((r["mismatches"], r["first"]["index"]), (1, (1, 3)))
        buf = torch.empty(2, 8, dtype=torch.bfloat16)
        buf.view(torch.int32).fill_(judge.SENTINEL)
        self.assertTrue(judge.rearmed("r", buf)["ok"])
        buf[1, 5] = 1.0
        self.assertFalse(judge.rearmed("r", buf)["ok"])


class Emulation(unittest.TestCase):
    def test_close_to_exact(self):
        for m in (1, 14):
            xs, res, gamma = inputs(m, m)
            normed, rout = judge.emulate_served(xs, res, gamma, 1e-5, served_warps_per_block(m))
            y = xs.double().sum(0) + res.double()
            torch.testing.assert_close(rout.double(), y, rtol=2e-2, atol=5e-2)
            ref = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + 1e-5) * gamma.double()
            torch.testing.assert_close(normed.double(), ref, rtol=2e-2, atol=5e-2)

    def test_residual_out_is_rounded_sum_plus_residual(self):
        xs, res, gamma = inputs(7, 3)
        _, rout = judge.emulate_served(xs, res, gamma, 1e-5, 7)
        acc = xs[0].float()
        for r in range(1, TP):
            acc = acc + xs[r].float()
        want = (acc.to(torch.bfloat16).float() + res.float()).to(torch.bfloat16)
        self.assertTrue(judge.compare("rout", rout, want)["ok"])

    def test_mutants_fail(self):
        """Each pre-registered mutant, applied to the emulation, must fail G1 against the true emulation."""
        m = 14
        xs, res, gamma = inputs(m, 5, wide=True)
        wb = served_warps_per_block(m)
        want = judge.emulate_served(xs, res, gamma, 1e-5, wb)
        mutants = {
            "order": judge.emulate_served(xs, res, gamma, 1e-5, wb, order=list(range(TP))[::-1]),
            "noresidual": judge.emulate_served(xs, torch.zeros_like(res), gamma, 1e-5, wb),
        }
        for name, (n, r) in mutants.items():
            checks = judge.judge_call(name, n[None], r[None], *want)
            self.assertFalse(judge.verdict(checks), name)
        self.assertTrue(judge.verdict(judge.judge_call("same", want[0][None], want[1][None], *want)))

    def test_layout_changes_some_bits(self):
        """The served block size (7 vs 14 warps) changes the sumsq tree; over 4 seeds at M 56 some normed bits move."""
        moved = 0
        for seed in range(4):
            xs, res, gamma = inputs(56, 100 + seed)
            a = judge.emulate_served(xs, res, gamma, 1e-5, 7)[0]
            b = judge.emulate_served(xs, res, gamma, 1e-5, 14)[0]
            moved += judge.compare("l", a, b)["mismatches"]
        self.assertGreater(moved, 0)

    def test_order_needs_wide_rows(self):
        """Measured property behind the salt design: plain randn rows give the same bits in either rank order."""
        xs, res, gamma = inputs(14, 5)
        a = judge.emulate_served(xs, res, gamma, 1e-5, 7)
        b = judge.emulate_served(xs, res, gamma, 1e-5, 7, order=list(range(TP))[::-1])
        self.assertTrue(judge.compare("o", a[1], b[1])["ok"])

    def test_g2_catches_copy_divergence(self):
        xs, res, gamma = inputs(1, 9)
        n, r = judge.emulate_served(xs, res, gamma, 1e-5, 7)
        nn, rr = n[None].repeat(8, 1, 1), r[None].repeat(8, 1, 1)
        nn.view(torch.int16)[5, 0, 7] ^= 1
        checks = judge.judge_call("g2", nn, rr, n, r)
        bad = [c["name"] for c in checks if not c["ok"]]
        self.assertEqual(bad, ["g2/G1/normed[5]", "g2/G2/normed[5]==[0]"])


if __name__ == "__main__":
    unittest.main()
