"""CPU tests of perfbot_kernels.nullarm (CARD-LENS §8 null arm). No GPU.

  python3 -I overlay/perfbot_kernels/nullarm/test_cpu.py
"""

import io
import os
import statistics
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

HERE = Path(__file__).resolve().parent
OVERLAY = HERE.parents[1]
sys.path.insert(0, str(OVERLAY))

import perfbot_kernels.nullarm as na  # noqa: E402


class NullArm(unittest.TestCase):
    def test_fresh_cache_env_gives_every_cache_a_new_empty_directory(self):
        with tempfile.TemporaryDirectory() as root:
            a, b = na.fresh_cache_env({"PATH": "/bin"}, root), na.fresh_cache_env({"PATH": "/bin"}, root)
            for var in na.CACHE_VARS:
                self.assertNotEqual(a[var], b[var])
                self.assertEqual(os.listdir(a[var]), [])
            self.assertEqual(a["PATH"], "/bin")

    def test_spread_is_the_median_shift_per_row(self):  # r17 at 40 rows: +0.416 / -0.447 against L
        self.assertEqual(na.spread({40: [10.0, 10.2, 10.1], 32: [8.0]}, {40: [10.5, 10.6, 10.4]}),
                         {40: abs(10.5 - 10.1), 32: None})

    def test_aa_shift_is_a_median_shift_and_the_gate_takes_no_dispersion_band(self):  # ruling 2026-10-20, like with like
        blocks = [([10.0, 10.1, 10.2], [10.3, 10.1, 10.4]), ([10.0, 9.0, 11.0], [10.0, 9.0, 11.0])]  # wide but unshifted
        self.assertAlmostEqual(na.aa_shift(blocks), 0.2 + (10.15 - 10.0))  # max in-block shift + block-median range
        self.assertNotIn("aa_band_us", na.gate.__code__.co_varnames)
        with self.assertRaises(KeyError):  # a caller that passes a dispersion band by its old name fails loudly
            na.emit({40: dict(gain_us=1.0, aa_band_us=0.1, null_spread_us=0.4, bar_us=0.5)}, lambda s, flush: None)

    def test_sm_clock_is_the_median_sampled_during_the_row_and_emit_always_carries_it(self):  # item 03 load clock
        clocks = iter([1785, 1484, 1786, 1783] + [1785] * 1000)
        with na.SmClock(read=lambda: next(clocks), period_s=0.001) as c:
            while len(c.samples) < 3:
                pass
        self.assertEqual(c.samples[:4], [1785, 1484, 1786, 1783])  # one at enter, one per period, one at exit
        self.assertEqual(c.mhz, statistics.median(c.samples))
        lines = []
        rec = na.emit({40: dict(gain_us=1.0, aa_shift_us=0.1, null_spread_us=0.4, bar_us=0.5, inc_mhz=c.mhz),
                       32: dict(gain_us=1.0, aa_shift_us=0.1, null_spread_us=0.4, bar_us=0.5)},
                      lambda s, flush: lines.append(s))
        self.assertEqual((rec["rows"]["40"]["inc_mhz"], rec["rows"]["32"]["inc_mhz"]), (c.mhz, None))
        self.assertEqual(na.read(lines[0])["rows"]["40"]["verdict"], "BUILD")  # the clock changes no verdict
        with unittest.mock.patch.dict(sys.modules, {"pynvml": None}), na.SmClock() as c:  # no NVML
            pass
        self.assertIsNone(c.mhz)

    def test_nvml_index_follows_cuda_visible_devices(self):
        with unittest.mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,3"}):
            self.assertEqual(na.nvml_index(1), 3)
        with unittest.mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-ab12"}):
            self.assertEqual(na.nvml_index(0), 0)

    def test_gate_takes_the_in_run_null_spread_never_a_floor(self):
        self.assertEqual(na.gate(0.5, 0.13, None, 0.2), "INCOMPLETE")
        self.assertEqual(na.gate(0.5, 0.13, 0.86, 0.2), "BELOW_NOISE")  # passes the A/A band, not the null arm
        self.assertEqual(na.gate(0.5, 0.13, 0.10, 0.2), "BUILD")
        self.assertEqual(na.gate(0.5, 0.13, 0.10, 0.6), "BELOW_BAR")
        self.assertEqual(na.gate(0.13, 0.13, 0.0, 0.0), "BELOW_NOISE")  # equal to the band is not above it
        self.assertEqual(na.gate(-1.288, 0.266, 0.0, 0.266), "REGRESS")  # i04b23occ b2
        self.assertEqual(na.gate(-0.5, 0.13, 0.86, 0.2), "BELOW_NOISE")  # a loss inside the null arm
        self.assertEqual(na.gate(-0.13, 0.13, 0.0, 0.0), "BELOW_NOISE")  # equal to the band is not past it

    def test_emit_and_read_round_trip_and_complete_needs_every_row(self):
        out = io.StringIO()
        rec = na.emit({24: dict(gain_us=1.0, aa_shift_us=0.1, null_spread_us=0.4, bar_us=0.5),
                       40: dict(gain_us=0.3, aa_shift_us=0.1, null_spread_us=0.86, bar_us=0.5)},
                      lambda s, flush: out.write(s + "\n"))
        self.assertEqual((rec["null_spread_us"], rec["rows"]["24"]["verdict"], rec["rows"]["40"]["verdict"]),
                         (0.86, "BUILD", "BELOW_NOISE"))
        self.assertEqual(na.read("noise\n" + out.getvalue() + "tail\n"), rec)
        self.assertTrue(na.complete(rec))
        rec2 = na.emit({24: dict(gain_us=1.0, aa_shift_us=0.1, null_spread_us=None, bar_us=0.5)}, lambda s, flush: None)
        self.assertFalse(na.complete(rec2))
        self.assertFalse(na.complete(na.read("no line\n")))

    def test_cli_exit_78_without_a_complete_record(self):
        with tempfile.TemporaryDirectory() as d:
            log, js = Path(d) / "log", Path(d) / "rec.json"
            env = {**os.environ, "PYTHONPATH": str(OVERLAY)}
            log.write_text("judge ran, no null arm\n")
            p = subprocess.run([sys.executable, "-B", "-m", "perfbot_kernels.nullarm", str(log), str(js)],
                               capture_output=True, text=True, env=env)
            self.assertEqual((p.returncode, p.stdout.strip()), (78, "PERFBOT-VERDICT INCOMPLETE: no null arm"))
            out = io.StringIO()
            na.emit({40: dict(gain_us=1.0, aa_shift_us=0.1, null_spread_us=0.4, bar_us=0.5)},
                    lambda s, flush: out.write(s + "\n"))
            log.write_text(out.getvalue())
            p = subprocess.run([sys.executable, "-B", "-m", "perfbot_kernels.nullarm", str(log), str(js)],
                               capture_output=True, text=True, env=env)
            self.assertEqual((p.returncode, p.stdout.strip()), (0, "PERFBOT-VERDICT null_spread_us=0.400 40=BUILD"))
            self.assertEqual(na.read(out.getvalue()), __import__("json").loads(js.read_text()))


if __name__ == "__main__":
    unittest.main()
