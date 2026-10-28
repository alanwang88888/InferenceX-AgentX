# PERFBOT P-07F: CPU test of the form R judge and frozen case (stdlib only).
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import judge  # noqa: E402
import make_case  # noqa: E402

CASE = json.loads((HERE / "frozen_case.json").read_text())


def arm(case, bump=None):
    out = {}
    for ch in case["chains"]:
        for t in ch["turns"]:
            k = (ch["chain"], t["turn"])
            out[k] = dict(ids=[ch["chain"], t["turn"], 7], top0={"5": -0.25, "9": -1.5}, hit=t["s"], wall_s=1.0)
    if bump:
        out[bump]["top0"] = {"5": -0.2500001, "9": -1.5}
    return out


def on_log(case):
    e = make_case.expected_rows(case)
    return "noise\nPERFBOT-P07F rows_suffix=1 rows_full=0 tokens_skipped=60032\n" + \
        f"PERFBOT-P07F rows_suffix={e['rows_suffix']} rows_full={e['rows_full']} tokens_skipped={e['tokens_skipped']}\n"


class Case(unittest.TestCase):
    def test_frozen_case_is_a_fresh_build(self):
        self.assertEqual((HERE / "frozen_case.json").read_text(), json.dumps(make_case.build(), indent=1) + "\n")

    def test_bucket(self):
        warm = [t for ch in CASE["chains"] for t in ch["turns"] if t["warm"]]
        self.assertTrue(all(1_024 <= t["new"] <= 4_095 and t["P"] < 100_000 for t in warm), [t["new"] for t in warm])
        self.assertIn("penalty", {ch["kind"] for ch in CASE["chains"]})


class Judge(unittest.TestCase):
    def test_pass(self):
        v = judge.judge(CASE, arm(CASE), arm(CASE), on_log(CASE), "no lines")
        self.assertEqual(v["verdict"], "PASS", v)
        self.assertEqual(v["notes"], [])

    def test_one_logprob_ulp_fails(self):
        v = judge.judge(CASE, arm(CASE), arm(CASE, bump=(2, 1)), on_log(CASE), "")
        self.assertEqual(v["gates"]["G1"], "FAIL")

    def test_nondeterministic_off_is_incomplete(self):
        v = judge.judge(CASE, arm(CASE), arm(CASE, bump=(0, 3)), on_log(CASE), "", off2=arm(CASE, bump=(0, 3)))
        self.assertEqual(v["verdict"], "INCOMPLETE", v)

    def test_never_fired_fails(self):
        self.assertEqual(judge.judge(CASE, arm(CASE), arm(CASE), "", "")["gates"]["G2"], "FAIL")
        v = judge.judge(CASE, arm(CASE), arm(CASE), "PERFBOT-P07F rows_suffix=0 rows_full=15 tokens_skipped=0\n", "")
        self.assertEqual(v["gates"]["G2"], "FAIL")

    def test_switch_leak_into_off_fails(self):
        self.assertEqual(judge.judge(CASE, arm(CASE), arm(CASE), on_log(CASE), on_log(CASE))["gates"]["G2"], "FAIL")

    def test_fallback_rows_missing_fails(self):
        v = judge.judge(CASE, arm(CASE), arm(CASE), "PERFBOT-P07F rows_suffix=10 rows_full=1 tokens_skipped=900000\n", "")
        self.assertEqual(v["gates"]["G3"], "FAIL")

    def test_missing_request_is_incomplete(self):
        on = arm(CASE)
        on.pop((1, 2))
        self.assertEqual(judge.judge(CASE, arm(CASE), on, on_log(CASE), "")["verdict"], "INCOMPLETE")


if __name__ == "__main__":
    unittest.main()
