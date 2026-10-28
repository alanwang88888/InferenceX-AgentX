# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the selftest runner (stdlib only; torch stubbed). python3 -B -m unittest test_cpu (in this dir)."""
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.modules.setdefault("torch", types.ModuleType("torch"))
_spec = importlib.util.spec_from_file_location("i15_run", Path(__file__).with_name("run.py"))
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)


class Runner(unittest.TestCase):
    def test_base_equals_the_manifest(self):
        m = json.loads((run.OVERLAY / "vllm_base.json").read_text())
        want = {r: m["base_sha256"][r] for r, e in m["intentional_edits"].items() if e["tag"] == run.TAG}
        self.assertEqual(run.BASE, want)

    def test_overlay_passes_and_a_stock_edit_fails(self):
        self.assertEqual(run.overlay_check()[0], 0)
        with tempfile.TemporaryDirectory() as d:
            for rel in run.BASE:
                p = Path(d) / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes((run.OVERLAY / rel).read_bytes().replace(b"import torch\n", b"import torch  \n", 1))
            self.assertEqual(run.overlay_check(Path(d))[0], 1)

    def test_untagged_copy_fails(self):  # a base copy without the P-15 lines is not this item's overlay
        with tempfile.TemporaryDirectory() as d:
            for rel in run.BASE:
                p = Path(d) / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                src = (run.OVERLAY / rel).read_bytes().splitlines(keepends=True)
                p.write_bytes(b"".join(l for l in src if run.TAG.encode() not in l))
            self.assertEqual(run.overlay_check(Path(d))[0], 1)

    def test_gate_verdict(self):
        ok = 'SUMMARY {"passed": true, "mutants_caught": true}'
        self.assertEqual(run.gate_verdict(0, ok)[0], 0)
        self.assertEqual(run.gate_verdict(1, 'SUMMARY {"passed": false, "mutants_caught": true}')[0], 1)
        self.assertEqual(run.gate_verdict(0, 'SUMMARY {"passed": true, "mutants_caught": false}')[0], 1)
        self.assertEqual(run.gate_verdict(1, "Traceback"), (2, None))
        self.assertEqual(run.gate_verdict(None, ok), (2, None))
        self.assertEqual(run.gate_verdict(-9, ok), (2, None))


if __name__ == "__main__":
    unittest.main()
