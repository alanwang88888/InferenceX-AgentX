"""CPU tests of the P-10 (iii-A) integration (perfbot_kernels.markov_sampler + the dspark speculator overlay edit):
switch off = image loop (run declines, warmup returns, nothing launched); the edit is lane P's PATCH.md lines plus the
warmup in capture(), placed where PATCH.md puts them. Bit-equality and timing are the kernel-only judge's (R-KO1c).

  python -I overlay/perfbot_kernels/markov_sampler/test_cpu.py
"""

import ast
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

OVERLAY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(OVERLAY))

import torch  # noqa: E402

from perfbot_kernels import markov_sampler as ms  # noqa: E402

SPEC = OVERLAY / "vllm/v1/worker/gpu/spec_decode/dspark/speculator.py"


def _method(tree, cls, name):
    c = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    return next((n for n in c.body if isinstance(n, ast.FunctionDef) and n.name == name), None)


class Switch(unittest.TestCase):
    def setUp(self):
        self._old = os.environ.pop("PERFBOT_EXP_MARKOV", None)

    def tearDown(self):
        if self._old is not None:
            os.environ["PERFBOT_EXP_MARKOV"] = self._old

    def test_off_by_default_run_declines_before_any_launch(self):
        spec = SimpleNamespace()  # no attribute is read when the switch is off
        self.assertEqual(ms._decline(spec, torch.zeros(1, 7, 8)), "switch off")
        real = torch.cuda.is_current_stream_capturing  # a CPU-only torch build has no capture query
        torch.cuda.is_current_stream_capturing = lambda: False
        try:
            self.assertIsNone(ms.run(spec, torch.zeros(1, 7, 8)))
        finally:
            torch.cuda.is_current_stream_capturing = real
        self.assertEqual(ms._WS, {})

    def test_off_warmup_returns_without_work(self):
        self.assertIsNone(ms.warmup(SimpleNamespace()))
        self.assertEqual(ms._WARM, set())

    def test_on_needs_exactly_one(self):
        for v in ("0", "true", "yes", ""):
            os.environ["PERFBOT_EXP_MARKOV"] = v
            self.assertEqual(ms._decline(SimpleNamespace(), torch.zeros(1, 7, 8)), "switch off", v)


class Edit(unittest.TestCase):
    tree = ast.parse(SPEC.read_text())

    def test_import(self):
        imports = [n for n in self.tree.body if isinstance(n, ast.ImportFrom) and n.module == "perfbot_kernels"]
        self.assertEqual([[a.name for a in n.names] for n in imports], [["markov_sampler"]])

    def test_fused_round_runs_right_after_sample_pos(self):
        """PATCH.md step 2: after `sample_pos = ...view(num_reqs, n_spec)`, return iff run() did the rounds."""
        body = _method(self.tree, "DSparkSpeculator", "_sample_sequential").body
        i = next(k for k, s in enumerate(body) if isinstance(s, ast.Assign)
                 and getattr(s.targets[0], "id", None) == "sample_pos")
        hook = body[i + 1]
        self.assertIsInstance(hook, ast.If)
        self.assertEqual(ast.unparse(hook.test), "markov_sampler.run(self, base_logits) is not None")
        self.assertEqual([ast.unparse(s) for s in hook.body], ["return"])
        self.assertEqual(hook.orelse, [])

    def test_warmup_precedes_the_draft_graph_capture(self):
        """PATCH.md step 3: warm once self.model is set (load_model ran) and before capture."""
        cap = _method(self.tree, "DSparkSpeculator", "capture")
        self.assertIsNotNone(cap)
        self.assertEqual([ast.unparse(s) for s in cap.body], ["markov_sampler.warmup(self)", "super().capture()"])


if __name__ == "__main__":
    unittest.main()
