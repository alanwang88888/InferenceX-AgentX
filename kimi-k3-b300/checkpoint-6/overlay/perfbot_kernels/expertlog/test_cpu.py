"""CPU tests of perfbot_kernels.expertlog: the switch-off apply() is the r3-318 one (static, AST), the device record on
CPU tensors (distinct counts, padding mask, logits path, inputs untouched, errors never escape), the host period
bookkeeping and the summary line. No vllm; the torch tests skip without torch.

  python -I overlay/perfbot_kernels/expertlog/test_cpu.py
"""

import ast
import contextlib
import io
import os
import sys
import types
import unittest
from pathlib import Path

OVERLAY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(OVERLAY))

import perfbot_kernels.expertlog as el  # noqa: E402

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

MOE = OVERLAY / "vllm/model_executor/layers/fused_moe/experts/trtllm_mxfp4_moe.py"
FLAG = "_PERFBOT_EXPERTLOG"


def kv(line):
    return dict(f.split("=", 1) for f in line.split()[1:])


class SwitchOffStatic(unittest.TestCase):
    """Off = no op added: the flag is a module constant False unless the env says "1", and every reference to the
    log sits in one `if FLAG:` (no else) whose body is the import and one discarded record() call."""

    def setUp(self):
        self.tree = ast.parse(MOE.read_text())

    def test_flag_is_env_constant_default_off(self):
        assigns = [n for n in self.tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == FLAG for t in n.targets)]
        self.assertEqual(len(assigns), 1)
        expr = compile(ast.Expression(assigns[0].value), "flag", "eval")
        for env, want in (({}, False), ({el.ENV: "0"}, False), ({el.ENV: ""}, False), ({el.ENV: "1"}, True)):
            self.assertIs(eval(expr, {"os": types.SimpleNamespace(environ=env)}), want, env)
        stores = [n for n in ast.walk(self.tree) if isinstance(n, ast.Name) and n.id == FLAG
                  and isinstance(n.ctx, ast.Store)]
        self.assertEqual(len(stores), 1)  # never reassigned

    def test_only_reference_is_one_guarded_record(self):
        guards = [n for n in ast.walk(self.tree) if isinstance(n, ast.If) and isinstance(n.test, ast.Name)
                  and n.test.id == FLAG]
        self.assertEqual(len(guards), 1)
        g = guards[0]
        self.assertEqual(g.orelse, [])
        imp, call = g.body
        self.assertIsInstance(imp, ast.ImportFrom)
        self.assertEqual((imp.module, [a.name for a in imp.names], imp.names[0].asname),
                         ("perfbot_kernels", ["expertlog"], None))
        self.assertIsInstance(call, ast.Expr)  # result discarded
        f = call.value.func
        self.assertEqual((f.value.id, f.attr), ("expertlog", "record"))
        for a in call.value.args:  # plain loads: evaluating the arguments has no effect
            self.assertIsInstance(a, (ast.Name, ast.Attribute))
        self.assertEqual(call.value.keywords, [])
        inside = {id(n) for n in ast.walk(g)}
        stray = [n.lineno for n in ast.walk(self.tree) if id(n) not in inside and (
            (isinstance(n, ast.Name) and n.id in ("expertlog", FLAG) and isinstance(n.ctx, ast.Load))
            or (isinstance(n, ast.ImportFrom) and "expertlog" in (n.module or "") + str([a.name for a in n.names])))]
        self.assertEqual(stray, [])
        apply = [n for c in self.tree.body if isinstance(c, ast.ClassDef) and c.name == "TrtLlmMxfp4ExpertsMonolithic"
                 for n in c.body if isinstance(n, ast.FunctionDef) and n.name == "apply"]
        self.assertEqual(len(apply), 1)
        self.assertIn(id(g), {id(n) for n in ast.walk(apply[0])})


class FakeState:
    def __init__(self, ring, rows):
        self.ring, self.top_k, self.experts, self.rows, self.steps, self.reads = ring, 16, 896, rows, [], []

    def set_step(self, slot, live):
        self.steps.append((slot, live))

    def read(self, n, layers):
        self.reads.append((n, layers))
        return [self.rows[slot] for slot, _ in self.steps[-n:]]


class HostBookkeeping(unittest.TestCase):
    def setUp(self):
        self.saved = dict(el._LAYERS)
        el._LAYERS.clear()
        el._LAYERS.update({1: 0, 2: 1, 3: 2})

    def tearDown(self):
        el._LAYERS.clear()
        el._LAYERS.update(self.saved)

    def test_periods_and_lines(self):
        out = []
        h = el.Host(7, out=out.append)
        st = FakeState(3, rows=[[100, 110, 120], [700, 710, -1], [90, 100, 110]])
        h.on_step({"a": 8}, st)  # decode, batch 1
        h.on_step({}, st)  # empty: no forward, no slot
        h.on_step({"a": 8, "b": 500}, st)  # mixed: b past 1 + K
        h.on_step({"a": 6}, st)  # decode, batch 1 (fewer drafts)
        self.assertEqual(out, [])
        h.on_step({"a": 8}, st)  # period full: flush first, then slot 0 again
        self.assertEqual(st.steps, [(0, 8), (1, 508), (2, 6), (0, 8)])
        self.assertEqual(st.reads, [(3, 3)])
        self.assertEqual(len(out), 2)
        dec, mix = (kv(x) for x in out)
        self.assertEqual((dec["kind"], dec["batch"], dec["steps"], dec["tokens"], dec["slots"], dec["layers"]),
                         ("decode", "1", "2", "7", "112", "3"))
        self.assertEqual((dec["distinct_min"], dec["distinct_max"], dec["distinct_p50"], dec["missing"]),
                         ("90", "120", "100", "0"))
        self.assertEqual((dec["distinct_mean"], dec["layer_mean_min"], dec["layer_mean_max"]),
                         ("105.00", "95.00", "115.00"))
        self.assertEqual((mix["kind"], mix["batch"], mix["tokens"], mix["distinct_mean"], mix["missing"]),
                         ("mixed", "2", "508", "705.00", "1"))

    def test_error_disables_never_raises(self):
        out = []
        h = el.Host(7, out=out.append)
        h.on_step({"a": 8}, object())  # no ring attribute
        self.assertFalse(h.on)
        self.assertIn("disabled", out[0])
        h.on_step({"a": 8}, object())
        self.assertEqual(len(out), 1)

    def test_step_kind_and_pct(self):
        self.assertIsNone(el.step_kind({}, 7))
        self.assertEqual(el.step_kind({"a": 8, "b": 1}, 7), "decode")
        self.assertEqual(el.step_kind({"a": 9}, 7), "mixed")
        v = list(range(1, 11))
        self.assertEqual((el._pct(v, 10), el._pct(v, 50), el._pct(v, 90), el._pct([5], 90)), (1, 5, 9, 5))


@unittest.skipIf(torch is None, "no torch")
class DeviceRecordOnCpu(unittest.TestCase):
    E, K = 896, 16

    def setUp(self):
        el._STATE.clear()
        el._LAYERS.clear()
        os.environ[el.STEPS_ENV] = "2"

    def tearDown(self):
        el._STATE.clear()
        el._LAYERS.clear()
        os.environ.pop(el.STEPS_ENV, None)

    def carrier(self, ids, w=0x3F80):
        m = ids.shape[0]
        packed = (ids.int() << 16) | w
        return packed, torch.randn(m, self.E)

    def test_counts_mask_and_inputs_untouched(self):
        g = torch.Generator().manual_seed(3)
        ids = torch.stack([torch.randperm(self.E, generator=g)[: self.K] for _ in range(8)])
        ids[1] = ids[0]  # duplicate rows share experts
        packed, logits = self.carrier(ids)
        pk0, lg0 = packed.clone(), logits.clone()
        owners = [object(), object()]
        el.record(owners[0], packed, logits, None, self.K)  # first call allocates; ctl at scratch row
        st = el.state(None)
        self.assertEqual((st.ring, st.experts, int(st.ctl[0])), (2, self.E, 2))
        st.set_step(0, 8)
        el.record(owners[0], packed, logits, None, self.K)
        st.set_step(1, 3)  # rows 3.. are padding
        el.record(owners[0], packed, logits, None, self.K)
        el.record(owners[1], packed[:2], logits[:2], None, self.K)
        self.assertTrue(torch.equal(packed, pk0) and torch.equal(logits, lg0))
        want = lambda rows: len(set(ids[:rows].reshape(-1).tolist()))
        self.assertEqual(st.read(2, 2), [[want(8), -1], [want(3), want(2)]])
        self.assertEqual(int(st.rec[:2].max()), -1)  # reset after the read

    def test_logits_path_uses_sigmoid_plus_bias(self):
        m = 300  # > router_topk.MAX_M: no packed top-K
        logits = torch.full((m, self.E), -4.0)
        bias = torch.zeros(self.E)
        logits[0::2, 16:32] = -3.0  # by logit alone even rows pick 16..31, odd rows 32..47: 32 distinct
        logits[1::2, 32:48] = -3.0
        bias[: self.K] = 0.5  # with the bias every row picks 0..15 (sigmoid(-4) + 0.5 > sigmoid(-3)) ...
        logits[0, 100: 100 + self.K] = 8.0  # ... except row 0, which picks 100..115 (sigmoid(8) = 0.9997)
        owner = object()
        el.record(owner, None, logits, bias, self.K)
        st = el.state(None)
        st.set_step(0, m)
        el.record(owner, None, logits, bias, self.K)
        self.assertEqual(st.read(1, 1), [[2 * self.K]])
        st.set_step(0, m)
        el.record(owner, None, logits, None, self.K)
        self.assertEqual(st.read(1, 1), [[3 * self.K]])  # row 0's 100..115, 16..31, 32..47

    def test_record_never_raises(self):
        with contextlib.redirect_stderr(io.StringIO()) as buf:
            el.record(object(), None, torch.zeros(4), None, self.K)  # 1-D logits
        self.assertIn("record skipped", buf.getvalue())

    def test_host_end_to_end(self):
        out = []
        h = el.Host(7, out=out.append)
        ids = torch.arange(8 * self.K).view(8, self.K) % 40  # 40 distinct over 8 rows
        packed, logits = self.carrier(ids)
        layers = [object() for _ in range(3)]
        for x in layers:
            el.record(x, packed, logits, None, self.K)  # profile run
        st = el.state(None)
        for _ in range(3):
            h.on_step({"a": 8}, st)
            for x in layers:
                el.record(x, packed, logits, None, self.K)
        self.assertEqual(len(out), 1)
        d = kv(out[0])
        self.assertEqual((d["steps"], d["layers"], d["distinct_mean"], d["distinct_p90"], d["missing"], d["slots"]),
                         ("2", "3", "40.00", "40", "0", "128"))


if __name__ == "__main__":
    unittest.main()
