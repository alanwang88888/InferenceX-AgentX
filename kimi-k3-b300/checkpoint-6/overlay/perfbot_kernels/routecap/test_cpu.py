"""CPU tests of perfbot_kernels.routecap: off is a no-op, on files each call in its own slot by a device counter
(wrap, padding rows -1, nb, latent ring), record() without prepare() is a no-op, the snapshot file round-trips, and
the overlay prepared.py calls prepare() in __init__ and record() right after both front.route() calls. The torch tests skip without torch.

  python -I overlay/perfbot_kernels/routecap/test_cpu.py
"""

import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

OVERLAY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(OVERLAY))
PREPARED = OVERLAY / "vllm/models/kimi_k3/nvidia/specialized_decode/cutedsl/moe/prepared.py"

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def load(**env):
    keys = ("PERFBOT_ROUTECAP", "PERFBOT_ROUTECAP_OUT", "PERFBOT_ROUTECAP_STEPS", "PERFBOT_ROUTECAP_ACT_STEPS")
    old = {k: os.environ.pop(k, None) for k in keys}
    os.environ.update(env)
    try:
        import perfbot_kernels.routecap as rc
        return importlib.reload(rc)
    finally:
        for k in keys:
            os.environ.pop(k, None)
            if old[k] is not None:
                os.environ[k] = old[k]


class Placement(unittest.TestCase):
    def test_record_follows_each_route_call(self):
        lines = PREPARED.read_text().splitlines()
        routes = [i for i, l in enumerate(lines) if "= self.front.route(x)" in l]
        self.assertEqual(len(routes), 2)
        for i in routes:
            self.assertIn("routecap.record(self.layer_idx, ids, weights, latent)", lines[i + 1])

    def test_prepare_runs_in_init_before_any_route(self):
        lines = PREPARED.read_text().splitlines()
        cls = next(i for i, l in enumerate(lines) if l.startswith("class PreparedMoE"))
        init = next(i for i, l in enumerate(lines) if i > cls and "def __init__" in l)
        prep = [i for i, l in enumerate(lines) if "routecap.prepare(" in l]
        self.assertEqual(len(prep), 1)
        nxt = next(i for i, l in enumerate(lines) if i > init and l.lstrip().startswith("def "))
        self.assertTrue(init < prep[0] < nxt)


@unittest.skipIf(torch is None, "torch not installed")
class Record(unittest.TestCase):
    def step(self, b, seed):
        g = torch.Generator().manual_seed(seed)
        ids = torch.randint(0, 896, (b, 16), generator=g, dtype=torch.int32)
        return ids, torch.rand(b, 16, generator=g).to(torch.bfloat16), torch.rand(b, 8, generator=g).to(torch.bfloat16)

    def test_off_allocates_nothing(self):
        rc = load()
        rc.record(3, *self.step(4, 0))
        self.assertEqual(rc._S, {})

    def test_on_without_prepare_is_a_no_op(self):
        rc = load(PERFBOT_ROUTECAP="1")
        rc.record(3, *self.step(4, 0))
        self.assertEqual(rc._S, {})

    def test_on_files_each_step_in_its_slot_and_wraps(self):
        rc = load(PERFBOT_ROUTECAP="1", PERFBOT_ROUTECAP_STEPS="3", PERFBOT_ROUTECAP_ACT_STEPS="2")
        rc.prepare("cpu", 16, torch.bfloat16, 8, torch.bfloat16)
        steps = [self.step(b, s) for s, b in enumerate((4, 6, 8, 5))]
        for ids, w, lat in steps:
            for layer in (7, 2):  # first-seen order sets the row: layer 7 -> 0, layer 2 -> 1
                rc.record(layer, ids, w, lat)
        snap = rc.snapshot()
        self.assertEqual(snap["layers"], {7: 0, 2: 1})
        self.assertEqual(snap["cnt"][:3].tolist(), [4, 4, 0])
        # 4 steps in 3 slots: slot 0 holds step 3 (b 5), slots 1, 2 hold steps 1, 2.
        self.assertEqual(snap["nb"][0].tolist(), [5, 6, 8])
        for slot, s in ((0, 3), (1, 1), (2, 2)):
            ids, w, lat = steps[s]
            b = ids.shape[0]
            self.assertTrue(torch.equal(snap["ids"][1, slot, :b], ids))
            self.assertTrue(bool((snap["ids"][1, slot, b:] == -1).all()))
            self.assertTrue(torch.equal(snap["weights"][0, slot, :b], w))
        self.assertTrue(torch.equal(snap["latent"][0, 1, :5], steps[3][2]))  # step 3 -> act slot 1
        self.assertTrue(torch.equal(snap["latent"][0, 0, :8], steps[2][2]))  # step 2 -> act slot 0

    def test_drain_writes_the_snapshot_atomically(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "cap", "routecap_rank0")
            rc = load(PERFBOT_ROUTECAP="1", PERFBOT_ROUTECAP_OUT=out, PERFBOT_ROUTECAP_STEPS="2")
            rc.PERIOD_S = 3600  # set before prepare() starts the thread: it must not race the explicit drain
            rc.prepare("cpu", 16, torch.bfloat16, 8, torch.bfloat16)
            rc.record(0, *self.step(4, 1))
            rc.drain()
            snap = torch.load(out + ".pt")
            self.assertEqual(snap["cnt"][0].item(), 1)
            self.assertFalse(os.path.exists(out + ".tmp.pt"))
            rc._S["off"] = True  # the atexit drain must not recreate the deleted temp dir


if __name__ == "__main__":
    unittest.main()
