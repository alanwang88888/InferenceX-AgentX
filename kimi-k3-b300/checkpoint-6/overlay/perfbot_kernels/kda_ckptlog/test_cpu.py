"""CPU tests of the PERFBOT-KDA-CKPT evidence log (switch PERFBOT_EXP_KDA_CKPT_LOG, P-07 risk c).

Needs vLLM 49cb2bbf importable (source tree on PYTHONPATH, CPU torch). The parent runs the state-free cases and starts
one child process per switch state. Each child installs the overlay finder (kda_ckpt/run_vllm_suite.install), so the
switch is read at import as in a served arm. The child replaces store_cache_checkpoints_kernel with a CPU copy of what
the Triton kernel stores, then drives the two real hook sites: MambaPrefillCheckpointExporter.export (checkpoint.py)
and MambaHybridModelState.preprocess_state (mamba_hybrid.py).

  cd <vLLM 49cb2bbf tree>; PYTHONPATH=. python <overlay>/perfbot_kernels/kda_ckptlog/test_cpu.py
"""

import contextlib
import io
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
OVERLAY = HERE.parents[1]
sys.path.insert(0, str(OVERLAY))
ENV = "PERFBOT_EXP_KDA_CKPT_LOG"
LOG_TAG = "# PERFBOT P-07 ckptlog"
LOG_LINES = {"vllm/model_executor/layers/mamba/checkpoint.py": 8,
             "vllm/v1/worker/gpu/model_states/mamba_hybrid.py": 2}
STATES = {"off": ({}, False), "on": ({ENV: "1"}, True), "not_one": ({ENV: "true"}, False)}
L, W, H, D, NB = 4, 6, 2, 4, 12  # conv state length, qkv width, heads, head dim, cache blocks


class Free(unittest.TestCase):
    """State-free: the digest and the tagged hook lines."""

    def test_digest_equal_bytes_equal_one_changed_byte_differs(self):
        import torch
        from perfbot_kernels.kda_ckptlog import digests

        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            x = torch.randn(3, 5, 7).to(dtype)
            d = digests(x)
            self.assertTrue(torch.equal(d, digests(x.clone())))
            self.assertTrue(torch.equal(digests(x.transpose(1, 2).contiguous().transpose(1, 2)), d))
            nbytes = x[0].numel() * x.element_size()
            for pos in range(nbytes):
                y = x.clone()
                y[1].view(-1).view(torch.uint8)[pos] ^= 1
                e = digests(y)
                self.assertEqual(e[0].item(), d[0].item())
                self.assertNotEqual(e[1].item(), d[1].item(), (dtype, pos))

    def test_hook_lines_are_tagged_and_guarded(self):
        for rel, n in LOG_LINES.items():
            lines = (OVERLAY / rel).read_text().splitlines()
            self.assertEqual(sum(LOG_TAG in line for line in lines), n, rel)
            for line in lines:
                if "PERFBOT_P07_CKPTLOG" in line or "kda_ckptlog" in line or ENV in line:
                    self.assertIn(LOG_TAG, line)
        src = (OVERLAY / "vllm/v1/worker/gpu/model_states/mamba_hybrid.py").read_text()
        body = src.split("def preprocess_state(", 1)[1].split('"""', 2)[2].lstrip().splitlines()
        self.assertIn("PERFBOT_P07_CKPTLOG is not None", body[0])  # first statement: before the pre-copy
        src = (OVERLAY / "vllm/model_executor/layers/mamba/checkpoint.py").read_text()
        export = src.split("    def export(", 1)[1].split("@triton.jit", 1)[0]
        self.assertLess(export.index("store_cache_checkpoints_kernel["), export.index("PERFBOT_P07_CKPTLOG.on_export"))

    def test_every_switch_state(self):
        for state, (env, _) in STATES.items():
            with self.subTest(state=state):
                child_env = {k: v for k, v in os.environ.items() if k != ENV}
                child_env.update(env, CKPTLOG_CHILD_STATE=state)
                run = subprocess.run([sys.executable, __file__], env=child_env, capture_output=True, text=True,
                                     timeout=600)
                out = run.stdout + run.stderr
                print(f"--- child {state}\n{out.strip()}", flush=True)
                self.assertEqual(run.returncode, 0, out[-3000:])
                self.assertRegex(out, r"(?m)^OK$")


# ---------------------------------------------------------------- child (one switch state, overlay installed)


def fake_kernel(grid):
    """CPU copy of store_cache_checkpoints_kernel's stores (checkpoint.py:226-291), by its positional arguments."""
    def run(x, conv, rec_ckpt, rec, qsl, offs, sidx, rrows, *a):
        L_, W_, R, null, has_rows = a[-6], a[-5], a[-4], a[-3], a[-2]
        for i in range(offs.numel()):
            s, o = int(sidx[i]), int(offs[i])
            if s == null or o <= 0:
                continue
            row = int(rrows[i]) if has_rows else i
            rec[s].view(-1)[:R] = rec_ckpt[row].reshape(-1)[:R]
            if o >= L_:
                end = int(qsl[i]) + o
                conv[s][:W_, :L_] = x[end - L_:end].T
    return run


class Child(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(HERE.parent / "kda_ckpt"))
        import run_vllm_suite

        run_vllm_suite.install()
        import torch
        import vllm.model_executor.layers.mamba.checkpoint as ckpt
        import vllm.v1.worker.gpu.model_states.mamba_hybrid as mh

        cls.torch, cls.ckpt, cls.mh = torch, ckpt, mh
        cls.on = STATES[os.environ["CKPTLOG_CHILD_STATE"]][1]
        ckpt.store_cache_checkpoints_kernel = type("K", (), {"__getitem__": lambda self, g: fake_kernel(g)})()
        cls.state = object.__new__(mh.MambaHybridModelState)
        cls.state._align_mode = False  # preprocess_state returns right after the hook

    def setUp(self):
        torch = self.torch
        if self.on:  # new cache tensors per case: watch the first layer seen again
            self.ckpt.PERFBOT_P07_CKPTLOG.LOG.layer = None
        g = torch.Generator().manual_seed(0)
        self.rec = torch.zeros(NB, H, D, D)
        self.conv = torch.zeros(NB, L, W).transpose(-1, -2)  # the kda.py view: (block, width, state_len)
        self.x = torch.randn(40, W, generator=g)
        self.ws = torch.randn(2, H, D, D, generator=g)  # FlashKDA workspace rows

    def export(self, blocks, offsets, cols, rows=(0, 1)):
        torch = self.torch
        ck = self.ckpt.MambaPrefillCheckpointMetadata(
            torch.tensor(offsets, dtype=torch.int32), torch.tensor(blocks, dtype=torch.int64),
            offsets=list(offsets), request_rows=torch.tensor(rows), block_cols=torch.tensor(cols))
        return self.ckpt.MambaPrefillCheckpointExporter().export(
            ck, raw_qkv=self.x, conv_state=self.conv, recurrent_checkpoint=self.ws, recurrent_state=self.rec,
            cu_seqlens=torch.tensor([0, 20, 40], dtype=torch.int32))

    def step(self, req_ids, table):
        batch = SimpleNamespace(req_ids=list(req_ids), num_reqs=len(req_ids))
        bt = self.torch.tensor(table, dtype=self.torch.int32)
        return self.state.preprocess_state(batch, (bt,), None, None)

    def run_out(self, fn):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            result = fn()
        return result, out.getvalue()

    def test_switch_and_engine_result(self):
        log = self.ckpt.PERFBOT_P07_CKPTLOG
        self.assertEqual(log is not None, self.on)
        self.assertEqual("perfbot_kernels.kda_ckptlog" in sys.modules, self.on)
        res, out = self.run_out(lambda: (self.step(["a", "b"], [[1, 5], [2, 7]]),
                                         self.export([5, 0], [8, 3], [1, -1]), self.step(["a"], [[1, 5]])))
        self.assertEqual(res, (None, None, None))
        self.assertTrue(torch_equal(self.rec[5], self.ws[0]))  # the export stored the state, log on or off
        if not self.on:
            self.assertEqual(out, "")
            return
        lines = re.findall(r"PERFBOT-KDA-CKPT export=\S+ req=a row=0 block=5 src=(\w+) dst=(\w+) csrc=(\w+) "
                           r"cdst=(\w+) eq=1", out)
        self.assertEqual(len(lines), 1, out)  # row 1 has the null block: not an export
        src, dst, csrc, cdst = lines[0]
        self.assertEqual((src, csrc), (dst, cdst))
        self.assertNotIn("-", (src, csrc))

    def test_log_cases(self):
        if not self.on:
            return
        log = self.ckpt.PERFBOT_P07_CKPTLOG
        log.EVERY = 2
        base = (log.LOG.exports, log.LOG.checked, log.LOG.mismatch, log.LOG.unchecked)
        _, out = self.run_out(lambda: self.step(["a", "b"], [[1, 5], [2, 7]]))
        # 1. An overwrite between the export and the next step: eq=0.
        _, out = self.run_out(lambda: (self.export([5, 7], [8, 2], [1, 1]),
                                       self.rec[5].view(-1).view(self.torch.uint8)[3].__ixor__(1),
                                       self.step(["a", "b"], [[1, 5], [2, 7]])))
        self.assertRegex(out, r"req=a row=0 block=5 .* eq=0")
        self.assertRegex(out, r"req=b row=1 block=7 .* csrc=- cdst=- eq=1")  # offset 2 < L: no conv store
        # 2. A conv overwrite alone: eq=0, recurrent digests equal.
        _, out = self.run_out(lambda: (self.export([5, 0], [8, 0], [1, -1]), self.conv[5].__setitem__((0, 0), 9.0),
                                       self.step(["a"], [[1, 5]])))
        m = re.search(r"block=5 src=(\w+) dst=(\w+) csrc=(\w+) cdst=(\w+) eq=0", out)
        self.assertIsNotNone(m, out)
        self.assertEqual(m[1], m[2])
        self.assertNotEqual(m[3], m[4])
        # 3. Unchecked: the request left the batch; another row holds the block; the column changed;
        #    a second export before a read-back. None of them is a mismatch.
        for req_ids, table in ((["b"], [[1, 7]]), (["a", "c"], [[1, 5], [5, 9]]), (["a"], [[1, 6]])):
            _, out = self.run_out(lambda: (self.export([5, 0], [8, 0], [1, -1]), self.step(req_ids, table)))
            self.assertRegex(out, r"block=5 .* eq=u", (req_ids, table))
        _, out = self.run_out(lambda: (self.export([5, 0], [8, 0], [1, -1]), self.export([5, 0], [8, 0], [1, -1]),
                                       self.step(["a"], [[1, 5]])))
        self.assertEqual(re.findall(r" eq=(\w)", out), ["u", "1"])
        self.assertEqual((log.LOG.exports, log.LOG.checked, log.LOG.mismatch, log.LOG.unchecked),
                         (base[0] + 8, base[1] + 4, base[2] + 2, base[3] + 4))
        self.assertRegex(out, rf"PERFBOT-KDA-CKPT-SUM exports={base[0] + 8} checked={base[1] + 4} "
                              rf"mismatch={base[2] + 2} unchecked={base[3] + 4}")
        # 4. Capture: nothing during a breakable capture.
        from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
        BreakableCUDAGraphCapture._tls.active = object()
        try:
            _, out = self.run_out(lambda: self.export([5, 0], [8, 0], [1, -1]))
        finally:
            BreakableCUDAGraphCapture._tls.active = None
        self.assertEqual((out, log.LOG.pending, log.LOG.exports), ("", [], base[0] + 8))

    def test_zz_error_disables_once(self):
        if not self.on:
            return
        log = self.ckpt.PERFBOT_P07_CKPTLOG
        good = log.digests
        log.digests = lambda rows: 1 / 0
        try:
            res, out = self.run_out(lambda: (self.export([5, 0], [8, 0], [1, -1]), self.step(["a"], [[1, 5]]),
                                             self.export([5, 0], [8, 0], [1, -1]), self.step(["a"], [[1, 5]])))
        finally:
            log.digests = good
        self.assertEqual(res, (None, None, None, None))
        self.assertTrue(torch_equal(self.rec[5], self.ws[0]))  # the engine's export still ran
        self.assertEqual(out.count("PERFBOT-KDA-CKPT disabled: ZeroDivisionError"), 1, out)
        self.assertEqual(out.count("PERFBOT-KDA-CKPT"), 1, out)
        self.assertFalse(log.LOG.on)


def torch_equal(a, b):
    import torch

    return torch.equal(a, b)


def load_tests(loader, tests, pattern):
    cases = (Child,) if os.environ.get("CKPTLOG_CHILD_STATE") else (Free,)
    return unittest.TestSuite(loader.loadTestsFromTestCase(c) for c in cases)


if __name__ == "__main__":
    unittest.main()
