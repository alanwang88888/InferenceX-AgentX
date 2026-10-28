"""CPU tests of P-07 form K in the overlay (switches PERFBOT_EXP_KDA_HASHFIX and PERFBOT_EXP_KDA_CKPT).

Needs vLLM 49cb2bbf importable (source tree on PYTHONPATH, CPU torch). The parent run starts one child process per
switch state. Each child installs the overlay finder (run_vllm_suite.install), so the switches are read at import as
in a served arm, and checks the truth table of its state:
- skip on iff HASHFIX=1 or CKPT=1; parts (1)+(2) on iff CKPT=1; a value other than "1" is off;
- parts (1)+(2) off: the predicate, the worker checkpoints and the scheduler split equal the 49cb2bbf functions
  on a grid (the 49cb2bbf function = the overlay copy minus its tagged lines, as test_overlay_base proves);
- parts (1)+(2) on: they differ from 49cb2bbf only on the col == init, mid-block shape, and only for a fresh row;
  the worker reads the fresh rows that the V2 model state sets.
Cache behaviour (wrong entries, wrong restores, steps per turn) is the vLLM suite's:
tests/v1/core/prefix_cache/test_mamba_checkpoint_at_initial_state.py (run_vllm_suite.py runs it on the overlay).

  cd <vLLM 49cb2bbf tree>; PYTHONPATH=. python <overlay>/perfbot_kernels/kda_ckpt/test_cpu.py
"""

import ast
import json
import os
import random
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
OVERLAY = HERE.parents[1]
TAG = "# PERFBOT P-07"
HASHFIX, CKPT = "PERFBOT_EXP_KDA_HASHFIX", "PERFBOT_EXP_KDA_CKPT"
# state -> (environment, (parts (1)+(2) on, skip on))
STATES = {
    "off": ({}, (False, False)),
    "hashfix": ({HASHFIX: "1"}, (False, True)),
    "ckpt": ({CKPT: "1"}, (True, True)),
    "both": ({HASHFIX: "1", CKPT: "1"}, (True, True)),
    "not_one": ({HASHFIX: "true", CKPT: "yes"}, (False, False)),
}
# Tagged (added) lines per overlay file. A change here is a reviewed change of the edit.
P07_LINES = {
    "vllm/v1/kv_cache_interface.py": 8,
    "vllm/v1/core/sched/scheduler.py": 2,
    "vllm/v1/core/single_type_kv_cache_manager.py": 12,
    "vllm/model_executor/layers/mamba/checkpoint.py": 14,  # 6 form K + 8 kda_ckptlog
    "vllm/v1/worker/gpu/model_states/mamba_hybrid.py": 10,  # 8 form K + 2 kda_ckptlog
}


class Edit(unittest.TestCase):
    """State-free: the five files are listed P-07 edits with the pinned number of tagged lines."""

    def test_manifest_and_tagged_line_counts(self):
        manifest = json.loads((OVERLAY / "vllm_base.json").read_text())
        p07 = {rel for rel, e in manifest["intentional_edits"].items() if e["tag"] == TAG}
        self.assertEqual(p07, set(P07_LINES))
        for rel, n in P07_LINES.items():
            lines = (OVERLAY / rel).read_text().splitlines()
            self.assertEqual(sum(TAG in line for line in lines), n, rel)


class States(unittest.TestCase):
    """One child per state; each must end with unittest's OK line."""

    def test_every_switch_state(self):
        for state, (env, _) in STATES.items():
            with self.subTest(state=state):
                child_env = {k: v for k, v in os.environ.items() if k not in (HASHFIX, CKPT)}
                child_env.update(env, P07_CHILD_STATE=state)
                run = subprocess.run([sys.executable, __file__], env=child_env, capture_output=True, text=True,
                                     timeout=600)
                out = run.stdout + run.stderr
                print(f"--- child {state}\n{out.strip()}", flush=True)
                self.assertEqual(run.returncode, 0, out[-3000:])
                self.assertRegex(out, r"(?m)^OK$")


# ---------------------------------------------------------------- child (one switch state, overlay installed)


def _base_functions(rel, names, module, cls=None):
    """The 49cb2bbf definitions of `names`: the overlay file minus its tagged lines, run in the module's globals."""
    src = "".join(line for line in (OVERLAY / rel).read_text().splitlines(True) if TAG not in line)
    body = ast.parse(src).body
    if cls is not None:
        body = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == cls).body
    defs = [n for n in body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert sorted(d.name for d in defs) == sorted(names), names
    ns = dict(vars(module))
    exec(compile(ast.Module(body=defs, type_ignores=[]), f"<49cb2bbf {rel}>", "exec"), ns)
    return ns


def _mid_block_eq(start, end, block):
    """Form K's new shape: the checkpoint column is the initial-state column, the start is not block-aligned."""
    return start % block != 0 and -(-end // block) - 2 == (start - 1) // block


class Child(unittest.TestCase):
    """Runs only in a child process, after the overlay finder is installed."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(HERE))
        import run_vllm_suite

        run_vllm_suite.install()
        import vllm.model_executor.layers.mamba.checkpoint as ckpt
        import vllm.v1.core.sched.scheduler as sched
        import vllm.v1.core.single_type_kv_cache_manager as stm
        import vllm.v1.kv_cache_interface as kci
        import vllm.v1.worker.gpu.model_states.mamba_hybrid as mh

        cls.ckpt, cls.sched, cls.stm, cls.kci, cls.mh = ckpt, sched, stm, kci, mh
        cls.state = os.environ["P07_CHILD_STATE"]
        cls.on, cls.skip = STATES[cls.state][1]
        base = _base_functions("vllm/v1/kv_cache_interface.py", ["is_mamba_prefill_checkpoint_valid"], kci)
        cls.base_valid = staticmethod(base["is_mamba_prefill_checkpoint_valid"])
        ns = _base_functions("vllm/model_executor/layers/mamba/checkpoint.py",
                             ["compute_mamba_prefill_checkpoints"], ckpt)
        ns["is_mamba_prefill_checkpoint_valid"] = base["is_mamba_prefill_checkpoint_valid"]
        cls.base_compute = staticmethod(ns["compute_mamba_prefill_checkpoints"])
        ns = _base_functions("vllm/v1/core/sched/scheduler.py", ["_mamba_block_aligned_split"], sched, "Scheduler")
        ns["is_mamba_prefill_checkpoint_valid"] = base["is_mamba_prefill_checkpoint_valid"]
        cls.base_split = staticmethod(ns["_mamba_block_aligned_split"])

    def test_switch_truth_table(self):
        self.assertEqual((self.kci.PERFBOT_P07_CKPT, self.kci.PERFBOT_P07_SKIP), (self.on, self.skip))
        self.assertIs(self.stm.PERFBOT_P07_SKIP, self.kci.PERFBOT_P07_SKIP)  # the manager's skip switch
        self.assertIs(self.mh.PERFBOT_P07_CKPT, self.kci.PERFBOT_P07_CKPT)  # the worker flag source

    def test_predicate_differs_from_base_only_on_the_new_shape(self):
        rng = random.Random(0)
        get_pos = self.kci.get_mamba_prefill_checkpoint_position
        new_shape = 0
        for block, unit in ((64, 16), (1536, 128)):
            for _ in range(20000):
                start = rng.randrange(0, 6 * block) // rng.choice((1, unit)) * rng.choice((1, unit))
                end = start + rng.randrange(1, 3 * block)
                for pos in {get_pos(end, unit, False), get_pos(end, unit, True)}:
                    args = dict(query_start=start, query_end=end, checkpoint_position=pos, hash_block_size=unit,
                                mamba_block_size=block, checkpoint_alignment=16)
                    got, want = self.kci.is_mamba_prefill_checkpoint_valid(**args), self.base_valid(**args)
                    if got != want:
                        self.assertTrue(self.on and got and _mid_block_eq(start, end, block), args)
                        new_shape += 1
        self.assertEqual(new_shape > 0, self.on)

    def test_worker_checkpoints_differ_only_on_fresh_new_shape_rows(self):
        rng = random.Random(1)
        changed = declined = 0
        for _ in range(4000):
            n = rng.randrange(1, 5)
            starts = [rng.randrange(0, 4000) // 128 * 128 + rng.choice((0, 0, 40)) for _ in range(n)]
            seq_lens = [s + rng.randrange(1, 4000) for s in starts]
            query_lens = [e - s for s, e in zip(starts, seq_lens)]
            fresh = [rng.random() < 0.5 for _ in range(n)]
            kw = dict(hash_block_size=128, mamba_block_size=1536, checkpoint_alignment=16,
                      drop_eagle_block=rng.random() < 0.5, cache_hit_alignment_tokens=128)
            got = self.ckpt.compute_mamba_prefill_checkpoints(seq_lens, query_lens, fresh_rows=fresh, **kw)
            want = self.base_compute(seq_lens, query_lens, **kw)
            relaxed = self.ckpt.compute_mamba_prefill_checkpoints(seq_lens, query_lens, **kw)  # None = all fresh
            for i in range(n):
                row, base_row = (got[0][i], got[1][i]), (want[0][i], want[1][i])
                if row != base_row:
                    self.assertTrue(self.on and fresh[i] and _mid_block_eq(starts[i], seq_lens[i], 1536))
                    changed += 1
                if (relaxed[0][i], relaxed[1][i]) != row:
                    self.assertTrue(self.on and not fresh[i])
                    declined += 1
        self.assertEqual((changed > 0, declined > 0), (self.on, self.on))

    def _sched(self, eagle):
        return SimpleNamespace(
            cache_config=SimpleNamespace(block_size=1536), hash_block_size=128, max_num_scheduled_tokens=16384,
            scheduler_config=SimpleNamespace(long_prefill_token_threshold=0), use_eagle_block_drop=eagle,
            mamba_partial_cache_hit=True, mamba_shared_prefix_checkpoint=True,
            mamba_has_prefill_checkpoint_blocks=True, mamba_prefill_checkpoint_alignment=16)

    def test_scheduler_split_differs_only_for_a_first_chunk(self):
        rng = random.Random(2)
        changed = 0
        for _ in range(20000):
            prompt = rng.randrange(1, 12000)
            computed = rng.choice((0, rng.randrange(0, prompt)))
            hit = rng.randrange(0, prompt - computed) // 128 * 128 if computed == 0 else 0
            req = SimpleNamespace(num_computed_tokens=computed, num_prompt_tokens=prompt, num_tokens=prompt,
                                  shared_prefix_boundary=rng.choice((0, rng.randrange(0, prompt + 1))))
            sched, budget = self._sched(rng.random() < 0.5), rng.randrange(1, 16384)
            got = self.sched.Scheduler._mamba_block_aligned_split(sched, req, budget, hit)
            if got != self.base_split(sched, req, budget, hit):
                self.assertTrue(self.on and computed == 0, vars(req))
                changed += 1
        self.assertEqual(changed > 0, self.on)

    def test_frozen_turn(self):
        """s 80000, P 82103 on the K3 config: 1408 tokens today (3 steps), 2103 on a first chunk with form K."""
        split = self.sched.Scheduler._mamba_block_aligned_split
        sched = self._sched(eagle=True)
        first = SimpleNamespace(num_computed_tokens=0, num_prompt_tokens=82103, num_tokens=82103,
                                shared_prefix_boundary=0)
        running = SimpleNamespace(**(vars(first) | {"num_computed_tokens": 80000}))
        self.assertEqual(split(sched, first, 2103, 80000), 2103 if self.on else 1408)
        self.assertEqual(split(sched, running, 2103), 1408)
        kw = dict(hash_block_size=128, mamba_block_size=1536, checkpoint_alignment=16, drop_eagle_block=True,
                  cache_hit_alignment_tokens=128)
        got = self.ckpt.compute_mamba_prefill_checkpoints([82103, 82103], [2103, 2103], fresh_rows=[True, False],
                                                          **kw)
        self.assertEqual(got, ([1920, 0], [52, -1]) if self.on else ([0, 0], [-1, -1]))

    def test_builder_reads_the_fresh_rows(self):
        import numpy as np
        import torch
        from vllm.v1.attention.backend import CommonAttentionMetadata
        from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
        from vllm.v1.kv_cache_interface import MambaSpec

        spec = MambaSpec(block_size=64, shapes=((1,),), dtypes=(torch.float32,), mamba_cache_mode="align",
                         num_prefill_checkpoint_blocks=1, prefill_checkpoint_alignment=1)
        config = SimpleNamespace(speculative_config=None, cache_config=SimpleNamespace(
            mamba_cache_mode="align", hash_block_size=16, cache_hit_alignment_tokens=16))
        builder = self.ckpt.MambaPrefillCheckpointBuilder(config, spec)
        # Two rows of 84 tokens from 96 to 180: initial-state column 1 == checkpoint column 1, checkpoint at 176.
        qsl = torch.tensor([0, 84, 168], dtype=torch.int32)
        seq_lens = torch.tensor([180, 180])
        m = CommonAttentionMetadata(
            query_start_loc=qsl, query_start_loc_cpu=qsl, seq_lens=seq_lens, seq_lens_cpu_upper_bound=seq_lens,
            num_reqs=2, num_actual_tokens=168, max_query_len=84, max_seq_len=180,
            block_table_tensor=torch.tensor([[11, 12, 13], [21, 22, 23]]), slot_mapping=torch.zeros(168))
        try:
            for fresh, want in ((np.array([True, False]), ([80, 0], [12, NULL_BLOCK_ID])),
                                (None, ([80, 80], [12, 22]))):
                self.ckpt.PERFBOT_P07_FRESH_ROWS = fresh
                out = builder.build(m, [0, 1])
                if not self.on:
                    self.assertIsNone(out)
                    continue
                self.assertEqual((out.offsets, out.state_indices.tolist()), want)
        finally:
            self.ckpt.PERFBOT_P07_FRESH_ROWS = None

    def test_v2_model_state_marks_the_first_chunk_after_add_as_fresh(self):
        import numpy as np
        import torch

        state = object.__new__(self.mh.MambaHybridModelState)
        state.cache_config = SimpleNamespace(block_size=1536, mamba_block_size=1536)
        state.vllm_config = SimpleNamespace(num_speculative_tokens=0)
        state.model_config = SimpleNamespace(max_model_len=1 << 20)
        state._align_mode, state.recoverssm, state.rope_state, state.prompt_embeds_state = True, None, None, None
        state._mamba_spec, state._mamba_group_ids = object(), []
        state.num_accepted_tokens_gpu = torch.ones(8, dtype=torch.int32)
        state._mamba_state_idx_gpu = torch.zeros(8, dtype=torch.int32)
        state._p07_add_start = np.full(8, -1, dtype=np.int64)
        state.add_request(3, SimpleNamespace(num_computed_tokens=80000))  # new or resumed: the row sent whole
        state.add_request(5, SimpleNamespace(num_computed_tokens=0))
        self.mh.build_attn_metadata = lambda **kw: kw  # the child process owns this module
        self.ckpt.PERFBOT_P07_FRESH_ROWS = None
        # Rows: req 3 first chunk from 80000; req 5 second chunk from 4096; req 6 never added (padding/dummy).
        computed = np.array([80000, 4096, 0], dtype=np.int32)
        sched = np.array([2103, 1536, 1], dtype=np.int32)
        qsl = np.concatenate([[0], np.cumsum(sched)]).astype(np.int32)
        batch = SimpleNamespace(
            num_reqs=3, num_reqs_after_padding=4, num_tokens=int(qsl[-1]), num_tokens_after_padding=int(qsl[-1]),
            query_start_loc_np=qsl, query_start_loc=torch.from_numpy(qsl), num_scheduled_tokens=sched,
            max_query_len=None, seq_lens_cpu_upper_bound=torch.from_numpy(computed + sched),
            seq_lens=torch.from_numpy(computed + sched), is_prefilling_np=np.array([True, True, False]),
            prefill_runs_as_decode_np=None, num_draft_tokens_per_req=None, dcp_local_seq_lens=None,
            positions=None, prompt_lens=None, idx_mapping_np=np.array([3, 5, 6]), num_computed_tokens_np=computed)
        args = dict(block_tables=(), slot_mappings={}, attn_groups=[], kv_cache_config=None)
        from vllm.config.compilation import CUDAGraphMode

        state.prepare_attn(batch, CUDAGraphMode.NONE, **args)
        fresh = self.ckpt.PERFBOT_P07_FRESH_ROWS
        if not self.on:
            self.assertIsNone(fresh)
            self.assertEqual(state._p07_add_start.tolist(), [-1] * 8)  # off: add_request is the base method
            return
        self.assertEqual(fresh.tolist(), [True, False, False])
        state.prepare_attn(batch, CUDAGraphMode.FULL, **args)  # padded to num_reqs_after_padding
        self.assertEqual(self.ckpt.PERFBOT_P07_FRESH_ROWS.tolist(), [True, False, False, False])
        state.prepare_attn(batch, CUDAGraphMode.NONE, for_capture=True, **args)
        self.assertIsNone(self.ckpt.PERFBOT_P07_FRESH_ROWS)  # capture: no worker gate


def load_tests(loader, tests, pattern):
    cases = (Child,) if os.environ.get("P07_CHILD_STATE") else (Edit, States)
    return unittest.TestSuite(loader.loadTestsFromTestCase(c) for c in cases)


if __name__ == "__main__":
    unittest.main()
