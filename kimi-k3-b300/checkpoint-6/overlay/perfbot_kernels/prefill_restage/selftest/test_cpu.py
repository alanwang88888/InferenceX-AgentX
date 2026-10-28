# PERFBOT P-07F: CPU self-test of form R (item 05) — stdlib only, no torch / numpy / vLLM needed.
"""Load the two overlay copies (vllm/v1/worker/gpu/states.py, sample/sampler.py) against stub modules and check the
staging decision. The stubs model only what the two files touch: StagedWriteTensor records each staged row write and
apply_write lays it into a per-row dict, so a test can read back exactly which columns the GPU row would hold.

  python3 -B -m unittest discover -s <overlay or bundle>/perfbot_kernels/prefill_restage/selftest -p 'test_*.py'

What it proves (the real-engine proof is the GPU judge, judge.py):
  R1 off (switch unset): every new row is staged whole, as at 49cb2bbf.
  R2 on, sampler declares the row prefix-free: only columns [num_computed, P) are staged; everything a prefill reader
     reads ([num_computed, num_computed + query + lookahead) within [0, P)) is equal to the off arm.
  R3 on, any history reader (penalties, bad words, logit bias, thinking budget, custom processor, prompt logprobs,
     watermark subclass, trace replay) or no sampler call at all: the row is staged whole (fail-safe default).
  R4 on, dense rows (ngram): whole row.
  R5 a row removed before apply_staged_writes is not staged; a free call for a row that is not pending is ignored.
  R6 evidence line: PERFBOT-P07F rows_suffix / rows_full / tokens_skipped counts are right.
  R7 the overlay copies are 49cb2bbf plus tagged added lines only (sha256 of the tag-stripped bytes = vllm_base.json).
"""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # overlay/ (worktree) or bundle/ (item bundle): both hold vllm/ and vllm_base.json
STATES = ROOT / "vllm/v1/worker/gpu/states.py"
SAMPLER = ROOT / "vllm/v1/worker/gpu/sample/sampler.py"
TAG = "# PERFBOT P-07F"


class _Any:
    """Stand-in for any object a stub module hands out: callable, attribute-rich, falsy."""

    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        return _Any()

    def __getattr__(self, name):
        return _Any()

    def __bool__(self):
        return False

    def __or__(self, other):
        return other

    def __getitem__(self, key):
        return _Any()


class _StubModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Any()


def _fake_numpy():
    np = _StubModule("numpy")
    np.zeros = lambda shape, dtype=None: [0] * (shape if isinstance(shape, int) else shape[0])
    return np


def _fake_torch():
    t = _StubModule("torch")
    t.int32, t.int64, t.float32 = "int32", "int64", "float32"
    t.zeros = lambda *a, **k: _Any()
    t.device = lambda *a, **k: "cpu"
    return t


class FakeStaged:
    """StagedWriteTensor model: stage_write / stage_write_elem / apply_write; rows = {row: {col: value}}."""

    def __init__(self, size, dtype=None, device=None, max_concurrency=None, uva_instead_of_gpu=False):
        self.write_contents = object() if uva_instead_of_gpu else None  # the real class: not None iff UVA
        self.staged = []
        self.rows = {}
        self.staged_tokens = 0  # cost model: tokens converted list -> buffer in apply_write

    def stage_write(self, index, start, x):
        if not x:
            return
        self.staged.append((index, start, list(x)))

    def stage_write_elem(self, index, x):
        self.staged.append((index, 0, [x]))

    def apply_write(self):
        for index, start, xs in self.staged:
            row = self.rows.setdefault(index, {})
            for i, v in enumerate(xs):
                row[start + i] = v
            self.staged_tokens += len(xs)
        self.staged.clear()


class FakeUva:
    def __init__(self, size, dtype=None):
        self.np = [0] * size

    def copy_to_uva(self, n=None):
        return None


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@contextlib.contextmanager
def stubbed(switch_on):
    """Fresh stub world + fresh module copies; the switch is read at import, so it is set before loading."""
    saved_mods, saved_env = dict(sys.modules), os.environ.get("PERFBOT_EXP_PREFILL_GRAPH")
    if switch_on:
        os.environ["PERFBOT_EXP_PREFILL_GRAPH"] = "1"
    else:
        os.environ.pop("PERFBOT_EXP_PREFILL_GRAPH", None)

    class Finder:
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in ("vllm", "numpy", "torch", "triton") and fullname not in sys.modules:
                return importlib.util.spec_from_loader(fullname, Loader())
            return None

    class Loader:
        def create_module(self, spec):
            m = _StubModule(spec.name)
            m.__path__ = []
            return m

        def exec_module(self, module):
            pass

    finder = Finder()
    sys.meta_path.insert(0, finder)
    try:
        sys.modules["numpy"] = _fake_numpy()
        sys.modules["torch"] = _fake_torch()
        bu = _StubModule("vllm.v1.worker.gpu.buffer_utils")
        bu.StagedWriteTensor, bu.UvaBackedTensor = FakeStaged, FakeUva
        sys.modules["vllm.v1.worker.gpu.buffer_utils"] = bu
        states = _load("vllm.v1.worker.gpu.states", STATES)
        sampler = _load("vllm.v1.worker.gpu.sample.sampler", SAMPLER)
        yield states, sampler
    finally:
        sys.meta_path.remove(finder)
        sys.modules.clear()
        sys.modules.update(saved_mods)
        if saved_env is None:
            os.environ.pop("PERFBOT_EXP_PREFILL_GRAPH", None)
        else:
            os.environ["PERFBOT_EXP_PREFILL_GRAPH"] = saved_env


class Feature:
    """A sampler sub-state whose add_request returns `used` (the real ones return bool)."""

    def __init__(self, used=False):
        self.used = used
        self.calls = 0

    def add_request(self, req_idx, sp):
        self.calls += 1
        return self.used

    def apply_staged_writes(self):
        pass


def SP(**k):
    sp = types.SimpleNamespace(prompt_logprobs=None, prompt_logprob_token_ids=None)
    sp.__dict__.update(k)
    return sp


def make_sampler(sampler_mod, req_states, readers=(), thinking=False, custom=(), subclass=False, trace=False):
    """Build a real Sampler (overlay copy) whose sub-states are Features; `readers` = names of built-ins in use."""
    m = sampler_mod
    feats = {n: Feature(n in readers) for n in ("logit_bias", "penalties", "bad_words")}
    it = iter([feats["penalties"], feats["logit_bias"], feats["bad_words"]])  # __init__ builds penalties first
    m.PenaltiesState = lambda *a, **k: next(it)
    m.LogitBiasState = lambda *a, **k: next(it)
    m.BadWordsState = lambda *a, **k: next(it)
    m.SamplingStates = lambda *a, **k: Feature(True)  # temperature etc.: "needs processing" but reads no history
    m.ThinkingBudgetState = lambda *a, **k: Feature(thinking)
    m.TraceReplayState = lambda *a, **k: Feature(False)
    m.LogprobTokenIdsState = lambda *a, **k: Feature(False)
    m.flashinfer_sampler_supported = lambda: False
    m.xpu_sampler_supported = lambda: False
    cls = type("GPUWatermarkSampler", (m.Sampler,), {}) if subclass else m.Sampler
    return cls(vllm_config=_Any(), max_num_reqs=8, vocab_size=1000, device="cpu", req_states=req_states,
               enable_trace_replay=trace, custom_logits_processors=[Feature(u) for u in custom])


def make_states(states_mod, dense=False):
    return states_mod.RequestState(max_num_reqs=8, max_model_len=1 << 20, max_num_batched_tokens=16384,
                                   num_speculative_steps=7, vocab_size=1000, device="cpu",
                                   use_dense_all_token_ids=dense)


def add(rs, smp, req, ids, num_computed, sp=None, call_sampler=True):
    """model_runner.add_requests order at 49cb2bbf: req_states.add_request, (sampler.add_request), then apply."""
    rs.add_request(req_id=req, prompt_len=len(ids), all_token_ids=ids, num_computed_tokens=num_computed, max_tokens=64)
    idx = rs.req_id_to_index[req]
    if call_sampler:
        smp.add_request(idx, sp or SP())
    return idx


def prefill_read(row, num_computed, P, query, lookahead=1):
    """What _prepare_prefill_inputs_kernel reads (input_batch.py:271-314): [num_computed, +query) and the lookahead."""
    cols = list(range(num_computed, num_computed + query)) + [c for c in range(num_computed + query, num_computed + query + lookahead) if c < P]
    return [row.get(c) for c in cols]


IDS = [1000 + (i * 7919) % 90000 for i in range(82_103)]  # the P-07 frozen turn's size: P 82,103, s 80,000
S = 80_000


class FormR(unittest.TestCase):
    def run_case(self, on, dense=False, call_sampler=True, sp=None, **sampler_kw):
        with stubbed(on) as (st, sm), contextlib.redirect_stderr(io.StringIO()) as err:
            rs = make_states(st, dense)
            smp = make_sampler(sm, rs, **sampler_kw)
            idx = add(rs, smp, "r0", IDS, S, sp=sp, call_sampler=call_sampler)
            rs.apply_staged_writes()
            return rs.all_token_ids.rows[idx], rs, err.getvalue()

    def test_r1_off_writes_the_whole_row(self):
        row, rs, log = self.run_case(on=False)
        self.assertEqual([row[c] for c in range(len(IDS))], IDS)
        self.assertEqual(rs.all_token_ids.staged_tokens, len(IDS))
        self.assertNotIn("PERFBOT-P07F", log)

    def test_r2_on_prefix_free_stages_the_suffix_and_readers_see_the_same_tokens(self):
        off, _, _ = self.run_case(on=False)
        on, rs, _ = self.run_case(on=True)
        self.assertEqual(sorted(on), list(range(S, len(IDS))))  # nothing below num_computed
        self.assertEqual(rs.all_token_ids.staged_tokens, len(IDS) - S)
        for q in (1, 1408, 2103):  # chunk plans of the frozen turn today (1,408 / 512 / 183) and merged (2,103)
            self.assertEqual(prefill_read(on, S, len(IDS), q), prefill_read(off, S, len(IDS), q))

    def test_r3_any_history_reader_or_no_sampler_call_keeps_the_whole_row(self):
        cases = {
            "penalties": dict(readers=("penalties",)), "bad_words": dict(readers=("bad_words",)),
            "logit_bias": dict(readers=("logit_bias",)), "thinking_budget": dict(thinking=True),
            "custom_processor": dict(custom=(True,)), "watermark_subclass": dict(subclass=True),
            "trace_replay": dict(trace=True), "prompt_logprobs": dict(sp=SP(prompt_logprobs=5)),
            "prompt_logprob_token_ids": dict(sp=SP(prompt_logprob_token_ids=[[1]])),
            "no_sampler_call": dict(call_sampler=False),
        }
        for name, kw in cases.items():
            with self.subTest(name):
                row, rs, _ = self.run_case(on=True, **kw)
                self.assertEqual(len(row), len(IDS))
                self.assertEqual([row[c] for c in range(len(IDS))], IDS)

    def test_r3_custom_processor_that_is_off_does_not_block(self):
        row, _, _ = self.run_case(on=True, custom=(False,))
        self.assertEqual(min(row), S)

    def test_r4_dense_rows_stay_whole(self):
        row, _, _ = self.run_case(on=True, dense=True)
        self.assertEqual(len(row), len(IDS))

    def test_r5_removed_or_stray_rows(self):
        with stubbed(True) as (st, sm), contextlib.redirect_stderr(io.StringIO()):
            rs = make_states(st)
            smp = make_sampler(sm, rs)
            rs.perfbot_p07f_free(3)  # not pending: ignored
            idx = add(rs, smp, "gone", IDS[:5000], 4000)
            rs.remove_request("gone")
            rs.apply_staged_writes()
            self.assertEqual(rs.all_token_ids.rows, {})
            self.assertEqual(rs._p07f_free, set())

    def test_r5_two_new_rows_one_pass_each_decided_alone(self):
        with stubbed(True) as (st, sm), contextlib.redirect_stderr(io.StringIO()) as err:
            rs = make_states(st)
            smp = make_sampler(sm, rs)
            a = add(rs, smp, "a", IDS[:3000], 2000)
            m = sm  # second request uses penalties: rebuild its sampler with the reader on
            smp_pen = make_sampler(m, rs, readers=("penalties",))
            b = add(rs, smp_pen, "b", IDS[:4000], 3500)
            rs.apply_staged_writes()
            self.assertEqual(min(rs.all_token_ids.rows[a]), 2000)
            self.assertEqual(min(rs.all_token_ids.rows[b]), 0)
            self.assertIn("PERFBOT-P07F rows_suffix=1 rows_full=0 tokens_skipped=2000", err.getvalue())
            self.assertIn("PERFBOT-P07F rows_suffix=1 rows_full=1 tokens_skipped=2000", err.getvalue())

    def test_r6_evidence_counts(self):
        """One line per row for rows 1-256 (the judge reads exact counts), then one per 64 rows."""
        with stubbed(True) as (st, sm), contextlib.redirect_stderr(io.StringIO()) as err:
            rs = make_states(st)
            smp = make_sampler(sm, rs)
            for k in range(320):
                add(rs, smp, f"r{k}", IDS[:2000], 1500 if k % 2 == 0 else 0)
                rs.apply_staged_writes()
                rs.remove_request(f"r{k}")
            lines = [l for l in err.getvalue().splitlines() if l.startswith("PERFBOT-P07F")]
            self.assertEqual(len(lines), 256 + 1)
            self.assertEqual(lines[0], "PERFBOT-P07F rows_suffix=1 rows_full=0 tokens_skipped=1500")
            self.assertEqual(lines[1], "PERFBOT-P07F rows_suffix=1 rows_full=1 tokens_skipped=1500")
            self.assertEqual(lines[-1], "PERFBOT-P07F rows_suffix=160 rows_full=160 tokens_skipped=240000")


class TaggedLinesOnly(unittest.TestCase):
    def test_r7_strip_tag_gives_the_base(self):
        base = json.loads((ROOT / "vllm_base.json").read_text())
        for rel in ("vllm/v1/worker/gpu/states.py", "vllm/v1/worker/gpu/sample/sampler.py"):
            with self.subTest(rel=rel):
                self.assertEqual(base["intentional_edits"][rel]["tag"], TAG)
                kept = b"".join(l for l in (ROOT / rel).read_bytes().splitlines(keepends=True) if TAG.encode() not in l)
                self.assertEqual(hashlib.sha256(kept).hexdigest(), base["base_sha256"][rel])


if __name__ == "__main__":
    unittest.main()
