"""CPU tests of perfbot_kernels.mla_ctx_segments (torch on CPU; flashinfer replaced by a ragged torch attention).

  python -B overlay/perfbot_kernels/mla_ctx_segments/test_cpu.py
"""
import importlib
import math
import os
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
ENV = "PERFBOT_EXP_MLA_CTX_SEGMENTS"


def load(mode):
    os.environ.pop(ENV, None)
    if mode is not None:
        os.environ[ENV] = mode
    import perfbot_kernels.mla_ctx_segments as m
    return importlib.reload(m)


def ragged_ref(q, k, v, cu_q, cu_kv, scale):
    """Per row b: softmax(q_b k_b^T * scale) v_b in fp64; returns out (T, H, V) and base-2 lse (T, H)."""
    outs, lses = [], []
    for b in range(len(cu_q) - 1):
        qb = q[cu_q[b]:cu_q[b + 1]].double()
        kb, vb = k[cu_kv[b]:cu_kv[b + 1]].double(), v[cu_kv[b]:cu_kv[b + 1]].double()
        s = torch.einsum("qhd,khd->hqk", qb, kb) * scale
        lse = torch.logsumexp(s, dim=-1)  # (H, q)
        outs.append(torch.einsum("hqk,khd->qhd", torch.softmax(s, dim=-1), vb))
        lses.append((lse / math.log(2.0)).transpose(0, 1))
    return torch.cat(outs), torch.cat(lses)


def fake_flashinfer(calls):
    def fmha(query, key, value, workspace_buffer, seq_lens, max_q_len, max_kv_len, bmm1_scale, bmm2_scale, o_sf_scale,
             batch_size, window_left, cum_seq_lens_q, cum_seq_lens_kv, enable_pdl, is_causal, return_lse, out=None,
             skip_all_rows_active_check=True):
        cq, ck = cum_seq_lens_q.tolist(), cum_seq_lens_kv.tolist()
        assert len(cq) == len(ck) == batch_size + 1 and seq_lens.tolist() == [ck[i + 1] - ck[i] for i in range(batch_size)]
        assert max_q_len == max(cq[i + 1] - cq[i] for i in range(batch_size)) and max_kv_len == max(seq_lens.tolist())
        assert not is_causal and return_lse and query.shape[0] == cq[-1]
        calls.append(dict(batch=batch_size, cu_q=cq, cu_kv=ck))
        o, lse = ragged_ref(query, key, value, cq, ck, bmm1_scale)
        out.copy_(o)
        return out, lse.float()
    mod = types.ModuleType("flashinfer.prefill")
    mod.trtllm_ragged_attention_deepseek = fmha
    pkg = types.ModuleType("flashinfer")
    pkg.prefill = mod
    sys.modules["flashinfer"], sys.modules["flashinfer.prefill"] = pkg, mod


class Plan(unittest.TestCase):
    def test_formula_and_clamps(self):
        m = load("auto")
        cases = {  # (q, kv) -> S at 148 SMs, 12 heads: argmin ceil(tiles*S/148)/S over S <= min(8, 16384 // q, kv // 4096)
            (183, 80000): 8, (256, 80000): 8, (512, 80000): 6, (1408, 80000): 2, (2100, 80000): 4, (3000, 80000): 1,
            (4000, 80000): 3, (183, 16464): 4, (183, 4095): 1, (1408, 65536): 2,
        }
        for (q, kv), want in cases.items():
            with self.subTest(q=q, kv=kv):
                self.assertEqual(m.plan(q, kv, 12, 148), want)
        self.assertEqual(m.plan(10000, 80000, 12, 148), 1)  # 16,384 // 10,000 = 1
        self.assertEqual(m.plan(183, 80000, 12, 148, mode="4"), 4)
        self.assertEqual(m.plan(183, 80000, 12, 148, mode="64"), 8)  # fixed N still clamped by S_max
        for off in ("0", "", "off"):
            self.assertEqual(m.plan(183, 80000, 12, 148, mode=off), 1)

    def test_bounds_cover_every_key_once(self):
        m = load("auto")
        for kv, s in ((80000, 13), (80000, 8), (16464, 4), (65536, 3), (4096, 1)):
            b = m.segment_bounds(kv, s)
            self.assertEqual((b[0], b[-1], len(b)), (0, kv, s + 1))
            self.assertTrue(all(b[j + 1] - b[j] >= kv // s for j in range(s)))


class Merge(unittest.TestCase):
    def test_merge_equals_one_softmax_fp64(self):
        m = load("auto")
        torch.manual_seed(0)
        H, D, V, q, kv = 3, 16, 8, 5, 97
        Q, K, Vv = torch.randn(q, H, D), torch.randn(kv, H, D), torch.randn(kv, H, V)
        one, lse_one = ragged_ref(Q, K, Vv, [0, q], [0, kv], 0.25)
        for s in (2, 3, 7):
            b = m.segment_bounds(kv, s)
            part, lse = ragged_ref(Q.repeat(s, 1, 1), K, Vv, [q * j for j in range(s + 1)], b, 0.25)
            o, l = m.merge_log2(part, lse, s, q)
            self.assertLess((o.double() - one).abs().max().item(), 1e-5)
            self.assertLess((l.double() - lse_one).abs().max().item(), 1e-5)


    def test_merge_is_shift_safe(self):
        """Large logits (base-2 lse ~ 300 > the fp32 exp2 range): the merge must stay finite and exact."""
        m = load("auto")
        part = torch.randn(4 * 3, 2, 5)
        lse = torch.randn(4 * 3, 2) + 300.0
        o, l = m.merge_log2(part, lse, 4, 3)
        w = torch.softmax(lse.view(4, 3, 2).double() * math.log(2.0), dim=0)
        ref = (part.view(4, 3, 2, 5).double() * w.unsqueeze(-1)).sum(0)
        self.assertTrue(torch.isfinite(o).all() and torch.isfinite(l).all())
        self.assertLess((o.double() - ref).abs().max().item(), 1e-5)

class Run(unittest.TestCase):
    def chunk(self, n_req=1, active=True):
        return SimpleNamespace(num_requests=n_req, all_rows_active=active)

    def backend(self):
        return SimpleNamespace(scale=0.2, _workspace_buffer=torch.empty(0))

    def setUp(self):
        self.calls = []
        fake_flashinfer(self.calls)
        torch.manual_seed(1)
        self.q, self.k, self.v = torch.randn(183, 12, 24), torch.randn(16464, 12, 24), torch.randn(16464, 12, 16)

    def test_segmented_equals_one_call(self):
        m = load("auto")
        m._SMS[self.q.device] = 148
        out, lse = m.run(self.backend(), self.chunk(), self.q, self.k, self.v, None, torch.float64)
        self.assertEqual(self.calls[-1]["batch"], 4)  # S = min(8, 16464 // 4096 = 4)
        self.assertEqual(self.calls[-1]["cu_q"], [0, 183, 366, 549, 732])
        one, lse_one = ragged_ref(self.q, self.k, self.v, [0, 183], [0, 16464], 0.2)
        self.assertLess((out - one).abs().max().item(), 1e-6)  # the merge runs in fp32, as merge_attn_states does
        self.assertLess((lse.double() - lse_one).abs().max().item(), 1e-5)
        self.assertEqual(tuple(out.shape), (183, 12, 16))
        self.assertEqual(tuple(lse.shape), (183, 12))

    def test_writes_given_out(self):
        m = load("2")
        m._SMS[self.q.device] = 148
        buf = torch.zeros(183, 12, 16, dtype=torch.float64)
        out, _ = m.run(self.backend(), self.chunk(), self.q, self.k, self.v, buf, torch.float64)
        self.assertEqual(out.data_ptr(), buf.data_ptr())
        self.assertEqual(self.calls[-1]["batch"], 2)

    def test_declines(self):
        for mode, chunk, why in ((None, self.chunk(), "switch unset"), ("0", self.chunk(), "switch 0"),
                                 ("auto", self.chunk(n_req=2), "R > 1"), ("auto", self.chunk(active=False), "inactive"),
                                 ("1", self.chunk(), "fixed S = 1")):
            with self.subTest(why=why):
                m = load(mode)
                m._SMS[self.q.device] = 148
                self.assertIsNone(m.run(self.backend(), chunk, self.q, self.k, self.v, None, torch.float64))
        m = load("auto")
        m._SMS[self.q.device] = 148
        self.assertIsNone(m.run(self.backend(), self.chunk(), torch.randn(3000, 12, 24), self.k, self.v, None, torch.float64))
        self.assertEqual(self.calls, [])

    def test_decision_logged_once_per_shape(self):
        m = load("auto")
        m._SMS[self.q.device] = 148
        with self.assertLogs("vllm.perfbot.mla_ctx_segments", level="INFO") as cm:
            for _ in range(3):
                m.run(self.backend(), self.chunk(), self.q, self.k, self.v, None, torch.float64)
        self.assertEqual(len(cm.output), 1)
        self.assertIn("q=183 kv=16464 S=4", cm.output[0])


if __name__ == "__main__":
    unittest.main()
