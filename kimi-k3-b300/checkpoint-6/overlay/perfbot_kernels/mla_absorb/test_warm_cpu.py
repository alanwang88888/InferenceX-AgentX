"""CPU tests of mla_absorb.served_pages and warmtest.py's pure parts (torch only, no GPU, no vllm):

  python3 -I overlay/perfbot_kernels/mla_absorb/test_warm_cpu.py
"""

import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # overlay/ (the package root on the server's PYTHONPATH)

from perfbot_kernels import mla_absorb  # noqa: E402
from perfbot_kernels.mla_absorb import warmtest as wt  # noqa: E402


class MultipleOf:
    base = 16


class Backend:
    """Image TokenspeedMLABackend semantics: get_supported_kernel_block_sizes() = [32, 64] (tokenspeed_mla.py:90)."""
    sizes = [64, 32]

    @classmethod
    def get_supported_kernel_block_sizes(cls):
        return cls.sizes


def select_common_block_size(manager, sizes):
    """Image v1/worker/utils.py:309-375 for one backend with int sizes: the manager block if listed, else the largest
    listed size that divides it."""
    return manager if manager in sizes else next(s for s in sorted(sizes, reverse=True) if manager % s == 0)


class ServedPages(unittest.TestCase):
    def test_int_sizes_sorted(self):
        self.assertEqual(mla_absorb.served_pages(Backend), [32, 64])

    def test_multiple_of_dropped(self):
        class Mixed(Backend):
            sizes = [64, MultipleOf(), 32]
        self.assertEqual(mla_absorb.served_pages(Mixed), [32, 64])

    def test_every_selection_is_warmed(self):
        # manager blocks: 32 / 64 (listed), 1536 (K3 hybrid alignment, P-01c), 16 * 2^k and 32 * 3
        for m in (32, 64, 128, 96, 1536, 4096):
            self.assertIn(select_common_block_size(m, Backend.sizes), mla_absorb.served_pages(Backend))
        self.assertEqual(select_common_block_size(1536, Backend.sizes), 64)


def rec(rank, served, rows, warm_keys=()):
    return {"rank": rank, "served_page": served, "rows": rows, "warm_keys": list(warm_keys)}


def key(page, q, heads=96, **kw):
    return {"page_size": page, "seq_len_q": q, "num_heads": heads, "use_pdl": False, **kw}


def arm_rows(page, jit_first, jit_repeat=0, wall=0.001):
    rows, seen = [], set()
    for heads in wt.HEADS:
        for n in wt.CASES:
            pad = mla_absorb.bucket(n)
            first = (heads, pad) not in seen
            seen.add((heads, pad))
            j = jit_first if first else jit_repeat
            rows.append({"heads": heads, "page": page, "n": n, "pad": pad, "jit": j, "misses": j, "wall_s": wall,
                         "key": key(page, pad, heads)})
    return rows


def all_keys(page):
    return [key(page, b, h) for h in wt.HEADS for b in (8, 16, 32, 64, 128, 256, 512)]


def world(warm_page=64, served=64, cold_first=1, stale_repeat=0):
    """Records as the GPU arms write them: calls at pages 32 and 64, the warm arm warmed at warm_page."""
    def arm(kind):
        out = []
        for r in range(wt.W):
            rows = []
            for page in (32, 64):
                if kind == "warm":
                    rows += arm_rows(page, 0 if page == warm_page else 1, wall=0.001 if page == warm_page else 1.4)
                else:
                    rows += arm_rows(page, cold_first, stale_repeat if kind == "stale16" else 0, 1.4)
            out.append(rec(r, served, rows, all_keys(warm_page) if kind == "warm" else ()))
        return out
    return {a: arm(a) for a in wt.ARMS}


def equal_all(v=True):
    return {(r, h, p): v for r in range(wt.W) for h in wt.HEADS for p in (32, 64)}


class Judge(unittest.TestCase):
    def test_cases_cover_every_bucket(self):
        pads = [mla_absorb.bucket(n) for n in wt.CASES]
        self.assertEqual(pads[:3], [16, 256, 512])
        self.assertEqual(sorted(set(pads)), [8, 16, 32, 64, 128, 256, 512])
        self.assertLessEqual(max(wt.CASES), mla_absorb.threshold())
        self.assertEqual(wt.mark_first([{"heads": 96, "page": 64, "pad": p} for p in pads]), [True] * 7 + [False] * 3)

    def test_local_len(self):
        for G in (1, 7, 8, 9613, 10112):
            for i in (1, 4):
                lens = [wt.local_len(G, r, 8, i) for r in range(8)]
                self.assertEqual(sum(lens), G)
                pos = mla_absorb.global_positions(8, max(lens), i)
                self.assertEqual(lens, [int((pos[r * max(lens):(r + 1) * max(lens)] < G).sum()) for r in range(8)])

    def test_key_diff(self):
        self.assertEqual(wt.key_diff(key(64, 16), [key(64, 16)]), {})
        self.assertEqual(wt.key_diff(key(64, 16), [key(32, 8), key(32, 16)]), {"page_size": [32, 64]})
        self.assertEqual(wt.key_diff(key(64, 16, use_pdl=True), [key(64, 16)]), {"use_pdl": [False, True]})
        self.assertIn("_", wt.key_diff(key(64, 16, heads=64), [key(64, 16)]))

    def test_pass(self):
        res = wt.judge(world(), equal_all())
        self.assertTrue(res["ok"], res)
        self.assertEqual(len(res["B_pairs"]), wt.W * len(wt.HEADS))

    def test_warm_at_wrong_page_fails_a_and_c_not_b(self):
        res = wt.judge(world(warm_page=32), equal_all())
        self.assertFalse(res["A"])
        self.assertFalse(res["C"])
        self.assertTrue(res["B"] and res["S"])  # page 32 pairs are warmed and equal
        self.assertEqual(res["keydiff"]["96:64:16"], {"page_size": [32, 64]})

    def test_slow_first_call_fails_a(self):
        w = world()
        w["warm"][3]["rows"][25]["wall_s"] = 0.2  # rows 20..39 are page 64
        self.assertFalse(wt.judge(w, equal_all())["A"])

    def test_insensitive_detector_fails_s(self):
        self.assertFalse(wt.judge(world(cold_first=0), equal_all())["S"])
        self.assertFalse(wt.judge(world(stale_repeat=1), equal_all())["S"])

    def test_b_needs_equal_and_a_warmed_pair(self):
        eq = equal_all()
        eq[(5, 64, 64)] = False
        self.assertFalse(wt.judge(world(), eq)["B"])
        w = world()
        for r in w["warm"]:
            for x in r["rows"]:
                x["jit"] = x["misses"] = 1  # nothing warmed: no valid pair
        self.assertFalse(wt.judge(w, equal_all())["B"])

    def test_incomplete_or_error_fails_all(self):
        w = world()
        w["cold"][2] = {"rank": 2, "error": "no record (exit -9)"}
        res = wt.judge(w, equal_all())
        self.assertFalse(res["complete"] or res["A"] or res["ok"])
        self.assertIn("cold:2", res["errors"])


if __name__ == "__main__":
    unittest.main(verbosity=1)
