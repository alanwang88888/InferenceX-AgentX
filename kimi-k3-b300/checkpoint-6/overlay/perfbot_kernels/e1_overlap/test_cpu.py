"""CPU tests of the E1 overlap harness's pure parts: python3 -m unittest perfbot_kernels.e1_overlap.test_cpu"""

import unittest

import torch

import perfbot_kernels.e1_overlap as e1
import perfbot_kernels.e1_overlap.selftest as st


class T(unittest.TestCase):
    def test_modes(self):
        self.assertEqual(e1.MS, (8, 48))
        self.assertEqual(set(e1.MODES), {"main", "aux", "both", "serial", "auxfirst"})
        for mode, (main_on, aux_on, placement) in e1.MODES.items():
            self.assertIn(placement, e1.PLACEMENTS)
            self.assertTrue(main_on or aux_on, mode)
        # exactly one served overlapped mode with both sides; main / aux differ from it only by the dropped side
        self.assertEqual(e1.MODES["both"], (True, True, "aux"))
        self.assertEqual(e1.MODES["main"][2], e1.MODES["aux"][2])
        self.assertEqual(e1.MODES["main"][2], e1.MODES["both"][2])

    def test_scopes(self):
        names = [p[0] for p in e1.PHASES]
        self.assertEqual(e1.SCOPES["layer"], tuple(names))
        for phases in e1.SCOPES.values():
            self.assertTrue(set(phases) <= set(names))

    def test_shapes_shared_with_served_packages(self):
        import perfbot_kernels.moe_stage as ms
        import perfbot_kernels.router_topk as rt
        import perfbot_kernels.skinny_ladder as sl

        self.assertEqual((e1.E, e1.TOPK, e1.LATENT), (ms.E, ms.TOPK, ms.H))
        self.assertTrue(set(e1.GEMMS.values()) <= set(sl.shapes()))
        site = {name: (n, k) for name, n, k, *_ in sl.SITES}
        self.assertEqual(site["E1 latent down"], e1.GEMMS["latent_down"])
        self.assertEqual(site["E1 shared gate_up"], e1.GEMMS["gate_up"])
        self.assertEqual(site["E3 shared down"], e1.GEMMS["shared_down"])
        w = torch.empty(e1.ROUTER, dtype=torch.bfloat16)
        self.assertIsNone(rt.eligible(w, torch.empty(e1.E, dtype=torch.float32), e1.TOPK))  # the fused router serves it
        self.assertLessEqual(max(e1.MS), ms.MAX_M)  # the stage serves every M here
        self.assertLessEqual(max(e1.MS), rt.MAX_M)

    def test_gemm_backend(self):
        plan = {m: ("dsv3_fused_a", None) for m in range(2, 9)}
        self.assertEqual(e1.gemm_backend(plan, 8), "dsv3_fused_a")
        self.assertEqual(e1.gemm_backend(plan, 48), "cublas")
        self.assertEqual(e1.gemm_backend({}, 8), "cublas")
        self.assertEqual(e1.gemm_backend(None, 8), "cublas")

    def test_summarize(self):
        s = e1.summarize({"main": 50.0, "aux": 30.0, "both": 62.0, "serial": 78.0, "auxfirst": 60.5})
        self.assertEqual(s, {"exposed_aux_us": 12.0, "hidden_aux_us": 18.0, "overlap_gain_us": 16.0,
                             "auxfirst_delta_us": -1.5})
        s = e1.summarize({"main": 50.0, "both": 62.0})
        self.assertEqual(s["exposed_aux_us"], 12.0)
        self.assertIsNone(s["hidden_aux_us"])
        self.assertIsNone(s["overlap_gain_us"])
        self.assertIsNone(e1.summarize({})["exposed_aux_us"])

    def test_same(self):
        a = torch.tensor([0.0, 1.5, float("nan")])
        self.assertTrue(st.same(a, a.clone(), "carrier"))  # NaN bits compare equal
        b = a.clone()
        b[0] = -0.0
        self.assertFalse(st.same(a, b, "carrier"))  # -0 != +0 bit for bit
        self.assertFalse(st.same(a, a.to(torch.float64), "latent"))
        r = torch.tensor([[1.0, -2.0], [4.0, 0.5]], dtype=torch.bfloat16)
        self.assertTrue(st.same(r, (r.float() * (1 + 2 ** -8)).to(torch.bfloat16), "rows"))
        self.assertFalse(st.same(r, r * 1.5, "rows"))
        self.assertTrue(st.agree({"rows": r, "wts": a}, {"rows": r.clone(), "wts": a.clone()}))
        self.assertFalse(st.agree({"rows": r}, {"rows": r, "wts": a}))


if __name__ == "__main__":
    unittest.main()
