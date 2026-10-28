"""CPU tests of the G ladder's pure parts: python3 -m unittest perfbot_kernels.skinny_ladder.test_cpu"""

import unittest

import perfbot_kernels.skinny_ladder as sl


class T(unittest.TestCase):
    def test_chunks(self):
        self.assertEqual(sl.chunks(48, 16), [(0, 16), (16, 32), (32, 48)])
        self.assertEqual(sl.chunks(40, 16), [(0, 16), (16, 32), (32, 40)])
        self.assertEqual(sl.chunks(8, 16), [(0, 8)])
        for m in range(1, 60):
            for c in (1, 7, 8, 16):
                cs = sl.chunks(m, c)
                self.assertEqual([a for a, _ in cs], list(range(0, m, c)))
                self.assertEqual(cs[-1][1], m)
                self.assertTrue(all(0 < b - a <= c for a, b in cs))
        self.assertRaises(ValueError, sl.chunks, 0, 8)

    def test_shapes(self):
        # 1536x7168 is shared by gate_up and g_proj: five distinct shapes, the image router's keys
        self.assertEqual(sl.shapes(), [(3584, 7168), (1536, 7168), (2304, 1536), (7168, 768), (2112, 7168)])

    def test_step_saving(self):
        us = {(3584, 7168, 48): {"served": 20.0, "chunk8": 15.0, "chunk16": 18.0},  # 92 calls x 5 us
              (2304, 1536, 48): {"served": 5.0, "chunk8": 6.0}}  # slower: clamped to 0
        self.assertAlmostEqual(sl.step_saving_ms(us, 6), 92 * 5.0e-3)
        self.assertEqual(sl.step_saving_ms(us, 5), 0.0)


if __name__ == "__main__":
    unittest.main()
