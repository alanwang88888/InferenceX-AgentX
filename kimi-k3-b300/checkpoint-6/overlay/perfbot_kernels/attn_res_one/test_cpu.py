"""CPU test of selftest.overlay_served: true through serve.sh's constant overlay link and through the tree itself.

No GPU, no torch: the function is read from selftest.py's source and run with that file's path.

  python3 -I overlay/perfbot_kernels/attn_res_one/test_cpu.py
"""

import ast
import os
import shutil
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = (HERE / "selftest.py").read_text()


def overlay_served(selftest_path: str):
    """selftest.overlay_served as if selftest.py were at `selftest_path`."""
    fn = next(n for n in ast.parse(SRC).body if isinstance(n, ast.FunctionDef) and n.name == "overlay_served")
    ns = {"os": os, "__file__": selftest_path}
    exec(compile(ast.Module([fn], []), "selftest.py", "exec"), ns)
    return ns["overlay_served"]


class OverlayServed(unittest.TestCase):
    def test_link_and_tree_paths_are_served_and_the_image_is_not(self):
        with tempfile.TemporaryDirectory() as d:
            tree = Path(d, "workspace", "bundles", "a", "overlay")
            mod = tree / "vllm" / "x.py"
            mod.parent.mkdir(parents=True)
            mod.write_text("")
            (tree / "perfbot_kernels" / "attn_res_one").mkdir(parents=True)
            shutil.copy(HERE / "selftest.py", tree / "perfbot_kernels" / "attn_res_one" / "selftest.py")
            link = Path(d, "perfbot-overlay")
            link.symlink_to(tree)
            image = Path(d, "dist-packages", "vllm", "x.py")
            image.parent.mkdir(parents=True)
            image.write_text("")
            for via in (link, tree):  # the selftest imported through the link, or from the tree
                f = overlay_served(str(via / "perfbot_kernels" / "attn_res_one" / "selftest.py"))
                with self.subTest(via=via.name):
                    self.assertTrue(f(str(link / "vllm" / "x.py")))
                    self.assertTrue(f(str(mod)))
                    self.assertFalse(f(str(image)))
                    self.assertFalse(f(str(tree) + "-other/vllm/x.py"))  # a sibling prefix is not inside


if __name__ == "__main__":
    unittest.main()
