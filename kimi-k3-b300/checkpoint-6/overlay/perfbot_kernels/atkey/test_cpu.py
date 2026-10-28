"""CPU tests of the narrow FlashInfer autotune key (PERFBOT-ATKEY, W5) and its binding to autotune-record/.

No GPU, no torch, no vLLM import: SimpleNamespace stands in for VllmConfig.

  python3 -I overlay/perfbot_kernels/atkey/test_cpu.py
"""
import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

HERE = Path(__file__).resolve().parent
OVERLAY = HERE.parents[1]
sys.path.insert(0, str(OVERLAY))

import perfbot_kernels.atkey as atkey  # noqa: E402

RECORD = OVERLAY.parent / "autotune-record" / "MANIFEST.json"


def k3(**over):
    """The served K3 config as cp4v2 c0/land4-6, conc1pair c1-c0/c1-land6 and pretraceaa pt-a0 resolve it (their
    server.log engine lines), plus fields the upstream key binds and the narrow key must ignore."""
    v = dict(architecture="KimiK3ForConditionalGeneration", dtype="torch.bfloat16", quantization="mxfp4", tp=8, pp=1, dp=1,
             dcp=8, ep=False, mnbt=16384, method="dspark", nst=7, kv="fp8", max_num_seqs=16,
             capture=[8, 16, 24], additional={"decode_backend": "latency_specialized"})
    v.update(over)
    return NS(model_config=NS(architecture=v["architecture"], dtype=v["dtype"], quantization=v["quantization"]),
              parallel_config=NS(tensor_parallel_size=v["tp"], pipeline_parallel_size=v["pp"], data_parallel_size=v["dp"],
                                 decode_context_parallel_size=v["dcp"], enable_expert_parallel=v["ep"]),
              scheduler_config=NS(max_num_batched_tokens=v["mnbt"], max_num_seqs=v["max_num_seqs"]),
              speculative_config=NS(method=v["method"], num_speculative_tokens=v["nst"]),
              cache_config=NS(cache_dtype=v["kv"]),
              compilation_config=NS(cudagraph_capture_sizes=v["capture"]), additional_config=v["additional"])


class Key(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ, {}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop(atkey.SALT_ENV, None)

    def test_conc_capture_sizes_and_additional_config_share_one_key(self):
        base = atkey.key(k3())
        for over in ({"max_num_seqs": 2}, {"capture": [7, 8, 14, 16, 21]}, {"additional": {}}):
            self.assertEqual(atkey.key(k3(**over)), base, over)

    def test_shape_fields_and_the_salt_each_move_the_key(self):
        base = atkey.key(k3())
        for over in ({"tp": 4}, {"quantization": "fp8"}, {"mnbt": 8192}, {"nst": 3}, {"kv": "auto"}, {"dcp": 1}):
            self.assertNotEqual(atkey.key(k3(**over)), base, over)
        os.environ[atkey.SALT_ENV] = "land-9"
        self.assertNotEqual(atkey.key(k3()), base)

    def test_no_drafter_is_a_key_of_its_own(self):
        cfg = k3()
        cfg.speculative_config = None
        self.assertNotEqual(atkey.key(cfg), atkey.key(k3()))

    def test_the_record_is_filed_under_the_served_config_key(self):
        m = json.loads(RECORD.read_text())
        self.assertEqual(atkey.key_of(m["fields"]), m["key"])
        self.assertEqual(atkey.fields(k3()), m["fields"])
        self.assertEqual(atkey.key(k3()), m["key"])


if __name__ == "__main__":
    unittest.main()
