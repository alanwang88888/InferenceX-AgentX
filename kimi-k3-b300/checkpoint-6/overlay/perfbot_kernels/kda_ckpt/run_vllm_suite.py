"""Run pytest on a vLLM 49cb2bbf source tree with the overlay's vllm/** copies shadowing it (P-07 form K).

The served path loads the copies through overlay/sitecustomize.py's finder, which turns on only when the installed
vLLM's version names the base. A source checkout has no installed version, so this runner puts the same finder first
on sys.meta_path itself. It then proves that the copies are the modules in use. The switches come from the
environment (PERFBOT_EXP_KDA_HASHFIX, PERFBOT_EXP_KDA_CKPT), as in a served arm.

  cd <vllm 49cb2bbf tree>; PYTHONPATH=. python <overlay>/perfbot_kernels/kda_ckpt/run_vllm_suite.py <pytest args>
"""

import importlib.util
import sys
from pathlib import Path

OVERLAY = Path(__file__).resolve().parents[2]
SHADOWED = ("vllm.v1.kv_cache_interface", "vllm.v1.core.single_type_kv_cache_manager",
            "vllm.v1.core.sched.scheduler", "vllm.model_executor.layers.mamba.checkpoint",
            "vllm.v1.worker.gpu.model_states.mamba_hybrid")


def install():
    spec = importlib.util.spec_from_file_location("perfbot_overlay_sitecustomize", OVERLAY / "sitecustomize.py")
    site = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(site)
    sys.meta_path.insert(0, site.PerfbotOverlayFinder())
    for name in SHADOWED:
        module = importlib.import_module(name)
        assert Path(module.__file__).is_relative_to(OVERLAY), (name, module.__file__)
    kci = sys.modules["vllm.v1.kv_cache_interface"]
    print(f"PERFBOT-OVERLAY test: {len(SHADOWED)} P-07 modules from {OVERLAY}; "
          f"CKPT={kci.PERFBOT_P07_CKPT} SKIP={kci.PERFBOT_P07_SKIP}", flush=True)


if __name__ == "__main__":
    install()
    import pytest

    sys.exit(pytest.main(sys.argv[1:]))
