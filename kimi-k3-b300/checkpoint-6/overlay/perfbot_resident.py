"""Resident evaluation (docs/expedition/ledger/RESIDENT-EVAL.md §2): rebuild the specialized decode from another
overlay bundle inside the running server, so one server measures several candidate readings.

serve.sh passes `--worker-extension-cls perfbot_resident.ResidentReload` only when PERFBOT_READINGS is set; vLLM
then mixes this class into every worker (vllm/v1/worker/worker_base.py:304-330) and POST /collective_rpc
{"method": "perfbot_reload", "args": [<bundle>]} runs it on all ranks. Inert unless called.

The order follows elastic EP's re-capture (vllm/distributed/elastic_ep/elastic_execute.py warm_and_capture):
graphs go first, the old decode's buffers are freed before the new decode allocates, the workspace is unlocked
for the warm-up and locked again after it.
"""

import gc
import hashlib
import json
import os
import sys
import time

# R1 scope: the only trees a reload may swap. Anything else is an R2 change (server restart).
RELOAD_TREES = ("vllm/models/kimi_k3/nvidia/specialized_decode", "perfbot_kernels")
RELOAD_MODULES = tuple(t.replace("/", ".") for t in RELOAD_TREES)
# R0: a bundle's {NAME: value} for switches that RELOAD_TREES modules read from os.environ at import.
READING_ENV_FILE, READING_ENV_PREFIX = "reading_env.json", "PERFBOT_EXP_"
_restore = {}  # the os.environ values the applied reading env replaced (None = unset)


def tree_shas(bundle: str) -> dict:
    """{bundle-relative path: sha256} of every file in RELOAD_TREES; the arm refuses a reading whose ranks differ."""
    out = {}
    for tree in RELOAD_TREES:
        for dirpath, dirs, files in os.walk(os.path.join(bundle, tree)):
            dirs[:] = sorted(d for d in dirs if d != "__pycache__")
            for name in sorted(files):
                path = os.path.join(dirpath, name)
                with open(path, "rb") as f:
                    out[os.path.relpath(path, bundle)] = hashlib.sha256(f.read()).hexdigest()
    return out


def apply_reading_env(bundle: str) -> dict:
    """Undo the previous reading's env, then set this bundle's; a bundle without the file runs with the server env."""
    for name, value in _restore.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    _restore.clear()
    path = os.path.join(bundle, READING_ENV_FILE)
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        env = json.load(f)
    bad = sorted(k for k, v in env.items() if not k.startswith(READING_ENV_PREFIX) or not isinstance(v, str))
    if bad:
        raise RuntimeError(f"{path}: {bad} are not {READING_ENV_PREFIX}* names with string values")
    for name, value in env.items():
        _restore[name] = os.environ.get(name)
        os.environ[name] = value
    return env


def repoint(bundle: str) -> list:
    """Make the next import of a RELOAD_MODULES module come from `bundle`; return the module keys dropped."""
    finders = [f for f in sys.meta_path if hasattr(f, "overlay_root")]
    if not finders:
        raise RuntimeError("PERFBOT-OVERLAY finder not installed: a reload has nothing to repoint")
    for finder in finders:
        finder.overlay_root = bundle
    sys.path.insert(0, bundle)  # perfbot_kernels is a top-level package found on sys.path
    dropped = sorted(k for k in sys.modules if any(k == m or k.startswith(m + ".") for m in RELOAD_MODULES))
    for key in dropped:
        del sys.modules[key]
    return dropped


def release_graphs(runner) -> None:
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager

    runner.cudagraph_manager.release_graphs()
    speculator = getattr(runner, "speculator", None)
    for attr in vars(speculator).values() if speculator is not None else ():
        if isinstance(attr, CudaGraphManager):
            attr.release_graphs()


class ResidentReload:
    _perfbot_bundle = None  # the bundle the live decode was built from; None = the server's own overlay

    def _perfbot_build(self, bundle: str, runner, lm) -> tuple:
        """Import the decode from `bundle` and build, warm and capture it, under the worker's vLLM config as at start
        (gpu_worker.py set_current_vllm_config; providers.py:161 dcp_box() reads it). A fresh runtime per build."""
        from vllm.config import set_current_vllm_config
        from vllm.v1.worker.gpu.eplb_utils import preserve_serving_state
        from vllm.v1.worker.workspace import lock_workspace, unlock_workspace

        env = apply_reading_env(bundle)
        dropped = repoint(bundle)
        with set_current_vllm_config(self.vllm_config):
            lm.build_specialized_decode()
            lm.specialized_decode.prepare_kv()
            t = time.monotonic()
            unlock_workspace()
            try:
                with preserve_serving_state(runner):
                    self.compile_or_warm_up_model()
            finally:
                lock_workspace()
        return env, dropped, t

    def _perfbot_release(self, runner, lm) -> None:
        import torch
        from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

        # A capture that raised in __enter__ keeps its thread guard set; the restore's capture then raised "Nested
        # BreakableCUDAGraphCapture" (rs1-dec, breakable_cudagraph.py:161-165).
        BreakableCUDAGraphCapture._tls.active = None
        release_graphs(runner)
        lm.specialized_decode = None
        gc.collect()
        torch.accelerator.synchronize()
        torch.accelerator.empty_cache()
        # The released graphs leave the shared pool's pinned-host twin at use_count 0 until the host cache is emptied;
        # a capture on that pool before then asserts (rs1-dec; torch 2.14 CachingHostAllocator.h:921).
        torch._C._host_emptyCache()

    def perfbot_reload(self, bundle: str) -> dict:
        """Rebuild from `bundle`. A failed build rebuilds from the previous bundle and returns {"error", "restored"}
        (serve.sh voids only this reading); a failed restore raises (serve.sh voids the rest)."""
        import torch

        t0 = time.monotonic()
        runner = self.model_runner
        model = runner.get_model()
        lm = model.get_language_model() if hasattr(model, "get_language_model") else model
        if lm.specialized_decode is None:
            raise RuntimeError("no specialized decode in this server: decode_backend is not latency_specialized")
        finders = [f for f in sys.meta_path if hasattr(f, "overlay_root")]
        previous = self._perfbot_bundle or (finders[0].overlay_root if finders else None)
        self._perfbot_release(runner, lm)
        before = torch.cuda.memory_allocated()
        t1 = time.monotonic()
        error = None
        try:
            env, dropped, t2 = self._perfbot_build(bundle, runner, lm)
            self._perfbot_bundle = bundle
        except Exception as e:  # noqa: BLE001 - the engine must outlive one bad bundle
            if previous is None:
                raise
            error = f"{type(e).__name__}: {e}"
            self._perfbot_release(runner, lm)
            env, dropped, t2 = self._perfbot_build(previous, runner, lm)
        out = {
            "rank": self.rank,
            "release_s": round(t1 - t0, 3),
            "rebuild_s": round(t2 - t1, 3),
            "recapture_s": round(time.monotonic() - t2, 3),
            "memory_allocated_before": before,
            "memory_allocated_after": torch.cuda.memory_allocated(),
            "modules_dropped": len(dropped),
            "file_shas": tree_shas(bundle if error is None else previous),
            "reading_env": env,
        }
        if error is not None:
            out.update(error=error, restored=previous)
        return out
