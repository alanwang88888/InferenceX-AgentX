"""PERFBOT-CUTECACHE: let cute.compile use the CuTe DSL's own file cache (E17c, ARM-TIME item 1).

Switch: PERFBOT_EXP_CUTE_CACHE=1. Only the value "1" is on. The overlay model.py (KimiK3ForCausalLM.
build_specialized_decode, tag "# PERFBOT BUILD") reads the switch and calls install() before SpecializedDecode is
built. With the switch off, this module does not load.

Why. A latency_specialized start spends ~147-197 s per rank in the DSL pass pipeline (dsl.passes, 1035 cute.compile
calls per rank) on every start, because cute.compile never reads or writes the DSL's file cache. Same code in
nvidia-cutlass-dsl 4.6.2 and 4.8.0 (the image's, UP-02):
- cute.compile (base_dsl/compiler.py, 4.8.0:1637-1638) forces compile_only=True and no_cache=True;
- BaseDSL._prepare_compilation (base_dsl/dsl.py, 4.8.0:2671-2690) also turns the cache off for compile_only.
serve.sh already points CUTE_DSL_CACHE_DIR at a fixed link into the arm's cache tree, which the driver exports and
seeds into the next carrier (E3); that tree stays empty today.

What install() changes. BaseDSL._prepare_compilation is wrapped in place: for a compile_only setup it sets
no_cache=False. It leaves the rest unchanged: a setup that already has the cache on, compile_to_precompiled_mlir, and
the DSL's own refusal when CUTE_DSL_KEEP_PTX / KEEP_CUBIN / KEEP_SASS is set. The DSL then does what it does for any
cached call (4.6.2: base_dsl/dsl.py compile_and_cache; 4.8.0: CompileContext.cache_enabled = not no_cache, then
base_dsl/compile_backend.py CompilerBackend._resolve_compile_cache / _finalize_cache):
- key: get_module_hash = the traced module's bytecode + every CUTE_DSL_* setting + the compile options + get_version()
  (cutlass_dsl/cutlass.py: the DSL's Python sources and its _cutlass_ir shared library);
- miss: the pass pipeline runs, then dump_cache_to_path writes the compiled module (its gpu.binary holds the cubin)
  with a CRC32;
- hit: load_cache_from_path reads it back and checks the CRC (a bad file is a warning and a miss); the loaded compiled
  module takes the place of the compiled one, so the cubin is the same bytes as on the miss.
A hit skips dsl.passes, never the trace: the key is the traced module.

Lines (engine stdout, flushed):
  PERFBOT-CUTECACHE on rank=<r> dsl=<nvidia-cutlass-dsl version> once, at install
  PERFBOT-CUTECACHE hit|miss <key[:12]> rank=<r>               one per cute.compile lookup
Any error at install prints "PERFBOT-CUTECACHE disabled: <error>" and leaves the DSL untouched.
"""

import functools
import importlib
import os

ENV = "PERFBOT_EXP_CUTE_CACHE"
TAG = "PERFBOT-CUTECACHE"
MARK = "_perfbot_cutecache"
KEEP = ("keep_ptx", "keep_cubin", "keep_sass")  # the DSL's own reasons to refuse the cache (_prepare_compilation)


def _rank():
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return str(dist.get_rank())
    except Exception:  # noqa: BLE001 - no rank is "-"
        pass
    return "-"


def _dsl_version():
    try:
        from importlib import metadata

        return metadata.version("nvidia-cutlass-dsl")
    except Exception:  # noqa: BLE001 - an unknown version is "-"
        return "-"


def cache_on(setup, envar):
    """True when this setup is a cute.compile (compile_only) that the DSL itself would cache but for compile_only."""
    return (setup.compile_only and setup.no_cache and not setup.compile_to_precompiled_mlir
            and not any(getattr(envar, k, False) for k in KEEP))


def lookup_module(dsl_module):
    """The module whose global load_cache_from_path the DSL calls: the sibling compile_backend (4.8.0), else dsl (4.6.2).

    4.8.0 dsl.py still star-imports cache_helpers, so its own load_cache_from_path exists but is never called.
    """
    try:
        return importlib.import_module(dsl_module.__name__.rpartition(".")[0] + ".compile_backend")
    except ImportError:
        return dsl_module


def install(env=os.environ, dsl_module=None):
    """Wrap BaseDSL._prepare_compilation and the DSL's load_cache_from_path; returns True when on."""
    if env.get(ENV) != "1":
        return False
    try:
        if dsl_module is None:
            from cutlass.base_dsl import dsl as dsl_module
        cls = dsl_module.BaseDSL
        if getattr(cls._prepare_compilation, MARK, False):
            return True
        rank = _rank()
        lookup = lookup_module(dsl_module)
        prepare, load = cls._prepare_compilation, lookup.load_cache_from_path

        @functools.wraps(prepare)
        def prepare_cached(self, *args, **kwargs):
            setup = prepare(self, *args, **kwargs)
            if cache_on(setup, self.envar):
                setup.no_cache = False
            return setup

        @functools.wraps(load)
        def load_logged(dsl_name, module_hash, *args, **kwargs):
            fn = load(dsl_name, module_hash, *args, **kwargs)
            print(f"{TAG} {'miss' if fn is None else 'hit'} {str(module_hash)[:12]} rank={rank}", flush=True)
            return fn

        setattr(prepare_cached, MARK, True)
        cls._prepare_compilation = prepare_cached
        lookup.load_cache_from_path = load_logged
        print(f"{TAG} on rank={rank} dsl={_dsl_version()}", flush=True)
        return True
    except Exception as e:  # noqa: BLE001 - the cache is an optimization; the engine builds without it
        print(f"{TAG} disabled: {type(e).__name__}: {e}", flush=True)
        return False
