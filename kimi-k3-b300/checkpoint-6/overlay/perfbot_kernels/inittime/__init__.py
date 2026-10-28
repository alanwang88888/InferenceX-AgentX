"""PERFBOT-INITTIME: evidence log of the specialized-decode build time (ARM-TIME item 1).

Switch: PERFBOT_EXP_INITTIME=1. Only the value "1" is on. The overlay model.py (KimiK3ForCausalLM.
build_specialized_decode, tag "# PERFBOT BUILD: INITTIME") reads the switch and calls install() just before it builds
SpecializedDecode. With the switch off, this module does not load and nothing prints.

Why. In a latency_specialized server about 292 s pass between "Model loading took" and the next engine line. Only
init_model_state runs there: KimiK3ModelState.__init__ -> model.build_specialized_decode() -> SpecializedDecode(...),
whose build_providers() does "all collective allocations, weight packing and kernel compiles". This log splits that time.

What install() times (NAMES). Each name is a path in the decode module's namespace
(vllm/models/kimi_k3/nvidia/specialized_decode/runtime/decode.py at 49cb2bbf):
- a class: its __init__ is wrapped in place. The class object, isinstance and the returned object do not change.
- a function in the namespace (pack_front): the namespace name is replaced by a wrapper that returns the same object.
- a method (Class.attr): the method is wrapped in place on the class.
Plus "cute.compile": CompileCallable.__call__ of cutlass.cute.compile, the one call that every explicit CuTe DSL
compile of the specialized decode goes through (cutedsl/**: `cute.compile(...)`). A cache hit in a compile_* @cache
function makes no cute.compile call.

Plus the two DSL phases under each cute.compile (E20; nvidia-cutlass-dsl 4.6.2 and 4.8.0, cutlass/base_dsl/compiler.py):
"dsl.passes" = Compiler.compile, the MLIR pass pipeline (lowering through the cubin), and "dsl.jit_engine" =
Compiler.jit, the host ExecutionEngine. The self_s of cute.compile is then the Python trace to IR plus the bookkeeping
around it. A compile cache that loads a cubin skips the two phases, never the trace.

Roots. ROOTS are the outer calls: SpecializedDecode (its __init__, the whole build) and SpecializedDecode.prepare_kv
(the per-KV-layout core compiles, after the build). A wrapped call outside a root, or on another thread, passes through
and prints nothing, so cute.compile calls of the rest of the engine are not logged.

Lines (engine stdout, flushed):
  PERFBOT-INITTIME <name> start=<unix s> dt_s=<s> rank=<r> self_s=<s>     one per wrapped call, at its end
  PERFBOT-INITTIME-SUM <name> count=<n> sum_s=<s> self_s=<s> max_s=<s> first_s=<s> rank=<r>
                                                                          one per name, at the end of each root call
dt_s is inclusive: nested wrapped calls (cute.compile inside PreparedMoE, ...) count in both. self_s is dt_s minus the
nested wrapped calls, so the self_s values of one root add up to the root's dt_s. first_s is the first call (it often
holds the compile). All times are host wall time: GPU work that a call launches and does not wait for is not in it.
Rank: torch.distributed global rank, else "-".

Cost when on: two perf_counter reads, one list push and pop and one print per wrapped call (a few thousand lines per
rank per build). No GPU work, no sync. Any error prints "PERFBOT-INITTIME disabled: <error>" once and turns the log
off; the wrapped calls then pass through. An error never reaches the engine.
"""

import functools
import importlib
import inspect
import os
import threading
import time

ENV = "PERFBOT_EXP_INITTIME"
TAG = "PERFBOT-INITTIME"
DECODE = "vllm.models.kimi_k3.nvidia.specialized_decode.runtime.decode"
NAMES = (
    "SpecializedDecode", "SpecializedDecode.build_providers", "SpecializedDecode.prepare_kv",
    "pack_front", "PrefetchController", "MoEScratch", "OprojMailbox", "MoeTailBox", "TwoShot",
    "small_batch_chain.SmallBatchBox", "Attention", "PreparedDenseMLP", "SharedFront", "PersistentExperts",
    "PreparedTail", "PreparedMoE", "small_batch_chain.SmallBatchChain", "PreparedEmbedding", "PreparedAttnRes",
    "PreparedAttnRes.bind_ring", "PreparedAttnRes.bind_aux",
)
COMPILE = "cute.compile"
PHASES = (("compile", "dsl.passes"), ("jit", "dsl.jit_engine"))  # Compiler attribute, logged name
ROOTS = ("SpecializedDecode", "SpecializedDecode.prepare_kv")
MARK = "_perfbot_inittime"


class Log:
    def __init__(self):
        self.on, self.rank, self.tid = True, "-", None
        self.stack = []  # [name, start unix s, start perf s, nested s]
        self.agg = {}  # name -> [count, sum, self, max, first]

    def off(self, e):
        if self.on:
            self.on, self.stack, self.agg = False, [], {}
            print(f"{TAG} disabled: {type(e).__name__}: {e}", flush=True)

    def enter(self, name):
        """A frame to close with leave(), or None: off, or outside a root, or on another thread."""
        if not self.stack:
            if name not in ROOTS:
                return None
            self.tid = threading.get_ident()
        elif threading.get_ident() != self.tid:
            return None
        frame = [name, time.time(), time.perf_counter(), 0.0]
        self.stack.append(frame)
        return frame

    def leave(self, frame):
        dt = time.perf_counter() - frame[2]
        if not self.stack or self.stack[-1] is not frame:
            raise RuntimeError(f"unbalanced frame {frame[0]}")
        self.stack.pop()
        own = dt - frame[3]
        if self.stack:
            self.stack[-1][3] += dt
        a = self.agg.setdefault(frame[0], [0, 0.0, 0.0, 0.0, dt])
        a[0], a[1], a[2], a[3] = a[0] + 1, a[1] + dt, a[2] + own, max(a[3], dt)
        print(f"{TAG} {frame[0]} start={frame[1]:.3f} dt_s={dt:.3f} rank={self.rank} self_s={own:.3f}", flush=True)
        if not self.stack:
            for name, (n, s, own_s, mx, first) in sorted(self.agg.items(), key=lambda kv: -kv[1][1]):
                print(f"{TAG}-SUM {name} count={n} sum_s={s:.3f} self_s={own_s:.3f} max_s={mx:.3f} "
                      f"first_s={first:.3f} rank={self.rank}", flush=True)
            self.agg = {}


LOG = Log()


def wrap(name, fn):
    """fn with the same arguments and result; the log part never raises into the caller."""

    @functools.wraps(fn)
    def timed(*args, **kwargs):
        frame = None
        if LOG.on:
            try:
                frame = LOG.enter(name)
            except Exception as e:  # noqa: BLE001 - a log must never stop the engine
                LOG.off(e)
        if frame is None:
            return fn(*args, **kwargs)
        try:
            return fn(*args, **kwargs)
        finally:
            if LOG.on:
                try:
                    LOG.leave(frame)
                except Exception as e:  # noqa: BLE001
                    LOG.off(e)

    setattr(timed, MARK, True)
    return timed


def patch(owner, attr, name):
    """Wrap owner.attr in place (a class attribute, or a module namespace name). Twice = once."""
    raw = inspect.getattr_static(owner, attr)
    inner = raw.__func__ if isinstance(raw, (classmethod, staticmethod)) else raw
    if getattr(inner, MARK, False):
        return
    timed = wrap(name, inner)
    setattr(owner, attr, type(raw)(timed) if inner is not raw else timed)


def target(root, path):
    """(owner, attr) of a NAMES path: a class means its __init__."""
    *head, last = path.split(".")
    owner = root
    for part in head:
        owner = getattr(owner, part)
    obj = getattr(owner, last)
    return (obj, "__init__") if isinstance(obj, type) else (owner, last)


def install(env=None, decode=None, compile_callable=None, compiler=None):
    """Wrap NAMES, cute.compile and the DSL PHASES when env[ENV] == "1"; else do nothing. Returns the number of wrapped
    names."""
    env = os.environ if env is None else env
    if env.get(ENV) != "1" or not LOG.on:
        return 0
    try:
        if decode is None:
            decode = importlib.import_module(DECODE)
        if compile_callable is None:
            import cutlass.cute as cute

            compile_callable = type(cute.compile)
        if compiler is None:
            from cutlass.base_dsl.compiler import Compiler as compiler
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                LOG.rank = str(dist.get_rank())
        except Exception:  # noqa: BLE001 - no rank is "-"
            pass
        sites = [(*target(decode, path), path) for path in NAMES] + [(compile_callable, "__call__", COMPILE)]
        sites += [(compiler, attr, name) for attr, name in PHASES]
        for owner, attr, name in sites:
            patch(owner, attr, name)
        print(f"{TAG} on: {len(sites)} names rank={LOG.rank}", flush=True)
        return len(sites)
    except Exception as e:  # noqa: BLE001
        LOG.off(e)
        return 0
