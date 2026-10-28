"""PERFBOT-PRETRACE: key each cute.compile before the trace; a hit loads the exported module and skips the trace
(ARM-TIME Option A, coordinator ruling 2026-10-20 18:45Z).

Why. With PERFBOT_EXP_CUTE_CACHE=1 the DSL pass pipeline is a file-cache hit, but every cute.compile still traces the
Python kernel to IR first, because the DSL's key is the traced module: 909 calls, 224 s self on rank 0, 231-234 s to
the last rank (restart-set-1 out-l4, 4 arms). A key made from the call's inputs needs no trace.

Switch: PERFBOT_EXP_CUTE_PRETRACE. The overlay model.py (KimiK3ForCausalLM.build_specialized_decode, tag
"# PERFBOT BUILD") reads it and calls install() before SpecializedDecode is built. Off ("" or "0"): this module does
not load.
- "1": a hit loads <dir>/<key>.o (cutlass.runtime.load_module); a miss traces as today and exports the result there.
- "export": every call traces and exports; never a hit (the proof's pass 1, perfbot_kernels/pretrace/proof.py).
<dir> = PERFBOT_PRETRACE_DIR, else $CUTE_DSL_CACHE_DIR/perfbot-pretrace (serve.sh: the arm's exported cache tree).

Key = sha256 of: the image (vllm, torch, nvidia-cutlass-dsl versions, CUDA_VERSION), the overlay tree sha256 (every
file under the overlay root), the arch (CUTE_DSL_ARCH, device capability and name), the env that can change a trace
(CUTE_DSL_* but paths / logging / profiling, PERFBOT_EXP_* but the build-time log and cache switches, VLLM_* but ports,
hosts, ranks, paths, timeouts, logging; never an absolute-path value: serve.sh gives each arm its own VLLM_CACHE_ROOT), the compile options, the kernel function (module, qualname, sha256 of its
source file, its closure cells) and a spec of every argument (values of scalars and config objects, dtype / shape /
stride of tensors, never data or addresses). An input that has no spec gives no key: that call traces and is not
exported, never a stale hit. A TVM-FFI compile (options carry enable-tvm-ffi) traces: its loaded form is another type.
A function whose trace reads a global that none of the above covers is the residual risk: the proof's collision check
(one key, two cubins) and the first served A/A pair measure it.

Lines (stdout, flushed):
  PERFBOT-PRETRACE on mode=<1|export> rank=<r> dir=<dir> overlay=<sha[:12]>         once, at install
  PERFBOT-PRETRACE build rank=<r> calls=<n> keyed=<n> hit=<n> miss=<n> unkeyed=<n> trace_s=<s> load_s=<s>
                   export_s=<s> key_s=<s> cubins=<n>:<sha[:12]>                        at the end of SpecializedDecode
  cubins = the distinct CUDA cubins of every keyed call's module object (a hit: the loaded file; a miss: the exported
  bytes) and sha256 over their sorted shas. An arm that traces every call (mode "export", an empty dir) and an arm that
  loads every call print the same value when the cache serves the bytes a trace makes.
  PERFBOT-PRETRACE disabled: <error>                                                   install failed; DSL untouched
PERFBOT_PRETRACE_RECORD=<dir>: one JSON record per call in <dir>/rank-<r>.jsonl and <dir>/rank-<r>.done.json at the
end of SpecializedDecode (proof only: the record mode also hashes each traced module and extracts its cubins).
"""

import functools
import hashlib
import importlib
import json
import os
import struct
import sys
import threading
import time

ENV = "PERFBOT_EXP_CUTE_PRETRACE"
DIR_ENV = "PERFBOT_PRETRACE_DIR"
RECORD_ENV = "PERFBOT_PRETRACE_RECORD"
TAG = "PERFBOT-PRETRACE"
MARK = "_perfbot_pretrace"
DECODE = "vllm.models.kimi_k3.nvidia.specialized_decode.runtime.decode"
OVERLAY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PACKAGES = ("vllm", "torch", "nvidia-cutlass-dsl", "nvidia-cutlass-dsl-libs-base")
ENV_DROP_WORDS = ("DIR", "PATH", "PORT", "HOST", "RANK", "TIMEOUT", "LOG", "PROFIL")
ENV_DROP = {ENV, "PERFBOT_EXP_INITTIME", "PERFBOT_EXP_CUTE_CACHE"}
MAX_NODES = 20000


class Unkeyable(Exception):
    """An input with no spec: the call traces."""


def env_part(env):
    """The env that can change a trace, as sorted (name, value) pairs."""
    keep = []
    for k, v in env.items():
        if k in ENV_DROP or not k.startswith(("CUTE_DSL_", "PERFBOT_EXP_", "VLLM_")) and k != "CUDA_VERSION":
            continue
        if any(w in k for w in ENV_DROP_WORDS) or os.path.isabs(v):
            continue
        keep.append((k, v))
    return sorted(keep)


def tree_sha(root):
    """sha256 over every file under root (relative path + bytes), __pycache__ and *.pyc excluded."""
    h = hashlib.sha256()
    for d, dirs, files in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x != "__pycache__")
        for f in sorted(files):
            if f.endswith(".pyc"):
                continue
            p = os.path.join(d, f)
            h.update(os.path.relpath(p, root).encode() + b"\0")
            with open(p, "rb") as fh:
                h.update(hashlib.sha256(fh.read()).digest())
    return h.hexdigest()


def file_sha(path):
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        raise Unkeyable("no source file %s" % path)


def unwrap(func, args):
    """The function cute.compile traces and its argument list (CompileCallable._compile's own resolution)."""
    import inspect

    args = list(args)
    if inspect.ismethod(func):
        args, func = [func.__self__] + args, func.__func__
    elif not inspect.isfunction(func) and hasattr(func, "__call__") and hasattr(func.__call__, "__func__"):
        args, func = [func] + args, func.__call__.__func__
    while hasattr(func, "__wrapped__"):
        func = func.__wrapped__
    return func, args


class Spec:
    """A canonical, address-free description of compile inputs. Raises Unkeyable for anything it cannot describe."""

    def __init__(self):
        self.nodes, self.active = 0, set()

    def fn(self, f):
        import inspect

        try:
            src = inspect.getsourcefile(f)
        except TypeError:
            raise Unkeyable("no source for %r" % type(f).__name__)
        cells = [self(c.cell_contents) for c in (getattr(f, "__closure__", None) or ())]
        return ["fn", getattr(f, "__module__", None), getattr(f, "__qualname__", None), file_sha(src), cells]

    def __call__(self, x):
        import enum
        import inspect

        self.nodes += 1
        if self.nodes > MAX_NODES:
            raise Unkeyable("more than %d nodes" % MAX_NODES)
        if x is None or isinstance(x, (bool, int, float, str, bytes)):
            return [type(x).__name__, repr(x)]
        t = type(x)
        name = "%s.%s" % (t.__module__, t.__qualname__)
        if isinstance(x, enum.Enum):
            return ["enum", name, x.name]
        if isinstance(x, type):
            return ["type", "%s.%s" % (x.__module__, x.__qualname__)]
        if inspect.isfunction(x) or inspect.ismethod(x):
            f, a = unwrap(x, ())
            return ["callable", self.fn(f), [self(v) for v in a]]
        if isinstance(x, functools.partial):
            return ["partial", self(x.func), [self(v) for v in x.args], self(dict(x.keywords))]
        if t.__name__ in ("CUstream", "cudaStream_t"):
            return ["stream"]  # a runtime argument: compile-time streams are fakes (make_fake_stream)
        if t.__module__.startswith("torch"):
            if t.__name__ in ("Tensor", "Parameter"):
                return ["tensor", str(x.dtype), list(x.shape), list(x.stride()), x.device.type]
            if t.__name__ in ("dtype", "device", "Size"):
                return [name, str(x)]
            raise Unkeyable(name)
        if any(c.__module__.startswith("torch") for c in t.__mro__[1:]):
            raise Unkeyable("torch subclass %s" % name)  # a module or tensor subclass: state is weights, not a spec
        if id(x) in self.active:
            raise Unkeyable("cycle at %s" % name)
        self.active.add(id(x))
        try:
            if isinstance(x, (list, tuple)):
                return [name, [self(v) for v in x]]
            if isinstance(x, dict):
                items = sorted(((self(k), self(v)) for k, v in x.items()), key=repr)
                return [name, items]
            if isinstance(x, (set, frozenset)):
                return [name, sorted((self(v) for v in x), key=repr)]
            state = getattr(x, "__dict__", None)
            slots = [s for c in t.__mro__ for s in getattr(c, "__slots__", ()) if s != "__dict__" and s != "__weakref__"]
            if state is not None or slots:
                d = dict(state or {})
                for s in slots:
                    if hasattr(x, s):
                        d[s] = getattr(x, s)
                return [name, sorted(([k, self(v)] for k, v in d.items()), key=repr)]
            text = str(x)  # C-extension values (MLIR types, cutlass numerics): their text, when it names no address
            if "0x" in text or len(text) > 4096:
                raise Unkeyable("no spec for %s" % name)
            return ["text", name, text]
        finally:
            self.active.discard(id(x))


def arch_part():
    try:
        import torch

        if torch.cuda.is_available():
            i = torch.cuda.current_device()
            return [list(torch.cuda.get_device_capability(i)), torch.cuda.get_device_name(i)]
    except Exception:  # noqa: BLE001 - no device: the env's CUTE_DSL_ARCH alone
        pass
    return None


def image_part():
    from importlib import metadata

    out = []
    for p in PACKAGES:
        try:
            out.append([p, metadata.version(p)])
        except Exception:  # noqa: BLE001 - an absent package is part of the image too
            out.append([p, None])
    return out + [["python", sys.version.split()[0]]]


def tvm_ffi(options_kw, compile_options, env):
    """True when this compile is TVM-FFI: the options string, the CompileCallable's EnableTVMFFI, or the DSL env."""
    on = [o.value for c, o in getattr(compile_options, "options", {}).items() if c.__name__ == "EnableTVMFFI"]
    return ("enable-tvm-ffi" in str(options_kw or "") or any(bool(v) for v in on)
            or env.get("CUTE_DSL_ENABLE_TVM_FFI", "0").lower() not in ("", "0", "false"))


def make_key(base, func, args, kwargs, options, parts=None):
    """sha256 hex of base (image, overlay, arch, env) + options + function + argument specs; Unkeyable if no spec.
    parts (a dict) receives a short sha per part: a cross-process miss names the part that moved."""
    f, a = unwrap(func, args)
    s = Spec()
    body = {"base": base, "options": s(options), "fn": s.fn(f), "args": [s(v) for v in a], "kwargs": s(dict(kwargs))}
    if parts is not None:
        parts.update({k: hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest()[:12]
                      for k, v in body.items()})
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def cubins(blob):
    """sha256 of every embedded CUDA cubin (ELF e_machine 190), sliced at its section-header table end."""
    out, i = [], blob.find(b"\x7fELF", 1)
    while i > 0:
        try:
            if struct.unpack_from("<H", blob, i + 18)[0] == 190:
                shoff = struct.unpack_from("<Q", blob, i + 40)[0]
                shentsize, shnum = struct.unpack_from("<HH", blob, i + 58)
                end = i + shoff + shentsize * shnum
                if end <= len(blob):
                    out.append(hashlib.sha256(blob[i:end]).hexdigest())
        except struct.error:
            pass
        i = blob.find(b"\x7fELF", i + 1)
    return out


def module_bytes(fn):
    import io

    buf = io.BytesIO()
    fn.ir_module.operation.write_bytecode(buf)
    return b"x" + buf.getvalue()  # a leading byte: cubins() looks past offset 0


def prefix(key):
    return "pt_" + key[:40]


class State:
    def __init__(self, mode, root, record, rank, base):
        self.mode, self.root, self.record, self.rank, self.base = mode, root, record, rank, base
        self.memo, self.lock = {}, threading.Lock()
        self.n = {"calls": 0, "keyed": 0, "hit": 0, "miss": 0, "unkeyed": 0}
        self.s = {"trace_s": 0.0, "load_s": 0.0, "export_s": 0.0, "key_s": 0.0}
        self.phase = threading.local()
        self.out = None
        self.served = set()

    def serve(self, blob):
        found = cubins(blob)
        with self.lock:
            self.served.update(found)

    def emit(self, rec):
        if self.record:
            if self.out is None:
                os.makedirs(self.record, exist_ok=True)
                self.out = open(os.path.join(self.record, "rank-%s.jsonl" % self.rank), "a")
            self.out.write(json.dumps(rec) + "\n")
            self.out.flush()

    def export(self, key, fn):
        path = os.path.join(self.root, key + ".o")
        tmp = "%s.%d.tmp" % (path, os.getpid())
        blob = fn.dump_to_object(prefix(key))
        with open(tmp, "wb") as fh:
            fh.write(blob)
        os.replace(tmp, path)
        self.serve(blob)  # ranks race to write the same key: same bytes, last rename wins

    def load(self, key):
        path = os.path.join(self.root, key + ".o")
        if not os.path.isfile(path):
            return None
        import cutlass.runtime as rt

        return rt.load_module(path)[prefix(key)]

    def call(self, orig, callable_self, args, kwargs):
        rec = {"rank": self.rank}
        t0 = time.time()
        func = args[0] if args else None
        options = [kwargs.get("options"), getattr(callable_self, "_compile_options", None)]
        key, parts = None, {} if self.record else None
        try:
            if tvm_ffi(options[0], options[1], os.environ):
                raise Unkeyable("tvm_ffi")
            key = make_key(self.base, func, args[1:], {k: v for k, v in kwargs.items() if k != "options"}, options,
                           parts)
            if parts:
                rec["parts"] = parts
        except Unkeyable as e:
            rec["unkeyed"] = str(e)[:200]
        except Exception as e:  # noqa: BLE001 - a spec bug is an unkeyed call, never a broken compile
            rec["unkeyed"] = "%s: %s" % (type(e).__name__, str(e)[:160])
        t1 = time.time()
        rec.update(key=key, fn=getattr(func, "__qualname__", type(func).__name__), key_s=round(t1 - t0, 4))
        with self.lock:
            self.n["calls"] += 1
            self.s["key_s"] += t1 - t0
            self.n["keyed" if key else "unkeyed"] += 1
        if key and self.mode == "1":
            if key in self.memo:
                rec["hit"] = "memo"
                self.emit(rec)
                return self.memo[key]
            try:
                fn = self.load(key)
            except Exception as e:  # noqa: BLE001 - an unreadable module is a miss
                fn, rec["load_error"] = None, "%s: %s" % (type(e).__name__, str(e)[:160])
            if fn is not None:
                with open(os.path.join(self.root, key + ".o"), "rb") as fh:
                    self.serve(fh.read())
                dt = time.time() - t1
                rec.update(hit=True, load_s=round(dt, 4))
                with self.lock:
                    self.n["hit"] += 1
                    self.s["load_s"] += dt
                self.memo[key] = fn
                self.emit(rec)
                return fn
        self.phase.passes = self.phase.hash = 0.0
        fn = orig(callable_self, *args, **kwargs)
        t2 = time.time()
        rec.update(hit=False, call_s=round(t2 - t1, 4), passes_s=round(self.phase.passes, 4),
                   hash_s=round(self.phase.hash, 4))
        with self.lock:
            self.s["trace_s"] += t2 - t1
            if key:
                self.n["miss"] += 1
        if key:
            try:
                self.export(key, fn)
                rec["export_s"] = round(time.time() - t2, 4)
                with self.lock:
                    self.s["export_s"] += time.time() - t2
                if self.mode == "1":
                    self.memo[key] = fn
            except Exception as e:  # noqa: BLE001 - not exported: the next start traces this call again
                rec["export_error"] = "%s: %s" % (type(e).__name__, str(e)[:160])
        if self.record:
            try:
                blob = module_bytes(fn)
                rec.update(module=hashlib.sha256(blob).hexdigest(), cubins=cubins(blob),
                           kernel_info=sorted((getattr(fn, "kernel_info", None) or {}).keys()))
            except Exception as e:  # noqa: BLE001
                rec["module_error"] = "%s: %s" % (type(e).__name__, str(e)[:160])
        self.emit(rec)
        return fn

    def summary(self):
        s = {k: round(v, 3) for k, v in self.s.items()}
        digest = hashlib.sha256("\n".join(sorted(self.served)).encode()).hexdigest()[:12]
        line = "%s build rank=%s %s %s cubins=%d:%s" % (TAG, self.rank, " ".join("%s=%d" % kv for kv in self.n.items()),
                                                    " ".join("%s=%.3f" % kv for kv in s.items()), len(self.served), digest)
        print(line, flush=True)
        if self.out is not None:
            self.out.close()
            self.out = None
        if self.record:
            os.makedirs(self.record, exist_ok=True)
            with open(os.path.join(self.record, "rank-%s.done.json" % self.rank), "w") as fh:
                json.dump({"rank": self.rank, "counts": self.n, "seconds": s, "base": self.base}, fh, sort_keys=True)
        return line


def _rank():
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return str(dist.get_rank())
    except Exception:  # noqa: BLE001 - no rank is "-"
        pass
    return "-"


def _timed(state, attr, fn):
    @functools.wraps(fn)
    def run(*a, **k):
        t0 = time.time()
        try:
            return fn(*a, **k)
        finally:
            setattr(state.phase, attr, getattr(state.phase, attr, 0.0) + time.time() - t0)

    return run


def install(env=None, decode=None, compile_callable=None):
    """Wrap cute.compile (and, at the end of SpecializedDecode.__init__, print the build line) when env[ENV] is "1" or
    "export"; returns the State, or None when off or failed."""
    env = os.environ if env is None else env
    mode = env.get(ENV, "")
    if mode not in ("1", "export"):
        return None
    try:
        if compile_callable is None:
            import cutlass.cute as cute

            compile_callable = type(cute.compile)
        if getattr(compile_callable.__call__, MARK, None) is not None:
            return getattr(compile_callable.__call__, MARK)
        root = env.get(DIR_ENV) or os.path.join(env.get("CUTE_DSL_CACHE_DIR") or os.path.join(OVERLAY, ".."),
                                                "perfbot-pretrace")
        os.makedirs(root, exist_ok=True)
        overlay = tree_sha(OVERLAY)
        base = {"image": image_part(), "overlay": overlay, "arch": arch_part(), "env": env_part(env)}
        state = State(mode, root, env.get(RECORD_ENV), _rank(), base)
        orig = compile_callable.__call__

        @functools.wraps(orig)
        def compile_keyed(self, *args, **kwargs):
            return state.call(orig, self, args, kwargs)

        setattr(compile_keyed, MARK, state)
        compile_callable.__call__ = compile_keyed
        try:
            from cutlass.base_dsl.compiler import Compiler
            from cutlass.base_dsl.dsl import BaseDSL

            Compiler.compile = _timed(state, "passes", Compiler.compile)
            BaseDSL.get_module_hash = _timed(state, "hash", BaseDSL.get_module_hash)
        except Exception:  # noqa: BLE001 - the split is evidence only
            pass
        if decode is None:
            decode = importlib.import_module(DECODE)
        cls = decode.SpecializedDecode
        init = cls.__init__

        @functools.wraps(init)
        def init_then_summary(self, *a, **k):
            init(self, *a, **k)
            state.summary()

        cls.__init__ = init_then_summary
        print("%s on mode=%s rank=%s dir=%s overlay=%s" % (TAG, mode, state.rank, root, overlay[:12]), flush=True)
        return state
    except Exception as e:  # noqa: BLE001 - the pre-trace key is an optimization; the engine builds without it
        print("%s disabled: %s: %s" % (TAG, type(e).__name__, e), flush=True)
        return None
