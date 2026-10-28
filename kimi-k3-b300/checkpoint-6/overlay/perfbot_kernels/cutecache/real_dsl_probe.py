"""E17c offline proof: one real cute.compile under the PERFBOT-CUTECACHE patch, CPU only (no CUDA context).

    python3 real_dsl_probe.py <cache dir> <overlay dir> <out json> <site dir>...

Each <site dir> is a pip --target tree: the deps (cuda-bindings, numpy, typing_extensions) and one nvidia-cutlass-dsl
version with protobuf. Run twice with one cache dir: run 1 misses and writes, run 2 hits. test_cpu.py RealDsl runs it.
"""
import hashlib, json, os, site, sys, time, types
cache, overlay, out, *sites = sys.argv[1:]
for d in sites:
    site.addsitedir(d)
sys.modules.setdefault("torch", types.ModuleType("torch"))
os.environ.update(CUTE_DSL_CACHE_DIR=cache, CUTE_DSL_ARCH="sm_103a", PERFBOT_EXP_CUTE_CACHE="1")
sys.path.insert(0, overlay)
import cutlass, cutlass.cute as cute
from cutlass.base_dsl.compiler import Compiler
from cuda.bindings.driver import CUstream
import perfbot_kernels.cutecache as cc

passes = []
orig = Compiler.compile
def counted(self, *a, **k):
    passes.append(1)
    return orig(self, *a, **k)
Compiler.compile = counted
on = cc.install()

@cute.kernel
def scale(x: cutlass.Int32):
    tidx, _, _ = cute.arch.thread_idx()
    if tidx == x:
        cute.printf("x %d\n", x)

@cute.jit
def host(x: cutlass.Int32, stream: CUstream):
    scale(x).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)

t0 = time.time()
fn = cute.compile(host, cutlass.Int32(3), CUstream(0))
dt = time.time() - t0
import io
from cutlass._mlir import ir
names, bins = set(), []
def walk(op):
    names.add(op.name)
    if "binary" in op.name:
        bins.append(str(op))
    return ir.WalkResult.ADVANCE
fn.ir_module.operation.walk(walk)
buf = io.BytesIO()
fn.ir_module.operation.write_bytecode(buf)
files = sorted(os.listdir(cache)) if os.path.isdir(cache) else []
json.dump({"on": on, "dsl_version": cutlass.__version__ if hasattr(cutlass, "__version__") else None, "compile_s": round(dt, 3), "dsl_passes_calls": len(passes), "binary_ops": sorted(n for n in names if "binary" in n or "module" in n), "binary_sha256": [hashlib.sha256(b.encode()).hexdigest() for b in bins], "binary_bytes": [len(b) for b in bins], "module_bytecode_sha256": hashlib.sha256(buf.getvalue()).hexdigest(),
           "cache_files": files}, open(out, "w"), indent=1)
print(open(out).read())
