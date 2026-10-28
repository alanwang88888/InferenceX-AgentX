"""PRETRACE offline proof: real cute.compile calls under perfbot_kernels.pretrace, CPU only (no CUDA context).

    python3 real_dsl_probe.py <module dir> <out json> <site dir>...

Run twice with one module dir: run 1 traces and exports (miss), run 2 loads (hit) with no trace. Each run compiles
the same kernel at x=3 twice (the second is a memo or a hit) and once at x=4 (another key). Each <site dir> is a pip
--target tree (deps + one nvidia-cutlass-dsl version), as for perfbot_kernels/cutecache/real_dsl_probe.py.
"""
import json, os, site, sys, types
mdir, out, *sites = sys.argv[1:]
for d in sites:
    site.addsitedir(d)
sys.modules.setdefault("torch", types.ModuleType("torch"))
os.environ.update(CUTE_DSL_ARCH="sm_103a")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import cutlass, cutlass.cute as cute
from cutlass.base_dsl.dsl import BaseDSL
from cuda.bindings.driver import CUstream
import perfbot_kernels.pretrace as pt

traces = []
orig = BaseDSL.generate_mlir
def counted(self, *a, **k):
    traces.append(1)
    return orig(self, *a, **k)
BaseDSL.generate_mlir = counted

class SpecializedDecode:
    pass

st = pt.install({pt.ENV: "1", pt.DIR_ENV: mdir, pt.RECORD_ENV: out + ".rec"}, decode=types.SimpleNamespace(SpecializedDecode=SpecializedDecode))

@cute.kernel
def scale(x: cutlass.Int32):
    tidx, _, _ = cute.arch.thread_idx()
    if tidx == x:
        cute.printf("x %d\n", x)

@cute.jit
def host(x: cutlass.Int32, stream: CUstream):
    scale(x).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)

fns = [cute.compile(host, cutlass.Int32(v), CUstream(0)) for v in (3, 3, 4)]
files = sorted(f for f in os.listdir(mdir) if f.endswith(".o"))
cub = {f[:-2]: pt.cubins(open(os.path.join(mdir, f), "rb").read()) for f in files}
traced = {}
for f in fns:
    if not getattr(f, "load_from_binary", False):
        traced.setdefault(id(f), pt.cubins(pt.module_bytes(f)))
res = {"on": st is not None, "counts": st.n if st else None, "traces": len(traces), "files": len(files),
       "file_cubins": cub, "traced_cubins": sorted(traced.values()), "types": [type(f).__name__ for f in fns],
       "load_from_binary": [bool(getattr(f, "load_from_binary", False)) for f in fns],
       "memo_same_object": fns[0] is fns[1],
       "parts": [json.loads(x).get("parts") for x in open(os.path.join(out + ".rec", "rank--.jsonl"))]}
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res, indent=1))
