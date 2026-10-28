"""CPU tests of perfbot_kernels.pretrace (ARM-TIME Option A). No GPU, no DSL (real_dsl_probe.py covers the DSL).

  python3 -I overlay/perfbot_kernels/pretrace/test_cpu.py
"""

import io
import json
import os
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import perfbot_kernels.pretrace as pt  # noqa: E402
from perfbot_kernels.pretrace import proof  # noqa: E402

BASE = {"image": [["vllm", "x"]], "overlay": "o", "arch": None, "env": []}


def kernel(a, b):
    return a + b


class Cfg:
    def __init__(self, tile, dtype="bf16"):
        self.tile, self.dtype = tile, dtype


class Addr:
    def __str__(self):
        return "<obj at 0x7f00>"

    __slots__ = ()


def key(*args, func=kernel, **kwargs):
    return pt.make_key(BASE, func, args, kwargs, ["opt", None])


class Key(unittest.TestCase):
    def test_same_inputs_same_key_and_any_changed_input_another(self):
        k = key(1, Cfg(128))
        self.assertEqual(k, key(1, Cfg(128)))
        for other in (key(2, Cfg(128)), key(1, Cfg(64)), key(1, Cfg(128, "fp8")), key(1, Cfg(128), x=1),
                      pt.make_key(dict(BASE, overlay="p"), kernel, (1, Cfg(128)), {}, ["opt", None]),
                      pt.make_key(BASE, kernel, (1, Cfg(128)), {}, ["opt2", None])):
            self.assertNotEqual(k, other)

    def test_a_closure_cell_and_a_bound_instance_are_inputs(self):
        def make(n):
            def inner(x):
                return x + n
            return inner

        self.assertNotEqual(key(1, func=make(1)), key(1, func=make(2)))

        class K:
            def __init__(self, tile):
                self.tile = tile

            def __call__(self, x):
                return x

        self.assertNotEqual(key(1, func=K(1)), key(1, func=K(2)))
        self.assertNotEqual(key(1, func=K(1).__call__), key(1, func=K(2).__call__))

    def test_an_address_a_cycle_or_a_huge_input_gives_no_key(self):
        for bad in (Addr(), [list(range(10))] * 3000):
            with self.assertRaises(pt.Unkeyable):
                key(bad)
        c = Cfg(1)
        c.self = c
        with self.assertRaises(pt.Unkeyable):
            key(c)

    def test_tvm_ffi_is_its_value_never_its_name(self):  # every default CompileOptions holds EnableTVMFFI(False)
        EnableTVMFFI = type("EnableTVMFFI", (), {})
        opts = lambda v: types.SimpleNamespace(options={EnableTVMFFI: types.SimpleNamespace(value=v)})
        self.assertFalse(pt.tvm_ffi(None, opts(False), {}))
        self.assertTrue(pt.tvm_ffi(None, opts(True), {}))
        self.assertTrue(pt.tvm_ffi("--enable-tvm-ffi", opts(False), {}))
        self.assertTrue(pt.tvm_ffi(None, None, {"CUTE_DSL_ENABLE_TVM_FFI": "1"}))

    def test_streams_are_runtime_arguments(self):
        CUstream = type("CUstream", (), {"__init__": lambda self, v: setattr(self, "v", v)})
        self.assertEqual(key(CUstream(0)), key(CUstream(7)))

    def test_env_part_keeps_trace_switches_and_drops_paths_ranks_and_build_logs_and_locations(self):
        env = {"CUTE_DSL_ARCH": "sm_103a", "CUTE_DSL_CACHE_DIR": "/x", "PERFBOT_EXP_I15": "1",
               "PERFBOT_EXP_INITTIME": "1", "PERFBOT_EXP_CUTE_CACHE": "1", pt.ENV: "1", "VLLM_KIMI_K3_AUX": "0",
               "VLLM_PORT": "1", "VLLM_DP_RANK": "3", "HOME": "/h", "CUDA_VERSION": "13.0.2",
               "VLLM_CACHE_ROOT": "/workspace/results/a/.arm-cache.Xy12/home/.cache/vllm",
               "CUTE_DSL_JIT_TIME_PROFILING": "1"}
        self.assertEqual(pt.env_part(env), [("CUDA_VERSION", "13.0.2"), ("CUTE_DSL_ARCH", "sm_103a"),
                                            ("PERFBOT_EXP_I15", "1"), ("VLLM_KIMI_K3_AUX", "0")])

    def test_cubins_finds_each_embedded_cuda_elf_by_its_own_length(self):
        def elf(machine, body):
            h = bytearray(64)
            h[0:4] = b"\x7fELF"
            h[18:20] = machine.to_bytes(2, "little")
            h[40:48] = (64 + len(body)).to_bytes(8, "little")
            h[58:60] = (64).to_bytes(2, "little")
            h[60:62] = (1).to_bytes(2, "little")
            return bytes(h) + body + bytes(64)

        a, b = elf(190, b"cubin-a"), elf(62, b"host")
        found = pt.cubins(b"pad" + b + a + b"tail")
        self.assertEqual(found, [__import__("hashlib").sha256(a).hexdigest()])


class FakeFn:
    def __init__(self, tag):
        self.tag = tag

    def dump_to_object(self, prefix):
        return ("obj:%s:%s" % (prefix, self.tag)).encode()


class Wrapper(unittest.TestCase):
    def state(self, mode, root, record=None):
        return pt.State(mode, root, record, "0", BASE)

    def test_mode_1_traces_and_exports_a_miss_then_loads_it_in_a_new_process_without_a_trace(self):
        with tempfile.TemporaryDirectory() as root:
            traced = []

            def orig(self, func, *a, **k):
                traced.append(a)
                return FakeFn(a)

            s1 = self.state("1", root)
            fn = s1.call(orig, None, (kernel, 1, Cfg(8)), {})
            self.assertEqual(len(traced), 1)
            [o] = os.listdir(root)
            self.assertTrue(o.endswith(".o") and open(os.path.join(root, o), "rb").read().startswith(b"obj:pt_"))
            self.assertIs(s1.call(orig, None, (kernel, 1, Cfg(8)), {}), fn)  # same process: memo
            s2 = self.state("1", root)
            loaded = []
            s2.load = lambda k: loaded.append(k) or ("LOADED" if os.path.exists(os.path.join(root, k + ".o")) else None)
            self.assertEqual(s2.call(orig, None, (kernel, 1, Cfg(8)), {}), "LOADED")
            self.assertEqual((len(traced), loaded, s2.n["hit"]), (1, [o[:-2]], 1))
            loaded.clear()
            s2.call(orig, None, (kernel, 2, Cfg(8)), {})  # another input: a miss, a trace
            self.assertEqual((len(traced), s2.n["miss"]), (2, 1))

    def test_a_tracing_arm_and_a_loading_arm_print_the_same_cubin_digest(self):
        def elf(body):
            h = bytearray(64)
            h[0:4], h[18:20] = b"\x7fELF", (190).to_bytes(2, "little")
            h[40:48], h[58:60], h[60:62] = (64 + len(body)).to_bytes(8, "little"), (64).to_bytes(2, "little"), (1).to_bytes(2, "little")
            return b"x" + bytes(h) + body + bytes(64)

        class ElfFn(FakeFn):
            def dump_to_object(self, prefix):
                return elf(repr(self.tag).encode())

        with tempfile.TemporaryDirectory() as root:
            orig = lambda self, func, *a, **k: ElfFn(a)
            calls = [(kernel, 1, Cfg(8)), (kernel, 2, Cfg(8))]
            trace = self.state("export", root)
            for c in calls:
                trace.call(orig, None, c, {})
            warm = self.state("1", root)
            warm.load = lambda k: "LOADED"
            for c in calls:
                warm.call(orig, None, c, {})
            self.assertEqual(warm.n["hit"], 2)
            line_t, line_w = trace.summary(), warm.summary()
            self.assertEqual(line_t.split("cubins=")[1], line_w.split("cubins=")[1])
            self.assertTrue(line_w.split("cubins=")[1].startswith("2:"))
            with open(os.path.join(root, os.listdir(root)[0]), "ab") as fh:
                fh.write(elf(b"other"))  # a module that serves another cubin moves the digest
            other = self.state("1", root)
            other.load = lambda k: "LOADED"
            for c in calls:
                other.call(orig, None, c, {})
            self.assertNotEqual(other.summary().split("cubins=")[1], line_t.split("cubins=")[1])

    def test_export_mode_always_traces_and_an_unkeyable_or_tvm_call_traces_unexported(self):
        with tempfile.TemporaryDirectory() as root:
            n = []
            s = self.state("export", root)
            orig = lambda self, func, *a, **k: n.append(1) or FakeFn(a)
            for _ in range(2):
                s.call(orig, None, (kernel, 1), {})
            s.call(orig, None, (kernel, Addr()), {})
            s.call(orig, None, (kernel, 1), {"options": "--enable-tvm-ffi"})
            self.assertEqual((len(n), s.n["keyed"], s.n["unkeyed"], len(os.listdir(root))), (4, 2, 2, 1))

    def test_a_record_line_per_call_and_a_done_file_at_the_build_end(self):
        with tempfile.TemporaryDirectory() as root:
            rec = os.path.join(root, "rec")
            s = self.state("export", root, rec)
            s.call(lambda self, func, *a, **k: FakeFn(a), None, (kernel, 1), {})
            with redirect_stdout(io.StringIO()) as out:
                s.summary()
            [r] = [json.loads(x) for x in open(os.path.join(rec, "rank-0.jsonl"))]
            self.assertEqual((r["hit"], r["fn"], bool(r["key"])), (False, "kernel", True))
            self.assertIn("module_error", r)  # a FakeFn has no IR module
            self.assertEqual(json.load(open(os.path.join(rec, "rank-0.done.json")))["counts"]["calls"], 1)
            self.assertTrue(out.getvalue().startswith("PERFBOT-PRETRACE build rank=0 calls=1 keyed=1"))

    def test_install_is_off_by_default_and_wraps_compile_and_the_build_end_when_on(self):
        self.assertIsNone(pt.install({}))

        class CompileCallable:
            def __call__(self, func, *a, **k):
                return FakeFn(a)

        class SpecializedDecode:
            def __init__(self):
                self.built = True

        decode = types.SimpleNamespace(SpecializedDecode=SpecializedDecode)
        with tempfile.TemporaryDirectory() as root, redirect_stdout(io.StringIO()) as out:
            st = pt.install({pt.ENV: "1", pt.DIR_ENV: root}, decode=decode, compile_callable=CompileCallable)
            CompileCallable()(kernel, 3)
            SpecializedDecode()
            self.assertIs(pt.install({pt.ENV: "1", pt.DIR_ENV: root}, decode=decode, compile_callable=CompileCallable),
                          st)
        lines = out.getvalue().splitlines()
        self.assertTrue(lines[0].startswith("PERFBOT-PRETRACE on mode=1 rank=-"), lines)
        self.assertIn("calls=1 keyed=1 hit=0 miss=1 unkeyed=0", lines[1])


def rec(rank, key, cubins, module, **k):
    return dict(rank=rank, key=key, hit=False, cubins=cubins, module=module, export_s=0.01, call_s=0.3,
                passes_s=0.05, hash_s=0.02, key_s=0.001, **k)


class Analyze(unittest.TestCase):
    P1 = {"world": 2, "ranks_done": 2, "seconds": 300.0}

    def records(self):
        return [rec("0", "k1", ["c1"], "m1"), rec("0", "k2", ["c1"], "m1"), rec("0", "k3", ["c3"], "m3"),
                rec("1", "k1", ["c1"], "m1"), dict(rank="0", key=None, hit=False, unkeyed="tvm_ffi", call_s=0.2)]

    def loaded(self, **over):
        rows = [{"key": k, "cubins": c, "load_s": 0.01, "to_s": 0.002} for k, c in
                (("k1", ["c1"]), ("k2", ["c1"]), ("k3", ["c3"]))]
        for r in rows:
            r.update(over.get(r["key"], {}))
        return rows

    def test_pass_counts_the_same_module_traced_twice_as_a_duplicate(self):
        s = proof.analyze(self.records(), self.loaded(), self.P1)
        self.assertEqual(s["verdict"], "PASS", s)
        c = s["counts"]
        self.assertEqual((c["calls"], c["keyed"], c["unkeyed"], c["keys"], c["rank0_dup_calls"], c["cubin_equal"]),
                         (5, 4, 1, 3, 1, "3/3"))
        self.assertEqual(s["unkeyed_reasons"], [("tvm_ffi", 1)])
        self.assertAlmostEqual(s["split_s"]["trace_s"], 3 * 0.3 + 0.2 - 3 * 0.07, places=2)  # rank 0 only
        self.assertTrue(proof.line(s).startswith("PERFBOT-PRETRACE verdict=PASS ranks=2/2 calls=5"))

    def test_one_key_with_two_cubin_sets_on_any_rank_fails(self):
        r = self.records() + [rec("1", "k3", ["c9"], "m9")]
        s = proof.analyze(r, self.loaded(), self.P1)
        self.assertEqual((s["verdict"], s["collision_keys"]), ("FAIL", ["k3"]))

    def test_a_load_error_a_missing_module_or_another_cubin_fails(self):
        for loaded in (self.loaded(k2={"error": "X"}), self.loaded()[:2], self.loaded(k3={"cubins": ["c4"]})):
            self.assertEqual(proof.analyze(self.records(), loaded, self.P1)["verdict"], "FAIL")

    def test_a_rank_short_or_pass_2_not_finished_is_incomplete(self):
        self.assertEqual(proof.analyze(self.records(), self.loaded(), dict(self.P1, ranks_done=1))["verdict"],
                         "INCOMPLETE")
        self.assertEqual(proof.analyze(self.records(), self.loaded(), self.P1, pass2_ok=False)["verdict"],
                         "INCOMPLETE")

    def test_extra_args_reads_like_serve_sh(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a")
            open(p, "w").write('# c\n--x 1  # y\n\n--j {"a":"b"}\n')
            self.assertEqual(proof.extra_args(p), ["--x", "1", "--j", '{"a":"b"}'])
            self.assertEqual(proof.world(["vllm", "--tensor-parallel-size", "8"]), 8)


if __name__ == "__main__":
    unittest.main()
