"""Kernel-only gate for the router top-K fusion (r3-301). Run in the server env (serve.sh PERFBOT_PRE_EXEC).

  python3 -B -m perfbot_kernels.router_topk.selftest           # identity, compile, equality, isolated timings
  python3 -B -m perfbot_kernels.router_topk.selftest --graph   # CUDA-graph capture + replay on every shape
  python3 -B -m perfbot_kernels.router_topk.selftest --iket    # a few launches only (run under run-iket)

Both gate modes update one record, $W/$PERFBOT_RESULTS/kernel-gate.json, and exit 1 on any failed check.

The paired baseline is the image path, run through the overlay's real TrtLlmMxfp4ExpertsMonolithic.apply on
K3's per-rank MoE shape (896 experts, top-16, sigmoid + fp32 bias, latent hidden 3584, I/partition 384,
SITU 4/25, synthetic MXFP4 weights):
  control   = router GEMM (GateLinear's cuBLAS tier, torch.mm out_dtype fp32) -> apply(logits)
              -> flashinfer trtllm_fp4_block_scale_moe (BlockScores + Cluster routing from logits)
  candidate = torch.ops.perfbot_router_topk.route (one launch) -> apply(carrier)
              -> flashinfer trtllm_fp4_block_scale_routed_moe (packed top-K; Cluster only)
Numerical contract, checked bit for bit: the candidate's packed top-K equals flashinfer's own routing of the
candidate's logits (ids from the routing replay buffer, bf16 weights from the deferred output), and the
candidate layer's output equals the control layer's output run on those logits. The candidate's logits are
checked against the control GEMM within fp32 sum-order tolerance (a different GEMM; not bitwise by design).
"""

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

E, TOPK, HID, H, I_PART = 896, 16, 7168, 3584, 384
SITU_BETA, SITU_LINEAR_BETA = 4.0, 25.0
# decode tokens per step = CONC x (1 + 7 drafts): 48 at conc6; plus capture sizes, plus 2 prefill sizes
SHAPES = [1, 3, 8, 16, 24, 32, 40, 48, 56, 64, 96, 128, 200, 256, 512, 2048]
TIMED_SHAPES = [8, 48, 128]
MODULE = "vllm.model_executor.layers.fused_moe.experts.trtllm_mxfp4_moe"
MODEL_MODULE = "vllm.models.kimi_k3.nvidia.model"


def results_path() -> Path:
    root = Path(os.environ.get("PERFBOT_WORKSPACE", "/workspace"))
    out = root / os.environ.get("PERFBOT_RESULTS", "results/measure")
    out.mkdir(parents=True, exist_ok=True)
    return out / "kernel-gate.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(mod) -> dict:
    """Prove the server imports the overlay modules and this kernel source."""
    import perfbot_kernels.router_topk as rt

    overlay = Path(rt.__file__).resolve().parents[2]
    finders = [type(f).__name__ for f in sys.meta_path if getattr(f, "overlay_root", None)]
    model_spec = importlib.util.find_spec(MODEL_MODULE)
    model_origin = model_spec.origin if model_spec else None
    files = sorted(p for p in overlay.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    apply_src = Path(mod.__file__).read_text()
    model_src = Path(model_origin).read_text() if model_origin else ""
    rec = {
        "overlay_root": str(overlay),
        "experts_module_file": mod.__file__,
        "experts_module_is_overlay": Path(mod.__file__).resolve().is_relative_to(overlay),
        "model_module_file": model_origin,
        "model_module_is_overlay": bool(model_origin) and Path(model_origin).resolve().is_relative_to(overlay),
        "overlay_finder_installed": finders,
        "experts_apply_consumes_carrier": "split_carrier" in apply_src,
        "model_calls_route": "perfbot_router_topk.route" in model_src,
        "default_on": os.environ.get("PERFBOT_R3_301_ROUTER", "1") != "0",
        "kernel_source_sha256": rt.source_sha256(),
        "overlay_sha256": {str(p.relative_to(overlay)): sha256(p) for p in files},
    }
    rec["ok"] = bool(rec["experts_module_is_overlay"] and rec["model_module_is_overlay"] and finders
                     and rec["experts_apply_consumes_carrier"] and rec["model_calls_route"])
    return rec


class Bench:
    """Synthetic K3 MoE layer (one TP rank) on the overlay's real apply(), plus the router inputs."""

    def __init__(self, mod, max_tokens: int):
        from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input

        dev = torch.device("cuda")
        g = torch.Generator(device=dev).manual_seed(1234)
        u8 = lambda *s: torch.randint(0, 256, s, dtype=torch.uint8, device=dev, generator=g)
        ue8 = lambda *s: torch.randint(119, 124, s, dtype=torch.uint8, device=dev, generator=g)
        self.w1 = u8(E, 2 * I_PART, H // 2)
        self.w2 = u8(E, H, I_PART // 2)
        w1_scale = ue8(E, 2 * I_PART, H // 32).view(torch.float8_e4m3fn)
        w2_scale = ue8(E, H, I_PART // 32).view(torch.float8_e4m3fn)
        self.gate_w = (torch.randn(E, HID, device=dev, generator=g) / HID ** 0.5 * 2.0).to(torch.bfloat16)
        self.bias = (torch.randn(E, device=dev, generator=g) * 0.05).float()
        self.gen, self.dev, self.mod = g, dev, mod
        self.quant = moe_kernel_quantize_input
        self.act = mod.MoEActivation.SITU
        self.replay = {}
        self.defer = True

        def make(capture: bool):
            s = object.__new__(mod.TrtLlmMxfp4ExpertsMonolithic)
            s.quant_config = SimpleNamespace(a1_scale=None, quant_dtype="mxfp8", per_act_token_quant=False,
                                             block_shape=None, is_scale_swizzled=False, mx_alignment=256,
                                             w1_scale=w1_scale, w2_scale=w2_scale, w1_bias=None, w2_bias=None)
            s.moe_config = SimpleNamespace(max_num_tokens=max_tokens, dp_size=1, defer_moe_finalize_max_num_tokens=-1,
                                           should_defer_moe_finalize=lambda n: self.defer)
            s.routing_method_type = mod.RoutingMethodType.DeepSeekV3
            s.topk, s.intermediate_size_per_partition = TOPK, I_PART
            s.hidden_dim = s.hidden_dim_unpadded = H
            s.local_num_experts, s.ep_rank = E, 0
            s.gemm1_alpha = torch.full((E,), SITU_BETA, dtype=torch.float32, device=dev)
            s.gemm1_beta = torch.full((E,), SITU_LINEAR_BETA, dtype=torch.float32, device=dev)
            s.gemm1_clamp_limit = None
            if capture:  # flashinfer writes its top-K ids here
                s.routing_replay_capture_fn = lambda ids: self.replay.__setitem__("ids", ids.clone())
                s._routing_replay_buffer = torch.empty((max_tokens, TOPK), dtype=torch.int16, device=dev)
            else:
                s.routing_replay_capture_fn, s._routing_replay_buffer = None, None
            return s

        self.capture = make(True)
        self.plain = make(False)

    def inputs(self, m: int, ties: bool):
        """(router input [m, 7168] bf16, gate weight, bias, latent x [m, 3584] bf16)."""
        if ties:  # exact small-integer logits, zero bias: many equal keys, exercises the lower-id rule
            xh = torch.randint(0, 2, (m, HID), device=self.dev, generator=self.gen).to(torch.bfloat16)
            w = (torch.randint(-1, 2, (E, HID), device=self.dev, generator=self.gen) * 0.25).to(torch.bfloat16)
            xh[:, 64:] = 0
            bias = torch.zeros_like(self.bias)
        else:
            xh = torch.randn(m, HID, device=self.dev, generator=self.gen).to(torch.bfloat16)
            w, bias = self.gate_w, self.bias
        x = (torch.randn(m, H, device=self.dev, generator=self.gen) * 0.5).to(torch.bfloat16)
        return xh.contiguous(), w, bias, x.contiguous()

    def quantize(self, x):
        return self.quant(x, None, quant_dtype="mxfp8", per_act_token_quant=False, block_shape=None,
                          is_scale_swizzled=False, mx_alignment=256)

    def apply(self, expert, router, bias, xq, xs):
        return expert.apply(xq, self.w1, self.w2, router, self.act, E, None, xs, False,
                            num_expert_group=1, e_score_correction_bias=bias, routed_scaling_factor=1.0,
                            topk_group=1)

    def control_layer(self, xh, w, bias, x, expert=None):
        logits = torch.mm(xh, w.T, out_dtype=torch.float32)  # GateLinear cuBLAS tier
        xq, xs = self.quantize(x)
        return self.apply(expert or self.plain, logits, bias, xq, xs)

    def candidate_layer(self, xh, w, bias, x):
        carrier = torch.ops.perfbot_router_topk.route(xh, w, bias, TOPK, 1.0)
        xq, xs = self.quantize(x)
        return self.apply(self.plain, carrier, bias, xq, xs), carrier


def unpack(out):
    if isinstance(out, torch.Tensor):
        return {"output": out}
    return {"gemm2_permuted": out.gemm2_permuted, "expert_weights": out.expert_weights,
            "permute": out.expanded_idx_to_permuted_idx}


def same(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return bool(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))


def gathered(o) -> torch.Tensor:
    """GEMM2 rows in (token, rank) order; the permutation is atomics-ordered, so compare through it."""
    idx = o["permute"].reshape(-1).long()
    return o["gemm2_permuted"][idx]


def compare_outputs(c, d) -> dict:
    c, d = unpack(c), unpack(d)
    if "output" in c:
        return {"finalized_output": same(c["output"], d["output"])}
    return {"expert_weights": same(c["expert_weights"], d["expert_weights"]),
            "gemm2_rows_by_token_rank": same(gathered(c), gathered(d))}


def check_shape(bench, m: int, ties: bool) -> dict:
    import perfbot_kernels.router_topk as rt

    xh, w, bias, x = bench.inputs(m, ties)
    carrier = torch.ops.perfbot_router_topk.route(xh, w, bias, TOPK, 1.0)
    packed, logits = rt.split_carrier(carrier, TOPK)
    ref = torch.mm(xh, w.T, out_dtype=torch.float32)
    err = (logits - ref).abs()
    tol = 1e-3 + 1e-3 * ref.abs()
    rec = {"fused_info": packed is not None,
           "logits_close": bool((err <= tol).all()),
           "logits_max_abs_err_info": float(err.max())}
    xq, xs = bench.quantize(x)
    if packed is not None:
        # flashinfer's own routing of the candidate's logits (control path, replay capture on)
        bench.defer = True
        out = unpack(bench.apply(bench.capture, logits.contiguous(), bias, xq, xs))
        ids = bench.replay["ids"][:m]
        cand_ids = (packed >> 16).to(torch.int16)
        cand_w = (packed & 0xFFFF).to(torch.int16).view(torch.bfloat16)
        rec["topk_ids"] = same(ids, cand_ids)
        rec["topk_weights"] = same(out["expert_weights"].reshape(cand_w.shape).to(torch.bfloat16), cand_w)
        if not (rec["topk_ids"] and rec["topk_weights"]):
            bad = ((ids != cand_ids).any(-1)).nonzero().flatten()[:3].tolist()
            rec["first_mismatch_rows_info"] = bad
        # routing decision vs the control GEMM's logits (information: a different GEMM, not the contract)
        bench.apply(bench.capture, ref, bias, xq, xs)
        rec["ids_equal_to_cublas_logits_routing_frac_info"] = float(
            (bench.replay["ids"][:m].sort(-1).values == cand_ids.sort(-1).values).all(-1).float().mean())
    for defer in (True, False):
        bench.defer = defer
        c = bench.apply(bench.plain, logits.contiguous(), bias, xq, xs)
        d = bench.apply(bench.plain, carrier, bias, xq, xs)
        rec["deferred" if defer else "finalized"] = compare_outputs(c, d)
    torch.cuda.synchronize()
    flat = [v for k, v in rec.items() if isinstance(v, bool) and not k.endswith("_info")]
    flat += [v for part in ("deferred", "finalized") for v in rec[part].values()]
    rec["ok"] = all(flat)
    return rec


def equality(bench) -> dict:
    shapes, ok = {}, True
    for m in SHAPES:
        for ties in (False, True):
            rec = check_shape(bench, m, ties)
            ok &= rec["ok"]
            shapes[f"M{m}{'-ties' if ties else ''}"] = rec
            print(f"EQUALITY M={m} ties={ties} ok={rec['ok']} {json.dumps(rec)}", flush=True)
    return {"ok": ok, "shapes": shapes,
            "compared_to": "flashinfer routing of the candidate's logits + control layer on them, bitwise"}


def graph_timed(fn, reps: int = 200) -> float:
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    for _ in range(10):
        graph.replay()
    times = []
    for _ in range(5):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(reps):
            graph.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1000.0 / reps)
    del graph
    return statistics.median(times)


def kernel_profile(bench, m: int) -> dict:
    from torch.profiler import ProfilerActivity, profile

    xh, w, bias, x = bench.inputs(m, False)
    bench.defer = True
    rec = {}
    for name, fn in (("control", lambda: bench.control_layer(xh, w, bias, x)),
                     ("candidate", lambda: bench.candidate_layer(xh, w, bias, x))):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        reps = 20
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(reps):
                fn()
            torch.cuda.synchronize()
        per = {}
        for evt in prof.events():
            if not str(evt.device_type).endswith("CUDA"):
                continue
            row = per.setdefault(evt.name[:120], {"calls_per_layer": 0.0, "us_per_layer": 0.0})
            row["calls_per_layer"] += 1.0 / reps
            row["us_per_layer"] += evt.time_range.elapsed_us() / reps
        rec[name] = per
    return rec


def timings(bench) -> dict:
    rec = {"unit": "us per call", "method": "CUDA-graph replay, median of 5 x 200", "per_shape": {}}
    for m in TIMED_SHAPES:
        xh, w, bias, x = bench.inputs(m, False)
        bench.defer = True
        row = {
            "layer_control_us": graph_timed(lambda: bench.control_layer(xh, w, bias, x)),
            "layer_candidate_us": graph_timed(lambda: bench.candidate_layer(xh, w, bias, x)),
            "router_control_us": graph_timed(lambda: torch.mm(xh, w.T, out_dtype=torch.float32)),
            "router_candidate_us": graph_timed(
                lambda: torch.ops.perfbot_router_topk.route(xh, w, bias, TOPK, 1.0)),
        }
        row["layer_delta_us"] = row["layer_candidate_us"] - row["layer_control_us"]
        row["router_delta_us"] = row["router_candidate_us"] - row["router_control_us"]
        rec["per_shape"][f"M{m}"] = row
        print(f"TIMING M={m} {json.dumps(row)}", flush=True)
    rec["kernels_M48"] = kernel_profile(bench, 48)
    return rec


def graph_probe(bench) -> dict:
    """Capture the candidate layer per shape, replay twice on fresh inputs, compare to eager and control."""
    import perfbot_kernels.router_topk as rt

    shapes, ok = {}, True
    for m in SHAPES:
        xh, w, bias, x = bench.inputs(m, False)
        bench.defer = m <= 512
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            bench.candidate_layer(xh, w, bias, x)
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out, carrier = bench.candidate_layer(xh, w, bias, x)
        rec = {}
        for rep in range(2):  # the second replay proves the arrival counter reset itself
            xh2, _, _, x2 = bench.inputs(m, False)
            xh.copy_(xh2)
            x.copy_(x2)
            graph.replay()
            torch.cuda.synchronize()
            got = {k: v.clone() for k, v in unpack(out).items()}
            got_carrier = carrier.clone()
            want, want_carrier = bench.candidate_layer(xh.clone(), w, bias, x.clone())
            want = unpack(want)
            _, logits = rt.split_carrier(want_carrier, TOPK)
            xq, xs = bench.quantize(x)
            ctl = unpack(bench.apply(bench.plain, logits.contiguous(), bias, xq, xs))
            if "output" in got:
                r_e = same(got["output"], want["output"])
                r_c = same(got["output"], ctl["output"])
            else:
                r_e = same(got["expert_weights"], want["expert_weights"]) and same(gathered(got), gathered(want))
                r_c = same(got["expert_weights"], ctl["expert_weights"]) and same(gathered(got), gathered(ctl))
            rec[f"replay{rep}_vs_eager"] = r_e
            rec[f"replay{rep}_vs_control_on_same_logits"] = r_c
            if m <= rt.MAX_M:
                rec[f"replay{rep}_carrier_vs_eager"] = same(got_carrier, want_carrier)
            else:  # packed region unused above MAX_M; logits only
                rec[f"replay{rep}_carrier_vs_eager"] = same(rt.split_carrier(got_carrier, TOPK)[1],
                                                            rt.split_carrier(want_carrier, TOPK)[1])
        counters = [v for v in rt._COUNTERS.values()]
        rec["counter_zero_after"] = all(bool((c == 0).all()) for c in counters)
        rec["ok"] = all(v for v in rec.values())
        ok &= rec["ok"]
        shapes[f"M{m}"] = rec
        print(f"GRAPH M={m} deferred={bench.defer} {json.dumps(rec)}", flush=True)
        del graph
    return {"ok": ok, "shapes": shapes}


def iket_run() -> int:
    """A handful of eager launches for run-iket (phase ranges compiled in with PERFBOT_ROUTER_TOPK_IKET=1)."""
    import perfbot_kernels.router_topk  # noqa: F401

    dev = torch.device("cuda")
    w = (torch.randn(E, HID, device=dev) / HID ** 0.5).to(torch.bfloat16)
    bias = (torch.randn(E, device=dev) * 0.05).float()
    for m in (48, 48, 48, 8, 128):
        xh = torch.randn(m, HID, device=dev).to(torch.bfloat16)
        torch.ops.perfbot_router_topk.route(xh, w, bias, TOPK, 1.0)
    torch.cuda.synchronize()
    print("PERFBOT-ROUTER-TOPK-IKET ok", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--iket", action="store_true")
    args = ap.parse_args()
    if args.iket:
        return iket_run()
    path = results_path()
    record = json.loads(path.read_text()) if args.graph and path.is_file() else {}
    record.setdefault("item", "r3-301")
    record["device"] = torch.cuda.get_device_name()
    ok = False
    try:
        mod = importlib.import_module(MODULE)
        import perfbot_kernels.router_topk as rt

        t0 = time.time()
        rt.compiled(E, HID, TOPK)
        if not args.graph:
            record["identity"] = identity(mod)
            record["build"] = {"seconds": round(time.time() - t0, 1), "sha256": rt.source_sha256()}
        bench = Bench(mod, max(SHAPES))
        if args.graph:
            record["graph"] = graph_probe(bench)
            ok = record["graph"]["ok"]
        else:
            record["equality"] = equality(bench)
            record["equality_verdict"] = record["equality"]["ok"]
            try:  # evidence, not the gate
                record["isolated_timings"] = timings(bench)
            except Exception:
                record["isolated_timings_error"] = traceback.format_exc()
                traceback.print_exc()
            ok = record["identity"]["ok"] and record["equality_verdict"]
    except Exception:
        record["error" if not args.graph else "graph_error"] = traceback.format_exc()
        traceback.print_exc()
    record["ok_graph" if args.graph else "ok_selftest"] = ok
    path.write_text(json.dumps(record, indent=1, sort_keys=True))
    print(f"PERFBOT-ROUTER-TOPK-SELFTEST mode={'graph' if args.graph else 'selftest'} ok={ok} record={path}",
          flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
