"""Kernel-only E1 overlap harness (package docstring). Run in the server env (PERFBOT_PRE_EXEC), one GPU:

  python3 -B -m perfbot_kernels.e1_overlap.selftest

One synthetic K3 MoE layer (TP8 local shapes, random bf16 weights, router [896, 7168] + fp32 bias, top-16, the
moe_stage self-test's synthetic MXFP4 experts) through the served functions and the served fork/join code:
maybe_execute_in_parallel (front) and SharedExperts (experts) on vLLM's aux_stream(). Per M in e1.MS, scope in
e1.SCOPES (front, experts, layer) and mode in e1.MODES (main, aux, both, serial, auxfirst): one eager call, a CUDA
graph of one call, GATE_REPLAYS replays compared to the eager outputs, then REPLAYS timed replays, each between its
own events after a 512 MB L2 flush; median us.
Lines: `E1 m= scope= mode= us= replays_equal=`, `E1-OVERLAP m= scope= main= aux= both= serial= auxfirst=
exposed_aux_us= hidden_aux_us= overlap_gain_us= auxfirst_delta_us=`, `E1-GATE pass=`.
Gate: every arm ran; replays bit-equal to the eager call (fc2's rows within moe_stage graph_probe's tolerance: its
cross-CTA fp32 red.add order varies run to run); on scope layer, serial and auxfirst outputs equal both's.
Exit 0 iff pass. Record e1_overlap.json under $PERFBOT_WORKSPACE/$PERFBOT_RESULTS.
PERFBOT_E1_PROFILE_RANGE=1: after timing, one cudaProfilerStart/Stop window around PROFILE_REPLAYS replays of each
scope-layer graph of modes both and serial, each inside an NVTX range `e1 m=<M> mode=<mode>` (for pic-c
--start-stop-method cuda-profiler-api, or nsys --capture-range=cudaProfilerApi).
"""

import json
import os
import statistics
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

import perfbot_kernels.e1_overlap as e1

REPLAYS, GATE_REPLAYS, PROFILE_REPLAYS = 60, 2, 8
FLUSH_BYTES = 512 << 20
ROWS_RTOL = 2 ** -6  # moe_stage.selftest graph_probe


def results_path(name) -> Path:
    root = Path(os.environ.get("PERFBOT_WORKSPACE", "/workspace"))
    out = root / os.environ.get("PERFBOT_RESULTS", "results/measure")
    out.mkdir(parents=True, exist_ok=True)
    return out / name


class Layer:
    """One layer's E1 front on the served code: weights, served callables, the served streams and events."""

    def __init__(self, dev):
        import perfbot_kernels.moe_stage as ms
        import perfbot_kernels.moe_stage.selftest as mst
        import perfbot_kernels.router_topk  # noqa: F401 - registers torch.ops.perfbot_router_topk.route
        from vllm.models.kimi_k3.nvidia import low_latency_gemm as llg
        from vllm.utils.torch_utils import aux_stream

        g = torch.Generator(device=dev).manual_seed(0)
        rn = lambda *s: torch.randn(*s, device=dev, generator=g)
        self.dev, self.g, self.aux = dev, g, aux_stream()
        table = llg._low_latency_table() if hasattr(llg, "_low_latency_table") else llg.KIMI_K3_PROJECTIONS
        self.lin, self.plans = {}, {}
        for name, (n, k) in e1.GEMMS.items():
            spec = (table or {}).get((n, k))
            self.plans[name] = llg._build_plan(spec) if spec is not None else {}
            method = llg.KimiK3LowLatencyLinearMethod(self.plans[name], {})
            w = (rn(n, k) * k ** -0.5).to(torch.bfloat16)
            self.lin[name] = (lambda x, mth=method, lay=SimpleNamespace(weight=w): mth.apply(lay, x))
        self.gate_w = (rn(*e1.ROUTER) * e1.HIDDEN ** -0.5 * 2.0).to(torch.bfloat16)  # as router_topk.selftest
        self.gate_b = (rn(e1.E) * 0.05).float()
        self.bench = mst.Bench()  # synthetic MXFP4 experts through the overlay's real TrtLlmMxfp4ExpertsMonolithic
        self.tail = mst.FakeTail(dev)
        ms.set_target(dev, self.tail.buf.data_ptr(), self.tail.flags, mst.SELF_RANK, mst.MAXM_TAIL)
        self.tail.arm(1)
        self.events = [torch.cuda.Event() for _ in range(4)]

    def route(self, x):  # overlay model.py:1435-1443
        return torch.ops.perfbot_router_topk.route(x, self.gate_w, self.gate_b, e1.TOPK, e1.ROUTE_SCALE)

    def shared(self, x):  # overlay model.py KimiMLP.forward:425-431 (reduce_results False: no collective)
        h = self.lin["gate_up"](x)
        a = torch.empty(h.shape[:-1] + (h.shape[-1] // 2,), dtype=h.dtype, device=h.device)
        torch.ops._C.situ_and_mul(a, h, e1.SITU_BETA, e1.SITU_LINEAR_BETA)  # SituAndMul.forward_cuda, activation.py
        return self.lin["shared_down"](a)

    def experts(self, latent, carrier):  # overlay trtllm_mxfp4_moe.py apply -> moe_stage (prepare defers the quant)
        out = self.bench._apply(latent, None, carrier, True)
        if not getattr(out, "perfbot_prefilled", False):
            raise RuntimeError("moe_stage declined: the image path ran, not the served stage")
        return out

    def static(self, x):
        """Eager results a side that is off returns instead of launching (inputs for the other side)."""
        carrier, latent = self.route(x), self.lin["latent_down"](x)
        return SimpleNamespace(carrier=carrier, latent=latent, shared=self.shared(x))

    def shared_experts(self, placement, side):
        from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExperts

        cfg = SimpleNamespace(moe_parallel_config=SimpleNamespace(
            enable_eplb=False, all2all_backend="", use_fi_nvl_two_sided_kernels=False))
        se = SharedExperts(side, cfg, False, lambda: False)
        if placement == "serial":
            se._stream = None  # = VLLM_DISABLE_SHARED_EXPERTS_STREAM=1 (shared_experts.py:64-66)
        return se

    def call(self, scope, mode, x, st):
        """fn() -> {name: tensor} running `scope` of the layer in `mode` (e1.MODES) on the served fork/join."""
        from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExpertsOrder
        from vllm.utils.multi_stream_utils import maybe_execute_in_parallel

        main_on, aux_on, placement = e1.MODES[mode]
        phases = e1.SCOPES[scope]
        stream = None if placement == "serial" else self.aux
        kw = {"launch_aux_first": True} if placement == "auxfirst" else {}
        side = self.shared if aux_on else (lambda _x: st.shared)
        se = self.shared_experts(placement, side) if placement != "auxfirst" else None
        ev = self.events

        def fn():
            out, carrier, latent = {}, st.carrier, st.latent
            if "front" in phases:  # overlay model.py:1466-1476
                carrier, latent = maybe_execute_in_parallel(
                    (lambda: self.route(x)) if main_on else (lambda: st.carrier),
                    (lambda: self.lin["latent_down"](x)) if aux_on else (lambda: st.latent),
                    ev[0], ev[1], stream, **kw)
                out.update(carrier=carrier, latent=latent)
            if "experts" in phases:
                main = (lambda: self.experts(latent, carrier)) if main_on else (lambda: None)
                if se is None:  # candidate: the same DAG as SharedExperts, aux side queued first
                    routed, shared = maybe_execute_in_parallel(main, lambda: side(x), ev[2], ev[3], stream, **kw)
                else:  # moe_runner.py _forward_impl:942, _apply_quant_method:638 / :680
                    se.maybe_sync_shared_experts_stream(x)
                    se(x, SharedExpertsOrder.NO_OVERLAP)
                    routed = main()
                    se(x, SharedExpertsOrder.MULTI_STREAM_OVERLAPPED)
                    shared = se.output
                out["shared"] = shared
                if routed is not None:  # as graph_probe: expert weights and fc2's rows (the map is not compared)
                    out.update(wts=routed.expert_weights, rows=self.tail.rows(x.shape[0]))
            return out

        return fn


def graph(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        res = fn()
    return g, res


def time_graph(g, flush):
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(REPLAYS)]
    for _ in range(3):
        g.replay()
    for a, b in ev:
        flush.zero_()
        a.record()
        g.replay()
        b.record()
    torch.cuda.synchronize()
    return round(statistics.median(a.elapsed_time(b) * 1e3 for a, b in ev), 2)


def same(a, b, name):
    """Bit-equal, except fc2's rows (moe_stage graph_probe tolerance)."""
    if name != "rows":
        return a.dtype == b.dtype and a.shape == b.shape and torch.equal(
            a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
    a, b = a.float(), b.float()
    return bool(torch.allclose(a, b, rtol=ROWS_RTOL, atol=1e-3 * b.abs().max().item()))


def agree(x, y):
    return set(x) == set(y) and all(same(x[k], y[k], k) for k in x)


def arm(fn, flush):
    eager = {k: v.clone() for k, v in fn().items()}
    torch.cuda.synchronize()
    g, res = graph(fn)
    ok = True
    for _ in range(GATE_REPLAYS):
        g.replay()
        torch.cuda.synchronize()
        ok &= agree({k: v.clone() for k, v in res.items()}, eager)
    return g, eager, {"us": time_graph(g, flush), "replays_equal": ok}


def main() -> int:
    torch.manual_seed(0)
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    profile = os.environ.get("PERFBOT_E1_PROFILE_RANGE", "0") == "1"
    flush = torch.empty(FLUSH_BYTES, dtype=torch.uint8, device=dev)
    rec = {"device": torch.cuda.get_device_name(dev), "capability": list(torch.cuda.get_device_capability(dev)),
           "phases": e1.PHASES, "modes": e1.MODES, "rows": [], "overlap": [], "agree": {}, "backends": {}}
    ok, keep = True, {}
    try:
        layer = Layer(dev)
        for m in e1.MS:
            x = torch.randn(m, e1.HIDDEN, device=dev, generator=layer.g).to(torch.bfloat16)
            st = layer.static(x)
            torch.cuda.synchronize()
            rec["backends"][m] = {n: e1.gemm_backend(p, m) for n, p in layer.plans.items()}
            print(f"E1-BACKENDS m={m} " + " ".join(f"{n}={b}" for n, b in rec["backends"][m].items()), flush=True)
            for scope in e1.SCOPES:
                us, outs = {}, {}
                for mode in e1.MODES:
                    try:
                        g, outs[mode], r = arm(layer.call(scope, mode, x, st), flush)
                        us[mode] = r["us"]
                        if profile and scope == "layer" and mode in ("both", "serial"):
                            keep[(m, mode)] = g
                    except Exception as e:  # noqa: BLE001 - a failed arm is a gate failure, recorded
                        r = {"error": f"{type(e).__name__}: {e}"[:300]}
                        traceback.print_exc()
                    row = {"m": m, "scope": scope, "mode": mode, **r}
                    ok &= "us" in r and r["replays_equal"]
                    rec["rows"].append(row)
                    print("E1 " + " ".join(f"{a}={b}" for a, b in row.items()), flush=True)
                ov = {"m": m, "scope": scope, **{k: us.get(k) for k in e1.MODES}, **e1.summarize(us)}
                rec["overlap"].append(ov)
                print("E1-OVERLAP " + " ".join(f"{a}={b}" for a, b in ov.items()), flush=True)
                if scope == "layer":
                    ag = {p: p in outs and "both" in outs and agree(outs[p], outs["both"]) for p in ("serial", "auxfirst")}
                    rec["agree"][m] = ag
                    ok &= all(ag.values())
                    print(f"E1-AGREE m={m} " + " ".join(f"{a}={b}" for a, b in ag.items()), flush=True)
        if profile and keep:
            torch.cuda.synchronize()
            torch.cuda.profiler.start()
            for (m, mode), g in keep.items():
                torch.cuda.nvtx.range_push(f"e1 m={m} mode={mode}")
                for _ in range(PROFILE_REPLAYS):
                    flush.zero_()
                    g.replay()
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
            torch.cuda.profiler.stop()
            rec["profile_ranges"] = [f"e1 m={m} mode={mode}" for m, mode in keep]
    except Exception:  # noqa: BLE001 - setup failure: recorded, gate fails
        rec["error"] = traceback.format_exc()
        traceback.print_exc()
        ok = False
    rec["pass"] = ok
    results_path("e1_overlap.json").write_text(json.dumps(rec, indent=1, default=str))
    print(f"E1-GATE pass={ok}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
