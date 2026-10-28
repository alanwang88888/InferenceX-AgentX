"""Kernel-only arm: the image KDA chain at served shapes, with real weights (the stage gate's baseline).

  python3 -B -m perfbot_kernels.kda_stage.control [--layers 4] [--shapes 1,2,3,6,8,16]

Writes $W/$PERFBOT_RESULTS/kda-control.json:
  facts     config (lower_bound), layer names, dtypes and shapes as loaded, image module paths
  numerics  per shape: the control's errors against the fp64 reference (the bar a candidate must meet), one round
            and a second round from the pools the first left (acceptance changes between rounds)
  timings   per shape, CUDA-graph replay of a chain over L distinct layers (L x ~112 MB of weights > L2, so weights
            stream cold as in the server), µs per layer: full (6 launches), in_proj, core (launches 2-5), o_proj
"""

import argparse
import json
import sys
import traceback

import torch

from . import common as c


def facts(layers):
    import vllm.model_executor.layers.mamba.ops.causal_conv1d as m1
    import vllm.models.kimi_k3.nvidia.ops.third_party.kda as m2

    lw = layers[0]
    return {"config": {k: v for k, v in c.model_config().items() if k != "raw"},
            "linear_attn_config": c.model_config()["raw"], "layers": [l.name for l in layers],
            "shapes": {k: [list(getattr(lw, k).shape), str(getattr(lw, k).dtype)]
                       for k in ("w_in", "w_fb", "w_conv", "a_log", "dt_bias", "w_norm", "w_o")},
            "modules": [m1.__file__, m2.__file__], "device": torch.cuda.get_device_name(),
            "sms": torch.cuda.get_device_properties(0).multi_processor_count}


def numerics(lw, n):
    out = {}
    st = c.make_state(n, seed=n, null_requests=1 if n >= 3 else 0)
    ctl = c.Control(lw)
    for rnd in range(2):
        before = st.clone()
        ref = c.reference(lw, before)
        core = torch.empty(1, st.t, c.H, c.D, dtype=torch.bfloat16, device="cuda")
        y = ctl(st, core)
        torch.cuda.synchronize()
        out[f"round{rnd}"] = c.errors(lw, before, st, y, core, ref)
        c.next_round(st, seed=100 + rnd)
    return out


def timings(layers, n):
    sts = [c.make_state(n, seed=10 * n + i) for i in range(len(layers))]
    ctls = [c.Control(lw) for lw in layers]
    cores = [torch.empty(1, sts[0].t, c.H, c.D, dtype=torch.bfloat16, device="cuda") for _ in layers]
    projs = [ctl.in_proj(st) for ctl, st in zip(ctls, sts)]
    L = len(layers)
    row = {
        "full_us": c.graph_timed(lambda: [ctl(st, co) for ctl, st, co in zip(ctls, sts, cores)]) / L,
        "in_proj_us": c.graph_timed(lambda: [ctl.in_proj(st) for ctl, st in zip(ctls, sts)]) / L,
        "core_us": c.graph_timed(lambda: [ctl.core(st, p, co) for ctl, st, p, co in zip(ctls, sts, projs, cores)]) / L,
        "o_proj_us": c.graph_timed(lambda: [ctl.o_proj(co) for ctl, co in zip(ctls, cores)]) / L,
    }
    return {k: round(v, 3) for k, v in row.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--shapes", default="1,2,3,6,8,16")
    args = ap.parse_args()
    shapes = [int(s) for s in args.shapes.split(",")]
    rec = {"item": "expedition-kda-control", "ok": False}
    path = c.results_path("kda-control.json")
    try:
        prefixes = c.kda_layers()
        layers = [c.load_layer(p) for p in prefixes[: args.layers]]
        rec["facts"] = facts(layers)
        print("FACTS", json.dumps(rec["facts"]), flush=True)
        rec["numerics"], rec["timings"] = {}, {}
        for n in shapes:
            rec["numerics"][f"N{n}"] = numerics(layers[0], n)
            print(f"NUMERICS N={n} {json.dumps(rec['numerics'][f'N{n}'])}", flush=True)
        for n in shapes:
            rec["timings"][f"N{n}"] = timings(layers, n)
            print(f"TIMING N={n} {json.dumps(rec['timings'][f'N{n}'])}", flush=True)
        rec["ok"] = True
    except Exception:
        rec["error"] = traceback.format_exc()
        print(rec["error"], file=sys.stderr, flush=True)
    path.write_text(json.dumps(rec, indent=1))
    return 0 if rec["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
