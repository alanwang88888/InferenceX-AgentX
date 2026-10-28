"""Kernel-only gate of the server adapter (serve.ServedCore), the wiring the overlay kda.py calls.

  python3 -B -m perfbot_kernels.kda_stage.selftest_serve

Checks, on real weights (layer 0, TP rank 0) and served-layout state (spec_idx = a row-strided block-table slice,
non-dense ssm slots, trailing null requests as in the padded FULL-graph batch):
  equal    run() launches core() and its out, conv and ssm are bit-equal to a direct core() call on a clone
  graph    run() captured in a CUDA graph, replayed twice on fresh state, bit-equal to eager core()
  decline  each decline class: no metadata, mixed batch, token-count mismatch, capture before warm, bad weight dtype
Writes $W/$PERFBOT_RESULTS/serve-gate.json; exit 0 = every check passed.
"""

import json
import logging
import sys
import traceback
from types import SimpleNamespace

import torch

from . import common as c
from . import core as k
from .serve import ServedCore

log = logging.getLogger("selftest_serve")
logging.basicConfig(level=logging.INFO)


def fake_layer(lw, st, prefix="layer0"):
    """The attributes of KimiK3DeltaAttention that ServedCore reads, in the server's dtypes and layouts."""
    w_norm_bf16 = lw.w_norm.to(torch.bfloat16)  # o_norm.weight is bf16; the loader copies it to fp32
    return SimpleNamespace(
        prefix=prefix, use_recoverssm=False, gate_lower_bound=lw.lower_bound,
        f_b_proj=SimpleNamespace(weight=lw.w_fb), conv1d=SimpleNamespace(weight=lw.w_conv.unsqueeze(1)),
        A_log=lw.a_log, dt_bias=lw.dt_bias, o_norm=SimpleNamespace(weight=w_norm_bf16),
        perfbot_norm_weight=w_norm_bf16.float(), kv_cache=(st.conv, st.ssm))


def metadata(st, n_real, prefix="layer0", **kw):
    m = SimpleNamespace(num_prefills=0, num_decodes=0, num_spec_decodes=n_real, spec_state_indices_tensor=st.spec_idx,
                        num_accepted_tokens=st.accepted)
    m.__dict__.update(kw)
    return {prefix: m}


def direct(layer, st, proj):
    lw = SimpleNamespace(w_fb=layer.f_b_proj.weight, w_conv=layer.conv1d.weight.view(3 * k.P, k.WIDTH),
                         a_log=layer.A_log, dt_bias=layer.dt_bias, w_norm=layer.perfbot_norm_weight,
                         lower_bound=layer.gate_lower_bound)
    out = torch.empty(1, st.t, k.H, k.D, dtype=torch.bfloat16, device="cuda")
    k.core(lw, st, proj, out)
    return out


def same(a, b):
    return bool(torch.equal(a, b))


def ssm_store(st):
    return st.ssm.as_strided((st.ssm.shape[0], st.ssm.stride(0)), (st.ssm.stride(0), 1))


def equal_check(lw, n, nulls):
    st = c.make_state(n, seed=7 * n, null_requests=nulls, idx_row_stride=33)
    proj = c.Control(lw).in_proj(st)
    ref = st.clone()
    want = direct(fake_layer(lw, ref), ref, proj)
    sc = ServedCore(fake_layer(lw, st), log)
    out = torch.empty(1, st.t, k.H, k.D, dtype=torch.bfloat16, device="cuda")
    r = sc.run(proj, out, metadata(st, n - nulls))
    torch.cuda.synchronize()
    return {"declined": r, "out": same(out, want), "conv": same(st.conv, ref.conv),
            "ssm": same(ssm_store(st), ssm_store(ref)), "ok": r is None and same(out, want)
            and same(st.conv, ref.conv) and same(ssm_store(st), ssm_store(ref))}


def graph_check(lw, n, nulls):
    st = c.make_state(n, seed=11 * n, null_requests=nulls, idx_row_stride=33)
    init = st.clone()
    proj = c.Control(lw).in_proj(st)
    sc = ServedCore(fake_layer(lw, st), log)
    sc.run(proj, torch.empty(1, st.t, k.H, k.D, dtype=torch.bfloat16, device="cuda"), None)  # profile run: warm
    out = torch.empty(1, st.t, k.H, k.D, dtype=torch.bfloat16, device="cuda")
    md = metadata(st, n - nulls)
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.cuda.graph(g, stream=s):
        r = sc.run(proj, out, md)
    torch.cuda.current_stream().wait_stream(s)
    rows = []
    for rep in range(2):
        st.conv.copy_(init.conv)
        ssm_store(st).copy_(ssm_store(init))
        g.replay()
        ref = init.clone()
        want = direct(fake_layer(lw, ref), ref, proj)
        torch.cuda.synchronize()
        rows.append(same(out, want) and same(st.conv, ref.conv) and same(ssm_store(st), ssm_store(ref)))
    return {"declined_in_capture": r, "replays_equal": rows, "ok": r is None and all(rows)}


def decline_checks(lw):
    n = 6
    st = c.make_state(n, seed=3, idx_row_stride=33)
    proj = c.Control(lw).in_proj(st)
    out = torch.empty(1, st.t, k.H, k.D, dtype=torch.bfloat16, device="cuda")
    res = {}
    sc = ServedCore(fake_layer(lw, st), log)
    res["no_metadata"] = sc.run(proj, out, None)
    res["mixed"] = sc.run(proj, out, metadata(st, n, num_decodes=1))
    proj56 = torch.cat([proj, proj[:8]])
    out56 = torch.empty(1, 56, k.H, k.D, dtype=torch.bfloat16, device="cuda")
    res["tokens"] = sc.run(proj56, out56, metadata(st, n))
    cold = ServedCore(fake_layer(lw, st), log)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        res["capture_before_warm"] = cold.run(proj, out, metadata(st, n))
    bad = fake_layer(lw, st)
    bad.f_b_proj = SimpleNamespace(weight=lw.w_fb.to(torch.float16))
    res["fp16_w_fb"] = ServedCore(bad, log).run(proj, out, metadata(st, n))
    want = {"no_metadata": "no metadata", "mixed": "not pure spec decode", "tokens": "check N=6",
            "capture_before_warm": "not warmed before capture", "fp16_w_fb": "disabled"}
    res["ok"] = all(res[key] == v for key, v in want.items())
    return res


def main() -> int:
    rec = {"item": "expedition-kda-serve-gate", "ok": False}
    path = c.results_path("serve-gate.json")
    try:
        lw = c.load_layer(c.kda_layers()[0])
        rec["core_sha"] = __import__("hashlib").sha256(open(k.__file__, "rb").read()).hexdigest()[:8]
        rec["equal"] = {f"N{n}null{nulls}": equal_check(lw, n, nulls) for n, nulls in ((1, 0), (6, 1), (6, 0), (16, 2))}
        print("EQUAL", json.dumps(rec["equal"]), flush=True)
        rec["graph"] = {f"N{n}null{nulls}": graph_check(lw, n, nulls) for n, nulls in ((6, 1), (16, 0))}
        print("GRAPH", json.dumps(rec["graph"]), flush=True)
        rec["decline"] = decline_checks(lw)
        print("DECLINE", json.dumps(rec["decline"]), flush=True)
        rec["ok"] = (all(v["ok"] for v in rec["equal"].values()) and all(v["ok"] for v in rec["graph"].values())
                     and rec["decline"]["ok"])
    except Exception:
        rec["error"] = traceback.format_exc()
        print(rec["error"], file=sys.stderr, flush=True)
    path.write_text(json.dumps(rec, indent=1))
    print("SERVE-GATE", "PASS" if rec["ok"] else "FAIL", flush=True)
    return 0 if rec["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
