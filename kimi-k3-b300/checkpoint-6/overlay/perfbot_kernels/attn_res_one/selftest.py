"""Kernel-only gate for the one-kernel AttnRes (r3-307). Run in the server env (serve.sh PERFBOT_PRE_EXEC):

  python3 -B -m perfbot_kernels.attn_res_one.selftest           # identity, compile, equality, production path, timings
  python3 -B -m perfbot_kernels.attn_res_one.selftest --graph   # CUDA-graph capture + 2 replays on every listed shape

Gate modes update one record, $W/$PERFBOT_RESULTS/kernel-gate.json, and exit 1 on any failed check.

Production path: `vllm.models.kimi_k3.nvidia.ops.attn_res.attn_res`, the overlay module model.py calls. The paired
baseline (control) is the same function with PERFBOT_R3_307_ATTN_RES=0: the image body (native
torch.ops._C.kimi_k3_attn_res for delta + output norm + no block write + k > 0, the image Triton _attn_res_kernel
otherwise). The control is checked bit-equal to the image module loaded from the image's own file.

Numerical contract (reduction order changed; precision unchanged): the prefix written back and the block written
are bit-equal to the control (both are RN-bf16 of the fp32 sum prefix + delta). The output is finite; against an fp64
reference its max error is at most the control's plus one bf16 ulp of the largest |reference|, its mean error at most
1.02x the control's, and every element is within max(1 bf16 ulp of max(|cand|, |ctrl|), 2 x control mean error) of
the control.

Timings: M=48 (the conc6 decode graph), KMAX = 9; a CUDA graph of CALLS calls rotating over ROT input sets, median
of REPS replays, per call. `step` is one replay of the 187-call mix of a decode step (nsys-16: k=2..8 x 22 and k=9 x 16
native-shaped calls, 8 block-write calls, 8 no-delta calls, 1 final call without the output norm).
"""

import argparse
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch
import triton
import triton.language as tl

H = 7168
EPS = 1e-6
MOD = "vllm.models.kimi_k3.nvidia.ops.attn_res"
SWITCH = "PERFBOT_R3_307_ATTN_RES"
EQ_M = [1, 7, 48, 64, 200]
KMAX_MAIN = 9
FLAGS = [(d, w, n) for d in (True, False) for w in (False, True) for n in (True, False)]
GRAPH_M = [1, 8, 16, 24, 32, 40, 48, 56, 64, 128]
CALLS, ROT, REPS = 20, 4, 30


def results_path() -> Path:
    root = Path(os.environ.get("PERFBOT_WORKSPACE", "/workspace"))
    out = root / os.environ.get("PERFBOT_RESULTS", "results/measure")
    out.mkdir(parents=True, exist_ok=True)
    return out / "kernel-gate.json"


def load_record() -> dict:
    try:
        return json.loads(results_path().read_text())
    except Exception:
        return {}


def save_record(rec: dict) -> None:
    results_path().write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def overlay_module():
    import importlib

    return importlib.import_module(MOD)


def image_module():
    """The image's own attn_res.py, loaded from its file under another name (no overlay)."""
    paths = [p for p in sys.path if "overlay" not in os.path.abspath(p or os.getcwd())]
    spec = importlib.machinery.PathFinder.find_spec("vllm", paths)
    path = Path(spec.submodule_search_locations[0]) / "models/kimi_k3/nvidia/ops/attn_res.py"
    s = importlib.util.spec_from_file_location("perfbot_image_attn_res", path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m, path


class Switch:
    def __init__(self, on: bool):
        self.on = on

    def __enter__(self):
        self.old = os.environ.get(SWITCH)
        os.environ[SWITCH] = "1" if self.on else "0"

    def __exit__(self, *a):
        if self.old is None:
            os.environ.pop(SWITCH, None)
        else:
            os.environ[SWITCH] = self.old


def make_inputs(m, kmax, seed, has_delta, has_norm, h=H):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"
    prefix = (torch.randn(m, h, generator=g, device=dev) * 2.0).bfloat16()
    delta = (torch.randn(m, h, generator=g, device=dev) * 0.7).bfloat16() if has_delta else None
    scale = 0.5 + torch.rand(1, kmax, 1, generator=g, device=dev) * 3.0
    blocks = (torch.randn(m, kmax, h, generator=g, device=dev) * scale).bfloat16()
    nw = (1.0 + 0.1 * torch.randn(h, generator=g, device=dev)).bfloat16()
    qk = (0.03 * torch.randn(h, generator=g, device=dev)).bfloat16()
    onw = (1.0 + 0.1 * torch.randn(h, generator=g, device=dev)).bfloat16() if has_norm else None
    return dict(prefix=prefix, delta=delta, blocks=blocks, norm_weight=nw, qk_weight=qk, output_norm_weight=onw)


def clone(inp):
    return {k: (None if v is None else v.clone()) for k, v in inp.items()}


def call(fn, inp, k, bwi):
    out = fn(inp["prefix"], inp["delta"], inp["blocks"], inp["norm_weight"], inp["qk_weight"],
             inp["output_norm_weight"], num_blocks=k, block_write_idx=bwi, eps=EPS,
             output_norm_eps=EPS if inp["output_norm_weight"] is not None else 0.0)
    return out, inp["prefix"], inp["blocks"]


def reference(inp, k, bwi):
    p = inp["prefix"].float()
    if inp["delta"] is not None:
        p = (p + inp["delta"].float()).bfloat16().float()
    pd = p.double()
    src = torch.cat([inp["blocks"][:, :k].double(), pd[:, None]], 1)
    w = inp["norm_weight"].double() * inp["qk_weight"].double()
    logits = (src * w).sum(-1) * torch.rsqrt((src * src).mean(-1) + EPS)
    mixed = (torch.softmax(logits, -1)[..., None] * src).sum(1)
    if inp["output_norm_weight"] is not None:
        mixed = mixed * torch.rsqrt((mixed * mixed).mean(-1, keepdim=True) + EPS) * inp["output_norm_weight"].double()
    return mixed


def ulp_bf16(x: torch.Tensor) -> torch.Tensor:
    e = torch.floor(torch.log2(x.double().abs().clamp_min(1e-30))).to(torch.int32) - 7
    return torch.ldexp(torch.ones_like(x, dtype=torch.float64), e)  # exact (torch.pow on CUDA is not)


def compare(cand, ctrl, ref):
    ec = (cand.double() - ref).abs()
    et = (ctrl.double() - ref).abs()
    d = (cand.double() - ctrl.double()).abs()
    tol = torch.maximum(ulp_bf16(torch.maximum(cand.double().abs(), ctrl.double().abs())), 2 * et.mean())
    r = dict(finite=bool(torch.isfinite(cand).all()), equal_frac=float((cand == ctrl).float().mean()),
             max_abs_vs_control=float(d.max()), cand_max_err=float(ec.max()), ctrl_max_err=float(et.max()),
             cand_mean_err=float(ec.mean()), ctrl_mean_err=float(et.mean()),
             elements_past_tol=int((d > tol).sum()))
    r["ok"] = (r["finite"] and r["cand_max_err"] <= r["ctrl_max_err"] + float(ulp_bf16(ref.abs().max()))
               and r["cand_mean_err"] <= 1.02 * r["ctrl_mean_err"] + 1e-12 and r["elements_past_tol"] == 0)
    if r["elements_past_tol"]:
        idx = torch.nonzero((d > tol).flatten()).flatten()[:5].tolist()
        r["past_tol_samples"] = [{"i": i, "cand": float(cand.flatten()[i]), "ctrl": float(ctrl.flatten()[i]),
                                  "ref": float(ref.flatten()[i])} for i in idx]
    return r


def cases(kmax, ms):
    for m in ms:
        for k in range(0, kmax + 1):
            for d, w, n in FLAGS:
                if w and k >= kmax:
                    continue
                yield m, kmax, k, (k if w else -1), d, n


def overlay_served(path: str) -> bool:
    """`path` is inside the overlay tree this file came from, after links resolve: serve.sh puts a constant link
    (/tmp/perfbot-overlay) to overlay/ or the arm's bundle first on PYTHONPATH."""
    root = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    return os.path.realpath(path).startswith(root + os.sep)


def run_selftest(rec: dict) -> bool:
    import perfbot_kernels.attn_res_one as aro

    ok = True
    ov = overlay_module()
    img, img_path = image_module()
    rec["identity"] = {
        "kernel_sha256": aro.source_sha256(), "kernel_file": aro.__file__, "overlay_module": ov.__file__,
        "overlay_module_sha256": sha256(ov.__file__), "image_module": str(img_path),
        "image_module_sha256": sha256(img_path), "native_op": hasattr(torch.ops._C, "kimi_k3_attn_res"),
        "overlay_served": overlay_served(ov.__file__), "num_warps": aro.NUM_WARPS,
        "triton": __import__("triton").__version__, "torch": torch.__version__,
    }
    print("IDENTITY", json.dumps(rec["identity"]), flush=True)
    ok &= rec["identity"]["overlay_served"]
    rec["device"] = torch.cuda.get_device_name()

    # Build: first call compiles each flag combination.
    t0 = time.time()
    for d, w, n in FLAGS:
        inp = make_inputs(2, KMAX_MAIN, 1, d, n)
        with Switch(True):
            call(ov.attn_res, inp, 1, 1 if w else -1)
    torch.cuda.synchronize()
    rec["build"] = {"seconds": round(time.time() - t0, 2), "sha256": aro.source_sha256()}
    print(f"PERFBOT-ATTN-RES compiled sha={aro.source_sha256()[:16]} seconds={rec['build']['seconds']}", flush=True)

    # Equality and production path, over every (M, k, flags) case.
    eq = {"contract": "prefix and written block bit-equal to control; out finite, max err vs fp64 <= ctrl + 1 bf16 "
                      "ulp(max|ref|), mean err <= 1.02 x ctrl, per element |cand-ctrl| <= max(1 ulp of "
                      "max(|cand|,|ctrl|), 2 x ctrl mean err)",
          "compared_to": "control = overlay attn_res with PERFBOT_R3_307_ATTN_RES=0 (image body) on the same inputs; "
                         "fp64 reference for error",
          "cases": 0, "failed": [], "control_unavailable": [], "control_vs_image_bit_equal": True, "worst": {}}
    prod = {"served_expected": 0, "served": 0, "unserved_checked": [], "ok": True}
    worst_ratio = 0.0
    all_cases = list(cases(KMAX_MAIN, EQ_M)) + list(cases(10, [48]))
    for i, (m, kmax, k, bwi, d, n) in enumerate(all_cases):
        base = make_inputs(m, kmax, 1000 + i, d, n)
        ref = reference(base, k, bwi)
        before = aro.CALLS["served"]
        with Switch(True):
            c_out, c_pre, c_blk = call(ov.attn_res, clone(base), k, bwi)
        prod["served_expected"] += 1
        prod["served"] += aro.CALLS["served"] - before
        try:
            with Switch(False):
                t_out, t_pre, t_blk = call(ov.attn_res, clone(base), k, bwi)
            i_out, i_pre, i_blk = call(img.attn_res, clone(base), k, bwi)
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001
            eq["control_unavailable"].append({"M": m, "KMAX": kmax, "k": k, "bwi": bwi, "delta": d, "norm": n,
                                              "error": f"{type(e).__name__}: {str(e)[:200]}"})
            continue
        img_eq = bool(torch.equal(t_out, i_out) and torch.equal(t_pre, i_pre) and torch.equal(t_blk, i_blk))
        eq["control_vs_image_bit_equal"] &= img_eq
        r = compare(c_out, t_out, ref)
        r["prefix_bit_equal"] = bool(torch.equal(c_pre, t_pre))
        r["blocks_bit_equal"] = bool(torch.equal(c_blk, t_blk))
        r["ok"] = r["ok"] and r["prefix_bit_equal"] and r["blocks_bit_equal"]
        eq["cases"] += 1
        tag = f"M{m}_KMAX{kmax}_k{k}_w{int(bwi >= 0)}_d{int(d)}_n{int(n)}"
        ratio = r["cand_mean_err"] / max(r["ctrl_mean_err"], 1e-30)
        if ratio > worst_ratio:
            worst_ratio, eq["worst"] = ratio, {"case": tag, **r}
        if m == 48 and kmax == KMAX_MAIN and (d, bwi >= 0, n) == (True, False, True):
            print(f"EQUALITY {tag} ok={r['ok']} {json.dumps(r)}", flush=True)
        if not r["ok"]:
            eq["failed"].append({"case": tag, **r})
            print(f"EQUALITY-FAIL {tag} {json.dumps(r)}", flush=True)
    eq["ok"] = not eq["failed"] and eq["control_vs_image_bit_equal"] and eq["cases"] > 0
    print(f"EQUALITY cases={eq['cases']} failed={len(eq['failed'])} control_unavailable={len(eq['control_unavailable'])}"
          f" control_vs_image_bit_equal={eq['control_vs_image_bit_equal']} worst={json.dumps(eq['worst'])}", flush=True)
    ok &= eq["ok"]

    # Not served: prefill split, a written block among the sources, bank above the cap -> image path, bit-equal.
    for m, kmax, k, bwi in [(256, 9, 4, -1), (48, 9, 3, 1), (48, 13, 4, -1)]:
        base = make_inputs(m, kmax, 7, True, True)
        before = aro.CALLS["served"]
        with Switch(True):
            c_out, c_pre, c_blk = call(ov.attn_res, clone(base), k, bwi)
        served = aro.CALLS["served"] - before
        try:
            i_out, i_pre, i_blk = call(img.attn_res, clone(base), k, bwi)
            same = bool(torch.equal(c_out, i_out) and torch.equal(c_pre, i_pre) and torch.equal(c_blk, i_blk))
        except Exception as e:  # noqa: BLE001
            same = f"image raised {type(e).__name__}"
        prod["unserved_checked"].append({"M": m, "KMAX": kmax, "k": k, "bwi": bwi, "served": served,
                                         "image_bit_equal": same})
        prod["ok"] &= served == 0 and same is True
    prod["ok"] &= prod["served"] == prod["served_expected"]
    print(f"PRODUCTION-PATH ok={prod['ok']} {json.dumps(prod)}", flush=True)
    ok &= prod["ok"]
    rec["equality"], rec["production_path"] = eq, prod

    rec["timing"] = timings(ov, aro.NUM_WARPS)
    rec["ok"] = bool(ok)
    return bool(ok)


@triton.jit
def _spin_kernel(x_ptr, n, USE_PDL: tl.constexpr):
    """A PDL predecessor: waits for the previous launch, lets its dependent launch at once, then spins ~n FMAs."""
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    pid = tl.program_id(0)
    x = tl.load(x_ptr + pid)
    for _ in range(n):
        x = x * 1.0000001 + 1e-7
    tl.store(x_ptr + pid, x)


SPIN_N = 3000
_SPIN_BUF = []


def spin():
    if not _SPIN_BUF:
        _SPIN_BUF.append(torch.zeros(148, device="cuda", dtype=torch.float32))
    _spin_kernel[(148,)](_SPIN_BUF[0], SPIN_N, USE_PDL=True, num_warps=1, launch_pdl=True)


def exposed_us(ov, specs, on):
    """Per-call time an attn_res call adds behind a long PDL predecessor that triggers early (as the MoE tail and
    the attention o_proj do in the server): (graph[spin, attn] x N - graph[spin] x N) / N."""
    calls = timed_calls(ov, specs, on)
    both = []
    for f in calls:
        both += [spin, f]
    t_both, _ = graph_time(both)
    t_spin, _ = graph_time([spin] * len(calls))
    return (t_both - t_spin) / len(calls), t_spin / len(calls)


def graph_time(fn_calls):
    """fn_calls: list of zero-arg callables run back to back inside one graph; returns median us per replay."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in fn_calls:
            f()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in fn_calls:
            f()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(REPS):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1000.0)
    return statistics.median(ts), min(ts)


def timed_calls(ov, specs, on):
    """specs: list of (k, bwi, delta, norm); ROT input sets; returns callables under the switch value."""
    sets = {}
    calls = []
    for i, (k, bwi, d, n) in enumerate(specs):
        key = (d, n, i % ROT)
        if key not in sets:
            sets[key] = make_inputs(48, KMAX_MAIN, 77 + len(sets), d, n)
        inp = sets[key]

        def f(inp=inp, k=k, bwi=bwi):
            with Switch(on):
                call(ov.attn_res, inp, k, bwi)

        calls.append(f)
    return calls


def step_specs():
    # nsys-16's native template k is num_blocks + 1 (the op accepts num_blocks 1..8): k=2..8 x 22, k=9 x 16.
    specs = []
    for nb in range(1, 9):
        specs += [(nb, -1, True, True)] * (22 if nb < 8 else 16)
    for idx in range(1, 9):
        specs += [(idx, idx, True, True), (idx + 1, -1, False, True)]
    specs.append((KMAX_MAIN, -1, True, False))
    return specs


def timings(ov, nw):
    out = {"M": 48, "KMAX": KMAX_MAIN, "calls_per_graph": CALLS, "rot": ROT, "reps": REPS, "num_warps": nw,
           "per_call": {}}
    singles = [(k, -1, True, True) for k in range(1, 9)] + [(k, k, True, True) for k in (1, 4, 8)] + \
              [(k, -1, False, True) for k in (2, 5, 9)] + [(9, -1, True, False), (0, 0, False, True)]
    for k, bwi, d, n in singles:
        tag = f"k{k}_w{int(bwi >= 0)}_d{int(d)}_n{int(n)}"
        try:
            cu, cmin = graph_time(timed_calls(ov, [(k, bwi, d, n)] * CALLS, True))
            tu, tmin = graph_time(timed_calls(ov, [(k, bwi, d, n)] * CALLS, False))
        except Exception as e:  # noqa: BLE001
            out["per_call"][tag] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            print(f"TIMING {tag} error {out['per_call'][tag]['error']}", flush=True)
            continue
        r = {"cand_us": cu / CALLS, "ctrl_us": tu / CALLS, "cand_min_us": cmin / CALLS, "ctrl_min_us": tmin / CALLS,
             "delta_us": (cu - tu) / CALLS}
        out["per_call"][tag] = r
        print(f"TIMING {tag} {json.dumps(r)}", flush=True)
    out["exposed"] = {}
    for k in range(1, 9):
        try:
            ce, sp = exposed_us(ov, [(k, -1, True, True)] * CALLS, True)
            te, _ = exposed_us(ov, [(k, -1, True, True)] * CALLS, False)
        except Exception as e:  # noqa: BLE001
            out["exposed"][f"k{k}"] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            continue
        out["exposed"][f"k{k}"] = {"cand_us": ce, "ctrl_us": te, "delta_us": ce - te, "spin_us": sp}
        print(f"EXPOSED k{k}_w0_d1_n1 {json.dumps(out['exposed'][f'k{k}'])}", flush=True)
    specs = step_specs()
    ce, sp = exposed_us(ov, specs, True)
    te, _ = exposed_us(ov, specs, False)
    out["exposed"]["step"] = {"calls": len(specs), "cand_us_per_call": ce, "ctrl_us_per_call": te,
                              "delta_us_per_call": ce - te, "delta_us_per_step": (ce - te) * len(specs),
                              "spin_us": sp}
    print(f"EXPOSED step {json.dumps(out['exposed']['step'])}", flush=True)
    cu, cmin = graph_time(timed_calls(ov, specs, True))
    tu, tmin = graph_time(timed_calls(ov, specs, False))
    out["step"] = {"calls": len(specs), "cand_us": cu, "ctrl_us": tu, "cand_min_us": cmin, "ctrl_min_us": tmin,
                   "delta_us": cu - tu, "cand_us_per_call": cu / len(specs), "ctrl_us_per_call": tu / len(specs)}
    print(f"TIMING step {json.dumps(out['step'])}", flush=True)
    return out


def run_graph(rec: dict) -> bool:
    import perfbot_kernels.attn_res_one as aro

    ov = overlay_module()
    res = {"shapes": 0, "failed": [], "served_in_capture": True}
    shapes = [(48, k, bwi, d, n) for (_, _, k, bwi, d, n) in cases(KMAX_MAIN, [48])]
    shapes += [(m, k, -1, True, True) for m in GRAPH_M if m != 48 for k in range(0, KMAX_MAIN + 1)]
    shapes += [(m, k, k, True, True) for m in GRAPH_M if m != 48 for k in (1, 8)]
    for i, (m, k, bwi, d, n) in enumerate(shapes):
        base = make_inputs(m, KMAX_MAIN, 5000 + i, d, n)
        with Switch(True):
            e_out, e_pre, e_blk = call(ov.attn_res, clone(base), k, bwi)
            st = clone(base)
            call(ov.attn_res, clone(base), k, bwi)  # warm on a side copy
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            before = aro.CALLS["served"]
            with torch.cuda.graph(g):
                g_out, _, _ = call(ov.attn_res, st, k, bwi)
            res["served_in_capture"] &= aro.CALLS["served"] - before == 1
        oks = []
        for _ in range(2):
            for key in ("prefix", "blocks"):
                st[key].copy_(base[key])
            g.replay()
            torch.cuda.synchronize()
            oks.append(bool(torch.equal(g_out, e_out) and torch.equal(st["prefix"], e_pre)
                            and torch.equal(st["blocks"], e_blk)))
        res["shapes"] += 1
        if not all(oks):
            res["failed"].append({"M": m, "k": k, "bwi": bwi, "delta": d, "norm": n, "replays": oks})
            print(f"GRAPH-FAIL M={m} k={k} bwi={bwi} d={d} n={n} {oks}", flush=True)
    res["ok"] = not res["failed"] and res["served_in_capture"] and res["shapes"] > 0
    print(f"GRAPH shapes={res['shapes']} failed={len(res['failed'])} served_in_capture={res['served_in_capture']}",
          flush=True)
    rec["graph"] = res
    return res["ok"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", action="store_true")
    a = ap.parse_args()
    rec = load_record()
    try:
        if a.graph:
            ok = run_graph(rec)
            rec["ok"] = bool(rec.get("ok", True) and ok)
        else:
            ok = run_selftest(rec)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        rec.setdefault("errors", []).append(traceback.format_exc()[-3000:])
        ok = False
        rec["ok"] = False
    save_record(rec)
    print(f"PERFBOT-ATTN-RES-SELFTEST mode={'graph' if a.graph else 'selftest'} ok={ok} record={results_path()}",
          flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
