"""Gate of the fused KDA core (core.py) against the image's launches 2-5 (common.Control.core), real weights.

  python3 -B -m perfbot_kernels.kda_stage.selftest_core [--check] [--graph] [--pdl-order] [--time]   (none = all)

Writes $W/$PERFBOT_RESULTS/core-gate.json after every section:
  --sweep    (only when asked; not part of the gate) time every core.CFGS entry and run the gate on the fastest
             at N 6 among those no slower than control at any N. Without it the gate runs on core.CFG.
  N 6 and 16 in check, graph and pdl_order use a spec_idx row stride of 33 (the server's block-table slice);
  check also requires core.check() to accept the tensors.
  check      N in {1,2,3,6,8,16} (one null request from N 3), 6 rounds per N, acceptance redrawn each round, the
             candidate's pools carried over. Per round both implementations start from the same state and the same
             bf16 proj; errors of each vs the fp64 reference on that bf16 proj (common.reference(..., proj)), the
             stage's true input; pass = candidate max_abs <= control max_abs on core, ssm, conv; slots outside the
             request's indices bit-unchanged; null rows 0. The errors vs the fp64 reference from x (in_proj not
             rounded) are recorded too ("vs_x"); they share the bf16 rounding of proj and do not gate.
  graph      capture + 2 replays per N bit-identical to eager; a replay after rewriting proj and accepted equals
             eager on those inputs.
  pdl_order  a PDL predecessor (launch_dependents first, a random spin, then it writes proj, accepted, spec_idx and
             the start state slots) before the candidate; output and pools bit-equal to the run with no
             predecessor, >= 50 random delays per N in {6, 16}.
  time       CUDA-graph replay of a chain over 4 distinct layers, us per layer, candidate vs Control.core.
gate PASS = check, graph, pdl_order pass, time N 6 <= 0.75 x control and no N slower than control.
"""

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

from . import common as c
from . import core as k

CHECK_N, GRAPH_N, PDL_N, TIME_N = (1, 2, 3, 6, 8, 16), (1, 2, 6, 16), (6, 16), (1, 2, 6, 8, 16)
ROUNDS, DELAYS, LAYERS = 6, 50, 4
ABLATE_N = (6, 16)
DIAG_VARIANTS = ("landed", "oldorder")  # core.py variants timed beside the candidate (diagnostics)
IDX_RS = 33  # check/graph/pdl_order at N 6 and 16: spec_idx = [:, :8] of an [N, 33] table (the server's block table)


def rs(n):
    return IDX_RS if n in (6, 16) else c.SLOTS
KEYS = ("core", "ssm", "conv")


def sha() -> str:
    d = Path(__file__).resolve().parent
    return hashlib.sha256(b"".join((d / f).read_bytes() for f in ("core.py", "selftest_core.py"))).hexdigest()[:16]


def nulls(n):
    return 1 if n >= 3 else 0


def ssm_store(st):
    s0 = st.ssm.stride(0)
    return st.ssm.as_strided((st.ssm.shape[0], s0), (s0, 1))


def restore(dst, src):
    dst.conv.copy_(src.conv)
    ssm_store(dst).copy_(ssm_store(src))
    dst.spec_idx.copy_(src.spec_idx)
    dst.accepted.copy_(src.accepted)


def new_out(st):
    return torch.empty(1, st.t, c.H, c.D, dtype=torch.bfloat16, device="cuda")


def snap(st, out):
    return out.clone(), st.conv.clone(), ssm_store(st).clone()


def same(a, b):
    return all(torch.equal(x, y) for x, y in zip(a, b))


# ---------------------------------------------------------------- check


def check(lw):
    ctl, rows, ok = c.Control(lw), {}, True
    for n in CHECK_N:
        st = c.make_state(n, seed=n, null_requests=nulls(n), idx_row_stride=rs(n))
        for rnd in range(ROUNDS):
            before = st.clone()
            proj = ctl.in_proj(before)
            ref = c.reference(lw, before, proj)
            ref_x = c.reference(lw, before)
            out_c = new_out(st)
            reason = k.check(lw, st, proj, out_c)
            k.core(lw, st, proj, out_c)
            stk, out_k = before.clone(), new_out(st)
            ctl.core(stk, proj, out_k)
            torch.cuda.synchronize()
            ec = c.errors(lw, before, st, ctl.o_proj(out_c), out_c, ref)
            ek = c.errors(lw, before, stk, ctl.o_proj(out_k), out_k, ref)
            vs_x = {"candidate": c.errors(lw, before, st, ctl.o_proj(out_c), out_c, ref_x),
                    "control": c.errors(lw, before, stk, ctl.o_proj(out_k), out_k, ref_x)}
            valid = before.valid.cpu()
            sidx = before.spec_idx.cpu()
            touched_ssm = {int(s) for i in range(n) if valid[i] for s in sidx[i] if s > 0}
            touched_conv = {int(sidx[i, 0]) for i in range(n) if valid[i] and sidx[i, 0] > 0}
            keep_ssm = [s for s in range(st.ssm.shape[0]) if s not in touched_ssm]
            keep_conv = [s for s in range(st.conv.shape[0]) if s not in touched_conv]
            untouched = (torch.equal(ssm_store(st)[keep_ssm], ssm_store(before)[keep_ssm])
                         and torch.equal(st.conv[keep_conv], before.conv[keep_conv]))
            null_rows = [r for i in range(n) if not valid[i] for r in range(i * c.SLOTS, (i + 1) * c.SLOTS)]
            nulls_zero = bool((out_c.view(-1, c.P)[null_rows] == 0).all()) if null_rows else True
            le = {key: ec[key]["max_abs"] <= ek[key]["max_abs"] for key in KEYS}
            good = all(le.values()) and untouched and nulls_zero and reason is None and st.spec_idx.stride(0) == rs(n)
            ok &= good
            rows[f"N{n}r{rnd}"] = {"candidate": ec, "control": ek, "le": le, "untouched": untouched,
                                   "null_rows_zero": nulls_zero, "accepted": before.accepted.tolist(), "ok": good,
                                   "idx_row_stride": st.spec_idx.stride(0), "check_reason": reason,
                                   "vs_x": vs_x}
            print(f"CHECK N={n} r={rnd} ok={good} " + " ".join(
                f"{key}:{ec[key]['max_abs']:.3e}/{ek[key]['max_abs']:.3e}" for key in KEYS), flush=True)
            c.next_round(st, seed=1000 * n + rnd)
    return {"ok": ok, "rows": rows}


# ---------------------------------------------------------------- graph


def graph(lw):
    ctl, rows, ok = c.Control(lw), {}, True
    for n in GRAPH_N:
        st0 = c.make_state(n, seed=100 + n, null_requests=nulls(n), idx_row_stride=rs(n))
        proj = ctl.in_proj(st0)
        s1 = st0.clone()
        o1 = new_out(st0)
        k.core(lw, s1, proj, o1)
        eager = snap(s1, o1)
        s2 = st0.clone()
        c.next_round(s2, seed=7 + n)
        proj2 = ctl.in_proj(s2)
        s2e, o2 = s2.clone(), new_out(st0)
        k.core(lw, s2e, proj2, o2)
        eager2 = snap(s2e, o2)
        sg, og, pg = st0.clone(), new_out(st0), proj.clone()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            k.core(lw, sg, pg, og)
        reps = []
        for _ in range(2):
            restore(sg, st0)
            og.zero_()
            g.replay()
            torch.cuda.synchronize()
            reps.append(same(snap(sg, og), eager))
        restore(sg, s2)
        pg.copy_(proj2)
        g.replay()
        torch.cuda.synchronize()
        rewritten = same(snap(sg, og), eager2)
        good = all(reps) and rewritten
        ok &= good
        rows[f"N{n}"] = {"replays_equal_eager": reps, "rewritten_equals_eager": rewritten,
                         "idx_row_stride": sg.spec_idx.stride(0),
                         "accepted": [st0.accepted.tolist(), s2.accepted.tolist()], "ok": good}
        print(f"GRAPH N={n} {rows[f'N{n}']}", flush=True)
        del g
    return {"ok": ok, "rows": rows}


# ---------------------------------------------------------------- pdl order

_PRED = {}


def predecessor():
    """grid (n), 256 threads: launch_dependents, spin `delay` clocks, then write proj, accepted, spec_idx (row
    stride irs) and the start state slots from staging copies (isrc is contiguous [n, 8])."""
    if "fn" in _PRED:
        return _PRED["fn"]
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32, Int64
    from cuda.bindings.driver import CUstream
    from cutlass.cute.runtime import make_ptr

    gm = cute.AddressSpace.gmem

    @cute.kernel
    def kern(delay: Int64, psrc: cute.Tensor, pdst: cute.Tensor, nw: Int32, asrc: cute.Tensor, adst: cute.Tensor,
             isrc: cute.Tensor, idst: cute.Tensor, irs: Int32, ssrc: cute.Tensor, ssm: cute.Tensor, sstride: Int64,
             n: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        b, _, _ = cute.arch.block_idx()
        cute.arch.griddepcontrol_launch_dependents()
        t0 = cute.arch.clock64()
        while cute.arch.clock64() - t0 < delay:
            pass
        for i in cutlass.range(b * 256 + tidx, nw, n * 256):
            pdst[i] = psrc[i]
        if b == 0:
            for i in cutlass.range(tidx, n * c.SLOTS, 256):
                idst[(i // c.SLOTS) * irs + i % c.SLOTS] = isrc[i]
            if tidx < n:
                adst[tidx] = asrc[tidx]
        slot = isrc[b * c.SLOTS + asrc[b] - 1]
        if slot > 0:
            per = c.H * c.D * c.D
            for i in cutlass.range(tidx, per, 256):
                ssm[Int64(slot) * sstride + i] = ssrc[Int64(b) * per + i]

    @cute.jit
    def launch(delay: Int64, psrc: cute.Pointer, pdst: cute.Pointer, nw: Int32, asrc: cute.Pointer,
               adst: cute.Pointer, isrc: cute.Pointer, idst: cute.Pointer, irs: Int32, ssrc: cute.Pointer,
               ssm: cute.Pointer, sstride: Int64, n: Int32, stream):
        big = cute.make_layout(1 << 30)
        T_ = lambda p: cute.make_tensor(p, big)  # noqa: E731
        kern(delay, T_(psrc), T_(pdst), nw, T_(asrc), T_(adst), T_(isrc), T_(idst), irs, T_(ssrc), T_(ssm), sstride,
             n).launch(grid=[n, 1, 1], block=[256, 1, 1], stream=stream)

    fake = lambda dt, i: make_ptr(dt, 4096 * (i + 1), gm, assumed_align=16)  # noqa: E731
    i32, f32 = cutlass.Int32, cutlass.Float32
    stream = CUstream(torch.cuda.current_stream().cuda_stream)
    fn = cute.compile(launch, Int64(0), fake(i32, 0), fake(i32, 1), Int32(1), fake(i32, 2), fake(i32, 3),
                      fake(i32, 4), fake(i32, 5), Int32(8), fake(f32, 6), fake(f32, 7), Int64(1), Int32(1), stream)

    def run(delay, psrc, pdst, asrc, adst, isrc, idst, ssrc, st):
        P_ = lambda t, dt: make_ptr(dt, t.data_ptr(), gm, assumed_align=16)  # noqa: E731
        fn(Int64(delay), P_(psrc, i32), P_(pdst, i32), Int32(psrc.numel() // 2), P_(asrc, i32), P_(adst, i32),
           P_(isrc, i32), P_(idst, i32), Int32(idst.stride(0)), P_(ssrc, f32), P_(st.ssm, f32),
           Int64(st.ssm.stride(0)), Int32(st.n),
           CUstream(torch.cuda.current_stream().cuda_stream))

    _PRED["fn"] = run
    return run


def pdl_order(lw):
    ctl, rows, ok = c.Control(lw), {}, True
    pred = predecessor()
    rng = torch.Generator().manual_seed(5)
    for n in PDL_N:
        st0 = c.make_state(n, seed=200 + n, null_requests=nulls(n), idx_row_stride=rs(n))
        proj = ctl.in_proj(st0)
        sr, orf = st0.clone(), new_out(st0)
        k.core(lw, sr, proj, orf)
        ref = snap(sr, orf)
        acc, sidx = st0.accepted.cpu(), st0.spec_idx.cpu()
        starts = [int(sidx[i, int(acc[i]) - 1]) for i in range(n)]
        stage = torch.stack([st0.ssm[s] if s > 0 else torch.zeros_like(st0.ssm[0]) for s in starts]).contiguous()
        junk = st0.clone()
        junk.accepted.copy_(st0.accepted % c.SLOTS + 1)
        junk.spec_idx.copy_(st0.spec_idx.roll(1, dims=0))
        for s in starts:
            if s > 0:
                junk.ssm[s].zero_()
        s, o, pdst = st0.clone(), new_out(st0), torch.zeros_like(proj)
        bad = []
        delays = torch.randint(0, 40000, (DELAYS,), generator=rng).tolist()
        for d in delays:
            restore(s, junk)
            pdst.zero_()
            o.fill_(1.0)
            torch.cuda.synchronize()
            pred(d, proj, pdst, st0.accepted, s.accepted, st0.spec_idx.contiguous(), s.spec_idx, stage, s)
            k.core(lw, s, pdst, o)
            torch.cuda.synchronize()
            if not same(snap(s, o), ref):
                bad.append(d)
        good = not bad
        ok &= good
        rows[f"N{n}"] = {"idx_row_stride": s.spec_idx.stride(0), "delays": len(delays),
                         "max_delay_clocks": max(delays), "mismatch_delays": bad, "ok": good}
        print(f"PDL N={n} {rows[f'N{n}']}", flush=True)
    return {"ok": ok, "rows": rows}


# ---------------------------------------------------------------- time


def sweep(layers):
    """Candidate us per layer for every core.CFGS entry at TIME_N (and without state stores at ABLATE_N); control
    once per N. A quick screen per cfg: one N 6 round's errors vs the proj reference, <= control's."""
    lw = layers[0]
    ctl = c.Control(lw)
    st0 = c.make_state(6, seed=77, null_requests=1)
    proj = ctl.in_proj(st0)
    ref = c.reference(lw, st0, proj)
    stk, ok_ = st0.clone(), new_out(st0)
    ctl.core(stk, proj, ok_)
    ek = c.errors(lw, st0, stk, ctl.o_proj(ok_), ok_, ref)
    screen = {"control": {key: ek[key]["max_abs"] for key in KEYS}}
    for i, cfg in enumerate(k.CFGS):
        s, o = st0.clone(), new_out(st0)
        k.core(lw, s, proj, o, cfg=cfg)
        ec = c.errors(lw, st0, s, ctl.o_proj(o), o, ref)
        screen[f"cfg{i}"] = {key: ec[key]["max_abs"] for key in KEYS}
        screen[f"cfg{i}"]["ok"] = all(ec[key]["max_abs"] <= ek[key]["max_abs"] for key in KEYS)
        print(f"SCREEN cfg{i}={cfg} {screen[f'cfg{i}']} control {screen['control']}", flush=True)
    rows = {"screen": screen}
    for n in TIME_N:
        ctls = [c.Control(lw) for lw in layers]
        sts = [c.make_state(n, seed=10 * n + i) for i in range(len(layers))]
        projs = [ctl.in_proj(st) for ctl, st in zip(ctls, sts)]
        oc = [new_out(st) for st in sts]
        L = len(layers)
        stk = [st.clone() for st in sts]
        row = {"control_core_us": round(c.graph_timed(
            lambda: [ctl.core(st, p, o) for ctl, st, p, o in zip(ctls, stk, projs, oc)]) / L, 3)}
        for i, cfg in enumerate(k.CFGS):
            stc = [st.clone() for st in sts]
            for lw, st, p, o in zip(layers, stc, projs, oc):  # eager first: compile outside capture
                k.core(lw, st, p, o, cfg=cfg)
            row[f"cfg{i}_us"] = round(c.graph_timed(
                lambda: [k.core(lw, st, p, o, cfg=cfg) for lw, st, p, o in zip(layers, stc, projs, oc)]) / L, 3)
            if n in ABLATE_N:
                row[f"cfg{i}_nostore_us"] = round(c.graph_timed(lambda: [
                    k.core(lw, st, p, o, cfg=cfg, variant="nostore")
                    for lw, st, p, o in zip(layers, stc, projs, oc)]) / L, 3)
        rows[f"N{n}"] = row
        print(f"SWEEP N={n} {row}", flush=True)

    def key(i):
        slower = any(r[f"cfg{i}_us"] > r["control_core_us"] for n_, r in rows.items() if n_ != "screen")
        return (not screen[f"cfg{i}"]["ok"], slower, rows["N6"][f"cfg{i}_us"])
    best = min(range(len(k.CFGS)), key=key)
    return {"ok": True, "rows": rows, "cfgs": [list(x) for x in k.CFGS], "best": best}


def timing(layers):
    rows = {}
    for n in TIME_N:
        ctls = [c.Control(lw) for lw in layers]
        sts = [c.make_state(n, seed=10 * n + i) for i in range(len(layers))]
        projs = [ctl.in_proj(st) for ctl, st in zip(ctls, sts)]
        stc, stk = [st.clone() for st in sts], [st.clone() for st in sts]
        oc, ok_ = [new_out(st) for st in sts], [new_out(st) for st in sts]
        L = len(layers)
        cand = c.graph_timed(lambda: [k.core(lw, st, p, o) for lw, st, p, o in zip(layers, stc, projs, oc)]) / L
        ctlt = c.graph_timed(lambda: [ctl.core(st, p, o) for ctl, st, p, o in zip(ctls, stk, projs, ok_)]) / L
        rows[f"N{n}"] = row = {"candidate_us": round(cand, 3), "control_core_us": round(ctlt, 3),
                               "ratio": round(cand / ctlt, 3)}
        # diagnostics, not the gate: the landed core (A-core-04) and the relaxed arrive with the landed load order in
        # the same chain; and per layer in_proj (F.linear, as the server) + core, minus in_proj alone
        for var in DIAG_VARIANTS:
            row[f"{var}_us"] = round(c.graph_timed(lambda: [k.core(lw, st, p, o, variant=var)
                                                           for lw, st, p, o in zip(layers, stc, projs, oc)]) / L, 3)
        xs = [st.x for st in sts]
        inp = c.graph_timed(lambda: [F.linear(x, lw.w_in) for lw, x in zip(layers, xs)]) / L
        row["inproj_only_us"] = round(inp, 3)
        for var in ("full",) + DIAG_VARIANTS:
            t_ = c.graph_timed(lambda: [k.core(lw, st, F.linear(x, lw.w_in), o, variant=var)
                                        for lw, st, x, o in zip(layers, stc, xs, oc)]) / L
            row[f"inproj_{var}_core_us"] = round(t_ - inp, 3)
        print(f"TIME N={n} {row}", flush=True)
    ratio6 = rows.get("N6", {}).get("ratio", 9.0)
    ok = ratio6 <= 0.75 and all(r["ratio"] <= 1.0 for r in rows.values())
    return {"ok": ok, "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    for f in ("check", "graph", "pdl-order", "time", "sweep"):
        ap.add_argument(f"--{f}", action="store_true")
    args = ap.parse_args()
    modes = [m for m in ("check", "graph", "pdl_order", "time") if getattr(args, m)] or \
        ["check", "graph", "pdl_order", "time"]
    path = c.results_path("core-gate.json")
    rec = {"item": "expedition-kda-core", "sha": sha(), "modes": modes, "device": torch.cuda.get_device_name(),
           "cfg": list(k.CFG)}

    def write():
        path.write_text(json.dumps(rec, indent=1))

    rc = 0
    try:
        prefixes = c.kda_layers()
        layers = [c.load_layer(p) for p in prefixes[:LAYERS if "time" in modes or args.sweep else 1]]
        t0 = time.time()
        st = c.make_state(1)
        k.core(layers[0], st, torch.zeros(st.t, c.COLS, dtype=torch.bfloat16, device="cuda"), new_out(st))
        torch.cuda.synchronize()
        rec["compile_s"] = round(time.time() - t0, 1)
        print(f"COMPILED {rec['compile_s']} s sha={rec['sha']}", flush=True)
        if args.sweep:
            rec["sweep"] = sweep(layers)
            k.CFG = k.CFGS[rec["sweep"]["best"]]
            rec["cfg"] = list(k.CFG)
            print(f"CFG {k.CFG}", flush=True)
            write()
        fns = {"check": lambda: check(layers[0]), "graph": lambda: graph(layers[0]),
               "pdl_order": lambda: pdl_order(layers[0]), "time": lambda: timing(layers)}
        for m in modes:
            try:
                rec[m] = fns[m]()
            except Exception:
                rec[m] = {"ok": False, "error": traceback.format_exc()}
                print(rec[m]["error"], file=sys.stderr, flush=True)
            write()
        if len(modes) == 4:
            rec["gate"] = "PASS" if all(rec[m]["ok"] for m in modes) else "FAIL"
        rc = 0 if all(rec[m]["ok"] for m in modes) else 1
    except Exception:
        rec["error"] = traceback.format_exc()
        print(rec["error"], file=sys.stderr, flush=True)
        rc = 1
    write()
    print("RESULT", json.dumps({m: rec.get(m, {}).get("ok") for m in modes}), rec.get("gate"), flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
