"""P-08 judge v5 (card §3: rulings R1-R7, R11a; review e94ff6a findings 1-11). Modes:

  check    --kernel K [--holdout] [--quick]   correctness of K vs fp64 (truth 1) and the incumbent (truth 2)
  controls [--quick]                          positive controls (control set members + clean fp32 emulator) pass
  power    [--quick]                          R3 power check: every mutant fails; writes the kill table (README.md)
  selftest [--quick]                          controls + identity pass; power check
  aa       --session S [--B --ctx | --case C]  one A/A block (incumbent vs incumbent) of seat session S
  time     --session S --kernel K [--B --ctx | --case C] one ABAB block incumbent vs K (void judged at verdict, R4)
  verdict  --session S --kernel K [--B --ctx | --case C, --bar]   R2/R4 bar reading over the session's blocks

Correctness bounds (R1 + R3: per case, per rank, PER DRAW, per metric, no averaging; the candidate runs all 4 draws).
env_d = the ENVELOPE (max) over the control set on draw d = the incumbent at the served split floor, the incumbent at
the other floor (kernels.ALT: B1 24, B2 12, B3 8, B5 4) and the fp32 split-KV emulator (R5), on the same case and draw:
  rel_peak  (rows with pmax >= 0.4)            cand_d <= 1.02 x env_d
  err_flat  (other live rows, value-rms units) cand_d <= 1.02 x env_d
  LSE max abs (base 2)                         cand_d <= max(1.02 x env_d, 2^-16)
  finite live rows; rows with no visible key merge-equivalent (finite out; lse <= live min - 64, -inf or NaN)
  combine (8 ranks, DcpValue merge emulated, value-rms units): cand_d <= 1.02 x env_d (same control set per rank)
  graph: capture on case A, copy case B's inputs, replay == eager on B (bit-equal); B >= 6: bit-equal at B6 and B8
  B3 split 4 and B5 split 2 at 341k are bar points (R11); B4 (split 3) is candidate range: planner, other-floor and
  graded cases at B3-5, the split is asserted per case; power reports robust kills per B (review b8b8f3e)
Power check (R3 + R5, pre-registered): every one of the 16 mutants has a robust kill, a (case, draw, metric) with
margin >= 1.5x (margin = cand / bound; > 1 fails a candidate; near 1.00x is noise for a kill); a survivor is
reported to the coordinator, the rule is not changed here.
Split salts: the seam takes a split FLOOR, so B1 can only be raised from the planner's 12 (24), B2 from 6 (12), B3 from 4 (8), B5 from 2 (4)
(LOG J-01); boundary-4095 runs split 12 with one empty split (not a single split; review 11b).

Timing (card §3 Timing (a) + R2): ring of >= 24 distinct KV shards and >= 4 x L2; in shard step k arm A reads shard k
and arm B reads shard (k + n/2) mod n, so every call is >= n/2 calls (>= 2 x L2 bytes) from the last read of its
shard (review 1); in-graph event pairs around each [attention, reducer]; two graphs (A first on even shard steps / B
first) replayed alternately (review 8); warm-up 20 replays discarded, median over >= 50 replays; ratio = B / A per
replay, denominator from the same block. After the replays every ring output of B is compared bit-equal with an eager
call on its shard (workspace state across back-to-back calls, review 10).
R2 + R4 + R6 session (per B, ctx): A/A blocks INTERLEAVED with the timed blocks (A/A, timed, A/A, ..., timed, A/A;
>= 5 A/A, the first and the last block are A/A; each block its own process) give: the A/A incumbent median (median of
the A/A blocks' incumbent medians), the absolute band [min, max] of those medians, the reference clock (median of their
clock medians), the in-block A/A band (max over them of max(|ratio median - 1|, (p90 - p10) / 2)) and
session_throttle_reasons (active reasons, idle 0x1 excluded, common to every A/A block: the seat's steady state).
Ratio OF RECORD (R6) = the timed block's candidate median / the A/A incumbent median; the in-block paired ratio is a
diagnostic only. Void is judged at verdict over the whole session: a timed block is void iff its median SM clock is
> 2 % off the reference, or its active reason set differs from session_throttle_reasons, or its incumbent median is
more than 1.0 % outside the absolute band (gross anomaly). Every block is written when it runs; the verdict record
lists every timed block with its void reason. Bar: >= 5 valid blocks, voids <= 50 % of the timed blocks, and
median of the valid ratios of record + A/A band width <= bar, where the width is the absolute band's relative width
(max - min) / A/A median (R7; the in-block A/A band is a diagnostic);
otherwise the session is "unstable" (no claim, no re-running to select blocks). session_throttle_reasons is reported
with every claim (a steady 0x4 on B300 is an integrator hazard, not a void).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import socket
import statistics
import subprocess
import sys
import time

import torch

import kernels
import p08

HERE = os.path.dirname(os.path.abspath(__file__))
SGRID = os.environ.get("P08_SGRID")  # var-split arm grid split override; written into every row (coordinator)
RES = os.environ.get("PERFBOT_MLA_BAR_RES") or os.path.join(os.path.dirname(HERE), "results")
NS = ("needle", "anti")
ALT = kernels.ALT  # the other split floor per B (planner: B1 12, B2 6, B3 4, B5 2)


def C(name, ctx, rank=0, stress=NS, min_split=None, want=None, splits=None):
    """want: the split the planner must pick for this case (R11 bar points); a different split aborts the judge.
    splits: per-request split counts of the var-split arm (v5.2); adds 'emu at splits' to this case's envelope."""
    return dict(name=name, ctx=ctx, rank=rank, stress=stress, min_split=min_split, want=want, splits=splits)


CASES = []
for _ctx in (200_003, 250_017, 341_000, 448_013):
    _r = (_ctx - 1) % 8 if _ctx == 250_017 else 0      # 250k on the rank holding the tail page
    for _B in (1, 2):
        CASES.append(C(f"c{_ctx // 1000}k-b{_B}-planner", [_ctx] * _B, _r))
        CASES.append(C(f"c{_ctx // 1000}k-b{_B}-split{ALT[_B]}", [_ctx] * _B, _r, min_split=ALT[_B]))
CASES += [
    C("tailrank-341k-b1", [341_000], rank=(341_000 - 1) % 8),  # rows see 42,625 + (i >= 7): the causal needle
    C("qdom-dc-341k-b1", [341_000], stress=("needle", "qdom", "dc")),
    C("boundary-4095-b1", [32_760]),                       # local shard 4,095 rows on rank 0 (split 12, one empty)
    C("emptyreq-b2", [341_000, 0]),                        # a padded request: length 0, total 0
    C("emptyrows-b2", [341_000, 9], rank=3),               # rows 0-1 of request 1 see no key on rank 3
    C("ragged-b2", [250_017, 200_003], rank=5),            # unequal lengths, neither page- nor split-divisible
    C("graded-341k-b1", [341_000], stress=("graded", "anti")),               # review 3: mass over all splits
    C("graded-341k-b2", [341_000] * 2, stress=("graded",)),
    C("graded-448k-b1-split24", [448_013], rank=7, stress=("graded", "anti"), min_split=24),
]
HOLDOUT = [C("holdout-300k-b1", [300_000])]
# v5 (R11a): the B3 / B5 bar points at 341k, planner split and other floor. Listed after HOLDOUT in the seed order, so
# every v4 case and the holdout keep their v4 seeds (a v4 kill stays a v5 kill).
PLAN = {3: 4, 4: 3, 5: 2}  # planner split at 341k (B >= 6 dispatches to the incumbent, so B3-5 are candidate range)
_bp = lambda Bs: [C(f"c341k-b{B}-{t}", [341_000] * B, min_split=f, want=w)
                  for B in Bs for t, f, w in (("planner", None, PLAN[B]), (f"split{ALT[B]}", ALT[B], ALT[B]))]
CASES_R11 = _bp((3, 5))
# review b8b8f3e findings 1-2 (appended after the R11 cases, so their seeds stay): B4 is in the candidate range;
# graded rows are the only stress that exposes a merge-weight error at the split counts 4 / 3 / 2.
CASES_R11 += _bp((4,)) + [C(f"graded-341k-b{B}", [341_000] * B, stress=("graded",), want=PLAN[B]) for B in (3, 4, 5)]
# v5.2 (var-split ruling): one mixed-length B5 step, served split 2 for all; the var-split arm uses VAR_SPLITS.


def var_splits(lens, units=12):
    """Rule of record (P-14 gate pass 20): one-wave greedy min-max. Every request starts at split 1; each further
    unit goes to the request with the largest len_i / split_i (lowest index on a tie) until `units` are used."""
    sp = [1] * len(lens)
    for _ in range(units - len(lens)):
        b = max(range(len(lens)), key=lambda i: (lens[i] / sp[i], -i))
        sp[b] += 1
    return tuple(sp)


_MIX = [345_000] + [88_000] * 4
VAR_SPLITS = var_splits(_MIX)  # (4, 2, 2, 2, 2): 12 units = 144 non-empty CTAs, one wave on 148 SMs
assert VAR_SPLITS == (4, 2, 2, 2, 2), VAR_SPLITS
CASES_R11 += [C("c345k-mixed-b5", _MIX, want=PLAN[5], splits=VAR_SPLITS),
              C("graded-345k-mixed-b5", _MIX, stress=("graded",), want=PLAN[5], splits=VAR_SPLITS)]
QUICK = ("c341k-b1-planner", "c341k-b2-planner", "emptyrows-b2", "graded-341k-b1", "graded-341k-b2",
         "c341k-b3-planner", "c341k-b5-planner", "graded-341k-b3", "graded-341k-b4", "graded-341k-b5", "c345k-mixed-b5")
COMBINE = [dict(name="combine-341k-b1", ctx=[341_000]), dict(name="combine-mixed-b2", ctx=[341_000, 9]),
           dict(name="combine-padded-b2", ctx=[341_000, 0])]
DRAWS = 4


def _seed(i, d):
    return 100 + 7 * i + 1000 * d


def _case(spec, i, d):
    split = p08.split_for(len(spec["ctx"]), spec["min_split"])[0] if "graded" in spec["stress"] else None
    if split is not None and spec["splits"]:
        split = max(spec["splits"])  # v5.2: plant over the finest var split (4), so every split of the long request is tested
    c = p08.make_case(spec["name"], _seed(i, d), spec["ctx"], spec["rank"], stress=spec["stress"],
                      min_split=spec["min_split"], split=split)
    if spec["want"] is not None and p08.split_of(c)[0] != spec["want"]:
        raise SystemExit(f"{spec['name']}: planner split {p08.split_of(c)[0]} != {spec['want']} (R11 bar point)")
    return c


def _bits_equal(a, b):
    return bool(torch.equal(a[0].view(torch.int16), b[0].view(torch.int16))) and bool(
        torch.equal(a[1].view(torch.int32), b[1].view(torch.int32)))


def _specs(holdout, quick):
    allspecs = list(enumerate(CASES + HOLDOUT + CASES_R11))   # seeds follow the index in the full list
    return [(i, s) for i, s in allspecs if (holdout or s not in HOLDOUT) and (not quick or s["name"] in QUICK)]


METRICS = ("rel_peak", "err_flat", "lse_max_abs")
_INC: dict = {}   # (case, draw) -> (case, truth, incumbent output, envelope); reused across kernels
_CMB: dict = {}   # (combine case, draw) -> (cases, lens, truth out, live, vrms, envelope)
_CTL: list = []


def _ctls():  # the control set besides the incumbent itself: the other floor (served <-> ALT[B]), fp32 emulator
    if not _CTL:
        _CTL.extend([kernels.ctl_otherfloor(), kernels.emulator()])
    return _CTL


def _inc_draw(s, i, d):
    key = (s["name"], d)
    if key not in _INC:
        c = _case(s, i, d)
        ref = p08.reference(c)
        oi = p08.run(kernels.inc(), c)
        # v5.2 mixed cases: + emu at splits (reference) + the composed incumbent at splits (ruling b revised)
        ctls = _ctls() + ([kernels.emulator(splits=s["splits"]), kernels.composed(s["splits"])] if s["splits"] else [])
        errs = [p08.errors(*oi, ref, c)] + [p08.errors(*p08.run(f, c), ref, c) for f in ctls]
        _INC[key] = (c, ref, oi, {m: max(getattr(e, m) for e in errs) for m in METRICS})
        if s["splits"]:  # v5.2: which member sets the bound on the mixed cases (measures the widening)
            names = ["inc", "otherfloor", "emu", "emu_splits", "composed"]
            print(json.dumps(dict(envelope_member=s["name"], d=d, argmax={
                m: names[max(range(len(errs)), key=lambda k: getattr(errs[k], m))] for m in METRICS})), flush=True)
    return _INC[key]


def _vrms(cs):
    return math.sqrt(statistics.mean(((c.kv[..., :512].double() * c.k_scale) ** 2).mean().item() for c in cs))


def _cmb_err(outs, lens, to, live, vr):
    m = p08.combine([o for o, _l in outs], [l for _o, l in outs], lens).double()
    return ((m[live] - to[live]).norm() / (math.sqrt(to[live].numel()) * vr)).item(), bool(torch.isfinite(m).all())


def _cmb_draw(s, j, d):
    key = (s["name"], d)
    if key not in _CMB:
        cs, refs, outs, lens = [], [], [], []
        for r in range(8):
            c = p08.make_case(s["name"], _seed(50 + j, d) + r, s["ctx"], r, stress=(), q_seed=_seed(50 + j, d))
            cs.append(c)
            refs.append(p08.reference(c))
            outs.append(p08.run(kernels.inc(), c))
            lens.append(c.lengths)
        to, tl = p08.combine_truth(refs)
        live = torch.isfinite(tl)
        vr = _vrms(cs)
        env = max([_cmb_err(outs, lens, to, live, vr)[0]] +
                  [_cmb_err([p08.run(f, c) for c in cs], lens, to, live, vr)[0] for f in _ctls()])
        _CMB[key] = (cs, lens, to, live, vr, env)
    return _CMB[key]


def _ratio(a, b):
    return a / b if b > 0 else (1.0 if a == 0 else math.inf)


def _bound(env: dict, m: str) -> float:
    return max(1.02 * env[m], 2.0 ** -16) if m == "lse_max_abs" else 1.02 * env[m]


def _margins(ce: p08.Err, env: dict) -> dict:
    """R3: cand / bound per metric (> 1 fails); non-finite and non-merge-equivalent empty rows count as inf."""
    mg = {m: _ratio(getattr(ce, m), _bound(env, m)) for m in METRICS}
    mg["finite"] = 0.0 if ce.finite else math.inf
    mg["empty"] = 0.0 if ce.empty_ok else math.inf
    return mg


def _fails(mg: dict) -> list:
    return [f"{m} {r:.3f}x bound" for m, r in mg.items() if r > 1 + 1e-9]


def check(kname: str, holdout: bool = False, quick: bool = False, core_only: bool = False, stop: bool = False,
          fn=None, rec: list = None) -> bool:
    """core_only: per-rank salts + combine (mutants, controls); graph and B >= 3 run for candidates. stop: return at
    the first failing case (mutants). fn: a kernel callable in place of kernels.get(kname). rec: appended with
    (case, draw, metric, margin) for every judged metric (the power check's kill table)."""
    cand = fn or kernels.get(kname)
    rec = [] if rec is None else rec
    ok = True
    for i, s in _specs(holdout, quick):
        draws, beq = [], True
        for d in range(DRAWS):
            c, ref, oi, env = _inc_draw(s, i, d)
            oc = p08.run(cand, c)
            ce = p08.errors(*oc, ref, c)
            beq &= _bits_equal(oi, oc)
            mg = _margins(ce, env)
            rec += [(s["name"], d, m, r) for m, r in mg.items()]
            why = _fails(mg)
            draws.append(dict(d=d, why=why, margin={m: round(r, 4) for m, r in mg.items() if m in METRICS},
                              cand={m: getattr(ce, m) for m in METRICS}, env=env,
                              n_peak=ce.n_peak, n_flat=ce.n_flat, n_empty=ce.n_empty))
            if why and stop:
                break
        bad = [x for x in draws if x["why"]]
        ok &= not bad
        print(json.dumps(dict(kernel=kname, sgrid=SGRID, case=s["name"], split=p08.split_of(c)[0], bit_equal_inc=beq,
                              pass_=not bad, draws=draws)), flush=True)
        if s["splits"] and kname == "cand:var-split":  # report line, not the gate (ruling b revised; review #3)
            arm = kernels.cand_module("var-split")
            oc = p08.run(cand, c)
            om = p08.run(kernels.composed(s["splits"]), c)
            print(json.dumps(dict(kernel=kname, sgrid=SGRID, case=s["name"], arm_splits=list(arm.LAST_SPLITS),
                                  want_splits=list(s["splits"]), bit_equal_composed=_bits_equal(oc, om),
                                  max_abs_diff_out=float((oc[0].float() - om[0].float()).abs().nan_to_num(0).max()))),
                  flush=True)
        if bad and stop:
            print("FAIL", kname, flush=True)
            return False
    for j, s in enumerate(COMBINE if not quick else COMBINE[:1]):
        res = []
        for d in range(DRAWS):
            cs, lens, to, live, vr, env = _cmb_draw(s, j, d)
            ce, fin = _cmb_err([p08.run(cand, c) for c in cs], lens, to, live, vr)
            mg = _ratio(ce, 1.02 * env) if fin else math.inf
            rec.append((s["name"], d, "combine", mg))
            res.append(dict(d=d, cand=ce, env=env, margin=mg, finite=fin, pass_=mg <= 1 + 1e-9))
        good = all(r["pass_"] for r in res)
        ok &= good
        print(json.dumps(dict(kernel=kname, sgrid=SGRID, case=s["name"], pass_=good, draws=res)), flush=True)
        if not good and stop:
            print("FAIL", kname, flush=True)
            return False
    if not core_only:
        ok &= graph_and_dispatch(kname, cand)
    print("OK" if ok else "FAIL", kname, flush=True)
    return ok


def graph_and_dispatch(kname, cand) -> bool:
    ca, cb = (p08.make_case("graph", sd, [341_000], 0, stress=NS) for sd in (7, 8))
    out = torch.empty(1, 8, p08.H, 512, dtype=torch.bfloat16, device="cuda")
    lse = torch.empty(1, 8, p08.H, dtype=torch.float32, device="cuda")
    p08.run(cand, ca, out, lse)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        p08.run(cand, ca, out, lse)
    cb.q_scale, cb.k_scale = ca.q_scale, ca.k_scale  # served scales are per-layer constants baked into the graph
    eager = [t.clone() for t in p08.run(cand, cb)]
    for t, u in ((ca.q, cb.q), (ca.kv, cb.kv), (ca.block_table, cb.block_table)):
        t.copy_(u)
    g.replay()
    torch.cuda.synchronize()
    geq = _bits_equal((out, lse), eager)
    print(json.dumps(dict(kernel=kname, sgrid=SGRID, case="graph-swap-replay", bit_equal=geq, pass_=geq)), flush=True)
    ok = geq
    for B in (6, 8):  # B >= 6 dispatches to the unmodified incumbent: bit-equal (R11a)
        c = p08.make_case(f"b{B}", 9, [100_003] * B, 0, stress=())
        eq = _bits_equal(p08.run(kernels.inc(), c), p08.run(cand, c))
        print(json.dumps(dict(kernel=kname, sgrid=SGRID, case=f"b{B}-dispatch", split=p08.split_of(c)[0], bit_equal=eq, pass_=eq)),
              flush=True)
        ok &= eq
    return ok


def composed_checks():
    """v5.2 [u] check of the composed member, before any arm reading (coordinator ruling b revised; vector 4,2,2,2,2):
    every split in VAR_SPLITS is reachable at B5 itself with floor = split, so the member needs no padded call.
    A failure stops the judge with the line to send to the coordinator."""
    B = len(VAR_SPLITS)
    reach = {sp: kernels.split_reachable(B, sp, p08.QLEN, "cuda") for sp in sorted(set(VAR_SPLITS))}
    ok = all(reach.values())
    print(json.dumps(dict(composed_check=dict(B=B, split_reachable=reach), pass_=ok)), flush=True)
    if not ok:
        raise SystemExit("v5.2 [u] composed-member check FAILED: send the composed_check line to the coordinator")


def controls(quick=False) -> bool:
    """Positive controls (R1 + R3): each member of the control set (the incumbent at the other floor; the incumbent
    itself is the identity check) and the clean fp32 split-KV emulator (a correct kernel outside the set) must pass.
    A failure is reported to the coordinator, not loosened."""
    composed_checks()
    res = dict(ctl_otherfloor=check("ctl_otherfloor", quick=quick, core_only=True, fn=_ctls()[0]),
               emu_clean=check("emu", quick=quick, core_only=True, fn=_ctls()[1]))
    print(json.dumps(dict(controls=res, pass_=all(res.values()))), flush=True)
    return all(res.values())


README = os.path.join(HERE, "README.md")
KT0, KT1 = "<!-- kill-table:begin (judge.py power writes this block) -->", "<!-- kill-table:end -->"


ROBUST = 1.5  # R5: a kill counts only at margin >= 1.5x (margins near 1.00x move between runs)


def power(quick=False) -> list:
    """R3 + R5 power check: every mutant runs the per-rank + combine check (no early stop) and must have at least one
    (case, draw, metric) with margin >= ROBUST. Kill table per mutant: robust kills / judged and the largest margin.
    Returns the surviving mutants."""
    rows, surv = [], []
    for m in list(kernels.MUTANTS) + list(kernels.EMU_FLAGS):
        kernels._STALE.clear()
        rec: list = []
        check(m, quick=quick, core_only=True, rec=rec)
        kills = sorted((r for r in rec if r[3] >= ROBUST), key=lambda r: r[3])
        if not kills:
            surv.append(m)
        perB = {B: sum(1 for r in kills if re.search(rf"-b{B}(-|$)", r[0])) for B in PLAN}
        rows.append(dict(mutant=m, n_robust=len(kills), n_judged=len(rec), max=max(rec, key=lambda r: r[3]), perB=perB))
    fmt = lambda r: f"{r[0]} d{r[1]} {r[2]} {r[3]:.4g}x"
    tab = [f"| mutant | robust kills (margin >= {ROBUST}x) / judged | robust kills at B{' / B'.join(map(str, PLAN))} "
           f"| largest margin (case, draw, metric) |", "|---|---|---|---|"] + [
        f"| {r['mutant']} | {r['n_robust']} / {r['n_judged']} | {' / '.join(str(v) for v in r['perB'].values())} | "
        f"{fmt(r['max'])}{'' if r['n_robust'] else ' SURVIVES'} |" for r in rows]
    head = (f"Power check ({'quick' if quick else 'full'} salt list, {time.strftime('%Y-%m-%d %H:%M')}, "
            f"{torch.cuda.get_device_name(0)}; envelope R5 = incumbent at served floor, other floor, fp32 emulator): "
            f"survivors {surv or 'none'}.")
    print(json.dumps(dict(power=rows, survivors=surv)), flush=True)
    if not quick:
        txt = open(README).read() if os.path.exists(README) else f"{KT0}\n{KT1}\n"
        a, b = txt.index(KT0) + len(KT0), txt.index(KT1)
        open(README, "w").write(txt[:a] + "\n" + head + "\n\n" + "\n".join(tab) + "\n" + txt[b:])
    return surv


def selftest(quick=False) -> bool:
    """Teeth: controls and the identity candidate pass; every mutant fails (power check)."""
    res = dict(controls=controls(quick), identity=check("cand:candidate", quick=quick))
    res["mutants_surviving"] = power(quick)
    good = res["controls"] and res["identity"] and not res["mutants_surviving"]
    print(json.dumps(dict(selftest=res, pass_=good)))
    print("OK selftest" if good else "FAIL selftest", flush=True)
    return good


# ---------------------------------------------------------------- timing (protocol of record + R2)

# the nvidia-smi index of torch device 0 (run.py sets CUDA_DEVICE_ORDER=PCI_BUS_ID so the two orders agree)
GPU = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]


class Clock:
    """nvidia-smi SM clock + throttle reasons every 100 ms, concurrent with a block."""

    def __enter__(self):
        self.p = subprocess.Popen(["nvidia-smi", "-i", GPU, "--query-gpu=clocks.sm,clocks_throttle_reasons.active",
                                   "--format=csv,noheader,nounits", "-lms", "100"], stdout=subprocess.PIPE, text=True)
        return self

    def __exit__(self, *a):
        self.p.terminate()
        out = self.p.communicate()[0]
        self.rows = [l.split(",") for l in out.strip().splitlines() if "," in l]
        self.mhz = [float(r[0]) for r in self.rows]
        self.reasons = sorted({r[1].strip() for r in self.rows})
        self.throttled = sorted({r for r in self.reasons if int(r, 16) & ~0x1})  # 0x1 = GPU idle, not a throttle


def _ctxs(args) -> list:
    """Global ctx per request of a timing block: --case names a judge case (its ctx list; B must match), else
    [--ctx] x B. The kernel sees ctx / 8 per rank (DCP8, rank 0)."""
    if not args.case:
        return [args.ctx] * args.B
    spec = next((s for s in CASES + HOLDOUT + CASES_R11 if s["name"] == args.case), None)
    if spec is None or len(spec["ctx"]) != args.B:
        sys.exit(f"--case {args.case}: no such case with B = {args.B}")
    return list(spec["ctx"])


def _ring(ctxs: list, seed: int = 11):
    """Ring of distinct KV shards (same q, block table, lengths; own KV pool each): shards >= 24 and bytes >= 4 x L2.
    q (442 KB per request) is shared: in serving it is written by MlaQuery just before the call, so L2-hot too."""
    base = p08.make_case("ring", seed, ctxs, 0, stress=())
    shard_bytes = base.kv.numel()
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    n = max(24, math.ceil(4 * l2 / shard_bytes))
    n += n % 2
    gen = torch.Generator(device="cuda").manual_seed(seed)
    shards = []
    for k in range(n):
        kv = (torch.randn(base.kv.shape, device="cuda", generator=gen) * 0.5).to(p08.FP8)
        shards.append(p08.Case(f"ring{k}", base.q, kv, base.block_table, base.lengths, base.totals, base.q_scale,
                               base.k_scale, 0))
    return shards, shard_bytes, l2


def block(fa, fb, ctxs: list, replays: int, warmup: int, soak_s: float = 0.0):
    B = len(ctxs)
    shards, sb, l2 = _ring(ctxs)
    n = len(shards)
    idx = {"a": list(range(n)), "b": [(k + n // 2) % n for k in range(n)]}
    outs = {a: [(torch.empty(B, 8, p08.H, 512, dtype=torch.bfloat16, device="cuda"),
                 torch.empty(B, 8, p08.H, dtype=torch.float32, device="cuda")) for _ in shards] for a in "ab"}
    fns = {"a": fa, "b": fb}
    for k in range(n):  # compile + size the workspaces outside capture
        for a in "ab":
            p08.run(fns[a], shards[idx[a][k]], *outs[a][k])
    torch.cuda.synchronize()
    graphs, evs = [], []
    for first in "ab":
        ev = [{a: (torch.cuda.Event(enable_timing=True, external=True),
                   torch.cuda.Event(enable_timing=True, external=True)) for a in "ab"} for _ in range(n)]
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(g, stream=s):
            for k in range(n):
                order = ("ab" if (k % 2 == 0) == (first == "a") else "ba")
                for a in order:
                    ev[k][a][0].record()
                    p08.run(fns[a], shards[idx[a][k]], *outs[a][k])
                    ev[k][a][1].record()
        graphs.append(g)
        evs.append(ev)
    for r in range(warmup):
        graphs[r % 2].replay()
    torch.cuda.synchronize()
    t_end = time.time() + soak_s  # optional soak, discarded like the warm-up
    while time.time() < t_end:
        for r in range(50):
            graphs[r % 2].replay()
        torch.cuda.synchronize()
    ta, tb = [], []
    with Clock() as clk:
        for r in range(replays):
            graphs[r % 2].replay()
            torch.cuda.synchronize()
            ev = evs[r % 2]
            ta.append(statistics.mean(ev[k]["a"][0].elapsed_time(ev[k]["a"][1]) * 1e3 for k in range(n)))
            tb.append(statistics.mean(ev[k]["b"][0].elapsed_time(ev[k]["b"][1]) * 1e3 for k in range(n)))
    ring_eq = all(_bits_equal(outs["b"][k], p08.run(fb, shards[idx["b"][k]])) for k in range(n))
    rs = sorted(b / a for a, b in zip(ta, tb))
    q = lambda v, f: sorted(v)[min(len(v) - 1, int(f * len(v)))]
    return dict(form="ring", shards=n, shard_bytes=sb, ring_bytes=sb * n, l2_bytes=l2, shard_offset_b=n // 2,
                replays=replays, warmup=warmup, soak_s=soak_s, first_arm="alternating",
                incumbent_median_us=q(ta, .5), candidate_median_us=q(tb, .5), incumbent_p10_us=q(ta, .1),
                incumbent_p90_us=q(ta, .9), ratio_median=q(rs, .5), ratio_p10=q(rs, .1), ratio_p90=q(rs, .9),
                clock_sm_median_mhz=statistics.median(clk.mhz) if clk.mhz else None,
                clock_sm_min_mhz=min(clk.mhz, default=None), clock_samples=len(clk.mhz),
                throttle_reasons=clk.reasons, throttle_active=clk.throttled, ring_outputs_bit_equal_eager=ring_eq,
                a_us=ta, b_us=tb)


def _meta(args, kname, role):
    return dict(session=args.session, role=role,
                block_id=f"{time.strftime('%Y%m%dT%H%M%S')}-{role}-{kname}-B{args.B}-{args.case or args.ctx}",
                node=socket.gethostname(), gpu=torch.cuda.get_device_name(0), B=args.B, ctx=args.ctx, rank=0,
                case=args.case, ctxs=_ctxs(args), sgrid=SGRID,
                split=p08.split_for(args.B, None)[0], incumbent="inc (image dcp.stock_decode)", candidate=kname)


def _session_file(args):
    return os.path.join(RES, f"session-{args.session}.jsonl")


def _rows(args):
    p = _session_file(args)
    rows = [json.loads(l) for l in open(p)] if os.path.exists(p) else []
    rows = [r for r in rows if r.get("sgrid") == SGRID]  # an S_grid override reading never mixes with the pair
    if args.case:  # a named case is its own shape; --ctx is ignored for it
        return [r for r in rows if r.get("case") == args.case]
    return [r for r in rows if (r["B"], r["ctx"]) == (args.B, args.ctx) and not r.get("case")]


def reference(args):
    """R2 + R4 reference from this session's interleaved A/A blocks at (B, ctx); None while fewer than 5 exist."""
    rows = [r for r in _rows(args) if r["role"] in ("aa", "timed")]
    aa = [r for r in rows if r["role"] == "aa"]
    if len(aa) < 5:
        return None
    meds = [r["incumbent_median_us"] for r in aa]
    steady = set.intersection(*(set(r["throttle_active"]) for r in aa))
    med = statistics.median(meds)
    return dict(n_aa=len(aa), aa_incumbent_median_us=med, band_us=[min(meds), max(meds)],
                band_width=(max(meds) - min(meds)) / med,  # R7: absolute width only
                clock_mhz=statistics.median(r["clock_sm_median_mhz"] for r in aa),
                aa_band=max(max(abs(r["ratio_median"] - 1), (r["ratio_p90"] - r["ratio_p10"]) / 2) for r in aa),
                session_throttle_reasons=sorted(steady),
                spans_session=rows[0]["role"] == "aa" and rows[-1]["role"] == "aa",
                aa_blocks=[r["block_id"] for r in aa])


def _void(r, ref) -> list:
    why = []
    lo, hi = ref["band_us"]
    if not 0.99 * lo <= r["incumbent_median_us"] <= 1.01 * hi:
        why.append(f"incumbent median {r['incumbent_median_us']:.3f} > 1 % outside the A/A band [{lo:.3f}, {hi:.3f}] us")
    if r["clock_sm_median_mhz"] is None or abs(r["clock_sm_median_mhz"] / ref["clock_mhz"] - 1) > 0.02:
        why.append(f"median SM clock {r['clock_sm_median_mhz']} vs reference {ref['clock_mhz']} > 2 %")
    if sorted(r["throttle_active"]) != ref["session_throttle_reasons"]:
        why.append(f"throttle reasons {r['throttle_active']} differ from the session's {ref['session_throttle_reasons']}")
    if not r["ring_outputs_bit_equal_eager"]:
        why.append("ring outputs differ from eager (workspace state across calls)")
    return why


def aa(args):
    inc = kernels.inc()
    r = block(inc, inc, _ctxs(args), args.replays, args.warmup, args.soak)
    r.update(_meta(args, "inc", "aa"))
    r["aa_band_block"] = max(abs(r["ratio_median"] - 1), (r["ratio_p90"] - r["ratio_p10"]) / 2)
    r["void"] = dict(value=False, reason="A/A block (defines the session reference)")
    _emit(r, _session_file(args))


def timed(args):
    r = block(kernels.inc(), kernels.get(args.kernel), _ctxs(args), args.replays, args.warmup, args.soak)
    r.update(_meta(args, args.kernel, "timed"))
    r["void"] = dict(value=None, reason="judged at verdict against the session's interleaved A/A band (R4)")
    _emit(r, _session_file(args))


def verdict(args):
    if SGRID is not None:
        sys.exit("verdict refuses to run with P08_SGRID set: the pair of record runs with it unset (coordinator)")
    ref = reference(args)
    rows = [r for r in _rows(args) if r["role"] == "timed" and r["candidate"] == args.kernel]
    blocks = []
    for r in rows:
        why = _void(r, ref) if ref else ["no reference (< 5 A/A blocks)"]
        rec = r["candidate_median_us"] / ref["aa_incumbent_median_us"] if ref else None
        blocks.append(dict(block_id=r["block_id"], ratio_of_record=rec, ratio_in_block=r["ratio_median"],
                           candidate_median_us=r["candidate_median_us"], incumbent_median_us=r["incumbent_median_us"], clock_sm_median_mhz=r["clock_sm_median_mhz"],
                           throttle_active=r["throttle_active"], void=bool(why), reason="; ".join(why)))
    valid = [b for b in blocks if not b["void"]]
    v = dict(session=args.session, role="verdict", kernel=args.kernel, B=args.B, ctx=args.ctx, case=args.case, sgrid=SGRID, bar=args.bar,
             reference=ref, timed=len(rows), valid=len(valid), void=len(rows) - len(valid), blocks=blocks)
    if ref is None or not ref["spans_session"]:
        v.update(state="no reference" if ref is None else "reference does not span the session (R4)", claim=None)
    elif len(valid) < 5 or (len(rows) - len(valid)) > 0.5 * len(rows):
        v.update(state="unstable" if rows else "no blocks", claim=None)
    else:
        med = statistics.median(b["ratio_of_record"] for b in valid)
        v.update(state="stable", ratio_of_record=med, band_width=ref["band_width"],
                 ratio_plus_band=med + ref["band_width"],
                 ratio_in_block_median=statistics.median(b["ratio_in_block"] for b in valid),
                 session_throttle_reasons=ref["session_throttle_reasons"],
                 bar_met=(med + ref["band_width"]) <= args.bar if args.bar else None)
    _emit(v, _session_file(args))
    return v


def _emit(r, path):
    print(json.dumps({k: v for k, v in r.items() if k not in ("a_us", "b_us")}), flush=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["check", "controls", "power", "selftest", "aa", "time", "verdict"])
    ap.add_argument("--kernel", default="cand:candidate")
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--ctx", type=int, default=341_000)
    ap.add_argument("--case", default=None, help="time a named judge case (its ctx list), e.g. c345k-mixed-b5")
    ap.add_argument("--replays", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--soak", type=float, default=0.0, help="seconds of discarded replays after the warm-up")
    ap.add_argument("--session", default=None, help="seat session id (R2 reference scope)")
    ap.add_argument("--bar", type=float, default=None)
    a = ap.parse_args()
    if a.mode in ("aa", "time", "verdict") and not a.session:
        sys.exit("--session is required for timing (R2)")
    if a.mode == "check":
        if not controls(quick=True):
            sys.exit("positive controls fail: no candidate is judged (R1)")
        sys.exit(0 if check(a.kernel, a.holdout, a.quick) else 1)
    if a.mode == "controls":
        sys.exit(0 if controls(a.quick) else 1)
    if a.mode == "power":
        sys.exit(1 if power(a.quick) else 0)
    if a.mode == "selftest":
        sys.exit(0 if selftest(a.quick) else 1)
    {"aa": aa, "time": timed, "verdict": verdict}[a.mode](a)


if __name__ == "__main__":
    main()
