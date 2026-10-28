"""Kernel-only bar arm of item 03 on the served image (B300 PERFBOT_PRE_EXEC; B200 dry run). Copies of the P-08 judge
(judge.py, p08.py, kernels.py; scratch/p08/judge) run the R4 timing session of record: per (B, ctx) A/A, timed, A/A, ...,
timed, A/A (NT timed (default 6) + NT+1 A/A blocks, each its own process), then the verdict. Candidate = this bundle's
perfbot_kernels.mla_core (kernel "pkg:mla_core"); incumbent = the image's dcp.stock_decode.

Order: numerics first (judge.py check --quick: controls + fp64 and served-truth envelope 1.02x). A failed check stops
the timing. Sessions: the bar points B1..B5 at 341k (planner split 12/6/4/3/2), then B3 and B5 at 256k (diagnostic, no
bar). Sessions are spread over --gpus devices, one session per device at a time.

Bars (coordinator ruling): B1, B2 <= 0.85x; B3, B5 <= 0.88x; B4 <= 1.00x; judged on ratio of record + A/A band width.
Exit: 0 every bar met (all five sessions stable); 1 a stable session missed its bar (no step crashed); 2 crash, numerics
fail, deadline, or a bar session without a stable verdict and no stable miss (no claim). A stable miss decides alone: "all
bars met" is false whatever the other sessions show. Unstable sessions print their in-block ratio median (diagnostic).

Run (cwd = the bundle root, so perfbot_kernels imports):
  python3 -B -m perfbot_kernels.mla_core.bar.run --gpus 8 --deadline 560
env: PERFBOT_MLA_BAR_RES (results dir; default $PERFBOT_RESULTS/mla_bar under /workspace, else ./mla_bar_results)."""
import argparse
import json
import os
import queue
import statistics
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))  # the bundle root that holds perfbot_kernels/
KERNEL = "pkg:mla_core"
BARS = {1: 0.85, 2: 0.85, 3: 0.88, 4: 1.00, 5: 0.88}
CTX, DIAG_CTX, DIAG_BS = 341_000, 256_000, (3, 5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--nt", type=int, default=6, help="timed blocks per session (verdict: >= 5 valid, voids <= 50 %)")
    ap.add_argument("--replays", type=int, default=1000)
    ap.add_argument("--deadline", type=float, default=560.0, help="seconds for the whole arm")
    ap.add_argument("--no-diag", action="store_true", help="skip the 256k diagnostic sessions")
    ap.add_argument("--session", default=time.strftime("bar-%Y%m%dT%H%M%S", time.gmtime()))
    a = ap.parse_args()
    if a.nt < 5:
        sys.exit("the verdict needs >= 5 valid timed blocks: --nt >= 5")
    res = os.environ.get("PERFBOT_MLA_BAR_RES") or (
        os.path.join("/workspace", os.environ["PERFBOT_RESULTS"], "mla_bar") if os.environ.get("PERFBOT_RESULTS")
        else os.path.abspath("mla_bar_results"))
    os.makedirs(res, exist_ok=True)
    env = dict(os.environ, PERFBOT_MLA_BAR_RES=res, CUDA_DEVICE_ORDER="PCI_BUS_ID",
               PYTHONPATH=os.pathsep.join(filter(None, [ROOT, os.environ.get("PYTHONPATH")])))
    env.pop("P08_SGRID", None)  # the verdict refuses an S_grid override
    t_end = time.time() + a.deadline
    log = open(os.path.join(res, f"{a.session}.log"), "a")

    def run(gpu, argv):
        left = t_end - time.time()
        if left <= 0:
            return 124
        t0 = time.time()
        try:
            rc = subprocess.run([sys.executable, "-B", "judge.py", *argv], cwd=HERE, stdout=log, stderr=subprocess.STDOUT,
                                env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)), timeout=left).returncode
        except subprocess.TimeoutExpired:
            rc = 124
        print(f"[bar] gpu{gpu} rc={rc} {time.time() - t0:6.1f}s judge.py {' '.join(argv)}", flush=True)
        return rc

    if run(0, ["check", "--kernel", KERNEL, "--quick"]) != 0:
        print("[bar] EXIT 2: numerics check failed or crashed; no timing", flush=True)
        return 2

    jobs = queue.Queue()
    for B in BARS:
        jobs.put((B, CTX))
    if not a.no_diag:
        for B in DIAG_BS:
            jobs.put((B, DIAG_CTX))
    rcs = {}
    sid = lambda B, ctx: f"{a.session}-B{B}-{ctx // 1000}k"

    def worker(gpu):
        while True:
            try:
                B, ctx = jobs.get_nowait()
            except queue.Empty:
                return
            sb = ["--session", sid(B, ctx), "--B", str(B), "--ctx", str(ctx)]  # one file per session: no shared appends
            steps = [["aa", *sb]] + [s for _ in range(a.nt) for s in
                                     (["time", "--kernel", KERNEL, "--replays", str(a.replays), *sb], ["aa", *sb])]
            rc = next((r for r in (run(gpu, s) for s in steps) if r != 0), 0)
            if rc == 0:
                bar = ["--bar", str(BARS[B])] if ctx == CTX else []
                rc = run(gpu, ["verdict", "--kernel", KERNEL, *sb, *bar])
            rcs[(B, ctx)] = rc

    th = [threading.Thread(target=worker, args=(g,)) for g in range(a.gpus)]
    for t in th:
        t.start()
    for t in th:
        t.join()

    verdicts = {}
    for B, ctx in rcs:
        p = os.path.join(res, f"session-{sid(B, ctx)}.jsonl")
        rows = [json.loads(l) for l in open(p)] if os.path.exists(p) else []
        verdicts.update({(B, ctx): r for r in rows if r["role"] == "verdict"})
    crash = missed = noclaim = False
    for (B, ctx), rc in sorted(rcs.items(), key=lambda kv: (kv[0][1] != CTX, kv[0][0])):
        v = verdicts.get((B, ctx), {})
        tag = f"bar {BARS[B]:.2f}" if ctx == CTX else "diagnostic"
        print(f"[bar] B{B} {ctx // 1000}k rc={rc} state={v.get('state')} ratio={v.get('ratio_of_record')} "
              f"+band={v.get('ratio_plus_band')} valid={v.get('valid')}/{v.get('timed')} {tag} met={v.get('bar_met')} "
              f"in-block={statistics.median([b['ratio_in_block'] for b in v['blocks']]) if v.get('blocks') else None}",
              flush=True)
        if ctx != CTX:
            continue
        crash |= rc != 0
        missed |= v.get("state") == "stable" and v.get("bar_met") is False
        noclaim |= v.get("state") != "stable"
    code = 2 if crash else 1 if missed else 2 if noclaim else 0
    print(f"[bar] EXIT {code} session {a.session} results {res}", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
