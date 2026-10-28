"""PERFBOT-PRETRACE proof filler (coordinator ruling 2026-10-20): the SpecializedDecode build, trace path vs hit path.

    python3 -B -m perfbot_kernels.pretrace.proof --out-dir <results dir> [--deadline 560] [--workspace /workspace]

Pass 1: one `vllm serve` with the served argv (probes/recipe.py + the arm overlay's server_extra_args, as serve.sh renders it)
plus `--load-format dummy`, in its own session, with PERFBOT_EXP_CUTE_PRETRACE=export and
CUTE_DSL_JIT_TIME_PROFILING=1: every cute.compile traces, exports its module and writes a record. The server is killed
once every rank has finished SpecializedDecode (records/rank-<r>.done.json). Compile keys hold layouts and config
values only, so dummy weights give the served modules [e: pretrace-map 2026-10-20].
Pass 2: a fresh interpreter on GPU 0 loads every exported module (cutlass.runtime.load_module, [prefix], .to()) and
takes the cubins it carries.
Verdict (analyze()): PASS = every rank done, no key with two cubin sets (across calls and ranks), every exported
module loaded and bound, every loaded cubin set equal to the traced one. One line, pretrace.json in --out-dir:
  PERFBOT-PRETRACE verdict=PASS|FAIL|INCOMPLETE ranks=<done>/<world> calls=... (split sums in seconds, rank 0)
rc: 0 PASS, 1 FAIL, 78 INCOMPLETE.
"""

import argparse
import collections
import glob
import json
import os
import signal
import statistics
import subprocess
import sys
import time

from perfbot_kernels import pretrace as pt

PASS2_RESERVE_S = 60


def extra_args(path):
    """overlay/server_extra_args as serve.sh reads it: `#` starts a comment, each line is word-split."""
    words = []
    with open(path) as fh:
        for line in fh:
            words += line.split("#", 1)[0].split()
    return words


def server_argv(workspace):
    conc = json.load(open(os.path.join(workspace, "probes/contract.json")))["point"]["conc"]
    extra = extra_args(os.path.join(pt.OVERLAY, "server_extra_args"))  # the arm's overlay (a bundle may carry one)
    out = subprocess.run([sys.executable, "-B", "probes/recipe.py", "argv", "--mode", "throughput", "--conc",
                          str(conc), "--"] + extra + ["--load-format", "dummy"], cwd=workspace, check=True,
                         stdout=subprocess.PIPE, universal_newlines=True).stdout
    return [x for x in out.splitlines() if x]


def world(argv):
    for flag in ("--tensor-parallel-size", "-tp"):
        if flag in argv:
            return int(argv[argv.index(flag) + 1])
    return 1


def pass1(argv, env, out_dir, deadline):
    records = os.path.join(out_dir, "records")
    n, t0 = world(argv), time.time()
    with open(os.path.join(out_dir, "pass1-server.log"), "ab") as log:
        p = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        done = []
        while time.time() < deadline:
            done = glob.glob(os.path.join(records, "rank-*.done.json"))
            if len(done) >= n or p.poll() is not None:
                break
            time.sleep(2)
        exit_before_kill = p.poll()
        for sig, wait in ((signal.SIGTERM, 20), (signal.SIGKILL, 20)):
            try:
                os.killpg(p.pid, sig)
            except ProcessLookupError:
                break
            try:
                p.wait(wait)
                break
            except subprocess.TimeoutExpired:
                continue
    return {"world": n, "ranks_done": len(done), "server_exit_before_kill": exit_before_kill,
            "seconds": round(time.time() - t0, 1)}


def load_pass(modules, out_path):
    """Pass 2 body (fresh interpreter): load, bind and read the cubins of every exported module."""
    import cutlass.runtime as rt

    with open(out_path, "w") as out:
        for path in sorted(glob.glob(os.path.join(modules, "*.o"))):
            key = os.path.basename(path)[:-2]
            rec = {"key": key}
            try:
                t0 = time.time()
                fn = rt.load_module(path)[pt.prefix(key)]
                t1 = time.time()
                fn.to()
                t2 = time.time()
                rec.update(load_s=round(t1 - t0, 4), to_s=round(t2 - t1, 4),
                           kernel_info=sorted((getattr(fn, "kernel_info", None) or {}).keys()))
            except Exception as e:  # noqa: BLE001 - a module that does not load is the finding
                rec["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
            with open(path, "rb") as fh:
                rec["cubins"] = pt.cubins(fh.read())
            out.write(json.dumps(rec) + "\n")


def read_jsonl(paths):
    out = []
    for p in paths:
        with open(p) as fh:
            out += [json.loads(x) for x in fh if x.strip()]
    return out


def analyze(records, loaded, p1, pass2_ok=True):
    """The verdict and its counts from pass-1 records (all ranks), pass-2 records and the pass-1 summary."""
    traced = [r for r in records if r.get("hit") is False]
    keyed = [r for r in traced if r.get("key")]
    by_key = collections.defaultdict(set)
    for r in keyed:
        if "cubins" in r:
            by_key[r["key"]].add(tuple(r["cubins"]))
    collisions = sorted(k for k, v in by_key.items() if len(v) > 1)
    rank0 = [r for r in traced if str(r.get("rank")) in ("0", "-")] or traced
    seen, dup = set(), 0
    for r in rank0:
        dup += r.get("module") in seen
        seen.add(r.get("module"))
    load_err = [x for x in loaded if "error" in x]
    equal = [x for x in loaded if "error" not in x and x["key"] in by_key and by_key[x["key"]] == {tuple(x["cubins"])}]
    exported = {r["key"] for r in keyed if "export_s" in r}
    missing = sorted(exported - {x["key"] for x in loaded})

    def total(rows, f):
        return round(sum(r.get(f, 0.0) or 0.0 for r in rows), 2)

    split = {"call_s": total(rank0, "call_s"), "passes_s": total(rank0, "passes_s"), "hash_s": total(rank0, "hash_s"),
             "key_s": total(rank0, "key_s"), "export_s": total(rank0, "export_s"),
             "load_s": total(loaded, "load_s"), "to_s": total(loaded, "to_s")}
    split["trace_s"] = round(split["call_s"] - split["passes_s"] - split["hash_s"], 2)
    unkeyed = collections.Counter(r.get("unkeyed", "").split(" ")[0] for r in traced if not r.get("key"))
    counts = {"ranks": "%d/%d" % (p1["ranks_done"], p1["world"]), "calls": len(traced), "keyed": len(keyed),
              "unkeyed": len(traced) - len(keyed), "keys": len(by_key), "rank0_calls": len(rank0),
              "rank0_modules": len(seen), "rank0_dup_calls": dup, "collisions": len(collisions),
              "exported": len(exported), "export_errors": sum("export_error" in r for r in keyed),
              "module_errors": sum("module_error" in r for r in keyed), "loaded": len(loaded) - len(load_err),
              "load_errors": len(load_err), "missing": len(missing), "cubin_equal": "%d/%d" % (len(equal), len(loaded))}
    if p1["ranks_done"] < p1["world"] or not pass2_ok or not traced:
        verdict = "INCOMPLETE"
    elif collisions or load_err or missing or len(equal) != len(loaded) or counts["export_errors"] \
            or counts["module_errors"]:
        verdict = "FAIL"
    else:
        verdict = "PASS"
    loads = [x["load_s"] + x["to_s"] for x in loaded if "error" not in x]
    return {"verdict": verdict, "counts": counts, "split_s": split, "pass1": p1,
            "median_load_to_s": round(statistics.median(loads), 4) if loads else None,
            "unkeyed_reasons": unkeyed.most_common(12), "collision_keys": collisions[:20], "missing": missing[:20],
            "load_errors": load_err[:5]}


def line(summary):
    c, s = summary["counts"], summary["split_s"]
    return " ".join(["%s verdict=%s" % (pt.TAG, summary["verdict"])] + ["%s=%s" % kv for kv in c.items()]
                    + ["%s=%s" % kv for kv in s.items()] + ["pass1_s=%s" % summary["pass1"]["seconds"]])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir")
    ap.add_argument("--deadline", type=float, default=560)
    ap.add_argument("--workspace", default="/workspace")
    ap.add_argument("--load-pass", nargs=2, metavar=("MODULES", "OUT"))
    a = ap.parse_args(argv)
    if a.load_pass:
        load_pass(*a.load_pass)
        return 0
    t0 = time.time()
    out = os.path.abspath(a.out_dir)
    modules = os.path.join(os.environ.get("TMPDIR", "/tmp"), "perfbot-pretrace-modules")
    os.makedirs(modules, exist_ok=True)
    env = dict(os.environ, **{pt.ENV: "export", pt.DIR_ENV: modules, pt.RECORD_ENV: os.path.join(out, "records"),
                              "CUTE_DSL_JIT_TIME_PROFILING": "1"})
    argv_ = server_argv(a.workspace)
    with open(os.path.join(out, "pass1-argv.json"), "w") as fh:
        json.dump(argv_, fh)
    p1 = pass1(argv_, env, out, t0 + a.deadline - PASS2_RESERVE_S)
    print("%s pass1 %s" % (pt.TAG, json.dumps(p1)), flush=True)
    p2_path = os.path.join(out, "pass2.jsonl")
    left = max(10, int(t0 + a.deadline - time.time()))
    try:
        p2_rc = subprocess.run([sys.executable, "-B", "-m", "perfbot_kernels.pretrace.proof", "--load-pass", modules,
                                p2_path], env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"), timeout=left).returncode
    except subprocess.TimeoutExpired:
        p2_rc = 124
    records = read_jsonl(sorted(glob.glob(os.path.join(out, "records", "rank-*.jsonl"))))
    loaded = read_jsonl([p2_path]) if os.path.exists(p2_path) else []
    summary = analyze(records, loaded, p1, pass2_ok=p2_rc == 0)
    summary["pass2_rc"] = p2_rc
    summary["pass2_s"] = round(time.time() - t0 - p1["seconds"], 1)
    with open(os.path.join(out, "pretrace.json"), "w") as fh:
        json.dump(summary, fh, indent=1, sort_keys=True)
    print(line(summary), flush=True)
    return {"PASS": 0, "FAIL": 1}.get(summary["verdict"], 78)


if __name__ == "__main__":
    sys.exit(main())
