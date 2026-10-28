# SPDX-License-Identifier: Apache-2.0
"""Item 15 integration selftest arm (kernel-only pre-exec, one 8-GPU B300 node, no B200 needed).

    python3 -B -m perfbot_kernels.draft_chain.selftest.run      (PYTHONPATH = overlay)

Steps, one JSON summary line each ("probe": "i15-selftest"):
  cpu/overlay   every P-15 overlay file minus its tagged lines has its 49cb2bbf sha256 (insert-only). The values are
                carried here (BASE): the staged overlay keeps the pinned vllm_base.json, since compose_bundle skips it;
                BASE equals the P-15 entries of this commit's vllm_base.json (checked by test_cpu.py).
  cpu/off       switch unset, fresh interpreter: the overlay mla / dspark_mla are the modules loaded; dspark_mla's AR
                global is the served function object; prepare and attention return None before any torch op.
  cpu/q_judge   lse_combine.selftest.test_q_g2 (identity passes, +3-code mutant fails).
  gate/k1       draft_ar.selftest.gpu --gate                 (GPU 0; one-GPU stand-in, mutants)
  gate/k2       draft_query.selftest.gpu --gate --lamport    (GPU 1; G1 / G2 / G3, mutants)
  gate/k3       lse_combine.selftest.gpu --gate              (GPU 2; latent bits, out envelope, mutants)
  chain         draft_chain.selftest.b300_chain              (8 GPUs, after the gates: prepare, routing, hook, parity)
A gate passes when its SUMMARY line says passed and every mutant was caught; no SUMMARY line = crash.
Exit: 0 every step passed; 1 a check failed; 2 a crash or a timeout. The last line is the summary, "rc" first.
Budget: gates in parallel <= 420 s, chain <= 600 s (its own budget), so <= ~1050 s with start-up: arm timeout 1200.
"""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

OVERLAY = Path(__file__).resolve().parents[3]
TAG = "# PERFBOT P-15"
BASE = {
    "vllm/models/kimi_k3/nvidia/mla.py": "f306e24238dca1fe55a635cf735dcbcacbfac98cef3a5d9c495e7a69a84b9467",
    "vllm/models/kimi_k3/nvidia/dspark_mla.py": "adfe621d7fbb89616e207118000c9f61201b2ff4af9b8d43387d71eec0344121",
}
GATES = {"gate/k1": ("perfbot_kernels.draft_ar.selftest.gpu", ["--gate"], "0"),
         "gate/k2": ("perfbot_kernels.draft_query.selftest.gpu", ["--gate", "--lamport"], "1"),
         "gate/k3": ("perfbot_kernels.lse_combine.selftest.gpu", ["--gate"], "2")}
GATE_TIMEOUT_S, CHAIN_TIMEOUT_S = 420, 660
RC_MEANING = {0: "ok", 1: "a check failed", 2: "crash / timeout"}
OFF_SNIPPET = """
import os, types
os.environ.pop("PERFBOT_EXP_DRAFT_CHAIN", None)
from perfbot_kernels import draft_chain
f = lambda *a: a
assert draft_chain.SWITCH is False
assert draft_chain.routed_ar(f) is f
assert draft_chain.prepare(None) is None and draft_chain._state is None
assert draft_chain.attention(types.SimpleNamespace(), None, None) is None
from vllm.models.common.ops.fused_allreduce_rms_norm import fused_allreduce_rms_norm as served
import vllm.models.kimi_k3.nvidia.dspark_mla as dspark
import vllm.models.kimi_k3.nvidia.mla as mla
assert dspark.draft_chain is draft_chain and mla.draft_chain is draft_chain  # the overlay copies are the ones loaded
assert dspark.fused_allreduce_rms_norm is served  # off: the module global is the served function object
print("off ok")
"""


def emit(row):
    print(json.dumps({"probe": "i15-selftest", **row}), flush=True)


def env(gpus=None):
    e = dict(os.environ)
    e.pop("PERFBOT_EXP_DRAFT_CHAIN", None)
    if gpus is not None:
        e["CUDA_VISIBLE_DEVICES"] = gpus
    return e


def overlay_check(root=OVERLAY):
    rows = {}
    for rel, sha in BASE.items():
        lines = (root / rel).read_bytes().splitlines(keepends=True)
        kept = b"".join(l for l in lines if TAG.encode() not in l)
        rows[rel] = hashlib.sha256(kept).hexdigest() == sha and len(kept) < sum(map(len, lines))
    return 0 if rows and all(rows.values()) else 1, {"files": rows}


def run(cmd, timeout, gpus=None, log=None):
    """(rc, full stdout, stderr tail, seconds); rc None on timeout. Output goes to files (no pipe can fill)."""
    t = time.time()
    log = Path(log or f"/tmp/i15-selftest/{abs(hash(tuple(cmd)))}")
    log.mkdir(parents=True, exist_ok=True)
    with open(log / "stdout", "w") as o, open(log / "stderr", "w") as e:
        p = subprocess.Popen([sys.executable, "-B", *cmd], env=env(gpus), stdout=o, stderr=e, text=True)
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
            rc = None
    return rc, (log / "stdout").read_text(), (log / "stderr").read_text()[-1500:], time.time() - t


def gate_verdict(proc_rc, text):
    """0 pass / 1 fail / 2 crash, from the gate's SUMMARY line (the gates exit 1 on a fail and on a crash alike)."""
    lines = [l for l in text.splitlines() if l.startswith("SUMMARY ")]
    if proc_rc not in (0, 1) or not lines:
        return 2, None
    s = json.loads(lines[-1][len("SUMMARY "):])
    ok = s.get("passed") is True and s.get("mutants_caught") is True
    return (0 if ok and proc_rc == 0 else 1), s


def main():
    t0 = time.time()
    results = {}

    rc, info = overlay_check()
    results["cpu/overlay"] = rc
    emit({"step": "cpu/overlay", "rc": rc, **info})

    prc, text, err, sec = run(["-c", OFF_SNIPPET], 120, log="/tmp/i15-selftest/off")
    results["cpu/off"] = (0 if prc == 0 and "off ok" in text
                          else 1 if prc == 1 and "AssertionError" in err else 2)
    emit({"step": "cpu/off", "rc": results["cpu/off"], "seconds": round(sec, 1), "tail": (text + err)[-300:]})

    prc, text, err, sec = run(["-m", "perfbot_kernels.lse_combine.selftest.test_q_g2"], 120, gpus="",
                              log="/tmp/i15-selftest/q_judge")
    results["cpu/q_judge"] = {0: 0, 3: 1}.get(prc, 2)
    emit({"step": "cpu/q_judge", "rc": results["cpu/q_judge"], "seconds": round(sec, 1), "tail": (text + err)[-300:]})

    procs = {}
    for name, (mod, args, gpu) in GATES.items():
        d = Path(f"/tmp/i15-selftest/{name}")
        d.mkdir(parents=True, exist_ok=True)
        f = open(d / "log", "w")
        procs[name] = (subprocess.Popen([sys.executable, "-B", "-m", mod, *args, "--out", str(d)], env=env(gpu),
                                        stdout=f, stderr=subprocess.STDOUT, text=True), time.time(), f, d)
    for name, (p, t, f, d) in procs.items():
        try:
            prc = p.wait(timeout=max(1, GATE_TIMEOUT_S - (time.time() - t)))
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
            prc = None
        f.close()
        out = (d / "log").read_text()
        rc, s = gate_verdict(prc, out)
        results[name] = rc
        fails = [l for l in out.splitlines() if l.startswith("FAIL")][:3]
        emit({"step": name, "rc": rc, "proc_rc": prc, "summary": s, "seconds": round(time.time() - t, 1),
              "fails": fails, "tail": "" if rc == 0 else out[-600:]})

    prc, text, err, sec = run(["-m", "perfbot_kernels.draft_chain.selftest.b300_chain"], CHAIN_TIMEOUT_S,
                              log="/tmp/i15-selftest/chain")
    lines = [json.loads(l) for l in text.splitlines() if l.startswith('{"probe": "i15-chain"')]
    for l in lines:
        if not l.get("summary"):
            emit({"step": "chain/" + l["check"], "rc": 0 if l["ok"] else 1,
                  **{k: v for k, v in l.items() if k not in ("probe", "check", "ok")}})
    summ = next((l for l in lines if l.get("summary")), None)
    results["chain"] = summ["rc"] if summ and prc == summ["rc"] else 2
    emit({"step": "chain", "rc": results["chain"], "proc_rc": prc, "summary": summ, "seconds": round(sec, 1),
          "tail": "" if results["chain"] == 0 else (text[-400:] + err[-800:])})

    worst = max(results.values())
    rc = 2 if worst == 2 else 1 if worst == 1 else 0
    emit({"summary": True, "rc": rc, "rc_meaning": RC_MEANING[rc], "steps": results,
          "seconds": round(time.time() - t0, 1)})
    return rc


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # an uncaught error must not exit 1 (1 = a check failed)
        emit({"summary": True, "rc": 2, "rc_meaning": RC_MEANING[2], "error": f"{type(e).__name__}: {e}"})
        sys.exit(2)
