#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Copy one measurement request's arms into results/<checkpoint>/ and write VERDICT.md.

An arm is a results/ subdirectory holding anything besides .arm-cache.* (the driver's compile cache, never a result).
Copies, per arm: metrics.json, identity.json, server_command.txt, phases.json, server-ready-seconds.txt,
point/<aggregate>.json, point/benchmark_command.txt, point/client-timing.json, point/client-runtime.txt,
point/aiperf_artifacts/{profile_export_aiperf.json, profile_export_aiperf.csv, profile_export.jsonl}, every lane's
point/lanes/<lane>/aiperf_artifacts/profile_export.jsonl,
eval/results_*.json, eval/eval-timing.json, and server.log with host names and internal paths replaced by <host> / <path>, gzip-compressed (server.log.gz: the hosting limit is 5 MiB per file).
Request-level: workload-revision.json, exit.json, and bundles.json (when present) and the arms json as arms.json with internal paths replaced.
Then it runs scripts/check_secrets.py on the destination and writes VERDICT.md from scripts/extract_metrics.py.
--guard SET,CTL_A,CTL_B adds the no-regression guards: TTFT p90 and output tok/s per GPU, SET - mean(CTL_A, CTL_B) with
the control spread |CTL_A - CTL_B| as the bound. A breach writes a GUARD FAIL line and suffixes the verdict enum -GUARD.
Liveness: every comma item of every on PERFBOT_EXP_* switch in an arm's identity.json env_perfbot (the arm env, bundle
arm_env.json included) needs a `PERFBOT-SWITCH <name>=<item> <file>:<line>` line in its server.log (the workload
overlay's sitecustomize.py marks each item a module reads); each item without one is a LIVENESS MISSING line.

    python3 scripts/ingest_results.py <request out dir> <checkpoint name> --arms <arms json> [--verdict "<one line>"] [--guard SET,CTL_A,CTL_B]
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GUARDS = (("ttft_p90_ms", "TTFT p90 ms", +1), ("output_tput_per_gpu", "output tok/s/GPU", -1))  # (key, label, sign of worse)
ARM_FILES = ["metrics.json", "identity.json", "server_command.txt", "phases.json", "server-ready-seconds.txt"]
POINT_FILES = ["benchmark_command.txt", "client-timing.json", "client-runtime.txt", "workload_distribution_summary.txt"]
AIPERF_FILES = ["profile_export_aiperf.json", "profile_export_aiperf.csv", "profile_export.jsonl"]
SANITIZE = [
    (re.compile(r"\b(umb-b300-\S*|umbriel-\S*|dp-\d{3}\b|computelab\S*|frontend-\d+|(?<!\S)\S+\.nvidia\.com)", re.I), "<host>"),
    (re.compile(r"""(/home/scratch\.[a-z_]+|/home/alanw|/tmp/k3-split-sync)[^\s"']*"""), "<path>"),  # stop at a quote: JSON stays valid
    (re.compile(r"(?<!\S)\S+\.sock\b"), "<socket>"),
]


def copy(src: Path, dst: Path):
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def sanitize(src: Path, dst: Path):
    if not src.is_file():
        return
    text = src.read_text(errors="replace")
    for pat, rep in SANITIZE:
        text = pat.sub(rep, text)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.suffix == ".gz":
        with gzip.open(dst, "wt") as f:
            f.write(text)
    else:
        dst.write_text(text)


def liveness_missing(arm: Path) -> list[str]:
    """NAME=item for each on switch item the arm env declares and server.log never marks (PERFBOT-SWITCH)."""
    env = (json.loads((arm / "identity.json").read_text()) if (arm / "identity.json").is_file() else {}).get("env_perfbot") or {}
    declared = {f"{k}={x.strip()}" for k, v in env.items() if k.startswith("PERFBOT_EXP_") and v not in ("", "0") for x in v.split(",") if x.strip()}
    log = (arm / "server.log").read_text(errors="replace") if (arm / "server.log").is_file() else ""
    return sorted(declared - set(re.findall(r"PERFBOT-SWITCH (\S+=\S+) ", log)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("request_dir", type=Path)
    ap.add_argument("checkpoint")
    ap.add_argument("--arms", type=Path, required=True)
    ap.add_argument("--verdict", default="VERDICT: <fill in: landed | not landed>, <reason>")
    ap.add_argument("--guard", help="SET,CTL_A,CTL_B arm names for the no-regression guards")
    a = ap.parse_args()
    dst = ROOT / "results" / a.checkpoint
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("workload-revision.json", "exit.json"):
        copy(a.request_dir / name, dst / name)
    sanitize(a.request_dir / "bundles.json", dst / "bundles.json")  # bundle source paths are internal
    sanitize(a.arms, dst / "arms.json")
    rows, live = [], []
    for arm in sorted(p for p in (a.request_dir / "results").iterdir()
                      if p.is_dir() and any(not c.name.startswith(".arm-cache.") for c in p.iterdir())):
        d = dst / arm.name
        for f in ARM_FILES:
            copy(arm / f, d / f)
        for f in POINT_FILES:
            copy(arm / "point" / f, d / "point" / f)
        for agg in arm.glob("point/*_conc*.json"):
            copy(agg, d / "point" / agg.name)
        for f in AIPERF_FILES:
            copy(arm / "point" / "aiperf_artifacts" / f, d / "point" / "aiperf_artifacts" / f)
        for f in arm.glob("point/lanes/*/aiperf_artifacts/profile_export.jsonl"):
            copy(f, d / f.relative_to(arm))
        for f in arm.glob("eval/results_*.json"):
            copy(f, d / "eval" / f.name)
        copy(arm / "eval" / "eval-timing.json", d / "eval" / "eval-timing.json")
        sanitize(arm / "server.log", d / "server.log.gz")
        if missing := liveness_missing(arm):
            live.append(f"- LIVENESS MISSING {arm.name}: {' '.join(missing)}")
        if (d / "metrics.json").is_file() or (d / "point").is_dir():
            m = json.loads(subprocess.run([sys.executable, ROOT / "scripts" / "extract_metrics.py", d, "--json"], check=True, capture_output=True, text=True).stdout)
            rows.append(m)
    verdict, guard = a.verdict, []
    if a.guard:
        by = {m["arm"]: m for m in rows}
        s, ca, cb = a.guard.split(",")
        for key, label, worse in GUARDS:
            v = {n: by[n][key] for n in (s, ca, cb)}
            spread, d = abs(v[ca] - v[cb]), v[s] - (v[ca] + v[cb]) / 2
            bad = worse * d > spread
            guard.append(f"- guard {label}: {s} {v[s]:.3f}, {ca} {v[ca]:.3f}, {cb} {v[cb]:.3f}; spread {spread:.3f}; "
                         f"{s} - mean(ctl) {d:+.3f}: {'GUARD FAIL' if bad else 'ok'}")
            if bad:
                verdict = re.sub(r"^(VERDICT: \S+?)([,.]?\s)", r"\1-GUARD\2", verdict, count=1)
        guard.append("")
    lines = [f"# {a.checkpoint}: verdict", "", verdict, "", *guard, *live, *([""] if live else []),
             "| arm | official p90 ITL ms | tok/s/user | full-decode p90 ms | tok/s/user | TTFT p90 ms | out tok/s/GPU | AL | GSM8K strict | gates |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for m in rows:
        lines.append(f"| {m['arm']} | {m.get('official_p90_itl_ms')} | {m.get('tok_s_user_official')} | {m.get('p90_full_decode_itl_ms')} | "
                     f"{m.get('tok_s_user_full_decode')} | {m.get('ttft_p90_ms')} | {m.get('output_tput_per_gpu')} | {m.get('accept_length')} | "
                     f"{m.get('gsm8k_exact_match_strict')} | {m.get('gate_failures')} |")
    lines += ["", "Source: this directory (copied by scripts/ingest_results.py; numbers by scripts/extract_metrics.py).", ""]
    (dst / "VERDICT.md").write_text("\n".join(lines))
    print((dst / "VERDICT.md").read_text())
    return subprocess.run([sys.executable, ROOT / "scripts" / "check_secrets.py", dst]).returncode


if __name__ == "__main__":
    sys.exit(main())
