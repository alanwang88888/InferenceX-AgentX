#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read one arm's client output and print the two p90 ITL statistics and tok/s/user.

Inputs (any that exist under <arm dir>):
  point/<name>.json                       the InferenceX aggregate (request_metrics.latency.itl.p90, seconds)
  point/aiperf_artifacts/profile_export.jsonl   per-request records (full_decode_duration ms, output_sequence_length)
  metrics.json                            the harness summary (p90_itl_ms, p90_full_decode_itl_ms) when present

Statistics:
  official p90 ITL  = p90 of aiperf inter_token_latency = (request_latency - ttft) / (osl - 1)   [the InferenceX objective]
  full-decode p90   = p90 of full_decode_duration / (osl - 1) per profiled request               [consistent statistic]
  tok/s/user        = 1000 / p90 ITL (ms)
  TTFT p90          = request_metrics.latency.ttft.p90 of the aggregate (ms)                     [no-regression guard]
  The aggregate holds lane 1 only; a metrics.json with all_lanes_<field> (the harness's all-lane reading) overrides
  TTFT p90 and output tok/s/GPU with all_lanes_ttft_p90_ms and all_lanes_output_tput_per_gpu.
p90 is linear interpolation over the sorted values (numpy default), as the harness does.

    python3 scripts/extract_metrics.py <arm dir> [--json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def q(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def full_decode(point: Path):
    paths = [point / "aiperf_artifacts" / "profile_export.jsonl", *sorted(point.glob("lanes/*/aiperf_artifacts/profile_export.jsonl"))]
    xs = []
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            r = json.loads(line)
            m, md = r.get("metrics", {}), r.get("metadata", {})
            if md.get("benchmark_phase") == "profiling" and "inter_token_latency" in m and "full_decode_duration" in m and not r.get("error"):
                xs.append(m["full_decode_duration"]["value"] / (m["output_sequence_length"]["value"] - 1))
    return {"p90_full_decode_itl_ms": q(xs, 0.9), "full_decode_itl_ms_median": q(xs, 0.5), "requests": len(xs)} if xs else None


def main() -> int:
    arm = Path(sys.argv[1])
    out = {"arm": arm.name}
    agg = next(iter(sorted((arm / "point").glob("*_conc*.json"))), None)
    if agg:
        a = json.loads(agg.read_text())
        lat = a["request_metrics"]["latency"]
        out["official_p90_itl_ms"] = round(lat["itl"]["p90"] * 1000, 3)
        out["official_median_itl_ms"] = round(lat["itl"]["p50"] * 1000, 3)
        out["ttft_p90_ms"] = round(lat["ttft"]["p90"] * 1000, 3)
        out["output_tput_per_gpu"] = a["request_metrics"]["throughput"]["per_gpu"]["output_tput_tps"]
        out["tput_per_gpu"] = a["request_metrics"]["throughput"]["per_gpu"]["total_tput_tps"]
        out["requests_successful"] = a["num_requests_successful"]
        out["requests_total"] = a["num_requests_total"]
    fd = full_decode(arm / "point")
    if fd:
        out.update(fd)
    m = arm / "metrics.json"
    if m.is_file():
        s = json.loads(m.read_text())
        out.setdefault("official_p90_itl_ms", s.get("p90_itl_ms"))
        out.setdefault("p90_full_decode_itl_ms", s.get("p90_full_decode_itl_ms"))
        if s.get("all_lanes_ttft_p90_ms") is not None:
            out["ttft_p90_ms"], out["output_tput_per_gpu"] = round(s["all_lanes_ttft_p90_ms"], 3), s["all_lanes_output_tput_per_gpu"]
        for k in ("accept_length", "gsm8k_exact_match_strict", "gsm8k_exact_match_flexible", "gate_failures", "comparison_key", "overlay_tree_sha256", "image_id"):
            if k in s:
                out[k] = s[k]
    if out.get("official_p90_itl_ms"):
        out["tok_s_user_official"] = round(1000 / out["official_p90_itl_ms"], 1)
    if out.get("p90_full_decode_itl_ms"):
        out["tok_s_user_full_decode"] = round(1000 / out["p90_full_decode_itl_ms"], 1)
    if "--json" in sys.argv:
        print(json.dumps(out, indent=1))
    else:
        for k, v in out.items():
            print(f"{k:28s} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
