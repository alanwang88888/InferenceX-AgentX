#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run the InferenceX AgentX client (agentx-harness 1.0.7) at the Kimi-K3 agentic point against a server
# started by scripts/serve.sh. Writes the aiperf artifacts and the per-point aggregate under OUT_DIR.
#
#   DURATION=fast (default)  1200 s, 1 warmup request per lane  (AIPERF_EXPERIMENTAL_FAST=1; this checkpoint's point)
#   DURATION=full            3600 s, the InferenceX protocol (their DURATION)
#   CONC=8 (default)         URL=http://127.0.0.1:8000 (default)   OUT_DIR=$PWD/point (default)
#
# Needs network for the first run: the public dataset semianalysisai/cc-traces-weka-062126 (revision 23f152f6)
# and the moonshotai/Kimi-K3 tokenizer are fetched from the Hugging Face Hub.
# The official entry point is `python3 -m infx.bench agentic` of InferenceX 7b28963e (inferencex-e2e/); it builds a
# venv with infx/bench/agentic/requirements.txt and runs exactly this `agentx profile` line (infx/bench/agentic/replay.py).
set -euo pipefail
DURATION=${DURATION:-fast}
CONC=${CONC:-8}
URL=${URL:-http://127.0.0.1:8000}
OUT_DIR=${OUT_DIR:-$PWD/point}
mkdir -p "$OUT_DIR"
case $DURATION in
  fast) SECONDS_=1200; WARMUP=(--warmup-requests-per-lane 1); export AIPERF_EXPERIMENTAL_FAST=1 ;;
  full) SECONDS_=3600; WARMUP=(); export AIPERF_EXPERIMENTAL_FAST=0 ;;
  *) echo "DURATION must be fast or full" >&2; exit 64 ;;
esac
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX='vllm:' AIPERF_HTTP_TCP_USER_TIMEOUT=900000
python3 -m pip install --quiet 'agentx-harness==1.0.7' 'datasets>=4.7.0'
agentx profile --url "$URL" --model moonshotai/Kimi-K3 --tokenizer moonshotai/Kimi-K3 \
  --concurrency "$CONC" --benchmark-duration "$SECONDS_" "${WARMUP[@]}" --tokenizer-trust-remote-code \
  --server-metrics "$URL/metrics" --output-artifact-dir "$OUT_DIR/aiperf_artifacts" \
  --public-dataset semianalysis_cc_traces_weka_062126 | tee "$OUT_DIR/benchmark.log"
echo "artifacts: $OUT_DIR/aiperf_artifacts  (score with: python3 scripts/extract_metrics.py <arm dir>)"
