#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Start the Kimi-K3 vLLM server of this checkpoint on one 8x B300 node, inside the pinned public image,
# with the overlay of this repository on PYTHONPATH. One server = one measurement point (InferenceX rule).
#
# Required environment:
#   MODEL_DIR   local copy of moonshotai/Kimi-K3 (revision 301be1b; ~1.5 TB)
#   DRAFT_DIR   local copy of Inferact/Kimi-K3-DSpark (revision cf6b8244)
# Optional:
#   MODE        throughput (default; synthetic acceptance at the golden AL 3.84, the InferenceX scored arm)
#               | eval_only (real block rejection sampling; the acceptance and GSM8K gate arm)
#   OVERLAY     default: the overlay/ directory beside this script's parent
#   PORT        default 8000
#   CACHE_DIR   default $PWD/cache (compile caches; the first start compiles CuTe DSL kernels for several minutes)
#
# The flags below are the recipe's override_c8 (InferenceX kimik3/vllm/b300-fp4-mtp/agentic.yaml) rendered as
# srtctl renders it, plus the landed flags of this checkpoint (overlay/server_extra_args) and the land-1b arm
# environment (overlay/arm_env.json). Deviations from the recipe: local model and drafter paths, no Mooncake
# KV offload (kv-transfer-config and MC_* / VLLM_MOONCAKE_* / WITH_NVIDIA_PEERMEM dropped), bind 127.0.0.1.
# results/*/server_command.txt holds the line the measured servers ran.
set -euo pipefail
: "${MODEL_DIR:?set MODEL_DIR}" "${DRAFT_DIR:?set DRAFT_DIR}"
MODE=${MODE:-throughput}
PORT=${PORT:-8000}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OVERLAY=${OVERLAY:-$HERE/overlay}
CACHE_DIR=${CACHE_DIR:-$PWD/cache}
IMAGE=vllm/vllm-openai:nightly-49cb2bbf71cfd36ee5aa33d1e184b1cddff55294
mkdir -p "$CACHE_DIR"

case $MODE in
  throughput) SPEC='{"method":"dspark","model":"/workspace/draft","num_speculative_tokens":7,"attention_backend":"TOKENSPEED_MLA","draft_sample_method":"probabilistic","rejection_sample_method":"synthetic","synthetic_acceptance_length":3.84}' ;;
  eval_only)  SPEC='{"method":"dspark","model":"/workspace/draft","num_speculative_tokens":7,"attention_backend":"TOKENSPEED_MLA","draft_sample_method":"probabilistic","rejection_sample_method":"block"}' ;;
  *) echo "MODE must be throughput or eval_only" >&2; exit 64 ;;
esac

# land-1b switches (overlay/arm_env.json). PERFBOT_SERVER_EXTRA_ARGS of that file is the second --compilation-config below.
ARM_ENV=(
  PERFBOT_EXP_DRAFT_CHAIN=1
  PERFBOT_EXP_GLUE=X2,S,D,G
  PERFBOT_EXP_MLA_CTX_SEGMENTS=auto
  PERFBOT_EXP_PREFILL_GRAPH=1
)
# recipe base.roles.agg.env (agentic.yaml), Mooncake entries dropped
RECIPE_ENV=(
  VLLM_ALLREDUCE_USE_FLASHINFER=1
  VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION=1
  VLLM_USE_V2_MODEL_RUNNER=1
  VLLM_USE_DIRECT_DCP_A2A=1
  VLLM_USE_DIRECT_DCP_Q_GATHER=1
  VLLM_USE_DIRECT_DCP_KV_GATHER=1
  VLLM_RPC_TIMEOUT=600000
  VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0
  VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
  PYTHONNOUSERSITE=1
  TORCH_CUDA_ARCH_LIST=10.0
  PYTHONHASHSEED=42
  VLLM_HTTP_TIMEOUT_KEEP_ALIVE=900
)

exec docker run --rm --gpus all --network=host --ipc=host --shm-size=64g \
  -v "$MODEL_DIR":/model:ro -v "$DRAFT_DIR":/workspace/draft:ro -v "$OVERLAY":/workspace/overlay:ro -v "$CACHE_DIR":/cache \
  -e HOME=/cache -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/workspace/overlay \
  -e HF_HOME=/cache/huggingface -e VLLM_CACHE_ROOT=/cache/vllm -e TRITON_CACHE_DIR=/cache/triton -e CUTE_DSL_CACHE_DIR=/cache/cute_dsl \
  "${ARM_ENV[@]/#/-e }" "${RECIPE_ENV[@]/#/-e }" \
  --entrypoint vllm "$IMAGE" serve /model \
  --served-model-name moonshotai/Kimi-K3 --tensor-parallel-size 8 --decode-context-parallel-size 8 --dcp-comm-backend a2a \
  --max-num-batched-tokens 16384 --trust-remote-code --language-model-only --enable-auto-tool-choice \
  --tool-call-parser kimi_k3 --reasoning-parser kimi_k3 --load-format fastsafetensors --moe-backend auto \
  --no-enable-flashinfer-autotune --enable-cumem-allocator --enable-prefix-caching --prefix-match-unit 128 \
  --kv-cache-dtype fp8 --stream-interval 10 --attention-backend TOKENSPEED_MLA \
  --attention-config '{"mla_prefill_backend":"TRTLLM_RAGGED","use_prefill_query_quantization":true}' \
  --disable-uvicorn-access-log --max-num-seqs 16 --gpu-memory-utilization 0.92 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,256,512,1024,2048,4096,8192]}' \
  --speculative-config "$SPEC" \
  --host 127.0.0.1 --port "$PORT" \
  --enable-flashinfer-autotune \
  --additional-config '{"decode_backend":"latency_specialized"}' \
  --mamba-ssm-cache-dtype float32 \
  --scheduler-cls perfbot_kernels.reqlog.stepdec_scheduler \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[7,8,14,16,21,24,28,32,35,40,48,56,64,72,80,88,96,104,112,120,128,256,512,1024,2048,4096,8192]}'
