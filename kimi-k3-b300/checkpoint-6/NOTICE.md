# Notices

This repository is licensed under the Apache License 2.0 (`LICENSE`).

## Third-party code

| path | origin | license |
|---|---|---|
| `overlay/vllm/**` | vLLM (https://github.com/vllm-project/vllm) at commit `49cb2bbf`, 32 files copied from the image; 26 carry inserted `# PERFBOT` lines, 6 are verbatim | Apache-2.0, "Copyright contributors to the vLLM project" (header kept in every file) |
| `overlay/perfbot_kernels/mla_core/kernel.py` | fork of tokenspeed 2.0.2 `mla_decode_fp8.py` (https://github.com/lightseekorg/tokenspeed) as installed in the image; an OFF member | MIT, Copyright (c) 2026 LightSeek Foundation (header kept) |
| `overlay/perfbot_kernels/experts_stage/{experts,experts_runtime,shared}.py`, `kdasplit_la/{kernel,prepared}.py` | forks of vLLM `49cb2bbf` CuTe DSL modules | Apache-2.0, vLLM header kept |
| everything else under `overlay/perfbot_kernels/`, `overlay/sitecustomize.py`, `overlay/perfbot_resident.py`, `scripts/`, documents | NVIDIA, this project | Apache-2.0 |

## Header census of the served overlay (168 files) [m]

37 Python files carry the vLLM Apache-2.0 header; 27 carry `SPDX-License-Identifier: Apache-2.0` without a copyright
line; 1 carries the MIT header above; 92 Python files and 10 JSON / Markdown files carry no header. The served bytes
are published as measured (`scripts/check_overlay_identity.py`), so headers are not added in this checkpoint; the
project license applies to every file without a header. A header pass is a change of the served tree and belongs to a
later checkpoint.

## Data and models referenced, not included

`moonshotai/Kimi-K3` and `Inferact/Kimi-K3-DSpark` (Hugging Face Hub, their own licenses);
`semianalysisai/cc-traces-weka-062126` (the InferenceX agentic trace corpus); `openai/gsm8k`.
InferenceX (https://github.com/SemiAnalysisAI/InferenceX) is Apache-2.0; its files are referenced, not copied.
