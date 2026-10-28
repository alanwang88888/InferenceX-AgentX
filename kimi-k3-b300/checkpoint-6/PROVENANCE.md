# Provenance of checkpoint 1

Every value below is copied from a file; the source column names it. `[u]` marks a value no file in `results/` carries.

| item | value | source |
|---|---|---|
| Engine | vLLM, commit `49cb2bbf71cfd36ee5aa33d1e184b1cddff55294`, version `0.32.0rc1.dev281+g49cb2bbf7` | `results/eval-gate-land-1b/l1-eval-on/identity.json` (`checks.image.vllm`, `env.VLLM_BUILD_COMMIT`) |
| Image | `vllm/vllm-openai:nightly-49cb2bbf71cfd36ee5aa33d1e184b1cddff55294`; registry config id `sha256:676ebb16b9aae58a1821e41fb907154f0e5a37084402350bbbf69a08035532d8`, amd64 manifest `sha256:8c1434bfee7d7626eef78ef70600434ffba4a1b8813cb1d1c8707ad25f47035e` | `metrics.json` `image`; the measured servers ran the same layers with the image `Entrypoint` cleared (loaded id `sha256:af62463b…`, `metrics.json` `image_id`) because the measurement harness passes its own command; `scripts/serve.sh` uses `--entrypoint vllm` on the public image instead |
| Image pin origin | InferenceX pins this image for its GB300 Kimi-K3 vLLM entries; the upstream PR adding Kimi-K3 specialized decode is vLLM #61867 | InferenceX `inferencex-e2e/configs/nvidia-master.yaml` (GB300 K3 entries) at `7b28963e` [u] |
| CUDA | 13.0.2, torch `2.14.0+cu130` | `identity.json` `checks.image` |
| CuTe DSL | `nvidia-cutlass-dsl[cu13]` 4.8.0 | vLLM `requirements/cuda.txt` at `49cb2bbf` [u]; the kernels in `overlay/perfbot_kernels/` compile against the DSL shipped in the image |
| Decode path | `--additional-config {"decode_backend":"latency_specialized"}` (vLLM's Kimi-K3 specialized decode, CuTe DSL) | `results/*/*/server_command.txt` |
| Model | `moonshotai/Kimi-K3`, revision `301be1b`, FP4 (MXFP4) checkpoint, 96 shards, 1,560,860,324,864 bytes | `identity.json` `checks.target`; revision from the measurement contract [u] |
| Drafter | `Inferact/Kimi-K3-DSpark`, revision `cf6b8244620e7ea4b0651d214f28e89eac75bed6`, `model.safetensors` sha256 `f9972a63…c053` | `identity.json` `checks.draft` |
| Speculation | DSpark, 7 draft tokens; scored arm: `rejection_sample_method=synthetic`, `synthetic_acceptance_length=3.84` (InferenceX golden AL for Kimi-K3 thinking-on, depth 7); gate arm: `rejection_sample_method=block` | `results/baseline-b0/b0-spec/server_command.txt`, `results/eval-gate-land-1b/*/server_command.txt` |
| Server flags | InferenceX recipe `kimik3/vllm/b300-fp4-mtp/agentic.yaml`, `base` + `override_c8`, rendered as srtctl renders it; deviations: local model and drafter paths, no Mooncake KV offload, bind 127.0.0.1:8000 | `server_command.txt`; `scripts/serve.sh` |
| Landed flags of this checkpoint | `--enable-flashinfer-autotune`; `--additional-config {"decode_backend":"latency_specialized"}`; `--mamba-ssm-cache-dtype float32`; `--scheduler-cls perfbot_kernels.reqlog.stepdec_scheduler` (a log-only scheduler wrapper; every scheduling decision is the base class's); land-1b: a second `--compilation-config` with capture sizes `7,8,14,16,21,24,28,32,35,40,…` (vLLM takes the last occurrence) | `overlay/server_extra_args`, `overlay/arm_env.json`, `results/eval-gate-land-1b/l1-eval-on/server_command.txt` |
| Node | 8x NVIDIA B300 SXM6 AC, driver 615.71.09, NVLink NV18 between every pair, Intel Xeon Platinum 8570, kernel 6.8.0 | `identity.json` `node` |
| Client | `agentx-harness` 1.0.7 (`agentx profile`), `datasets` 5.1.0, `lm_eval` 0.4.9.2 | `metrics.json` `client_runtime`, `point/benchmark_command.txt` |
| Dataset | `semianalysisai/cc-traces-weka-062126` (loader `semianalysis_cc_traces_weka_062126`), 393 traces, revision `23f152f6` | `point/kimik3_fp4_b300_vllm_agentic_dspark_conc8.json` `dataset`; revision from the measurement contract [u] |
| InferenceX | repository `SemiAnalysisAI/InferenceX`, commit `7b28963e6f81da9cf1863928c442a04a1fd92118`, config key `kimik3-fp4-b300-vllm-agentic-dspark` | `metrics.json` `inferencex_sha` |
| Point | agentic coding, concurrency 8, fast protocol (`AIPERF_EXPERIMENTAL_FAST=1`: 1200 s, 1 warmup request per lane) | `point/benchmark_command.txt`; `metrics.json` `duration_mode` |
| Scoring | official: aiperf `inter_token_latency` p90 (`request_metrics.latency.itl.p90`), tok/s/user = 1000 / p90 ITL; consistent: p90 of `full_decode_duration / (osl - 1)`; throughput per GPU from `request_metrics.throughput.per_gpu` | `scripts/extract_metrics.py` |

## Baseline B0 (6.57 ms)

`results/baseline-b0/b0-spec/`: throughput mode, fast, conc 8, on the same image id. Its server ran the measurement
harness's earlier overlay (65 files, tree `0c98b4e8…`, workload revision `00c1e4a8…`, `metrics.json` `overlay_tree_sha256`):
instrumentation and the landed flags listed above, no kernel change. Its server command is byte-identical to the
land-1b command up to the land-1b additions (`--scheduler-cls …`, the second `--compilation-config`).

## Control of the gate (ctl)

`results/eval-gate-land-1b/l1-eval-off/`: the harness overlay at workload revision `3703b94…` (85 files, tree
`1d1f6567…`), the same landed flags, no land-1b member. Land-1b = that tree plus the bundle (`CHANGESET.md`).
