# Inference-engine patch waiver — PR #TBD (the InferenceX PR is not yet opened)

Filed per `docs/PR_REVIEW_CHECKLIST.md` (engine-patch item), in the form of `docs/waiver/2508.md`.

## Config covered

- Master config entry: `kimik3-fp4-b300-vllm-agentic-dspark` in `configs/nvidia-master.yaml`
- Pinned image: `vllm/vllm-openai:nightly-49cb2bbf71cfd36ee5aa33d1e184b1cddff55294@sha256:8c1434bf…`
- Patch entrypoint: an overlay directory on the server's `PYTHONPATH`; `overlay/sitecustomize.py` installs an import
  finder that replaces 35 `vllm.*` modules by the overlay copies (whole-module replacement; tagged inserted lines plus 4 tagged replaced lines listed in `patches/image-replaced-lines.json`)
  and adds the `perfbot_kernels` package. Published at https://github.com/alanwang88888/InferenceX-AgentX, tag `checkpoint-4-land-8`;
  `patches/overlay-vllm-vs-49cb2bbf.diff` is the unified diff against the image.

## What is patched

| member | files | change |
|---|---|---|
| fused rejection-sampler tail, draft prep, next-step prologue | `v1/worker/gpu/spec_decode/rejection_sampler.py`, `sample/sampler.py`, `states.py`, `spec_decode/dflash/speculator.py`; `perfbot_kernels/step_tail` | launch fusion, bit-equal |
| DSpark draft chain | `models/kimi_k3/nvidia/dspark_mla.py`, `mla.py`; `perfbot_kernels/{draft_chain,draft_ar,draft_query,lse_combine}` | Lamport all-reduce + fused query / LSE-combine kernels, bit-equal; draft weights and precision unchanged |
| MoE experts stage (X2, S) | `specialized_decode/cutedsl/moe/{prepared,front}.py`; `perfbot_kernels/{glue,experts_stage}` | shared expert on the stage grid at 24-40 rows; fp64 envelope <= 1.02 |
| MLA context segments | `v1/attention/backends/mla/prefill/trtllm_ragged.py`; `perfbot_kernels/mla_ctx_segments` | segmented context attention + LSE merge; fp64 envelope |
| prefill suffix restage | `v1/worker/gpu/model_runner.py`; `perfbot_kernels/prefill_restage` | keeps the step graph in collider steps; ids and top-20 logprobs bit-equal |
| MLA decode variable split (L6) | `.../cutedsl/mla/{prepared,dcp}.py`, `.../runtime/providers.py`; `perfbot_kernels/mla_varsplit` | per-request split counts and a fused reduction; fp64 envelope 1.02 |
| MoE front pin, lean experts stage | `.../cutedsl/moe/front.py`; `perfbot_kernels/{moe_front,experts_stage}` | bit-equal at the served rows |
| MLA decode core (E2b, MLA-LOOP) | `.../cutedsl/mla/prepared.py`; `perfbot_kernels/mla_core` | one-pass online softmax, split PV MMA; fp64 envelope 1.02, MLA-LOOP bit-exact to E2b |
| mamba checkpoint copy-on-write (CKPT-COW) | `v1/kv_cache_interface.py`, `v1/core/single_type_kv_cache_manager.py`, `v1/core/sched/scheduler.py`, `model_executor/layers/mamba/checkpoint.py` | exports the prefix-cache checkpoint when its column meets the initial-state column; tolerance (different chunking of the same history) |
| log-only instrumentation | `--scheduler-cls perfbot_kernels.reqlog.stepdec_scheduler` and tagged log lines | scheduling decisions are the base class's |

## Why the unmodified upstream image cannot run this benchmark

It can. This waiver covers a performance patch, not a correctness one: the unmodified image serves the point at
6.57 ms official p90 ITL (152 tok/s/user). The patched engine reaches 5.24 ms (190.8 tok/s/user) at checkpoint 4 (`results/checkpoint-4/VERDICT.md`). At checkpoint 3 the two
unmodified-image controls of the same request measured 6.45 / 7.11 ms, and the pre-registered three-way rule returned CONFIRMED-ITL (`results/checkpoint-3/VERDICT.md`).
If the maintainers hold the waiver to correctness-only cases, this PR waits for the upstreaming below.

## Upstream PR / issue links

- vLLM Kimi-K3 specialized decode (the code the members extend): https://github.com/vllm-project/vllm/pull/61867
- Member PRs to vLLM: none opened yet

## Removal plan

Upstream each member to vLLM behind its own PR; when a nightly ships them, bump `image:` to that nightly, drop the
overlay and the `PERFBOT_EXP_*` env from the recipe, and delete this waiver in the same PR.
