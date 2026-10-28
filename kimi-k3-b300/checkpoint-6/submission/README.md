# InferenceX submission files for checkpoint 4 (land-8)

Source followed: the InferenceX repository `SemiAnalysisAI/InferenceX` at commit `7b28963e6f81da9cf1863928c442a04a1fd92118`
(https://github.com/SemiAnalysisAI/InferenceX): `CONTRIBUTING.md` (PR review flow, performance changelog requirement,
draft-model precision), `inferencex-e2e/docs/PR_REVIEW_CHECKLIST.md` (the CODEOWNER checklist), `inferencex-e2e/docs/waiver/2508.md`
(the one filed engine-patch waiver, the template we follow), `inferencex-e2e/perf-changelog.yaml` (entry format),
`inferencex-e2e/configs/nvidia-master.yaml` (config key `kimik3-fp4-b300-vllm-agentic-dspark`),
`inferencex-e2e/benchmarks/single_node/srt-slurm-recipes/kimik3/vllm/b300-fp4-mtp/agentic.yaml` (the recipe).

InferenceX has no separate "submission package". A submission is a pull request to that repository that changes these files:

| file in InferenceX | what checkpoint 4 changes | draft here |
|---|---|---|
| `inferencex-e2e/configs/nvidia-master.yaml`, key `kimik3-fp4-b300-vllm-agentic-dspark` | `image:` pinned to `vllm/vllm-openai:nightly-49cb2bbf71cfd36ee5aa33d1e184b1cddff55294@sha256:8c1434bf…` (the key at `7b28963e` names `nightly-dev-x86_64-cu13-3696c77`) | `nvidia-master.proposed.yaml` |
| `…/kimik3/vllm/b300-fp4-mtp/agentic.yaml`, `override_c8` | the landed flags and the land-8 capture sizes and switches (`overlay/arm_env.json`) | `recipe-override_c8.proposed.yaml` |
| `inferencex-e2e/perf-changelog.yaml` | one appended entry (never edit history) | `perf-changelog-entry.yaml` |
| `inferencex-e2e/docs/waiver/<PR number>.md` | required: the overlay patches the engine's Python inside the pinned image | `waiver-DRAFT.md` |
| PR description | summary, validation from a real run, draft-precision statement, AI model disclosure in a collapsed block, bilingual title | `PR-DESCRIPTION-DRAFT.md` |

## What their rules say about this change set

1. **Engine patch.** Checklist item: "Verified that this PR does not patch the inference engine or serving stack — the
   pinned image must run as shipped … The only exception is a patch covered by a filled-out waiver at `docs/waiver/<PR_NUMBER>.md`
   … stating what is patched, why the unmodified upstream image cannot run this benchmark, the upstream PR/issue link,
   and the removal plan." The overlay is such a patch (Python modules shadowed by `sitecustomize.py`, new CuTe DSL
   kernels). The filed precedent (waiver 2508) is a correctness waiver: the stock image could not serve. Ours is a
   performance patch; a waiver may be refused on that ground [u]. The compliant path is to upstream the members to
   vLLM (the Kimi-K3 specialized decode lives upstream since PR #61867) and bump the image pin; the waiver draft
   states that as the removal plan.
2. **Draft as it ships.** The DSpark drafter, its precision and its computation are unchanged; item 15 replaces
   launches with bit-equal kernels (`CHANGESET.md`). The PR must state the drafter revision, its stored precision
   and the effective precision (checklist item on draft precision).
3. **Synthetic acceptance.** The scored arm uses the committed golden AL 3.84 for Kimi-K3 thinking-on at depth 7; unchanged.
4. **Validation.** The PR description reports the real run (both p90 statistics, throughput per GPU, GSM8K) from
   `results/checkpoint-4/` and links it; no local-check inventories.
5. **Pareto coverage.** Their checklist recommends >= 5 points per curve; checkpoints 1-3 measure conc 8 only; checkpoint 4 adds conc 1. A
   submission needs the other `conc-list` points of the config (1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 70) or a core
   maintainer's bypass [u].
6. **Engine ordering.** Kimi-K3's primary engine is native/upstream vLLM (`MODELS.md`); this submission is vLLM. OK.
7. **Image pin.** A new pin needs the digest (`image@sha256:…`), as recent entries do.
