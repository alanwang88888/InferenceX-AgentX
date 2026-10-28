Title: `Kimi-K3 B300 vLLM agentic: pin 49cb2bbf, latency_specialized decode and conc-8 landing set / Kimi-K3 B300 vLLM 智能体：固定 49cb2bbf 镜像、latency_specialized 解码与 conc-8 落地集`

Problem: the Kimi-K3 B300 vLLM agentic entry is pinned to a 2026-08 nightly (3696c77) that predates the upstream
specialized decode path; the GB300 entries already run 49cb2bbf with `latency_specialized`.

Change: pin the B300 entry to the same image, enable the specialized decode backend and flashinfer autotune, add
conc-8 capture sizes 7-35, and ship a conc-8 landing set as an engine overlay under a waiver (`docs/waiver/<PR>.md`):
fused rejection tail and prologue, bit-equal DSpark draft-chain kernels, MoE experts stage at 24-40 rows, segmented
context MLA prefill, prefill-suffix restage, MLA decode variable split and a one-pass MLA decode core, MoE front pin and lean
experts stage, mamba checkpoint copy-on-write. Server code only; client, recipe methodology, acceptance target and ITL
definition unchanged.

Result (local 1200 s fast runs, 8x B300, one node): conc 8 official p90 ITL 5.24 ms (190.8 tok/s/user) vs 6.59 ms (151.7) for the unmodified image of record (1.258x, cross-run); full-decode p90 5.86 ms. The checkpoint-3 set in the same request read 5.29 ms (NO_REGRESSION). conc 1: 3.02 ms (331.1 tok/s/user) vs 3.08 ms on the unmodified image (1.021x, cross-run); same-request ABA vs the checkpoint-3 set: NO_REGRESSION. TTFT p90 is higher at both points (conc 8 +218 ms tail only, conc 1 +12.6 ms inside the A/A), noted in the run README.
GSM8K with real acceptance: 0.9697 strict (floor 0.90); greedy responses identical to checkpoint 3 on 1319/1319. Run: https://github.com/alanwang88888/InferenceX-AgentX/tree/main/kimi-k3-b300/checkpoint-6/results/checkpoint-4.

Risks: engine patch (waiver); conc 8 only in this PR (other `conc-list` points to follow); the fast protocol (1200 s)
was used for the numbers above, the 3600 s protocol run is not run.

Draft model: `Inferact/Kimi-K3-DSpark` revision cf6b8244, served as shipped (bf16 weights as stored, vLLM default
handling, fp8 KV inherited from the target as in the recipe); the overlay changes launch structure only, bit-equal
to the image's draft computation. `SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE` is not applicable (vLLM).

<details><summary>Validation details</summary>

- `results/checkpoint-4/VERDICT.md`: `land8` (land-8) and `land5` (land-5) arms, both p90 statistics, throughput per GPU; `conc1/`: the conc 1 ABA.
- `results/checkpoint-3/VERDICT.md`: the land-5 ABA against two unmodified-image controls (CONFIRMED-ITL).
- `results/checkpoint-4/eval/`: eval_only arm, GSM8K 1319 questions, real acceptance, greedy parity with checkpoint 3.

</details>

<details><summary>AI model disclosure</summary>

Claude (Anthropic), model id `claude-fable-5-1`, prepared this PR text and the repository scaffolding from measurement
files under the direction of the author; the kernels and measurements were produced by the author's PerfBot
agent sessions (Claude models; exact ids per session are recorded in the project's internal logs and are not verified here).

</details>
