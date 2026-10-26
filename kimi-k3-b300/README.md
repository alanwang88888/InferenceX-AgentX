# Kimi-K3 on 8x B300 under InferenceX AgentX: results, method, optimizations

This document is written for authors and reviewers who need to align their own measurements with ours. It gives:

1. the headline results against two baselines,
2. the exact measurement method, as published by InferenceX and as run by us,
3. a checklist for aligning other models' numbers with these,
4. the list of optimizations behind each pushed checkpoint, with their numerics class and evidence.

All runs use the unmodified InferenceX AgentX client and dataset. The model, its quantization and its expert routing are
the published ones; every change is server-side code or a server flag on the public vLLM image `nightly-49cb2bbf`.
Numbers in sections 1-3 carry a record reference [Rn] resolved in Appendix A; section 4 has its own appendix.


## 1. Results

Per-user decode speed of Kimi-K3 (fp4) on one node of 8x NVIDIA B300 under the InferenceX AgentX workload, in tokens per second per user (p90 interactivity, higher is better) [R1].

| concurrency | A: InferenceX public B300 (2026-08-16 run, vLLM 5894fdf of 2026-08-14), full protocol (3600 s, warmup 10/lane) | B: stock vLLM nightly-49cb2bbf (2026-10-16), the build behind InferenceX's GB300 AgentX rows of 2026-10-18¹, fast protocol (1200 s, warmup 1/lane) | ours: checkpoint-6 (49cb2bbf + landed changes), fast protocol (1200 s, warmup 1/lane) | speedup vs B |
|---|---|---|---|---|
| 1 | 202.8 | 324.7 | 351.5 | 1.083x |
| 8 | 63.9 | 151.7 | 207.6 | 1.368x |
| 32 | 17.6 | 17.3 | 18.4 | 1.061x (1.044x key-matched) |

Speedup vs B is same-build, 1200 s fast protocol; A is cross-build and cross-protocol. At conc 1 and 8, B is one stock run from an earlier session; at conc 32, B is the mean of eight stock runs, with the key-matched sensitivity in brackets.
B and ours share model, quantization and routing; A is InferenceX's published B300 configuration of the same model (fields that differ or are unpublished: vLLM build 5894fdf vs 49cb2bbf; decode executor stock vs latency_specialized (a vLLM option); protocol 3600 s / warmup 10 vs 1200 s / warmup 1; KV offload (Mooncake DRAM) vs none; recipe fingerprint 61ba8917 vs unpublished; conc-32 speculation unpublished (API label 'mtp')).

¹ InferenceX listed GB300 AgentX rows on this build on 2026-10-18 (GB300 conc 4 (agg, no offload)). Its public results now list GB300 rows only from 2026-10-23 on 1b2a6663.
At conc 32 the stock control is the mean of eight stock runs fixed before any candidate run was read. One very long agent turn (input 659,451 tokens) happened to be sent in 4/8 stock runs and in 0/3 of our runs; it slows the runs that receive it. Without those stock runs the control is 56.825 ms (n 4), which gives the bracketed speedup [R1].

## 2. How we measure

### 2.1 The InferenceX AgentX workload, as published

We use the InferenceX AgentX benchmark unchanged ([methodology](https://inferencex.semianalysis.com/agentx/methodology), [glossary](https://inferencex.semianalysis.com/glossary), [client](https://github.com/SemiAnalysisAI/InferenceX/blob/418d8d48a3bc98ca715c091033199603c9c5a9b9/inferencex-e2e/docs/agentx-standalone.md)).

- **Closed-loop agent clients.** The client replays recorded coding-agent sessions. Concurrency is the number of live agent clients (lanes). A lane sends its next turn only after the server has answered the previous one, so a faster server receives more turns in the same time.
- **Start position.** Each lane joins its recorded session part way through. In our runs the logged start positions span 31 % to 75 % of the session time at conc 32 [R18] (35-74 % at conc 8 [R16], 53 % at conc 1 [R14]), and they are the same in every run of a concurrency.
- **Primer and warmup.** Before measurement the client sends one cache primer per active session (34 at conc 32 [R19]) and then a fixed number of warmup requests per lane. Primer and warmup requests are not counted.
- **Measured window.** The client then measures for a fixed time. The full protocol measures for 3600 s with 10 warmup requests per lane; the published B300 rows ran 3586-3629 s [R20]. The fast protocol measures for 1200 s with 1 warmup request per lane.
- **Statistics** ([code](https://github.com/SemiAnalysisAI/InferenceX/blob/418d8d48a3bc98ca715c091033199603c9c5a9b9/inferencex-e2e/infx/results/agentic/request_metrics.py)). p90 ITL is the 90th percentile over requests of each request's mean inter-token latency. Interactivity (tok/s/user) = 1000 / p90 ITL in ms. TTFT p90 is the 90th percentile time to first token. E2E normalized interactivity ([glossary](https://inferencex.semianalysis.com/glossary/e2e-normalized-interactivity), [definition](https://github.com/ai-dynamo/aiperf/pull/1270)) = 1 / p90 of (request latency / output tokens): it also charges queueing and prefill.

### 2.2 Our runs, exactly as run

Every setting below is read from the server and client command of each run. Where a flag appears twice, the last one applies, as in vLLM. "Stock" is the unmodified image; "ours" is the same image with our changes (checkpoint 6). The published recipe is [here](https://github.com/SemiAnalysisAI/InferenceX/blob/418d8d48a3bc98ca715c091033199603c9c5a9b9/inferencex-e2e/benchmarks/single_node/srt-slurm-recipes/kimik3/vllm/b300-fp4-mtp/agentic.yaml).

| setting | conc 1 stock [R2][R3] | conc 1 ours [R4][R5] | conc 8 stock [R6][R7] | conc 8 ours [R8][R9] | conc 32 stock [R10][R11] | conc 32 ours [R12][R13] |
|---|---|---|---|---|---|---|
| image | nightly-49cb2bbf | nightly-49cb2bbf | nightly-49cb2bbf | nightly-49cb2bbf | nightly-49cb2bbf | nightly-49cb2bbf |
| tensor parallel | 8 | 8 | 8 | 8 | 8 | 8 |
| decode context parallel | 8 | 8 | 8 | 8 | 8 | 8 |
| expert parallel | off | off | off | off | off | off |
| data parallel | 1 | 1 | 1 | 1 | 1 | 1 |
| max batched tokens per step | 16384 | 16384 | 16384 | 16384 | 16384 | 16384 |
| max sequences | 2 | 2 | 16 | 16 | 64 | 64 |
| GPU memory fraction | 0.92 | 0.92 | 0.92 | 0.92 | 0.92 | 0.90 |
| KV cache dtype | fp8 | fp8 | fp8 | fp8 | fp8 | fp8 |
| prefix caching | on | on | on | on | on | on |
| KV offload | none | none | none | none | none | none |
| speculative decoding | DSpark, 7 draft tokens, synthetic acceptance | DSpark, 5 draft tokens, synthetic acceptance | DSpark, 7 draft tokens, synthetic acceptance | DSpark, 5 draft tokens, synthetic acceptance | none | none |
| decode executor | latency_specialized | latency_specialized | latency_specialized | latency_specialized | latency_specialized | latency_specialized |
| client concurrency | 1 | 1 | 8 | 8 | 32 | 32 |
| measured window (s) | 1200 | 1200 | 1200 | 1200 | 1200 | 1200 |
| warmup requests per lane | 1 | 1 | 1 | 1 | 1 | 1 |
| dataset | semianalysis_cc_traces_weka_062126 | semianalysis_cc_traces_weka_062126 | semianalysis_cc_traces_weka_062126 | semianalysis_cc_traces_weka_062126 | semianalysis_cc_traces_weka_062126 | semianalysis_cc_traces_weka_062126 |

- **Hardware.** One node, 8x NVIDIA B300 SXM, all eight GPUs serve one model instance (tensor parallel 8, decode context parallel 8, no expert or data parallelism).
- **Protocol.** All our runs use the fast protocol (1200 s, 1 warmup request per lane). Section 3 explains how to compare fast and full numbers.
- **Speculative decoding.** At conc 1 and 8 both stock and ours draft with DSpark (stock 7 draft tokens, ours 5 draft tokens: the draft length is one of our changes). Acceptance is synthetic, at a fixed acceptance length, as in the published throughput recipe. At conc 32 nobody drafts: the recipe turns drafting off above conc 16.
- **Controls.** Every result is judged against stock or previous-checkpoint runs made in the same session on the same node, in an A-B-A order where possible. Each run is a fresh server start.
- **Decision rule** [R21]. Gain = mean(candidate p90 ITL) - mean(control p90 ITL). The band is the 95 % half-width of a null bootstrap: resample the control runs alone (2000 draws, fixed seed), and take the difference of means. A change counts as a gain only if the gain is beyond the band. The rule and the control set are fixed before the candidate runs are read.
- **Quality gate** [R22]. Each checkpoint also passes a GSM8K accuracy run (strict match at least 0.9; checkpoint 6 scored 0.9682). Throughput runs use synthetic acceptance, so the accuracy run is the check that outputs stay correct.

## 3. How to align your numbers with ours

1. **Hold fixed:** model and quantization; hardware and parallel layout (one node of 8 GPUs, tensor parallel 8, decode context parallel 8); the client, dataset and concurrency; the speculative-decoding setting per concurrency; the protocol (fast or full); KV offload on or off.
2. **Publish per row:** image or commit and its date; the full server command; the client command (concurrency, measured window, warmup per lane); the number of runs and how many server starts they came from; p90 ITL per run and the mean; TTFT p90 and E2E normalized interactivity; the failed-request rate.
3. **Fast vs full.** A fast run (1200 s) and a full run (3600 s) do not sample the same turns. The client is closed loop, so the measured window and the server's own speed decide how far each lane gets into its session. Compare fast with fast and full with full. Treat a fast-vs-full ratio as descriptive only.
4. **Request mix.** At conc 32, single very long turns (hundreds of thousands of input tokens) enter some runs and not others, depending on where the lanes are when the window closes. Such a turn slows every other request while it is prefilled. Use several runs per side. Before you call a difference real, compare runs on the requests both sides served (match on conversation and turn). Section 1 shows this effect at conc 32.
5. **Same-build controls.** Our speedup is measured against stock runs of the same image in the same session. A ratio against a public row made with another build, protocol or recipe mixes those differences into the result.

## 4. Optimizations

This section lists every change that landed in checkpoints 1 to 6, and the conc-32 landing set that we measured after
checkpoint 6 but have not yet pushed. Every change is server code (an overlay on the public vLLM image
`nightly-49cb2bbf`) or a server flag. We did not change the model, its quantization, the routing, the draft model, the
client, the recipe's acceptance method or the ITL definition.

### How to read the gains

- We measure gains **per checkpoint, not per change**. Each checkpoint is a set of changes, and each set was judged at
  the real InferenceX point against same-run controls. The "evidence" column gives the per-change reading that we had
  before the set went to the real point (a kernel judge or a step-level proxy). It is not an end-to-end gain.
- "Paired gain" = mean control p90 ITL / mean candidate p90 ITL, both from the same run. "vs stock" = stock p90 ITL /
  candidate p90 ITL; at conc 1 and conc 8 the stock reading is from an earlier run.
- Numerics class: **bit-equal** = the output bits equal the image path on the judge cases. **Tolerance** = the output is
  inside a stated bound against an fp64 reference (most use a max error ratio of 1.02 vs the image's own error), and the
  checkpoint's GSM8K gate passed (strict >= 0.90).
- Every number below is measured. The appendix gives the record for each one.

### 4.1 Gain per checkpoint (official p90 ITL, fast protocol)

| checkpoint | conc | candidate p90 ITL (ms) | same-run controls (ms) | delta (ms) | verdict | paired gain | vs stock | GSM8K strict |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| 1 | 8 | 6.10 | 6.61 / 6.89 | -0.650 | NOT_RESOLVED (interval contains 0) | 1.107x | 1.080x | 0.9697 |
| 2 | 8 | 5.88 | 6.56 / 6.30 | -0.550 | NOT_RESOLVED (interval contains 0) | 1.094x | 1.121x | 0.9697 |
| 3 | 8 | 5.15 | 6.45 / 7.11 | -1.630 | CONFIRMED | 1.317x | 1.280x | 0.9697 |
| 4 | 8 | 5.24 | 5.29 (checkpoint 3) | -0.050 | NO_REGRESSION | 1.010x | 1.258x | 0.9697 (1319/1319 responses equal checkpoint 3) |
| 4 | 1 | 3.02 | 2.99 / 2.99 (checkpoint 3) | +0.023 | NO_REGRESSION | 0.990x | 1.020x | as above |
| 5 | 8 | 4.82 | 5.39 (checkpoint 4, mean of 8) | -0.567 | GAIN | 1.118x | 1.368x | 0.9727 |
| 5 | 1 | 2.845 | 3.02 (checkpoint 4) | -0.174 | GAIN | 1.062x | 1.083x | as above |
| 6 | 32 | 54.43 | 57.76 (stock, mean of 8) | -3.331 | GAIN (key-matched 1.044x) | 1.061x | 1.061x | 0.9682 |
| conc-32 landing set (**measured, not yet pushed**) | 32 | 48.32 | 57.76 (stock, mean of 8) | -9.44 | GAIN | - | 1.195x | 0.9674 |

Checkpoint 6 changed no code at conc 1 and conc 8; those points carry checkpoint 5 by identity. The conc-1 and conc-8
reads of the conc-32 landing set are running now; until they read, that set is a conc-32 result only.

### 4.2 Changes, by checkpoint

#### Checkpoint 1

| change | what it changes | numerics | evidence before the real point |
|---|---|---|---|
| MoE experts stage, no JOIN + shared expert beside routed experts | At 17-40 decode rows the JOIN step is off, and the shared expert's down projection gets 4-5 SMs so it runs beside the routed experts instead of after them. | X2 part bit-equal (12/12); shared-expert part tolerance (ratio <= 1.02) | about -0.14 ms per token on the mixed proxy |
| Fused step tail | One persistent CuTe DSL kernel replaces the rejection-sampler launches; one launch replaces four draft-prep launches; one replaces 17 next-step prologue launches. Same math, fewer launches. | bit-equal tokens (42/42 cases x 3 arms) | - |
| Dead draft-gather guard | Skips the gather of rejected draft tokens. | bit-equal | - |
| Draft chain | Three kernels in the draft MLA layer at 17-64 padded rows: all-reduce fused with residual add and RMSNorm; one launch for the q path; one for the combine. 22 launches per draft layer become 8. Draft weights and math unchanged. | bit-equal (1319/1319 responses equal the image path) | -0.120 ms per step on the long-context proxy |
| Extra CUDA-graph capture sizes 7-35 | Decode batches of 1-5 requests x 7 tokens hit a captured graph instead of padding up to the next multiple of 8. | graph shape only | - |
| Prefill graph, restaged suffix | A step that runs a prefill chunk beside decoders keeps its CUDA graph. | bit-equal token ids and top-20 logprobs, server on vs off | -14 ms per such step (median, proxy) |
| MLA context segments | A prefill chunk with few new tokens over a long cached context fills only 12-72 of 148 SMs. The keys are cut into up to 8 segments, run as rows of one call, and merged by log-sum-exp. | tolerance (fp64 envelope, 96/96 cases) | 0.20x-0.93x of the one-call time at 183-2,100 new tokens (B200 kernel judge) |

Prior art: split-KV with log-sum-exp merge, [Flash-Decoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html).

#### Checkpoint 2

| change | what it changes | numerics | evidence |
|---|---|---|---|
| MLA decode variable split | The MLA decode kernel picks the number of KV splits per batch shape from a rule table instead of one fixed split. | tolerance (ratio <= 1.02); GSM8K PASS | - |
| MoE front GEMM pin | Pins the MoE front warp-specialized GEMM at rows 16/24/32/40. | bit-equal at rows 16/24/32/40 | - |
| Lean experts stage | A lean experts-stage kernel at rows 24/32/40 (drain 8 at row 40). | bit-equal at rows 24/40 (B300), row 32 (B200); GSM8K PASS | - |

Prior art: [Flash-Decoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html) (split-KV decode).

#### Checkpoint 3

| change | what it changes | numerics | prior art |
|---|---|---|---|
| MLA decode core, one-pass softmax | The correction warpgroup and its pipeline are removed; the softmax is one-pass online; the variable split is ported into the core. | tolerance (ratio <= 1.02); GSM8K PASS | [FlashMLA](https://github.com/deepseek-ai/FlashMLA) |
| MLA decode core, split PV MMA | The PV matrix multiply is split so it does not wait on P; the loop period moves toward the tensor-core floor. | bit-equal to the previous core (0 mismatches in 410,531,328 elements) | [FlashAttention-4 sm100](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/flash_fwd_sm100.py), [CUTLASS Blackwell FMHA](https://github.com/NVIDIA/cutlass/tree/main/examples/77_blackwell_fmha) |
| Mamba-state checkpoint copy-on-write | When the prefix-cache checkpoint column meets the initial-state column, the image skips the checkpoint export; this exports it through a fresh block or a copy-on-write of the hit block. More prefix hits for the linear-attention state. | tolerance (a different chunking of the same history); GSM8K PASS | [vllm#60888](https://github.com/vllm-project/vllm/pull/60888), [vllm#52789](https://github.com/vllm-project/vllm/pull/52789), [PagedAttention](https://arxiv.org/abs/2309.06180) |

#### Checkpoint 4

| change | what it changes | numerics | evidence / prior art |
|---|---|---|---|
| KDA decode, 2 CTAs per SM | The KDA (linear attention) decode kernel runs 2 CTAs per SM at the 5-token verify row; the gate warp folds into the compute warps; per-token fp order is unchanged. | bit-equal (37/37 + 14/14 cases) | kernel 14.26 -> 10.48 us. [Kimi Linear](https://arxiv.org/abs/2510.26692), [FLA KDA](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/kda) |
| KDA decode, 4-token row schedule | The 4-token verify row takes the same 2-CTA schedule. | bit-equal (5 cases) | kernel -3.28 us |

#### Checkpoint 5

| change | what it changes | numerics | prior art |
|---|---|---|---|
| 5 draft tokens per step (was 7) | Speculative decoding drafts 5 tokens; the per-K constants of every landed kernel follow. The recipe's synthetic accept length moves 3.84 -> 3.62, against us in every ITL read. | real accept length 1.5388 vs reference 1.5513 (band 0.042): PASS; GSM8K 0.9727 | [speculative decoding](https://arxiv.org/abs/2211.17192), [draft length](https://arxiv.org/abs/2405.19715) |
| MLA variable-split rules at the new batch shapes | Adds rule rows for the 5-draft batch shapes. | tolerance (as checkpoint 2) | [Flash-Decoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html) |
| 57 CUDA-graph capture sizes (was 31) | Fewer padded rows per graph replay. | graph shape only | [vLLM CUDA graphs](https://docs.vllm.ai/en/latest/design/cuda_graphs.html) |
| Shared-expert SM counts and lean-stage rows retuned | Shared-expert SMs per row count retuned for the new shapes; lean experts stage at rows 12/18/24/30. | as checkpoints 1-2 | checkpoint 1 |

#### Checkpoint 6 (no code change)

| change | what it changes | numerics |
|---|---|---|
| Per-point recipe at conc 32 | The checkpoint-5 code at conc 32 with the recipe's conc-32 fields: 64 sequences, no draft, graph sizes 1-64 and 128-8192, and `--gpu-memory-utilization 0.90`. At 0.92 the server died on a CUDA out-of-memory in two runs; at 0.90 the KV cache is 36.95 GiB instead of 42.3 GiB and the prefix-cache hit rate does not change (0.835-0.840 vs stock 0.838-0.839). | no kernel change; GSM8K 0.9682 |

### 4.3 Conc-32 landing set — measured, not yet pushed

Built on checkpoint 6. Measured at conc 32 as two parts. Each part was judged against the same 8 stock arms.

| part | change | what it changes | numerics | cumulative p90 ITL (ms) | vs stock |
|---|---|---|---|---:|---:|
| start | checkpoint 6 | - | - | 54.43 | 1.061x |
| 1 | MLA decode variable split at 23-32 rows | The split rule table gets rows for the conc-32 batch shapes. | tolerance | | |
| 1 | Context reorg by view | The kv_b projection reads a strided view of the cached latent instead of a concatenated copy, so the copy kernel is removed. No transferred mechanism. | bit-equal: kv_b output bits equal the image path on 5/5 B300 rows | 50.21 (part 1) | 1.150x |
| 2 | Context kv_b projection GEMM | A CuTe DSL GEMM with fused fp8 pack replaces the served projection at large token counts; the served path stays below the cutover. | bit-equal (7/7) | | |
| 2 | Sequence-parallel residual block | The attention-residual block (including o_proj) runs sequence-parallel between the reduce-scatter and the all-gather, at 1024 tokens and above. | tolerance: at 1024 tokens and above the fused GEMM + reduce-scatter path is off and the MoE tail GEMM runs at a different shape (full N 7168 on T/8 rows vs the image's N 896 shard), so the reduction layout differs; judge PASS 5/5 | | |
| 2 | KDA chunked prefill kernels | New chunked-prefill kernels for the linear-attention layers at 2048 tokens and above; workspace allocated after KV sizing. | bit-equal (44/44) | 48.32 (part 2) | **1.195x** |

- Official p90 ITL, two arms: 48.20 and 48.44 ms (mean 48.32). Gain vs stock -9.44 ms (band 2.03): GAIN.
- p90 TTFT guard: +541.8 ms vs stock, inside the stock A/A bound 648.5 ms. GSM8K strict 0.9674 (floor 0.90).
- Part 1 alone failed the TTFT guard (+2,903 ms). We trace that to which very long turns entered the sample (same-key
  ITL gain on every shared key), not to the kernels. The landing set as a whole passes.
- Status: the conc-1 and conc-8 reads of this set are running. It is pushed only after those read and the user approves.

Prior art: [Flash-Decoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html) (variable split);
[sequence parallelism, Korthikanti et al.](https://arxiv.org/abs/2205.05198) (residual block between reduce-scatter and
all-gather); [CUTLASS](https://github.com/NVIDIA/cutlass) (CuTe DSL GEMM); [FLA KDA](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/kda)
and [Kimi Linear](https://arxiv.org/abs/2510.26692) (chunked KDA prefill).

### 4.4 Tried and not landed (short list)

| change | result |
|---|---|
| Four kernel changes in the first set (KDA split, MLA core v0, three graph edges, prefetch tables) | Each won kernel-only; as a group the decode step classes regressed +0.27 to +0.96 ms. Removed from checkpoint 1. |
| Short-query context-attention fast path at conc 32 | Measured on top of the landing set: 51.43 ms vs 48.32 ms. Not landed; it returns only with a key-matched read. |

---

### Appendix 4: record of every number

Paths in the checkpoint repository are relative to its root at each tag. Paths marked "internal" are not published.

| number(s) | record |
|---|---|
| checkpoint 1-6 p90 ITL, controls, delta, verdict | `README.md` and `results/checkpoint-<n>/VERDICT.md` at tag `checkpoint-6-land-9-q1-c32` |
| paired gain, vs stock | paper table generator (`PAPER-TABLE.md`, rows per tag), notes 3-4 |
| GSM8K per checkpoint | `results/checkpoint-<n>/eval/`; checkpoint 1: `results/eval-gate-land-1b/` |
| per-change mechanism, numerics, evidence, prior art (checkpoints 1-6) | `CHANGESET.md` at tag `checkpoint-6-land-9-q1-c32` |
| checkpoint 6 recipe, 0.90 memory, KV 36.95 GiB, hit rate | `CHANGESET.md` "Checkpoint 6 additions"; `results/checkpoint-6/README.md` |
| conc-32 landing set 48.20 / 48.44, gain -9.44, band 2.03, TTFT +541.8 | internal: `requests/restart-set-1/out-c32b2r7ab/` (metrics.json, judge output); ledger `side/C32-SPRINT.md` row "part 2"; `side/C32-BUNDLE-2.md` rev 12-13 |
| part 1 50.21, TTFT +2,903 | internal: `requests/restart-set-1/out-c32b2vg/`; `side/C32-SPRINT.md` row "part 1" |
| GSM8K 0.9674 (landing set) | internal: `requests/restart-set-1/out-c32s1pc3/metrics.json` (`gsm8k_exact_match_strict` 0.96740) |
| per-member gates of the landing set (7/7, 5/5, 44/44, judge PASS) | internal: `side/C32-BUNDLE-2.md` §1 |
| fast path 51.43 ms | internal: `side/C32-SPRINT.md` row "tsnf"; `requests/restart-set-1/out-c32r8ab/` |
| context reorg bit-equal 5/5 | internal: `requests/restart-set-1/out-i03ctxg1b/` (`I03CTXG1 G_PASS`, bits_kv 5/5 rows, B300; CPU 12/12) |
| prior-art links | all links in §4.2 MLA rows and §4.3 return HTTP 200 (curl, 2026-10-26) |

---

## Appendix A (sections 1-3): records

Every [Rn] in sections 1-3 points to one record. Paths are relative to the program root; 'tag <name>' means the file at that git tag of the published checkpoint repository.

| ref | record | sha256 | value read |
|---|---|---|---|
| R1 | side/PAPER-TABLE.md | 1fb8207c | headline table rows conc 1 / 8 / 32 + the two caveat sentences (generated by tools/paper_table.py) |
| R2 | requests/restart-set-1/out-conc1pair/results/c1-c0/server_command.txt | d30b7a9a | conc 1 stock: server flags (last occurrence wins), identical in all 1 arms of the row |
| R3 | requests/restart-set-1/out-conc1pair/results/c1-c0/point/benchmark_command.txt | 298cf046 | conc 1 stock: client command |
| R4 | tag checkpoint-5-land-9/results/checkpoint-5/conc1/pair1-cp5c1r2/c1-land9/server_command.txt | cf2e9c44 | conc 1 ours: server flags (last occurrence wins), identical in all 2 arms of the row |
| R5 | tag checkpoint-5-land-9/results/checkpoint-5/conc1/pair1-cp5c1r2/c1-land9/point/benchmark_command.txt | b18ca862 | conc 1 ours: client command |
| R6 | requests/restart-set-1/out-cp4v2/results/c0/server_command.txt | 5b779c9f | conc 8 stock: server flags (last occurrence wins), identical in all 1 arms of the row |
| R7 | requests/restart-set-1/out-cp4v2/results/c0/point/benchmark_command.txt | 0d41b8a6 | conc 8 stock: client command |
| R8 | tag checkpoint-5-land-9/results/checkpoint-5/conc8/aba1-cp5abar22/land9/server_command.txt | 73458bd2 | conc 8 ours: server flags (last occurrence wins), identical in all 4 arms of the row |
| R9 | tag checkpoint-5-land-9/results/checkpoint-5/conc8/aba1-cp5abar22/land9/point/benchmark_command.txt | b7acbb02 | conc 8 ours: client command |
| R10 | tag checkpoint-6-land-9-q1-c32/results/checkpoint-6/conc32/stock-c32aa1/c32-s1/server_command.txt | 46f278be | conc 32 stock: server flags (last occurrence wins), identical in all 8 arms of the row |
| R11 | tag checkpoint-6-land-9-q1-c32/results/checkpoint-6/conc32/stock-c32aa1/c32-s1/point/benchmark_command.txt | fb277325 | conc 32 stock: client command |
| R12 | tag checkpoint-6-land-9-q1-c32/results/checkpoint-6/conc32/q1-c32q1c3/c32-l9q1-a/server_command.txt | 7208739e | conc 32 ours: server flags (last occurrence wins), identical in all 3 arms of the row |
| R13 | tag checkpoint-6-land-9-q1-c32/results/checkpoint-6/conc32/q1-c32q1c3/c32-l9q1-a/point/benchmark_command.txt | d6898bbf | conc 32 ours: client command |
| R14 | requests/restart-set-1/out-cp5c1r3/results/c1-land9/point/benchmark.log | fd26b4b0 | conc 1: lane start positions (sample_time) min 53 % max 53 %; first-dispatch start list identical across 3 runs |
| R15 | requests/restart-set-1/out-cp5c1r3/results/c1-land9/point/benchmark.log | fd26b4b0 | conc 1: 1 snapshot primers + 1 warmup request(s) per lane |
| R16 | requests/restart-set-1/out-cp5abar24/results/land9/point/benchmark.log | 8301ab59 | conc 8: lane start positions (sample_time) min 35 % max 74 %; first-dispatch start list identical across 5 runs |
| R17 | requests/restart-set-1/out-cp5abar24/results/land9/point/benchmark.log | 8301ab59 | conc 8: 7 snapshot primers + 1 warmup request(s) per lane |
| R18 | requests/restart-set-1/out-c32q1c2/results/c32-l9q1-c/point/benchmark.log | 8ddde8b8 | conc 32: lane start positions (sample_time) min 31 % max 75 %; first-dispatch start list identical across 11 runs |
| R19 | requests/restart-set-1/out-c32q1c2/results/c32-l9q1-c/point/benchmark.log | 8ddde8b8 | conc 32: 34 snapshot primers + 1 warmup request(s) per lane |
| R20 | side/sources/infx-api-kimik3-2026-10-25T2119Z.json | 06f3eb7d | public B300 rows of 2026-08-16: duration_seconds 3586-3629 |
| R21 | items/item-05-prefill-step-floor/judge-c32-6/poolread.py | 9b667aee | band rule: null bootstrap over control readings, 2000 draws, seed 0, 95 % half-width |
| R22 | tag checkpoint-6-land-9-q1-c32/results/checkpoint-6/conc32/eval-c32q1ev/c32-l9q1-ev/metrics.json | 6a950771 | GSM8K strict 0.9682, floor 0.9 |

- 2026-10-26T05:16Z [date -u] written by items/item-05-prefill-step-floor/stage0/readme_sec.py.
