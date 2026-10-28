# Change set of checkpoint 1: land-1b

## Composition

| layer | tree digest (`overlay_tree_sha256`) | files | what |
|---|---|---:|---|
| image | — | — | vLLM `49cb2bbf`, public |
| control (ctl) | `1d1f656755b271535602c1f99780e0c101eb3bf00ca00a187db297eaecf9ab2a` [m] | 85 | the measurement harness overlay at workload revision `3703b94`: the import hook, landed flags, log-only instrumentation, earlier inert kernels |
| land-1b (served) | `2afe24889cf47673bcff994438994341a6ce124497ffa536539da72a6c2a4def` [m] | 168 | ctl plus the land-1b bundle (86 files incl. `arm_env.json`) laid on top; this is `overlay/` plus `internal/overlay-redacted/` (`scripts/check_overlay_identity.py`) |

Every change to a vLLM file is an inserted line tagged `# PERFBOT <item>`; removing those lines gives the image file
byte for byte (`scripts/make_image_diff.py --check`; 27 of the 33 `overlay/vllm` files differ from the image, 6 are
verbatim copies). New code lives under `overlay/perfbot_kernels/`. Members are switched on by environment variables
(`overlay/arm_env.json`); code of a member that is off is inert (the call site returns the image path).

## Members that are ON (`overlay/arm_env.json`)

| member | switch | code | rows live | correctness class |
|---|---|---|---|---|
| L1 = X2 + S (items 06, 08) | `PERFBOT_EXP_GLUE=X2,S` | `perfbot_kernels/glue.py`, `experts_stage/`, `vllm/.../cutedsl/moe/prepared.py`, `front.py` | 24 / 32 / 40 | X2 bit-equal; S within an fp64 envelope (ratio <= 1.02) |
| tail D (item 02) | `PERFBOT_EXP_GLUE=D` | `perfbot_kernels/step_tail/`, `step_tail_hooks.py`, `vllm/v1/worker/gpu/spec_decode/rejection_sampler.py`, `sample/sampler.py`, `states.py`, `spec_decode/dflash/speculator.py` | all decode steps | bit-equal |
| deadgather G (item 06) | `PERFBOT_EXP_GLUE=G` | `glue.py` guard in the draft gather | all | bit-equal |
| draft chain (item 15) | `PERFBOT_EXP_DRAFT_CHAIN=1` | `perfbot_kernels/draft_chain/`, `draft_ar/`, `draft_query/`, `lse_combine/`, `vllm/models/kimi_k3/nvidia/dspark_mla.py`, `mla.py` | draft steps with 16 < M <= 64 padded rows | bit-equal |
| cs35 capture sizes (item 06) | second `--compilation-config` | server flag | graph shapes 7..35 step 7 | tolerance by nature (graph shapes) |
| prefill graph, form R (item 05) | `PERFBOT_EXP_PREFILL_GRAPH=1` | `vllm/v1/worker/gpu/model_runner.py` (`# PERFBOT P-07F`), `perfbot_kernels/prefill_restage/` | collider steps (a prefill chunk beside decoders) | equivalence judge: token ids and top-20 logprobs bit-equal |
| MLA context segments L1 (item 13) | `PERFBOT_EXP_MLA_CTX_SEGMENTS=auto` | `perfbot_kernels/mla_ctx_segments/`, `vllm/v1/attention/backends/mla/prefill/trtllm_ragged.py` | prefill chunks | fp64 envelope (<= 1.02) |

### L1 = X2 + S (items 06 and 08): MoE experts stage at 24-40 rows
What it changes: the routed-experts stage of the specialized decode at 17-40 decode rows. Mechanism: X2 turns the
JOIN step off at 9-40 rows; S gives the TP8 verify SharedDown 5 SMs at 17-24 rows and 4 at 33-40 rows, so the shared
expert runs beside the routed experts on the stage grid instead of after it (`glue.py` docstring; `experts_stage/__init__.py`).
Proxy evidence: the whole land-1 set read -0.937 ms per step at the d5 step class (40 rows, 83 % of steps), of which
L1 is about -0.14 ms per token at the mixed proxy [m, ATTRIBUTION reading, cited in `internal/README.md`].
Gate: `judge_split.py`, rows 16/24/32/40 x 3 layers with determinism replay: X2S bit-equal to the image on 12/12 layer
checks of the run; S changes 31-104 latent words at 20/40 rows within the fp64 ratio bound 1.02 [m, selftest table].

### Tail D (item 02): fused rejection-sampler tail, draft prep and next-step prologue
What it changes: the end of every decode step. Mechanism: one persistent CuTe DSL kernel replaces gather_draft_sampled,
the cu slice, the four launches of rejection_sample (stats, rejection, resample, insert) and get_num_sampled_and_rejected
(unit R); one launch replaces the four draft-prep launches (unit C) and one the 17 next-step prologue launches (unit E).
Fewer launches, same math (`step_tail/__init__.py`, `prep.py`, `prologue.py`). Declines to the image path on any
unexpected shape or dtype. Proxy evidence: priced 0.021 ms per p90 token [e, items ledger]. Gate: lane judges
p11c/p11d/p11e, 42/42 cases in each of three arms (off, D, whole set): tokens bit-equal; logsumexp floats within the
fp32 bound; `r_fp64.py` second truth PASS [m].

### Deadgather G (item 06)
What it changes: the draft-token gather. Mechanism: a guard that skips the gather of dead (rejected) draft tokens
(`glue.py`: "the dead draft-token gather guard"). Gate: `g_selftest.py`, r 1-8 x n 1-7, bit-equal, no launch [m].

### Draft chain (item 15): the DSpark draft layer outside the main graph
What it changes: three kernels in the served draft (DSpark MLA) at 17-64 padded draft rows. Mechanism: kernel 1 is a
two-shot Lamport all-reduce fused with the residual add and RMSNorm; kernel 2 replaces the fused q/kv RMSNorm, q_b,
W_UK bmm, q-concat with FP8 cache insert and the DCP q gather by one launch that writes a q mailbox the tokenspeed MLA
kernel reads; kernel 3 replaces the DCP combine and the W_UV bmm. 22 launches per draft layer become 8
(`draft_chain/__init__.py`). The draft weights, their precision and the draft computation are unchanged: the kernels
are bit-equal to the image path (InferenceX draft-as-shipped rule). Proxy evidence: d5 -0.120 ms per step on the
long-context decode proxy [m, LAND-2 table]. Gate: `draft_ar` 1342 checks, `draft_query` 449, `lse_combine` 2364,
mutants caught; 1319 responses identical to the image path (R-M2) [m]; the 8-rank integration self-test
(`draft_chain/selftest/run.py`) needs one 8-GPU node and was not run on the one-GPU keeper [m].

### cs35 capture sizes (item 06)
What it changes: the CUDA-graph capture sizes. Mechanism: sizes 7, 14, 21, 28, 35 are added so decode batches of
1-5 requests x 7 tokens (the p90 request runs with 3-5 decoders alive) hit a captured graph instead of padding up
to the next multiple of 8. Priced 0.029 ms per p90 token [e, items ledger]. Gate: a server flag; no numerics change
beyond graph shape. Start-up: PIECEWISE sizes 22 -> 27 and dspark FULL sizes 14 -> 15 (graph capture 19+30 s -> 25+45 s) [m].

### Prefill graph, form R (item 05): restage the prefill suffix
What it changes: the collider step, in which a prefill chunk of a re-prefilling request runs beside the decoders.
Mechanism: the prefill rows are restaged so the step keeps its graph (`# PERFBOT P-07F` lines in `model_runner.py`;
log lines `PERFBOT-P07F rows_suffix=… rows_full=… tokens_skipped=…` show it firing). Proxy evidence: colliders
(class m6) read -14 ms per collider step median in land-1 [m, ATTRIBUTION]. Gate: CPU tests 9+9 OK; equivalence judge
G1: token ids and top-20 logprobs bit-equal across an off and an on server (e2e) [m].

### MLA context segments L1 (item 13): adaptive segments for the chunked-context prefill fmha
What it changes: the context attention call of a prefill chunk with few new tokens over a long cached context.
Mechanism: one trtllm_ragged_attention_deepseek call per context chunk fills only ceil(q/256) x H CTAs (12-72 of 148
SMs for 183-1,408 new tokens); the request's keys are cut into S segments run as S rows of one call and merged by
log-sum-exp (Flash-Decoding), S chosen to minimise waves/S, S <= 8 (`mla_ctx_segments/__init__.py`). Proxy evidence:
0.20x/0.26x/0.58x/0.93x of the one-call time at q 183/512/1408/2100 on a 64k chunk [m, B200 judge]. Gate: fp64
envelope on all rows, 96/96 judge cases, mutants 9/9 [m].

## Members carried OFF (code present, inert)

KDA_LA (item 14, `kdasplit_la/`), MLACORE (item 03, `mla_core/`), edges T1/T3/T11 (item 02) and K48 prefetch tables
(item 09). Reason [m, ATTRIBUTION]: with these on as a group (land-1), step classes d1-d4 regressed by
+0.27 / +0.73 / +0.96 / +0.42 ms per step and d5 gained only -0.094; kernel-only each kernel wins, so the regression is
an integration effect, member unknown [u]. land-1b (this set) is land-1 with the four switches off; its code is byte-equal to land-1.

## Evidence ledger

| reading | ctl-0 | ctl-end | land-1 | land-1b | unit |
|---|---:|---:|---:|---:|---|
| full-decode p90, mixed proxy (5 long-context decoders + one re-prefilling lane) | 6.737 | 6.743 | -0.262 | **-0.244** | ms per token [m] |
| d5 step median (40 rows) | 20.451 | 20.468 | 19.483 | 19.577 | ms per step [m] |
| d1 / d2 / d3 / d4 step median | 10.662 / 12.530 / 15.542 / 17.397 | 10.679 / 12.584 / 15.548 / 17.467 | 10.597 / 13.148 / 15.899 / 17.790 | 10.323 / 12.415 / 14.938 / 17.373 | ms per step [m] |

The proxy reads a pure-decode gain at 0.55-0.6x (decode-only steps hold about 60 % of wall time in the proxy) [m].
The checkpoint number is the real InferenceX point, not the proxy: `results/checkpoint-1/` when it lands.

### Correctness gate at the real point (`results/eval-gate-land-1b/`, `eval_only`, real acceptance, fast, conc 8) [m]

| arm | p90 ITL ms | full-decode p90 ms | out tok/s/GPU | accept length | GSM8K strict / flexible | failed requests | gates |
|---|---:|---:|---:|---:|---:|---:|---|
| l1-eval-off (ctl) | 19.86 | 20.10 | 25.92 | 1.570 | 0.9704 / 0.9697 | 0.0070 | none |
| l1-eval-on (land-1b) | 17.67 | 18.14 | 25.80 | 1.523 | 0.9697 / 0.9689 | 0.0071 | none |

GSM8K: 1319 questions, 5-shot, concurrency 16, floor 0.90 (InferenceX `thresholds.yaml` default). Accept length is
`1 + accepted / drafts` from vLLM's spec-decode counters over the point. The eval_only numbers are a gate, not the
score: InferenceX scores the throughput arm with synthetic acceptance at 3.84.

## Rules this set keeps

- No model-architecture change, no reduced FLOPs, no precision change of target or draft (InferenceX review checklist).
- The InferenceX client, recipe flags, acceptance methodology and ITL definition are untouched; the overlay changes server code only.
- The overlay is an engine patch under InferenceX rules ("the pinned image must run as shipped"): see `submission/README.md`.

## Checkpoint 2 additions: land-2 = land-1b + L6 + pin + 08L

Files that differ between the land-1b and land-2 bundles (`diff -rq`; membership per the land-2 composition record, LAND-2.md §1b'):

| member | switch | files | class |
|---|---|---|---|
| L6 variable split (item 01) | `PERFBOT_EXP_B35=F,O` | `perfbot_kernels/mla_varsplit/` (new), `vllm/models/kimi_k3/nvidia/specialized_decode/cutedsl/mla/{prepared.py,dcp.py}`, `.../runtime/providers.py` | fp64 tolerance 1.02; eval PASS (`results/checkpoint-2/eval/`) |
| pin (item 17) | `PERFBOT_EXP_FRONTPIN=1` | `perfbot_kernels/moe_front/` (new), `.../cutedsl/moe/front.py` | bit-equal at rows 16/24/32/40 |
| 08L lean experts stage (item 08) | `PERFBOT_EXP_STAGE=L` | `perfbot_kernels/experts_stage/{__init__.py,experts.py,experts_runtime.py,test_cpu.py}` (replaced) | bit-equal to the stage at rows 24/40 on B300, row 32 on B200 only; eval PASS |

`arm_env.json` adds the three switches; `server_extra_args` is unchanged. Open items: `results/checkpoint-2/README.md`.

## Checkpoint 3 additions: land-5 = land-2 + E2b + MLA-LOOP + CKPT-COW

Files that differ between the land-2 and land-5 bundles (`diff -rq`):

| member | switch | files | mechanism | gate | prior art |
|---|---|---|---|---|---|
| E2b (item 03), in land-3 | `PERFBOT_EXP_MLACORE=1` | `perfbot_kernels/mla_core/{__init__.py,kernel.py}`, `.../cutedsl/mla/prepared.py` (the var-split args move above the fork) | MLA decode core fork: the correction warpgroup and the p_cor pipeline are deleted; one-pass online softmax; the variable-split port of L6 | fp64 tolerance 1.02 (judge v5.3); timed and frozen cores bit-equal on the timed path, splits=None (E2BBITS EQUAL); eval PASS | FlashMLA, one-pass online softmax with split-KV and LSE combine: https://github.com/deepseek-ai/FlashMLA |
| MLA-LOOP (item 03), in land-4 | `PERFBOT_MLA_CORE_SPLITMMA=1` (= code default) | `perfbot_kernels/mla_core/{__init__.py,kernel.py}` | the PV MMA of the MLA core is split so that it does not wait on P; the loop period moves toward the tensor floor | bit-exact vs E2b: 0 mismatches over 410,531,328 elements, graph replay B1-B5 | FlashAttention-4 sm100 MMA/softmax overlap: https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/flash_fwd_sm100.py ; CUTLASS 77 Blackwell MLA: https://github.com/NVIDIA/cutlass/tree/main/examples/77_blackwell_fmha |
| CKPT-COW (item 04), in land-5 | `PERFBOT_EXP_CKPT_COW=1` | `vllm/v1/kv_cache_interface.py`, `vllm/v1/core/single_type_kv_cache_manager.py`, `vllm/v1/core/sched/scheduler.py`, `vllm/model_executor/layers/mamba/checkpoint.py` | mamba prefix-cache checkpoint copy-on-write: when the checkpoint column collides with the initial-state column (cases EQ, EQ_ALIGNED, LT), the base skips the checkpoint export (it needs checkpoint column > initial-state column). CKPT-COW exports it through a fresh block (LT) or a copy-on-write of the hit block (EQ, EQ_ALIGNED). All other rows take the served path | tolerance (a different chunking of the same history); bar-1 PASS on B200; eval PASS | vllm#60888 (checkpoint at the initial-state column, CoW slot): https://github.com/vllm-project/vllm/pull/60888 ; vllm#52789 (mamba prefix caching, align mode): https://github.com/vllm-project/vllm/pull/52789 ; PagedAttention copy-on-write of shared KV blocks: https://arxiv.org/abs/2309.06180 |

- `arm_env.json` adds `PERFBOT_EXP_MLACORE=1`, `PERFBOT_MLA_CORE_SPLITMMA=1` and `PERFBOT_EXP_CKPT_COW=1`.
- `perfbot_kernels/glue.py` (from land-3) gains one liveness mark per expanded `PERFBOT_EXP_GLUE` alias item. It prints only, and the computation does not change.
- **Overlay rule from checkpoint 3:** an overlay file is the image file plus tagged additions plus recorded tagged replacements, each one listed.
  - CKPT-COW replaces 4 image lines in 2 files. `patches/image-replaced-lines.json` records them as exact text.
  - `scripts/make_image_diff.py --check` reverses them before the strip and reproduces the image hashes.
  - With the switch off, the replaced code computes the base condition and adds no block.
  - Owner item 04 makes CKPT-COW insert-only in a later set; then the record is empty again.
- Rule for member entry from checkpoint 3: a member's source commit passes the driver suite, including `test_overlay_base`, or carries a
  recorded replaced-lines entry.
- Open items: `results/checkpoint-3/README.md`.

## Checkpoint 4 additions: land-8 = land-5 + La + B4 occ2

Files that differ between the land-5 and land-8 bundles: `arm_env.json`, `perfbot_kernels/kdasplit_la/__init__.py`,
`perfbot_kernels/kdasplit_la/prepared.py` (bundle sums; every other member's bytes equal land-5).

| member | switch | files | mechanism | gate | prior art |
|---|---|---|---|---|---|
| La (items 14, 04) | `PERFBOT_EXP_KDA_LA=1` (read at import) | `perfbot_kernels/kdasplit_la/{__init__.py,kernel.py,prepared.py}` | KdaSplit lane L candidate `qbatch` as the served PreparedKda: B5 runs at 2 CTAs/SM (schedule row 5:1.8.2.4), the gate warp folds into the four compute warps at the occupancy-3 rows, and at T > 1 each token's q butterfly runs one token late. The fp order per token is unchanged | bit-equal to the 49cb2bbf incumbent, 37/37 cases (B200) and 14/14 (item 14 check); B5 14.26 -> 10.48 us kernel-only | vLLM CuTe DSL KDA decode (the base): https://github.com/vllm-project/vllm/blob/49cb2bbf71cfd36ee5aa33d1e184b1cddff55294/vllm/models/kimi_k3/nvidia/specialized_decode/cutedsl/kda/kernel.py ; FlashInfer multi-token KDA decode: https://raw.githubusercontent.com/flashinfer-ai/flashinfer/main/flashinfer/kda_kernels/fused_kda_decode_multitoken.py ; KDA: Kimi Linear https://arxiv.org/abs/2510.26692 , Gated Delta Networks https://arxiv.org/abs/2412.06464 , FLA https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/kda |
| B4 occ2 (item 04) | `PERFBOT_EXP_KDA_B4OCC=1` | `perfbot_kernels/kdasplit_la/prepared.py` (6 tagged lines) | the 12-head B4 verify row takes B5's schedule 1.8.2.4 (2 CTAs/SM, cluster 4) | on/off bit-equal on 5 cases; B4 kernel-only gain 3.28 us vs bar 0.55 | same kernel and links as La |

- `arm_env.json` adds `PERFBOT_EXP_KDA_LA=1` and `PERFBOT_EXP_KDA_B4OCC=1` (keys now sorted).
- Each switch prints one liveness line on its path only (`PERFBOT-SWITCH kda_la=on`, `kda_b4occ=on B4=1.8.2.4`).
  Every land-8 arm of record shows both, 8 ranks each; every land-5 arm shows neither.
- Harness overlay changes since checkpoint 3 (workload `9e593b8` -> `48c6a26`; in both arms, not members of land-8):
  - ATKEY, `vllm/model_executor/warmup/flashinfer_autotune_cache.py` (2 tagged lines) + `perfbot_kernels/atkey/`: the FlashInfer
    autotune cache key is a narrow key, so all arms load one tuning of record (`a4b725cd`). Its tactics are byte-identical to the
    checkpoint-3 key `7fa8e43d` (dev-46, `GAP-M1` §2.3).
  - CGPROFILE, `vllm/v1/worker/gpu_worker.py` (2 tagged lines): vLLM's CUDA-graph memory estimate applies only when
    `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` is set.
  - LOADTIME, `vllm/models/kimi_k3/nvidia/model.py` (2 tagged lines) + `perfbot_kernels/loadtime/`: a weight-load trace, off unless
    `PERFBOT_EXP_LOADTIME=1`.
  - `pretrace/`, `nullarm/`, `attn_res_one/selftest.py`, `vllm_base.json`: harness and test code, not on the serving path.
- Redacted: `perfbot_kernels/loadtime/test_cpu.py` (internal fixture path) moves to `internal/overlay-redacted/` (now 12 files).
- Open items: `results/checkpoint-4/README.md`.

## Checkpoint 5 additions: land-9 r2 = land-8 + K5 + R28 + PAD + S moves / lean 12

Served files that differ between land-8 and land-9 (`git diff` of `overlay/`, 12 files): `arm_env.json`,
`perfbot_kernels/k5q.py` (new), `perfbot_kernels/graphkit.py` (new), `draft_chain/__init__.py`, `experts_stage/__init__.py`,
`mla_varsplit/{__init__,kernel}.py`, `step_tail/prologue.py`, `step_tail_hooks.py`, `cutedsl/mla/prepared.py`,
`cutedsl/moe/{front,prepared}.py`.

| member | switch | mechanism | prior art |
|---|---|---|---|
| K5 (item 01) | `--speculative-config` `num_speculative_tokens` 5 (was 7) | 5 draft tokens per step; the per-K constant tables of the existing members come from `k5q.consts(5)` | https://arxiv.org/abs/2211.17192 ; https://arxiv.org/abs/2405.19715 |
| R28 (item 01) | `mla_varsplit` rules | MLA var-split rule rows at the K5 batch shapes, `RULES[3]` (16, 8) | https://crfm.stanford.edu/2023/10/12/flashdecoding.html |
| PAD (item 05) | `--compilation-config` `cudagraph_capture_sizes` | 57 capture sizes (max 8192) instead of 31: fewer padded rows per graph replay | https://docs.vllm.ai/en/latest/design/cuda_graphs.html |
| S moves + lean 12 (item 01) | `PERFBOT_EXP_GLUE` S, `PERFBOT_EXP_STAGE=L` (unchanged switches) | SharedDown SMs per row count 9-12:8, 13-18:6, 19-24:5, 25-30:5; lean experts-stage rows {12, 18, 24, 30} | checkpoint 1 L1 = X2 + S (above) |

- The throughput recipe's synthetic accept length is 3.62 (was 3.84) with K5; the eval arm measures real acceptance.
- Autotune key: land-9 arms load the K5 tuning of record `6b30cb81`; land-8 arms load `a4b725cd` (ATKEY, checkpoint 4).
- Redacted: the same 12 files as checkpoint 4 under `internal/overlay-redacted/`.
- Open items: `results/checkpoint-5/README.md`.

## Checkpoint 6 additions: per-point recipe (conc 1, 8, 32); no code change

No served code changes. The overlay code is land-9 r2 at every point. Per point, the switch file and the server
arguments change (InferenceX per-point fields; table in `results/checkpoint-6/README.md`):

| point | switch file | server arguments |
|---|---|---|
| conc 1, conc 8 | `overlay/arm_env.json` (1242c82a, checkpoint 5) | checkpoint 5 command, byte for byte |
| conc 32 | `results/checkpoint-6/served-conc32/arm_env.json` (22877606) = the same switches without `PERFBOT_SERVER_EXTRA_ARGS` | recipe `override_c32` (64 sequences, captures 1..64 + 128..8192, no draft) + `--gpu-memory-utilization 0.90` |

- Why 0.90: at 0.92 the land-9 conc 32 server died on a fatal CUDA out-of-memory in two runs (3.27-3.28 GiB MoE
  workspace request, 2.84-2.99 GiB free). The KV cache is 36.95 GiB instead of 42.3; the prefix-cache hit rate at conc 32
  does not change (q1 0.835-0.840 vs stock 0.838-0.839).
- Conc 32 served tree `d77685c1…` = the default tree with 3 files replaced (`scripts/check_overlay_identity.py --point conc32`):
  the switch file and the log-only STEPDEC record `perfbot_kernels/reqlog/__init__.py` (+ its CPU test under
  `internal/overlay-redacted-conc32/`), which adds two fields `pn`, `pnmax` per step record.
- Autotune key: the conc 32 arms (stock and land-9 q1) load the tuning record `bdcc573b` (ATKEY `hit=35 tuned=0`);
  the stock reference arm `c32-stock` tuned cold.
- The conc 32 MLA varsplit rule rows (23-32 rows) are not in this checkpoint: HELD (`results/checkpoint-6/README.md`).
