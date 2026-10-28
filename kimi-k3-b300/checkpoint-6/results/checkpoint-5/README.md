# checkpoint-5: land-9 r2 (land-8 + K5 + R28 + PAD + item 01 S moves), pooled conc 8 ABAs + pooled conc 1 pairs + eval

Status: COMPLETE 2026-10-23 (every slot filled from landed runs read by the frozen judge). Not published: the coordinator
pushes after PUSH GO. Tags: [m] measured,
[d] arithmetic (by a tool), [c] code read, [u] unverified.

User rule (USER DIRECTIVE 2026-10-23T07:34Z, the push gate of this checkpoint): push when the solid rule holds on the
frozen judge:
(a) the conc 8 pool clears its band on official AND consistent p90 ITL;
(b) the conc 1 pool over >= 2 pairs clears its band on both;
(c) eval PASS, the TTFT and tok/s/GPU guards, error parity and peak memory are stated;
(d) the reads of item 02 and item 03 are byte-equal.
A miss on any slot keeps the cap rule (<= 4 conc 8 ABAs, <= 3 conc 1 pairs) and reports the band table.
tok/s/user = 1000 / official p90 ITL.

## The set: land-9 r2

- Workload revision (pin) of every run below: `bd6a5f454ef90b6cf8f13d28373e6a090dae42e5` [m `workload-revision.json`].
- Bundle land-9-r2: 111 files; harness tree 8025b6c8... (`conc8/*/bundles.json`); eval bundle land-9-eval-r2,
  harness tree b654772e... (`eval/bundles.json`). The two differ only in the speculative-config sampler fields.
- Served tree: `overlay/` + `internal/overlay-redacted/` = 205 files, `overlay_tree_sha256`
  32533b8745eca301a572dfa86de42cc59419893b562dae19928d306b6e687aeb [m `scripts/check_overlay_identity.py` PASS;
  equal to `metrics.json` of every land9 / c1-land9 arm]. The eval arm serves 1153067e (eval bundle; sampler fields).
- vs land-8, 12 served files change [m git diff]: `arm_env.json`, `perfbot_kernels/k5q.py` (new), `graphkit.py` (new),
  `draft_chain/__init__.py`, `experts_stage/__init__.py`, `mla_varsplit/{__init__,kernel}.py`, `step_tail/prologue.py`,
  `step_tail_hooks.py`, `cutedsl/mla/prepared.py`, `cutedsl/moe/{front,prepared}.py`.

| lever | what | where |
|---|---|---|
| K5 | 5 draft tokens per step instead of 7 (`num_speculative_tokens` 7 -> 5); per-K constant tables in `k5q.consts(5)`; synthetic acceptance length of the throughput recipe 3.84 -> 3.62 (the measured real-acceptance ratio, eval below) | `arm_env.json`, `k5q.py`, consumers below |
| R28 (B5 rules) | MLA var-split rule rows for the K5 batch shapes, `RULES[3]` (16, 8) | `mla_varsplit/` |
| PAD | 57 CUDA-graph capture sizes (max 8192) instead of 31: fewer padded rows per graph replay | `arm_env.json` `--compilation-config` |
| S moves + lean 12 (item 01) | SharedDown SM split per row count `S_TAIL` 9-12:8, 13-18:6, 19-24:5, 25-30:5; lean experts-stage rows {12, 18, 24, 30} | `moe/front.py`, `experts_stage/` |

Prior art (each fetched HTTP 200 through the egress host at 2026-10-23T07:39Z [m]; titles read [m]):
- Draft length choice: Leviathan et al., "Fast Inference from Transformers via Speculative Decoding",
  https://arxiv.org/abs/2211.17192 ; SpecDec++, "Boosting Speculative Decoding via Adaptive Candidate Lengths",
  https://arxiv.org/abs/2405.19715 .
- Capture-size padding: vLLM CUDA graphs design, https://docs.vllm.ai/en/latest/design/cuda_graphs.html ;
  https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/ .
- Split-KV decode rows (R28): Flash-Decoding, https://crfm.stanford.edu/2023/10/12/flashdecoding.html .
- S / lean rows: the members of checkpoints 1-4 (`CHANGESET.md`); only their tables change.

## Liveness (read first; an arm without it is VOID)

Every arm below [m, `judge/cpread-*.txt` LIVE rows]: `PERFBOT-SWITCH-CHECK set=11 live=11 dead=-`; ATKEY
`hit=35 tuned=0 intact=yes`, key a4b725cd on the land-8 control arms and 6b30cb81 (K5 tuning of record) on the land-9 arms;
`gate_failures []` -> VALID. `scripts/ingest_results.py` writes no LIVENESS MISSING line for any arm.

## The judge (frozen)

Item 03 `cpread.py` ab5260c6 (per run: cp4judge dfe1ad11, c1pair a46dca57, c1aba f8f9fc15, split 3c7cb5d6, evalread 55fdad48)
and `poolgen.py` 5f7dfa15 (pooled band, method poolread 9b667aee). Band: conc 8 = null bootstrap over the control rows;
conc 1 = max(sampling, control cross-run A/A). Each output is in `judge/` with its sha256 prefix; item 02 and item 03 each
ran every read and posted the sha (slot (d)).

| read | file | sha256 | item 03 sha | equal |
|---|---|---|---|---|
| cp5abar22 | `judge/cpread-cp5abar22.txt` | 31be4bdf | 31be4bdf | yes [m] |
| cp7aba4b (land9 arms) | `judge/cpread-cp7aba4b.txt` | 78bc1169 | 78bc1169 | yes [m] |
| cp5abar23 | `judge/cpread-cp5abar23.txt` | 6d1717a4 | 6d1717a4 | yes [m] |
| cp5abar24 | `judge/cpread-cp5abar24.txt` | 9543b298 | 9543b298 | yes [m] |
| cp5c1r2 | `judge/cpread-cp5c1r2.txt` | 26bd8397 | 26bd8397 | yes [m] |
| cp5c1r3 | `judge/cpread-cp5c1r3.txt` | 99da41b0 | 99da41b0 | yes [m] |
| cp5evalr2b | `judge/cpread-cp5evalr2b.txt` | 51e175cf | 51e175cf | yes [m] |
| pool c8 x3 + c1 x2 | `judge/pool-c8x3-c1x2.txt` | 56845843 | 56845843 | yes [m] |
| pool c8 x4 + c1 x2 (of record) | `judge/pool-c8x4-c1x2.txt` | 75f90ec3 | 75f90ec3 | yes [m] |

## The number: conc 8, pooled same-request ABAs (control land-8 = checkpoint 4 content)

Runs: `conc8/aba1-cp5abar22`, `conc8/aba2-cp7aba4b` (land9 + its two land-8 controls; the request's other arms are not
checkpoint 5), `conc8/aba3-cp5abar23`, `conc8/aba4-cp5abar24`. Arm order ctl-land8 / land9 / ctl8-end.

| ABA | ctl-land8 | land9 | ctl8-end | official, consistent p90 ITL ms [m] |
|---|---|---|---|---|
| 1 cp5abar22 | 5.15 / 6.062 | 4.82 / 5.621 | 4.95 / 6.067 | |
| 2 cp7aba4b | 6.36 / 6.642 | 4.74 / 5.309 | 5.32 / 6.210 | |
| 3 cp5abar23 | 5.08 / 6.181 | 4.77 / 5.449 | 5.64 / 6.235 | |
| 4 cp5abar24 | 5.39 / 6.141 | 4.94 / 5.944 | 5.19 / 5.705 | |

| pool | statistic | controls n | land9 n | land9 mean | ctl mean | gain | band | verdict |
|---|---|---|---|---|---|---|---|---|
| 3 ABA (56845843) | official | 6 | 3 | 4.777 | 5.417 | -0.640 | 0.653 | INSIDE (short 0.013) |
| 3 ABA (56845843) | consistent | 6 | 3 | 5.460 | 6.233 | -0.773 | 0.272 | GAIN |
| **4 ABA (75f90ec3, of record)** | official | 8 | 4 | 4.818 | 5.385 | -0.567 | 0.510 | **GAIN** |
| **4 ABA (75f90ec3, of record)** | consistent | 8 | 4 | 5.581 | 6.155 | -0.575 | 0.293 | **GAIN** |

Margin note [d]: the official statistic clears its band by 0.057 ms only (0.567 vs 0.510); the consistent one by 0.282.
In cp5abar24 alone the official read is INSIDE (-0.350, bar 0.660) and the consistent read is +0.021 INSIDE; the pool of
record is the 4-ABA pool. Land9 official p90 per ABA: 4.82 / 4.74 / 4.77 / 4.94 ms.

Control spread note [m, side read of cp7aba4b]: the control rows spread 4.95-6.36 ms official at conc 8. Item 03 classes
the drift as LOAD_REALIZATION (conf 0.8): the closed-loop client gives about 1 ms of official p90 spread at conc 8; the
server is not the cause. The consistent statistic has a band of 0.272 on the same rows.

## The number: conc 1, pooled pairs (slot (b) CLEAR)

| pair | c1-land8 | c1-land9 | official delta (bar 0.171) | consistent delta (bar 0.062) |
|---|---|---|---|---|
| 1 cp5c1r2 | 3.027 / 3.106 | 2.837 / 2.898 | -0.190 GAIN | -0.208 GAIN |
| 2 cp5c1r3 | 3.010 / 3.094 | 2.852 / 2.881 | -0.158 NO_REGRESSION | -0.212 GAIN |
| pool (56845843) official | ctl 3.019 | land9 2.845 | -0.174, band 0.125 (sampling 0.125, A/A 0.017) | **GAIN** |
| pool (56845843) consistent | ctl 3.100 | land9 2.890 | -0.210, band 0.098 (sampling 0.098, A/A 0.013) | **GAIN** |

Pairs used 2 of cap 3.

## Two-row table (README template, user rule)

| conc | official p90 ITL (ms) | tok/s/user (= 1000 / ITL) | ratio vs stock | ratio vs previous checkpoint (same request, land-8 controls) |
|---|---|---|---|---|
| 8 | 4.818 (mean of 4 land9 arms) | 207.6 | 1.368x (vs 6.59) | 1.118x (ctl mean 5.385) |
| 1 | 2.845 (mean of 2 c1-land9 arms) | 351.6 | 1.083x (vs 3.080; cross-pin) | 1.061x (ctl mean 3.019) |

## Guards, error parity, peak memory (slot (c))

Bound per guard = max(control spread, this run's A/A); land9 - mean(ctl) [m `judge/cpread-*.txt` GUARD rows]:

| run | TTFT p90 land9 - ctl (bound) | out tok/s/GPU land9 - ctl (bound) | failed fraction ctl-land8 / land9 / ctl8-end | peak device memory ctl / land9 |
|---|---|---|---|---|
| cp5abar22 | -66.2 ms (34.3) OK | +0.135 (0.095) OK | 0.0000 / 0.0042 / 0.0042 | 262.93 / 267.65 GiB |
| cp7aba4b | -65.8 ms (127.9) OK | +0.110 (0.399) OK | 0.0000 / 0.0042 / 0.0042 | 262.93 / 267.65 GiB |
| cp5abar23 | -68.1 ms (50.2) OK | +0.144 (0.037) OK | 0.0021 / 0.0021 / 0.0021 | 262.93 / 267.65 GiB |
| cp5abar24 | -56.7 ms (18.4) OK | +0.461 (0.049) OK | 0.0000 / 0.0000 / 0.0043 | 262.93 / 267.65 GiB |

- Error rule (coordinator 2026-10-23T04:49Z): failed fraction <= 10 % and land9 not above its run's controls. Each run:
  land9 count = the larger control count, and land9 fraction <= the larger control fraction [m `metrics.json`
  failed_request_fraction: 2/476 vs 0, 2/472 (aba 1); 2/476 vs 0, 2/471 (aba 2); 1/476 vs 1/471, 1/469 (aba 3); 0 vs 0, 2/463 (aba 4)].
  conc 1: 1/156 failed on every arm of both pairs (0.0064) [m].
- Peak memory: land9 267.65 GiB, +4.72 GiB over land-8 [m `peak dev` rows]; cause not split [u]; no OOM on any arm.
- conc 1 TTFT (no guard bar: a pair has one control) [m]: c1-land9 1556.7 / 1569.2 ms vs c1-land8 1525.6 / 1534.5 ms
  (+31 / +35 ms, +2.1 %); out tok/s/GPU 17.593 / 17.587 vs 17.813 / 17.770 (-1.1 % [d]; cause not split [u]). NOTE, not a gate.

## Eval of record (ev-land9, eval_only, real acceptance; slot (c)) [m `judge/cpread-cp5evalr2b.txt`]

- GSM8K strict 0.9727 (gate >= 0.90), gate_failures [].
- Real acceptance length 1.5388 vs the K5 reference 1.5513 (-0.0125, band 0.042) and vs stock 1.5898 (-0.0510, cap 0.06): PASS.
  Per position 0.2687 / 0.1277 / 0.0739 / 0.0435 / 0.0249.
- Liveness: 23 items seen, none missing. RESULT PASS.
- Disclosure: K5 lowers acceptance per step by design (5 draft tokens, not 7). The throughput recipe's synthetic acceptance
  length moves with it (3.84 -> 3.62, `metrics.json` accept_length 3.618-3.619 on land9 arms vs 3.839-3.841 on controls).
  This works against land9 in every ITL read above.

## Verdict

SOLID-AB conc8 CLEAR conc1 CLEAR (poolgen 5f7dfa15, 75f90ec3): checkpoint-5 = land-9 r2. conc 8 pooled 4 ABAs official -0.567
ms (band 0.510), consistent -0.575 (band 0.293); conc 1 pooled 2 pairs -0.174 (0.125) / -0.210 (0.098); eval PASS
(GSM8K 0.9727, AL 1.5388); guards OK, error parity holds. Official conc 8 4.818 ms (207.6 tok/s/user, 1.368x vs stock
6.59); conc 1 2.845 ms (351.6 tok/s/user, 1.083x vs 3.080).

## Publish steps (none run until the coordinator posts PUSH GO; push pre-authorized by USER DIRECTIVE 07:34Z on a clear)

1. At cp5abar24 exit: `stage_ingest.sh cp5abar24 conc8/aba4-cp5abar24 ctl-land8,land9,ctl8-end land9,ctl-land8,ctl8-end`;
   cpread + 4-ABA poolgen into `judge/`; fill every slot; compare shas with item 03. DONE 2026-10-23T08:45Z.
2. Checks: `check_secrets.py`, `check_overlay_identity.py` (32533b87), `make_image_diff.py --check`.
3. Commit on branch `checkpoint-5-land-9-staging` (base a04a41e = origin/main; the local W9 commit 9132cff is NOT in it).
4. Annotated tag `checkpoint-5-land-9` (message = the Verdict line). The coordinator pushes branch + tag to
   alanw/k3_inferenceX main through the egress host.

## Appendix after the push (single-arm lines; not a gate; the verdict above is unchanged)

Two single-arm runs on the checkpoint-5 pins (measure-2 bd6a5f45, served tree 32533b87, ATKEY a4b725cd / 6b30cb81 intact,
all arms VALID) were added to the same frozen pool (poolgen 5f7dfa15; a single arm = one row, same weight as an ABA arm;
band = null bootstrap over control rows):

| run | arm | official p90 ITL ms | consistent p90 ms | dir |
|---|---|---:|---:|---|
| cp5ctl1 | ctl-land8 | 5.13 | 6.074 | `conc8/single1-cp5ctl1` |
| cp5l91 | land9 | 4.94 | 5.642 | `conc8/single2-cp5l91` |

| pool | rows ctl / land9 | official gain (band) | consistent gain (band) | SOLID-AB | file |
|---|---|---|---|---|---|
| of record (4 ABAs) | 8 / 4 | -0.567 (0.510) | -0.575 (0.293) | c8 CLEAR, c1 CLEAR | `judge/pool-c8x4-c1x2.txt` |
| + cp5ctl1 | 9 / 4 | -0.539 (0.478) | -0.565 (0.273) | c8 CLEAR, c1 CLEAR | `judge/pool-c8x5-c1x2.txt` |
| + cp5ctl1 + cp5l91 | 9 / 5 | -0.515 (0.446) | -0.553 (0.252) | c8 CLEAR, c1 CLEAR | `judge/pool-c8x6-c1x2.txt` |

Both added lines keep GAIN on both statistics with a larger margin over the band (official 0.057 -> 0.069). conc 1 is
unchanged (same 2 pairs). Pooled land9 official over 5 rows: mean 4.842 ms.
