# K3 InferenceX checkpoints: Kimi-K3 on 8x B300, upstream vLLM

Reproducible landing sets over the InferenceX Kimi-K3 vLLM configuration (`kimik3-fp4-b300-vllm-agentic-dspark`).
One git tag per checkpoint. Each tag holds the server code change (an overlay on the public vLLM image), the exact
server and client commands, the measured results, and the InferenceX submission files.

Provenance tags in this repository: `[m]` = copied from a measurement file in `results/`; `[u]` = stated by a document, not re-measured.

## Checkpoint 6 (land-9, per-point recipe): the numbers at conc 1, 8 and 32

| point: agentic coding, 8x B300, 1200 s fast protocol | official p90 ITL (ms) | tok/s/user = 1000 / p90 ITL | full-decode p90 ITL (ms) | source |
|---|---:|---:|---:|---|
| conc 32, checkpoint 6: land-9 q1 (no draft, `--gpu-memory-utilization 0.90`), mean of 3 arms | **54.43** [m] | **18.4** [m] | **55.83** [m] | `results/checkpoint-6/conc32/q1-*/`; tag `checkpoint-6-land-9-q1-c32` |
| conc 32, public image (stock recipe), mean of 8 arms | 57.76 [m] | 17.3 [m] | 59.18 [m] | `results/checkpoint-6/conc32/stock-*/` |
| conc 8, checkpoint 6 = checkpoint 5 (same server) | 4.82 [m] | 207.6 [m] | 5.58 [m] | `results/checkpoint-5/conc8/` |
| conc 1, checkpoint 6 = checkpoint 5 (same server) | 2.84 [m] | 351.6 [m] | 2.89 [m] | `results/checkpoint-5/conc1/` |

Checkpoint 6 verdict: **GAIN vs the unmodified image at conc 32 (SOLID-AB conc32 CLEAR NO_REGRESS); conc 1 and conc 8 unchanged from checkpoint 5; eval PASS.**
- conc 32 (3 land-9 q1 arms vs 8 stock arms, frozen judge `judge-c32-4`): -3.331 ms official (band 1.847) and -3.358 ms
  full-decode (band 2.460): GAIN on both. 57.761 / 54.430 = **1.061x**.
- Key-matched INFO (not a verdict): one very long turn is sent in some runs and not in others (4 of 8 stock arms, 0 of
  3 q1 arms). Against the 4 stock arms without it: -2.395 ms official (band 1.574), **1.044x**.
- Guard: TTFT p90 +391 ms, inside the stock A/A bound 648.5 ms. Correctness: GSM8K strict 0.9682 (floor 0.90); the conc
  32 recipe has no draft, so there is no accept-length gate.
- conc 1 and conc 8 run the checkpoint 5 server byte for byte: 1.368x (conc 8) and 1.083x (conc 1), no regression by identity.
- Per-point recipe, runs, judges, audit and the HELD varsplit note: `results/checkpoint-6/README.md`.

`overlay/` plus `internal/overlay-redacted/` is the served conc 1 / conc 8 tree (`32533b87…`, 205 files). The conc 32
served tree is the same with 3 files replaced (`scripts/check_overlay_identity.py --point conc32`: `d77685c1…`).

## Checkpoint 5 (land-9): the number

| point: agentic coding, 8x B300, 1200 s fast protocol | official p90 ITL (ms) | tok/s/user = 1000 / p90 ITL | full-decode p90 ITL (ms) | source |
|---|---:|---:|---:|---|
| conc 8, checkpoint 5: land-9 overlay (arm `land9`) = land-8 + K5 + R28 + PAD + S moves; mean of 4 same-request ABAs | **4.82** [m] | **207.6** [m] | **5.58** [m] | `results/checkpoint-5/conc8/`; tag `checkpoint-5-land-9` |
| conc 8, checkpoint 4 content, same requests (arms `ctl-land8`, `ctl8-end`), mean of 8 | 5.39 [m] | 185.7 [m] | 6.16 [m] | `results/checkpoint-5/conc8/` |
| conc 1, land-9 (arm `c1-land9`) vs checkpoint 4 content (arm `c1-land8`), mean of 2 same-request pairs | **2.84** vs 3.02 [m] | **351.6** vs 331.3 [m] | 2.89 vs 3.10 [m] | `results/checkpoint-5/conc1/` |

Checkpoint 5 verdict: **GAIN vs checkpoint 4 at conc 8 and conc 1 (SOLID-AB CLEAR / CLEAR); eval PASS.** The judge is frozen and pooled; its outputs are in `results/checkpoint-5/judge/`.
- conc 8 (4 ABAs, pooled): -0.567 ms official (band 0.510) and -0.575 ms full-decode (band 0.293): GAIN on both.
- conc 1 (2 pairs, pooled): -0.174 ms official (band 0.125) and -0.210 ms full-decode (band 0.098): GAIN on both.
- vs the unmodified image: conc 8 6.59 / 4.818 = 1.368x; conc 1 3.080 / 2.845 = 1.083x (cross-run stock readings).
- Guards: TTFT p90 and output tok/s/GPU ok in every conc 8 ABA; error parity holds; peak memory 267.65 GiB (+4.72 GiB).
- Correctness (`results/checkpoint-5/eval/`): GSM8K strict 0.9727 (floor 0.90); real accept length 1.5388 vs the K5
  reference 1.5513 (band 0.042) and stock 1.5898 (cap 0.06): PASS. K5 drafts 5 tokens, not 7; the throughput recipe's
  synthetic accept length moves with it (3.84 -> 3.62), against land-9 in every ITL read.
- Membership, judge and notes: `results/checkpoint-5/README.md`.

`overlay/` plus `internal/overlay-redacted/` is the served land-9 tree (`scripts/check_overlay_identity.py`: `32533b87…`, 205 files).

## Checkpoint 4 (land-8): the number

| point: agentic coding, 8x B300, 1200 s fast protocol | official p90 ITL (ms) | tok/s/user = 1000 / p90 ITL | full-decode p90 ITL (ms) | source |
|---|---:|---:|---:|---|
| conc 8, checkpoint 4: land-8 overlay (arm `land8`) = land-5 + La + B4 occ2 | **5.24** [m] | **190.8** [m] | **5.86** [m] | `results/checkpoint-4/land8/metrics.json`; tag `checkpoint-4-land-8` |
| conc 8, checkpoint 3 content, same request (arm `land5`) | 5.29 [m] | 189.0 [m] | 6.01 [m] | `results/checkpoint-4/land5/metrics.json` |
| conc 1, land-8 (arm `c1-land8`) vs checkpoint 3 content (arms `c1-land5a`, `c1-land5b`), same request | **3.02** vs 2.99 / 2.99 [m] | **331.1** vs 334.4 / 334.4 [m] | 3.05 vs 3.07 / 3.05 [m] | `results/checkpoint-4/conc1/` |

Checkpoint 4 verdict: **NO_REGRESSION vs checkpoint 3 at conc 8 and conc 1; eval parity PASS.**
- conc 8 (`results/checkpoint-4/VERDICT.md`): land8 - land5 -0.050 ms official (bar 0.66), -0.151 ms full-decode (bar 0.625): both inside.
- conc 1 (`results/checkpoint-4/conc1/VERDICT.md`, pre-registered ABA): +0.023 ms official inside the band 0.171; -0.003 full-decode inside 0.062.
- vs the unmodified image: conc 8 6.59 / 5.24 = 1.258x (stock of record); conc 1 3.080 / 3.017 = 1.021x. Both stock readings are
  from earlier requests (cross-run).
- Guards: TTFT p90 is higher on land-8 (conc 8 +218 ms, tail only; conc 1 +12.6 ms, inside the same-request A/A). Both are
  PASS-WITH-NOTE; the mechanical `GUARD FAIL` line at conc 1 stays in its VERDICT.md. Output tok/s/GPU ok.
- The M1 bar (official p90 ITL <= 5.05 ms at conc 8) is not met (5.24).
- Correctness (`results/checkpoint-4/eval/`): the greedy GSM8K responses are identical to checkpoint 3 on 1319/1319
  questions; GSM8K strict 0.9697 (floor 0.90). The replay accept length 1.5328 is a note, not a verdict (its window is a
  duration-bounded, sampled request mix; `results/checkpoint-4/README.md`).
- Membership, audits and notes: `results/checkpoint-4/README.md`.

At tag `checkpoint-4-land-8`, `overlay/` plus `internal/overlay-redacted/` is the served land-8 tree (`6503addb…`, 203 files).

## Checkpoint 3 (land-5): the number

| point: agentic coding, conc 8, 8x B300, 1200 s fast protocol | official p90 ITL (ms) | tok/s/user = 1000 / p90 ITL | full-decode p90 ITL (ms) | source |
|---|---:|---:|---:|---|
| checkpoint 3: land-5 overlay (arm `land5`) = land-2 + E2b + MLA-LOOP + CKPT-COW | **5.15** [m] | **194.2** [m] | **6.20** [m] | `results/checkpoint-3/land5/metrics.json`; tag `checkpoint-3-land-5` |
| checkpoint 3 controls, same request: public image, no code change (arms `c0`, `c1`) | 6.45 / 7.11 [m] | 155.0 / 140.6 [m] | 7.55 / 7.74 [m] | `results/checkpoint-3/{c0,c1}/metrics.json` |

Checkpoint 3 verdict (`results/checkpoint-3/VERDICT.md`): **CONFIRMED-ITL** (no `-GUARD`) under the pre-registered rule.
- land-5 beats both controls by more than the control spread on both statistics.
- land5 - mean(controls): -1.630 ms official (95 % [-2.211, -0.896]) and -1.446 ms full-decode (95 % [-2.413, -0.834]).
- Guards: TTFT p90 -284.1 ms ok; output tok/s/GPU +2.468 ok.
- The M1 bar (official p90 ITL <= 5.05 ms) is missed by 0.10 ms.
- Membership, gate 0 v2, the item-04 audit diffs and open items: `results/checkpoint-3/README.md`.

Correctness gate of land-5 (real acceptance, `eval_only`) [m] (`results/checkpoint-3/eval/`):
- GSM8K strict 0.9697 (floor 0.90).
- Accept length 1.5568: PASS under eval rule v2 (ref = mean of land-3 and land-4, 1.5798; band 0.042).
- Under rule v1 (ref = land-4 alone) it reads FAIL by 0.0019. Rule v2 was set after the data; the caveat is in the checkpoint README.

`overlay/` plus `internal/overlay-redacted/` is the served land-5 tree (`scripts/check_overlay_identity.py`: `e471ad7a…`, 196 files).

## Checkpoint 2 (land-2): the number

| point: agentic coding, conc 8, 8x B300, 1200 s fast protocol | official p90 ITL (ms) | tok/s/user = 1000 / p90 ITL | full-decode p90 ITL (ms) | source |
|---|---:|---:|---:|---|
| checkpoint 2: land-2 overlay (arm `land2`) = land-1b + L6 + pin + 08L | **5.88** [m] | **170.1** [m] | **7.15** [m] | `results/checkpoint-2/land2/metrics.json`; tag `checkpoint-2-land-2` |
| checkpoint 2 controls, same request: public image, no code change (arms `c0`, `c1`) | 6.56 / 6.30 [m] | 152.4 / 158.7 [m] | 7.51 / 7.28 [m] | `results/checkpoint-2/{c0,c1}/metrics.json` |

Checkpoint 2 verdict (`results/checkpoint-2/VERDICT.md`): **NOT_RESOLVED** (no `-GUARD`) under the same pre-registered rule.
land2 - mean(controls) is -0.550 ms official (95 % [-1.222, +0.286]) and -0.244 ms full-decode (95 % [-0.916, +0.418]); TTFT p90 guard
-95.5 ms ok, output tok/s/GPU guard +1.985 ok. Membership, protocol, deviation and open items: `results/checkpoint-2/README.md`;
independent re-judge audit: `results/checkpoint-2/AUDIT.md`.

The correctness gate of land-2 has landed (real acceptance, `eval_only`): GSM8K strict 0.9697 / flexible 0.9689 (floor 0.90),
accept length 1.5597 vs 1.570 on the stock control (floor 1.510) and 1.523 on land-1b (+0.037, above the +-0.03 band on the high
side, ruled not a regression), no gate failure [m] (`results/checkpoint-2/eval/`). `overlay/` plus `internal/overlay-redacted/` is the
served land-2 tree (`scripts/check_overlay_identity.py`: `912da4804d8d…`, 189 files).

## Checkpoint 1 (land-1b): the number

| point: agentic coding, conc 8, 8x B300, 1200 s fast protocol | official p90 ITL (ms) | tok/s/user = 1000 / p90 ITL | full-decode p90 ITL (ms) | source |
|---|---:|---:|---:|---|
| baseline B0: public image, recipe flags, no code change | 6.57 [m] | 152.2 [m] | 7.53 [m] | `results/baseline-b0/b0-spec/metrics.json` |
| checkpoint 1: land-1b overlay (arm `lb`) | **6.10** [m] | **163.9** [m] | **7.25** [m] | `results/checkpoint-1/lb/metrics.json`; tag `checkpoint-1-land-1b` |
| checkpoint 1 controls, same request: public image, no code change (arms `c0`, `c1`) | 6.61 / 6.89 [m] | 151.3 / 145.1 [m] | 7.43 / 7.62 [m] | `results/checkpoint-1/{c0,c1}/metrics.json` |

Two statistics are always reported. The official p90 ITL is InferenceX's objective (aiperf `inter_token_latency`
p90). The full-decode p90 is `full_decode_duration / (osl - 1)` per request; it counts decode time that the official
statistic drops on requests whose last chunks carry no content. A landing needs a gain on both.

Checkpoint 1 verdict (`results/checkpoint-1/VERDICT.md`): **NOT_RESOLVED** at the real point under the pre-registered
three-way rule (CONFIRMED only if the set beats both controls by more than the control-to-control spread on both statistics;
FAIL if the interval is above 0 on either). land-1b reads below both controls on both statistics: lb - mean(controls) is
-0.650 ms official (95 % [-1.181, +0.487]) and -0.270 ms full-decode (95 % [-1.338, +0.488]); the intervals contain 0 at this
power (per-arm p90 bootstrap sd about 0.27 ms; the predicted effect was -0.11 ms). The evidence of record for land-1b is
step-level (admission at the dl5 and mixed_p07 proxies) plus the GSM8K gate below.

The correctness gate of land-1b has landed (real acceptance, `eval_only`): GSM8K strict 0.9697 / flexible 0.9689 (floor 0.90),
accept length 1.523 vs 1.570 on the control, no gate failure [m] (`results/eval-gate-land-1b/`). See `CHANGESET.md`.

## Reproduce

1. Hardware: one node with 8x NVIDIA B300 (SXM6), NVLink, driver 615.71.09 as measured [m], Docker with the NVIDIA runtime.
2. Pull the public image: `docker pull vllm/vllm-openai:nightly-49cb2bbf71cfd36ee5aa33d1e184b1cddff55294`
   (vLLM commit `49cb2bbf`; registry manifest `sha256:8c1434bf…` for amd64, see `PROVENANCE.md`).
3. Download the model and the drafter from the Hugging Face Hub:
   `moonshotai/Kimi-K3` revision `301be1b` (about 1.5 TB, 96 shards) and `Inferact/Kimi-K3-DSpark` revision `cf6b8244`.
4. Clone this repository and check out the checkpoint tag.
5. Start the server (the overlay goes on `PYTHONPATH`; `overlay/sitecustomize.py` makes `overlay/vllm/...` files replace the image's modules):
   `MODEL_DIR=/path/to/Kimi-K3 DRAFT_DIR=/path/to/Kimi-K3-DSpark MODE=throughput bash scripts/serve.sh`
   Wait for `Application startup complete` on port 8000. The first start compiles CuTe DSL kernels (several minutes; a
   warm cache is reused from `CACHE_DIR`). `MODE=throughput` is the scored arm (synthetic acceptance at the golden
   acceptance length 3.84, the InferenceX rule for agentic speculative decoding). `MODE=eval_only` is the gate arm (real acceptance).
6. Run the client: `DURATION=fast OUT_DIR=$PWD/point bash scripts/client.sh` (1200 s, one warmup request per lane; the
   checkpoint's point). `DURATION=full` is the 3600 s InferenceX protocol. The client is `agentx-harness==1.0.7`,
   dataset `semianalysisai/cc-traces-weka-062126`, exactly the `agentx profile` line in `results/*/*/point/benchmark_command.txt`.
7. Read the numbers: `python3 scripts/extract_metrics.py <dir holding point/>`. It prints the official p90 ITL, the
   full-decode p90, tok/s/user, output tok/s/GPU.
8. Gate (optional, `MODE=eval_only` server): run InferenceX's GSM8K eval `python3 -m infx.bench eval` (repository
   `SemiAnalysisAI/InferenceX` at `7b28963e`, `inferencex-e2e/`), concurrency 16, floor 0.90 (`infx/evals/thresholds.yaml`).

To run without the overlay (the baseline), start the server with `OVERLAY=` pointing at an empty directory, or set
`PERFBOT_OVERLAY_OFF=1` in the server environment (the overlay then prints `PERFBOT-OVERLAY off` once and the image's own modules run).

## Checks (run before you trust the tree)

| command | proves |
|---|---|
| `python3 scripts/check_overlay_identity.py` | `overlay/` plus `internal/overlay-redacted/` hashes to the served tree digest the measured arms recorded in `metrics.json` (`overlay_tree_sha256`; `patches/served-overlay-identity.json`): the conc 1 / conc 8 tree |
| `python3 scripts/check_overlay_identity.py --point conc32` | the same union with the conc 32 files of `results/checkpoint-6/served-conc32/` and `internal/overlay-redacted-conc32/` laid on top hashes to the conc 32 served tree (`points.conc32` of the same record) |
| `python3 scripts/make_image_diff.py --check` | every `overlay/vllm/**/*.py` file, with the recorded replacements reversed (`patches/image-replaced-lines.json`) and its `# PERFBOT` lines removed, is the image file byte for byte (hashes in `patches/image-base-sha256.json`) |
| `python3 scripts/check_secrets.py` | no secret, internal host name or internal path outside `internal/` |

## Layout

| path | content |
|---|---|
| `overlay/` | the served land-9 r2 code tree (conc 1 / conc 8): 193 files here plus 12 in `internal/overlay-redacted/` = 205 served. `vllm/` = 35 modified or copied vLLM files (tagged `# PERFBOT` additions, plus the tagged replacements listed in `patches/image-replaced-lines.json`); `perfbot_kernels/` = CuTe DSL kernels and their self-tests; `sitecustomize.py` = the import hook; `server_extra_args` = the landed server flags; `arm_env.json` = the land-9 switches of conc 1 / conc 8 (conc 32: `results/checkpoint-6/served-conc32/arm_env.json`) |
| `patches/overlay-vllm-vs-49cb2bbf.diff` | unified diff, image -> overlay, for the 29 files that differ (generated) |
| `patches/*.json` | the recorded image hashes, the recorded tagged replacements (`image-replaced-lines.json`) and the served tree digest the checks compare against |
| `scripts/` | `serve.sh`, `client.sh`, `extract_metrics.py`, the three checks, `ingest_results.py` (copies a measurement into `results/`) |
| `results/baseline-b0/` | the baseline measurement (6.57 ms) [m] |
| `results/eval-gate-land-1b/` | the land-1b correctness gate: both arms, GSM8K, aggregates [m] |
| `results/checkpoint-1/` | the checkpoint ABA: arms `c0` (control), `lb` (land-1b), `c1` (control), `VERDICT.md` [m] |
| `results/checkpoint-2/` | the land-2 ABA: arms `c0` (control), `land2`, `c1` (control), `VERDICT.md`, `AUDIT.md`; `eval/` = the land-2 correctness gate (GSM8K, real acceptance) [m] |
| `results/checkpoint-3/` | the land-5 ABA: arms `c0` (control), `land5`, `c1` (control), `VERDICT.md`, `README.md` (gate 0 v2, audit diffs); `eval/` = the land-5 correctness gate [m] |
| `results/checkpoint-6/` | the conc 32 point: 8 stock arms (`conc32/stock-*`), 3 land-9 q1 arms (`conc32/q1-*`), the eval arm (`conc32/eval-c32q1ev`), `served-conc32/` (the conc 32 switch file and the replaced log file), judge outputs (`judge/`), `AUDIT.md`, `VERDICT.md`, `README.md` [m] |
| `results/checkpoint-5/` | the land-9 ABAs at conc 8 (`conc8/`) and conc 1 (`conc1/`), `eval/`, judge outputs (`judge/`), `VERDICT.md`, `README.md` [m] |
| `results/checkpoint-4/` | the land-8 pair: arms `land8`, `land5` (checkpoint 3 content), `VERDICT.md`, `README.md`; `conc1/` = the conc 1 ABA (`c1-land5a`, `c1-land8`, `c1-land5b`); `eval/` = the land-8 correctness gate (greedy parity) [m] |
| `CHANGESET.md` | every member of land-1b: what it changes, mechanism, evidence, gate |
| `PROVENANCE.md` | every pin: image, commit, model, drafter, dataset, client, InferenceX revision |
| `submission/` | the InferenceX submission files and what their process requires |
| `internal/` | NVIDIA-only notes, the audit reports of checkpoints 3 and 4, the twelve overlay files that name internal hosts or paths (same bytes as served; `overlay-redacted-conc32/` = the one conc 32 replacement of that kind), and the frozen checkpoint 6 judges (`judge-checkpoint-6/`, unchanged bytes; their default paths are internal) |
| `CITATION.cff`, `LICENSE`, `NOTICE.md` | how to cite; Apache-2.0; third-party notices |

## Checkpoints

| tag | set | status |
|---|---|---|
| `checkpoint-1-land-1b` | land-1b (see `CHANGESET.md`) | landed, real point NOT_RESOLVED, tag checkpoint-1-land-1b |
| `checkpoint-2-land-2-provisional` | land-2 = land-1b + L6 + pin + 08L (`results/checkpoint-2/README.md`) | superseded: published before the eval pass; `overlay/` at that tag is still land-1b's |
| `checkpoint-2-land-2` | land-2 (same set) | landed, eval PASS, real point NOT_RESOLVED, `overlay/` = served land-2 tree |
| `checkpoint-3-land-5` | land-5 = land-2 + E2b + MLA-LOOP + CKPT-COW (`CHANGESET.md`) | landed, eval PASS (rule v2), real point CONFIRMED-ITL, `overlay/` = served land-5 tree |
| `checkpoint-4-land-8` | land-8 = land-5 + La + B4 occ2 (`CHANGESET.md`) | landed, NO_REGRESSION vs checkpoint 3 at conc 8 and conc 1, eval parity 1319/1319, `overlay/` = served land-8 tree |
| `checkpoint-5-land-9` | land-9 r2 = land-8 + K5 + R28 + PAD + S moves (`CHANGESET.md`) | landed, GAIN vs checkpoint 4 at conc 8 and conc 1, eval PASS, `overlay/` = served land-9 tree |
| `checkpoint-6-land-9-q1-c32` | land-9 r2 with a per-point recipe; conc 32 = land-9 q1 at 0.90 (`results/checkpoint-6/README.md`) | landed, GAIN vs the image at conc 32 (1.061x; key-matched INFO 1.044x), conc 1 / conc 8 = checkpoint 5, eval PASS |

## Omitted from the public copy

This directory is the tag `checkpoint-6-land-9-q1-c32` (commit `6a894e14`) of the checkpoint repository without its `internal/` subtree. Omitted (names only):

- `internal/README.md`, `internal/audit-checkpoint-3/` (2 files), `internal/audit-checkpoint-4/CP4-REJUDGE-RESULT.md`.
- `internal/judge-checkpoint-6/judge-c32-4/` and `judge-c32-5/` (the frozen conc 32 judges: `c32.py`, `cpread.py`, `evalpool.py`, `poolgen.py`, `poolguard.py`, `poolread.py`, `README.md`, `SHA256SUMS`). Their verbatim outputs are in `results/checkpoint-6/judge/out/`.
- `internal/overlay-redacted/` (12 served files: `perfbot_kernels/{cutecache,experts_stage,inittime,loadtime,mla_mixbench,mla_split,mla_splitkv,mla_varsplit,reqlog}/test_cpu.py`, `perfbot_kernels/mla_core/DESIGN.md`, `perfbot_kernels/moe_front/r0.py`, `perfbot_kernels/nullarm/__init__.py`) and `internal/overlay-redacted-conc32/perfbot_kernels/reqlog/test_cpu.py`. These are CPU self-tests, one design note, one benchmark script and the harness null-arm module; none runs in the served path.
- `CITATION.cff` `repository-code`, `submission/PR-DESCRIPTION-DRAFT.md` and `submission/waiver-DRAFT.md`: the internal origin URL is replaced by this public repository.

Checks in this copy: `python3 scripts/make_image_diff.py --check` PASS and `python3 scripts/check_secrets.py` PASS. `python3 scripts/check_overlay_identity.py` (and `--point conc32`) reads FAIL here: it hashes 193 of the 205 served files, so the recorded digests `32533b87…` / `d77685c1…` cannot be reproduced without the 12 omitted files.
