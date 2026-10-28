# checkpoint-2: the real-point ABA of land-2 (throughput, synthetic acceptance 3.84, fast protocol, conc 8)

Checkpoint of record: the land-2 eval_only pass landed (see Eval of record below; `eval/`).

Three arms on one request, same image, same node, same flags, run in this order: `c0` = control (harness overlay only),
`land2` = land-2 (the checkpoint overlay), `c1` = control (harness overlay only). Per arm: `metrics.json` (summary),
`identity.json`, `server_command.txt`, `phases.json`, `server-ready-seconds.txt`, `point/` (InferenceX aggregate
`kimik3_…_conc8.json`, `benchmark_command.txt`, timings, `aiperf_artifacts/` summary export, csv and per-request jsonl),
`server.log.gz` (gzip; the raw logs are 4.3-5.6 MiB, above the 5 MiB hosting limit) with host names and internal paths replaced.
Request-level: `arms.json`, `bundles.json` (internal paths replaced), `workload-revision.json`, `exit.json` (exit 0, 2026-10-20T11:05:14Z; all arms rc 0).
`AUDIT.md` is the independent read-only re-judge audit, copied as written except for one internal path replaced by `<path>` on line 5.
`eval/` is the eval_only request (one arm, `ev-land2`, real acceptance, `rejection_sample_method block`; exit 0, 2026-10-20T12:45:04Z, rc 0),
same per-arm files plus `eval/results_….json` (lm-eval GSM8K) and `eval/eval-timing.json`; `eval/VERDICT.md` holds the gate table.
`VERDICT.md` lines `LIVENESS MISSING` are expected for this workload revision: the `PERFBOT-SWITCH` server-log markers that
`scripts/ingest_results.py` looks for were added to the workload after this run (audit finding 5); switch liveness here rests on `identity.json` `env_perfbot` and the served tree digest.

## The set: land-2 = land-1b + L6 + pin + 08L

Membership per the land-2 composition record (§1b, §1b'); `CHANGESET.md` describes the land-1b members in full.

| member | switch (`land2/identity.json` `env_perfbot`) | what it is |
|---|---|---|
| land-1b: X2 (item 06) | `PERFBOT_EXP_GLUE=X2,S,D,G` | experts JOIN off at decode rows 9-40; bit-equal |
| land-1b: S (items 06/08) | same `GLUE` value | SharedDown SM counts at verify rows 17-24 and 33-40; bit-equal |
| land-1b: D = R,C,E (item 02) | same `GLUE` value; `D` expands to `R,C,E` | fused rejection tail R (ON), C and E step-tail fusions; falls back to the image path outside plain sampling |
| land-1b: G (item 02) | same `GLUE` value | draft-token gather; off with structured output |
| land-1b: draft chain (item 15) | `PERFBOT_EXP_DRAFT_CHAIN=1` | two-shot draft chain for 16 < M <= 64 |
| land-1b: cs35 capture sizes (item 06) | `PERFBOT_SERVER_EXTRA_ARGS=--compilation-config {…capture sizes 7,8,14,16,21,24,28,32,35,…}` | config only, no code |
| land-1b: prefill graph + P-07F (item 05) | `PERFBOT_EXP_PREFILL_GRAPH=1` | staging of long cached prompts; bit-equal |
| land-1b: MLA ctx segments (item 13) | `PERFBOT_EXP_MLA_CTX_SEGMENTS=auto` | prefill chunk segmentation; fp64 tolerance |
| L6 (item 01) | `PERFBOT_EXP_B35=F,O` | MLA decode variable split; fp64 tolerance 1.02; needs the eval pass |
| pin (item 17) | `PERFBOT_EXP_FRONTPIN=1` | MoE front WsGemm pin at rows 16/24/32/40; bit-equal |
| 08L (item 08) | `PERFBOT_EXP_STAGE=L` | lean experts stage at rows 24/32/40, drain 8 at 40; needs the eval pass |

KC (item 17) is excluded: it read FAIL at the dl5 admission (d1 +0.105, d5 +0.155 ms/step vs band 0.100; `AUDIT.md` row KC).
The eight switches above, plus the harness pair `PERFBOT_EXP_CUTE_CACHE=1` / `PERFBOT_EXP_INITTIME=1` set in every arm (`arms.json`), are the whole membership.
Bundle tree sha256 `e48f54bebcdd1762e01ca23a540e7a42bea8126e3666837b7e17fe30a2cc6be0` (`bundles.json`, class R2); served tree
`912da4804d8d…` over 189 files (`land2/metrics.json` `overlay_tree_sha256`, = bundle + harness overlay). The server command of `land2` is
byte-equal to checkpoint 1's `lb` (`results/checkpoint-1/lb/server_command.txt`); the land-2 members are switched on by environment only.
`overlay/` plus `internal/overlay-redacted/` of this tag is the served land-2 tree: `python3 scripts/check_overlay_identity.py` recomputes
`912da4804d8d53878b54919b7f22c466076a92bd202d30843aca72623c7f8cf5` over 189 files (`patches/served-overlay-identity.json`), equal to
`land2/metrics.json` and `eval/ev-land2/metrics.json` `overlay_tree_sha256`. Composition: the harness overlay at workload revision
`ace1a726` (88 files, tree `6004dbd1…` = arms `c0` and `c1`) with the land-2 bundle (104 files) on top.

## Protocol

Official InferenceX fast protocol: `AIPERF_EXPERIMENTAL_FAST` (1200 s window, one warmup request per lane), as
`point/benchmark_command.txt` shows (`--benchmark-duration 1200 --warmup-requests-per-lane 1`). The leaderboard rows ran 3600 s
(`results/checkpoint-1/LEADERBOARD-CONTEXT.md`). Same protocol as checkpoint 1. Throughput mode: synthetic acceptance at AL 3.84
(`server_command.txt`: `rejection_sample_method synthetic, synthetic_acceptance_length 3.84`; measured AL 3.839 in every arm, `VERDICT.md`).

## The number

p90 tok/s/user = 1000 / official p90 ITL (ms). From `VERDICT.md` (numbers by `scripts/extract_metrics.py`):

| arm | official p90 ITL ms | p90 tok/s/user | source |
|---|---:|---:|---|
| land2 | 5.880 | 170.1 | `land2/metrics.json` |
| c0 / c1 (stock controls, this request) | 6.560 / 6.300 | 152.4 / 158.7 | `c0/metrics.json`, `c1/metrics.json` |
| checkpoint 1 baseline (arm `c0`) | 6.610 | 151.3 | `results/checkpoint-1/VERDICT.md` |

The leaderboard reference rows are unchanged from checkpoint 1 (`results/checkpoint-1/LEADERBOARD-CONTEXT.md`).

## Verdict

`VERDICT.md` holds the verdict line and the per-arm table (written by `scripts/ingest_results.py` from `scripts/extract_metrics.py`).
Rule, pre-registered before the request ran: CONFIRMED if land-2 beats both controls by more than the control-to-control spread on
both p90 statistics; FAIL if the 95 % interval of land2 - mean(controls) is above 0 on either; NOT_RESOLVED otherwise. Guards: TTFT p90 and
output tok/s per GPU must not regress by more than the control spread. Result: **NOT_RESOLVED**, no `-GUARD`.

| statistic | c0 | land2 | c1 | ctl spread | land2 - mean(ctl) [95 %] | pre-registered prediction |
|---|---:|---:|---:|---:|---|---|
| official p90 ITL ms | 6.560 | 5.880 | 6.300 | 0.260 | -0.550 [-1.222, +0.286] | -0.88 +- 0.83 |
| consistent (full-decode) p90 ms | 7.509 | 7.151 | 7.281 | 0.228 | -0.244 [-0.916, +0.418] | -0.49 +- 0.91 |
| guard TTFT p90 ms | 1331.260 | 1253.390 | 1366.430 | 35.170 | -95.455 | ok |
| guard output tok/s/GPU | 42.811 | 44.836 | 42.890 | 0.079 | +1.985 | ok |

Official beats both controls by more than the spread (-0.68 / -0.42 vs 0.26); consistent does not (land2 - c1 = -0.130 vs 0.228).
Both points sit inside prediction +- resolution. The intervals come from the per-request bootstrap of the land-2 composition record (§3d) and are
reproduced in `AUDIT.md` (row cp2); the point values are reproduced by `scripts/ingest_results.py` in this directory.

Deviation from the pre-registration: the official delta -0.550 is above checkpoint 1's land-1b delta -0.650, which pre-registered a SPLIT request
(control / land-1b / land-2 / control). The SPLIT was deferred: the added-member effect is about 0.23 ms and the real-point control spread is
0.228-0.260 ms, so a four-arm split has resolution of the same size and reads NOT_RESOLVED by construction; attribution rides checkpoint 3.

c0 and land2 each had 1 of about 450 profiled requests end with an empty completion (aiperf `InvalidInferenceResultError`, `point/aiperf_artifacts/profile_export.jsonl`);
c1 had none. This class occurs at 0-0.7 % in every earlier real-point arm, including the stock image, and is not caused by the landing set.

## Eval of record (`eval/ev-land2`, eval_only, real acceptance; coordinator ruling in LAND-2.md §4)

| gate | floor / band | land-2 | source | result |
|---|---|---:|---|---|
| GSM8K exact_match strict-match (1319 samples, 5-shot, conc 16) | >= 0.90 | 0.9697 (flexible 0.9689) | `eval/ev-land2/eval/results_2026-10-20T12-44-07.949988.json`; `eval/ev-land2/metrics.json` | PASS |
| real-acceptance AL, cumulative vs stock control 1.570 | >= 1.510 (cap 0.06) | 1.5597 (-0.010) | `eval/ev-land2/metrics.json` `accept_length`; control `results/eval-gate-land-1b/l1-eval-off/metrics.json` | PASS |
| real-acceptance AL, incremental vs land-1b 1.523 | +-0.03 | +0.037 | land-1b `results/eval-gate-land-1b/l1-eval-on/metrics.json` | above the band on the high side; ruled not a regression (LAND-2.md §4) |

The +0.037 is an increase in acceptance, so it is not a regression; the band was written for drops. The run-to-run AL noise of two control
evals is unknown; a stock A/A eval is queued to calibrate the band (LAND-2.md §4). Gate failures: none (`eval/VERDICT.md`).
The eval server command is byte-equal to land-1b's eval arm (`results/eval-gate-land-1b/l1-eval-on/server_command.txt`).

## Open items (from `AUDIT.md`, findings 4-7; every member re-judged MATCH; finding 3, the eval pass, is closed above)

- No §10 Productization section for 08L or pin; the land-1b members have one-line statements with owner [u] (finding 4).
- Liveness of L6, pin and 08L in the `land2` arm is proven by `identity.json` and the overlay digest only; `server.log` has no import-time marker line for these three switches (finding 5).
- 08L row 32 is bit-equal on B200 only; the B300 judge covers rows 24 and 40 (finding 6).
- L6 judge v5.4 "72/72" is cited but its raw log path was not found [u] (finding 7).

## Reproduce the verdict

    python3 scripts/ingest_results.py <request out dir> checkpoint-2 --arms <arms json> --guard land2,c0,c1 --verdict "<line>"
    python3 scripts/extract_metrics.py results/checkpoint-2/land2
