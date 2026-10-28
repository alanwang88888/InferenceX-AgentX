# checkpoint-2: verdict

VERDICT: NOT_RESOLVED at the real point (pre-registered rule); checkpoint of record after the land-2 eval_only PASS (eval/VERDICT.md). land-2 official p90 ITL 5.880 ms vs controls 6.560 / 6.300 (land2 - mean(ctl) -0.550, 95 % [-1.222, +0.286]); consistent p90 7.151 vs 7.509 / 7.281 (-0.244, 95 % [-0.916, +0.418]). Official beats both controls by more than the control spread (0.260); consistent does not (land2 - c1 = -0.130 vs spread 0.228); neither interval excludes 0. Evidence of record: step-level admit at dl5 (admit.py PASS, d5 -1.350 ms/step), GSM8K strict 0.9697 and real-acceptance AL 1.5597 (eval/).

- guard TTFT p90 ms: land2 1253.390, c0 1331.260, c1 1366.430; spread 35.170; land2 - mean(ctl) -95.455: ok
- guard output tok/s/GPU: land2 44.836, c0 42.811, c1 42.890; spread 0.079; land2 - mean(ctl) +1.985: ok

- LIVENESS MISSING c0: PERFBOT_EXP_CUTE_CACHE=1 PERFBOT_EXP_INITTIME=1
- LIVENESS MISSING c1: PERFBOT_EXP_CUTE_CACHE=1 PERFBOT_EXP_INITTIME=1
- LIVENESS MISSING land2: PERFBOT_EXP_B35=F PERFBOT_EXP_B35=O PERFBOT_EXP_CUTE_CACHE=1 PERFBOT_EXP_DRAFT_CHAIN=1 PERFBOT_EXP_FRONTPIN=1 PERFBOT_EXP_GLUE=D PERFBOT_EXP_GLUE=G PERFBOT_EXP_GLUE=S PERFBOT_EXP_GLUE=X2 PERFBOT_EXP_INITTIME=1 PERFBOT_EXP_MLA_CTX_SEGMENTS=auto PERFBOT_EXP_PREFILL_GRAPH=1 PERFBOT_EXP_STAGE=L

| arm | official p90 ITL ms | tok/s/user | full-decode p90 ms | tok/s/user | TTFT p90 ms | out tok/s/GPU | AL | GSM8K strict | gates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| c0 | 6.56 | 152.4 | 7.50904603207987 | 133.2 | 1331.26 | 42.8114 | 3.839994428452038 | None | [] |
| c1 | 6.3 | 158.7 | 7.281007091840694 | 137.3 | 1366.43 | 42.89014 | 3.839227282418265 | None | [] |
| land2 | 5.88 | 170.1 | 7.151042657814491 | 139.8 | 1253.39 | 44.83625 | 3.8392190063889955 | None | [] |

Source: this directory (copied by scripts/ingest_results.py; numbers by scripts/extract_metrics.py).
