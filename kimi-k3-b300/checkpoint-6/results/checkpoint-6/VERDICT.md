# checkpoint-6: verdict

VERDICT: GAIN vs the unmodified image at conc 32 (SOLID-AB conc32 CLEAR NO_REGRESS, judge-c32-4); conc 1 and conc 8 = checkpoint 5 (same served tree and server command, NO_REGRESSION by identity). checkpoint-6 = land-9 r2 with a per-point recipe: conc 1 / 8 = the checkpoint-5 land-9 env; conc 32 = land-9 q1 (no draft, gpu-memory-utilization 0.90). conc 32, 3 candidate arms vs 8 stock arms: official p90 ITL 54.430 vs 57.761 (-3.331, band 1.847) GAIN, consistent 55.825 vs 59.183 (-3.358, band 2.460) GAIN; 1.061x. TTFT guard OK (d +391.1, bound 648.5). Key-matched INFO (4 stock arms without turn 351): -2.395 (band 1.574), 1.044x. Eval: GSM8K strict 0.9682 (PASS, conc32/eval-c32q1ev/).

| conc | candidate official p90 ITL ms | tok/s/user | control official | tok/s/user | band | gain | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| 32 | 54.430 (land-9 q1, n 3) | 18.4 | 57.761 (stock, n 8) | 17.3 | 1.847 | -3.331 | GAIN |
| 32, INFO key-matched | 54.430 (n 3) | 18.4 | 56.825 (stock without turn 351, n 4) | 17.6 | 1.574 | -2.395 | GAIN |
| 8 | 4.818 (land-9, n 4) | 207.6 | 5.385 (land-8, n 8) | 185.7 | 0.510 | -0.567 | GAIN (checkpoint 5, carried) |
| 1 | 2.845 (land-9, n 2) | 351.6 | 3.019 (land-8, n 2) | 331.3 | 0.125 | -0.174 | GAIN (checkpoint 5, carried) |

Per-run arms and numbers: conc32/*/VERDICT.md. Judge outputs: judge/out/. Independent re-judge: AUDIT.md. Record: README.md.
