# checkpoint-3: verdict

CONFIRMED-ITL at the real point (pre-registered rule, cp1itl.py 7c89402); checkpoint of record. land-5 official p90 ITL 5.150 ms vs controls 6.450 / 7.110 (land5 - mean(ctl) -1.630, 95 % [-2.211, -0.896]); consistent p90 6.202 vs 7.555 / 7.741 (-1.446, 95 % [-2.413, -0.834]). land-5 beats both controls by more than the control spread on both statistics (0.660 / 0.186). M1 bar 5.05 ms missed by 0.10 ms. Eval: GSM8K strict 0.9697, real-acceptance AL 1.5568 PASS under eval rule v2 (eval/).

- guard TTFT p90 ms: land5 1006.425, c0 1287.619, c1 1293.441; spread 5.822; land5 - mean(ctl) -284.105: ok
- guard output tok/s/GPU: land5 45.322, c0 42.860, c1 42.848; spread 0.012; land5 - mean(ctl) +2.468: ok

| arm | official p90 ITL ms | tok/s/user | full-decode p90 ms | tok/s/user | TTFT p90 ms | out tok/s/GPU | AL | GSM8K strict | gates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| c0 | 6.45 | 155.0 | 7.554872803694213 | 132.4 | 1287.619 | 42.85999669053799 | 3.8396681707868487 | None | [] |
| c1 | 7.11 | 140.6 | 7.7409409420714335 | 129.2 | 1293.441 | 42.847858042957306 | 3.8384353426736504 | None | [] |
| land5 | 5.15 | 194.2 | 6.201527195276246 | 161.3 | 1006.425 | 45.32226056091322 | 3.839327125790583 | None | [] |

Source: this directory (copied by scripts/ingest_results.py; numbers by scripts/extract_metrics.py).
