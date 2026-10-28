# checkpoint-4: verdict

VERDICT: NO_REGRESSION vs checkpoint 3 (same-run land-5): checkpoint-4 = land-8. Official p90 ITL land8 5.240 vs land5 5.290 (-0.050, bar 0.66, INSIDE); consistent 5.863 vs 6.014 (-0.151, bar 0.625, INSIDE). 1.258x vs the stock official 6.59 (cp4v2 c0). M1 bar 5.05 not met. TTFT p90 +218.5 ms: PASS-WITH-NOTE (tail only). Eval: greedy GSM8K parity 1319/1319 vs land-5, GSM8K 0.9697 (eval/). conc 1: ABA NO_REGRESSION (conc1/).

| arm | official p90 ITL ms | tok/s/user | full-decode p90 ms | tok/s/user | TTFT p90 ms | out tok/s/GPU | AL | GSM8K strict | gates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| land5 | 5.29 | 189.0 | 6.014490017456433 | 166.3 | 969.699 | 45.11133868146378 | 3.8399778418683175 | None | [] |
| land8 | 5.24 | 190.8 | 5.863390474547656 | 170.5 | 1188.231 | 45.061975800600614 | 3.8396725750861647 | None | [] |

Source: this directory (copied by scripts/ingest_results.py; numbers by scripts/extract_metrics.py).
