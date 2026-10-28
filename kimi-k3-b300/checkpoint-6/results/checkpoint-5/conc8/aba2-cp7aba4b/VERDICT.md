# checkpoint-5/conc8/aba2-cp7aba4b: verdict

VERDICT: POOLED (checkpoint 5 is judged on the pool, not per run): this run's frozen read is judge/cpread-cp7aba4b.txt; the checkpoint verdict is in results/checkpoint-5/VERDICT.md

- guard TTFT p90 ms: land9 917.920, ctl-land8 1047.632, ctl8-end 919.767; spread 127.865; land9 - mean(ctl) -65.780: ok
- guard output tok/s/GPU: land9 45.395, ctl-land8 45.484, ctl8-end 45.086; spread 0.399; land9 - mean(ctl) +0.110: ok

| arm | official p90 ITL ms | tok/s/user | full-decode p90 ms | tok/s/user | TTFT p90 ms | out tok/s/GPU | AL | GSM8K strict | gates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ctl-land8 | 6.36 | 157.2 | 6.641778758443644 | 150.6 | 1047.632 | 45.48417149277517 | 3.840335915517421 | None | [] |
| ctl8-end | 5.32 | 188.0 | 6.209750037693918 | 161.0 | 919.767 | 45.085624883410254 | 3.838888888888889 | None | [] |
| land9 | 4.74 | 211.0 | 5.3093480890654625 | 188.3 | 917.92 | 45.39524092499421 | 3.618775435926281 | None | [] |

Source: this directory (copied by scripts/ingest_results.py; numbers by scripts/extract_metrics.py).
