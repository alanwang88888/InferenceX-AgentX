# checkpoint-5/conc8/aba4-cp5abar24: verdict

VERDICT: POOLED (checkpoint 5 is judged on the pool, not per run): this run's frozen read is judge/cpread-cp5abar24.txt; the checkpoint verdict is in results/checkpoint-5/VERDICT.md

- guard TTFT p90 ms: land9 1080.596, ctl-land8 1146.547, ctl8-end 1128.113; spread 18.434; land9 - mean(ctl) -56.734: ok
- guard output tok/s/GPU: land9 45.122, ctl-land8 44.686, ctl8-end 44.637; spread 0.049; land9 - mean(ctl) +0.461: ok

| arm | official p90 ITL ms | tok/s/user | full-decode p90 ms | tok/s/user | TTFT p90 ms | out tok/s/GPU | AL | GSM8K strict | gates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ctl-land8 | 5.39 | 185.5 | 6.141370688033458 | 162.8 | 1146.547 | 44.68644240375211 | 3.838710250804316 | None | [] |
| ctl8-end | 5.19 | 192.7 | 5.7046603836477985 | 175.3 | 1128.113 | 44.63732568150158 | 3.839068304214137 | None | [] |
| land9 | 4.94 | 202.4 | 5.944125316848878 | 168.2 | 1080.596 | 45.12239850790563 | 3.618383610798181 | None | [] |

Source: this directory (copied by scripts/ingest_results.py; numbers by scripts/extract_metrics.py).
