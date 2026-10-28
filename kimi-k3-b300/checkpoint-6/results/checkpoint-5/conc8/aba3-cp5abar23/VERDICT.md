# checkpoint-5/conc8/aba3-cp5abar23: verdict

VERDICT: POOLED (checkpoint 5 is judged on the pool, not per run): this run's frozen read is judge/cpread-cp5abar23.txt; the checkpoint verdict is in results/checkpoint-5/VERDICT.md

- guard TTFT p90 ms: land9 937.769, ctl-land8 1030.949, ctl8-end 980.767; spread 50.182; land9 - mean(ctl) -68.089: ok
- guard output tok/s/GPU: land9 45.394, ctl-land8 45.268, ctl8-end 45.231; spread 0.037; land9 - mean(ctl) +0.144: ok

| arm | official p90 ITL ms | tok/s/user | full-decode p90 ms | tok/s/user | TTFT p90 ms | out tok/s/GPU | AL | GSM8K strict | gates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ctl-land8 | 5.08 | 196.9 | 6.181194211970871 | 161.8 | 1030.949 | 45.267865942852666 | 3.8406102976740097 | None | [] |
| ctl8-end | 5.64 | 177.3 | 6.235044007283425 | 160.4 | 980.767 | 45.2306989685192 | 3.839558543969584 | None | [] |
| land9 | 4.77 | 209.6 | 5.44875017089983 | 183.5 | 937.769 | 45.393739610072004 | 3.617795418837344 | None | [] |

Source: this directory (copied by scripts/ingest_results.py; numbers by scripts/extract_metrics.py).
