# checkpoint-5/conc8/aba1-cp5abar22: verdict

VERDICT: POOLED (checkpoint 5 is judged on the pool, not per run): this run's frozen read is judge/cpread-cp5abar22.txt; the checkpoint verdict is in results/checkpoint-5/VERDICT.md

- guard TTFT p90 ms: land9 951.722, ctl-land8 1035.104, ctl8-end 1000.785; spread 34.319; land9 - mean(ctl) -66.223: ok
- guard output tok/s/GPU: land9 45.342, ctl-land8 45.255, ctl8-end 45.159; spread 0.095; land9 - mean(ctl) +0.135: ok

| arm | official p90 ITL ms | tok/s/user | full-decode p90 ms | tok/s/user | TTFT p90 ms | out tok/s/GPU | AL | GSM8K strict | gates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ctl-land8 | 5.15 | 194.2 | 6.061576329268294 | 165.0 | 1035.104 | 45.25464599360548 | 3.840822684549583 | None | [] |
| ctl8-end | 4.95 | 202.0 | 6.067333162629855 | 164.8 | 1000.785 | 45.15931053022152 | 3.8394563918757467 | None | [] |
| land9 | 4.82 | 207.5 | 5.62086562100825 | 177.9 | 951.722 | 45.34204644083264 | 3.6188156480026468 | None | [] |

Source: this directory (copied by scripts/ingest_results.py; numbers by scripts/extract_metrics.py).
