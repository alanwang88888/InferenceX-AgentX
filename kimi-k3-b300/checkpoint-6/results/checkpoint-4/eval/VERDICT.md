# checkpoint-4/eval: verdict

VERDICT: PASS. Greedy GSM8K (temperature 0) responses of ev-land8 are identical to ev-land5 (checkpoint 3) on 1319/1319 docs (evparity.py 0bb4c8cc); GSM8K strict 0.9697 >= 0.90. NOTE: replay AL 1.5328 (not a verdict: duration-bounded sampled replay window).

| arm | official p90 ITL ms | tok/s/user | full-decode p90 ms | tok/s/user | TTFT p90 ms | out tok/s/GPU | AL | GSM8K strict | gates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ev-land8 | 15.52 | 64.4 | 16.764499333333333 | 59.6 | 1175.518 | 28.05965409149595 | 1.5327611541544215 | 0.9696739954510993 | [] |

Source: this directory (copied by scripts/ingest_results.py; numbers by scripts/extract_metrics.py).
