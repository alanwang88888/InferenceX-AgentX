# checkpoint-3/eval: verdict

PASS under eval rule v2 (eval_only, real acceptance; evalread.py 6a5fbcd). GSM8K strict 0.9697 (floor 0.90); real-acceptance AL 1.5568 vs ref 1.5798 = mean(land-3 1.5590, land-4 1.6007), one-sided band 0.042 -> floor 1.5378: PASS (-0.0231). Under rule v1 (ref = land-4 1.6007 alone, floor 1.5587) it reads FAIL by 0.0019; rule v2 was set after the data (coordinator ruling 2026-10-20).

| arm | official p90 ITL ms | tok/s/user | full-decode p90 ms | tok/s/user | TTFT p90 ms | out tok/s/GPU | AL | GSM8K strict | gates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ev-land5 | 16.07 | 62.2 | 16.441007015360572 | 60.8 | 1307.708 | 26.664795959648202 | 1.5567628156384303 | 0.9696739954510993 | [] |

Source: this directory (copied by scripts/ingest_results.py; numbers by scripts/extract_metrics.py).
