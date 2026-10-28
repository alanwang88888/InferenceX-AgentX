# checkpoint-2/eval: verdict

VERDICT: PASS (eval_only, real acceptance). GSM8K strict 0.9697 (floor 0.90); real-acceptance AL 1.5597 (floor 1.510; stock control 1.570, land-1b 1.523).

- LIVENESS MISSING ev-land2: PERFBOT_EXP_B35=F PERFBOT_EXP_B35=O PERFBOT_EXP_DRAFT_CHAIN=1 PERFBOT_EXP_FRONTPIN=1 PERFBOT_EXP_GLUE=D PERFBOT_EXP_GLUE=G PERFBOT_EXP_GLUE=S PERFBOT_EXP_GLUE=X2 PERFBOT_EXP_MLA_CTX_SEGMENTS=auto PERFBOT_EXP_PREFILL_GRAPH=1 PERFBOT_EXP_STAGE=L

| arm | official p90 ITL ms | tok/s/user | full-decode p90 ms | tok/s/user | TTFT p90 ms | out tok/s/GPU | AL | GSM8K strict | gates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ev-land2 | 18.03 | 55.5 | 18.153530663215758 | 55.1 | 1336.42 | 26.41784 | 1.5597247311827958 | 0.9696739954510993 | [] |

Source: this directory (copied by scripts/ingest_results.py; numbers by scripts/extract_metrics.py).
