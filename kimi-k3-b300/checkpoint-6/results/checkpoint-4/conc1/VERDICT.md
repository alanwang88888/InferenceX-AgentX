# checkpoint-4/conc1: verdict

VERDICT: NO_REGRESSION-GUARD at conc 1 (pre-registered ABA, c1aba.py f8f9fc15): official p90 ITL c1-land5a 2.994 / c1-land8 3.017 / c1-land5b 2.995; band 0.171 (floor; A/A 0.000); delta +0.023. Consistent 3.069 / 3.056 / 3.049, band 0.062, delta -0.003. 1.021x vs the stock conc-1 official 3.080 (cross-pin).

- guard TTFT p90 ms: c1-land8 1666.872, c1-land5a 1658.021, c1-land5b 1650.488; spread 7.533; c1-land8 - mean(ctl) +12.618: GUARD FAIL
  - Ruling (coordinator, 2026-10-21): PASS-WITH-NOTE. The guard bar per statistic = max(the control spread, the same-run
    same-content A/A of that arm shape). Paired TTFT by (conversation, turn), equal ISL: c1-land8 - c1-land5a n 35 p50 0
    p90 +8 ms; c1-land8 - c1-land5b n 33 p50 +2 p90 +9 ms; A/A c1-land5b - c1-land5a n 43 p50 -2 p90 +32 ms. The +12.6 ms
    (0.8 %) is inside the A/A band. Server prefill mean 753.4 / 752.3 ms (c1-land5a / c1-land8).
- guard output tok/s/GPU: c1-land8 17.765, c1-land5a 17.753, c1-land5b 17.765; spread 0.012; c1-land8 - mean(ctl) +0.006: ok

| arm | official p90 ITL ms | tok/s/user | full-decode p90 ms | tok/s/user | TTFT p90 ms | out tok/s/GPU | AL | GSM8K strict | gates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| c1-land5a | 2.99 | 334.4 | 3.0684060372261097 | 325.9 | 1658.021 | 17.75279544502999 | 3.842912360305857 | None | [] |
| c1-land5b | 2.99 | 334.4 | 3.0466906893727264 | 328.2 | 1650.488 | 17.76511575003203 | 3.8426040465457594 | None | [] |
| c1-land8 | 3.02 | 331.1 | 3.053867104235727 | 327.5 | 1666.872 | 17.764857261799218 | 3.842613385942452 | None | [] |

Source: this directory (copied by scripts/ingest_results.py; numbers by scripts/extract_metrics.py).
