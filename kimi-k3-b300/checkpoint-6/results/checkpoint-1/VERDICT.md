# checkpoint-1: verdict

VERDICT: NOT_RESOLVED at the real point (pre-registered three-way rule). land-1b official p90 ITL 6.100 ms vs controls 6.610 / 6.890 (lb - mean(ctl) -0.650, 95 % [-1.181, +0.487]); consistent p90 7.255 vs 7.430 / 7.618 (-0.270, 95 % [-1.338, +0.488]). Favourable direction on both statistics; the interval contains 0 at this power (per-arm sd ~0.27 ms). Evidence of record: step-level admit at dl5/mixed_p07 and GSM8K PASS.

- guard TTFT p90 ms: lb 1156.600, c0 1413.700, c1 1292.780; spread 120.920; lb - mean(ctl) -196.640: ok
- guard output tok/s/GPU: lb 44.137, c0 43.775, c1 43.083; spread 0.692; lb - mean(ctl) +0.708: ok

| arm | official p90 ITL ms | tok/s/user | full-decode p90 ms | tok/s/user | TTFT p90 ms | out tok/s/GPU | AL | GSM8K strict | gates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| c0 | 6.61 | 151.3 | 7.430116825740419 | 134.6 | 1413.7 | 43.7751 | 3.8388694823753573 | None | [] |
| c1 | 6.89 | 145.1 | 7.618485169230772 | 131.3 | 1292.78 | 43.08265 | 3.838606839020267 | None | [] |
| lb | 6.1 | 163.9 | 7.254668994856962 | 137.8 | 1156.6 | 44.13657 | 3.838948707997509 | None | [] |

Source: this directory (copied by scripts/ingest_results.py; numbers by scripts/extract_metrics.py).
