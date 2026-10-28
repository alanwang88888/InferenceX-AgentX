# checkpoint-5: verdict

VERDICT: GAIN vs checkpoint 4 (same-request land-8 controls), SOLID-AB conc8 CLEAR conc1 CLEAR: checkpoint-5 = land-9 r2. conc 8, 4 ABAs pooled (poolgen 5f7dfa15, judge/pool-c8x4-c1x2.txt 75f90ec3): official p90 ITL land9 4.818 vs ctl 5.385 (-0.567, band 0.510), consistent 5.581 vs 6.155 (-0.575, band 0.293). conc 1, 2 pairs pooled: official 2.845 vs 3.019 (-0.174, band 0.125), consistent 2.890 vs 3.100 (-0.210, band 0.098). 1.368x vs the stock official 6.59 at conc 8; 1.083x vs 3.080 at conc 1. Guards OK in every ABA; error parity holds; peak memory 267.65 GiB. Eval: GSM8K strict 0.9727, real AL 1.5388 (PASS, eval/).

| conc | land9 official p90 ITL ms | tok/s/user | land-8 control official | tok/s/user | band | gain | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| 8 | 4.818 (n 4) | 207.6 | 5.385 (n 8) | 185.7 | 0.510 | -0.567 | GAIN |
| 1 | 2.845 (n 2) | 351.6 | 3.019 (n 2) | 331.3 | 0.125 | -0.174 | GAIN |

Per-run arms and numbers: conc8/*/VERDICT.md, conc1/*/VERDICT.md, eval/VERDICT.md. Judge outputs: judge/. Record: README.md.
