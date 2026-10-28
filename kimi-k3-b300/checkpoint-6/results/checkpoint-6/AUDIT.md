# Checkpoint 6 audit: land-9 q1 at conc 32 (item 03)

Item 03 re-judged checkpoint 6 from the raw arm files. It used read-only copies of the frozen judges,
run with `python3.12 -I -B`. No judge file was changed. Every hash in this file is sha256 (first 8 hex digits).
Claim tags: [m] measured or re-run here, [d] arithmetic, [c] code or log read.

## Verdict

| read | statistic | control mean | candidate mean | gain ms | band ms | verdict |
|---|---|---|---|---|---|---|
| of record (judge-c32-4, 8 stock vs 3 q1) | official p90 ITL | 57.761 | 54.430 | -3.331 | 1.847 | GAIN |
| of record | consistent p90 full-decode ITL | 59.183 | 55.825 | -3.358 | 2.460 | GAIN |
| of record | TTFT guard | 3287.111 | | d +391.099 | bound 648.510 | OK |
| INFO, key-matched (4 stock arms without turn 351) | official | 56.825 | 54.430 | -2.395 | 1.574 | GAIN |
| INFO, key-matched | consistent | 57.895 | 55.825 | -2.070 | 1.314 | GAIN |

- [m] Of record: `SOLID-AB conc32 CLEAR NO_REGRESS`. Ratio 57.761 / 54.430 = 1.061x [d].
- [m] Key-matched ratio: 56.825 / 54.430 = 1.044x [d]. This is INFO only. It never replaces the verdict.
- [m] Gate 2 (accuracy, arm c32-l9q1-ev): `PASS: gsm8k exact_match,strict-match = 0.9682 (>= 0.9 from default)`.
  The acceptance-length floor does not apply: the conc-32 recipe has no speculative drafts.

**Audit result: PASS.** The judge-c32-4 re-run gives the same output, byte for byte, as read 1 of record.

## 1. Judge integrity [m]

```
judge-c32-4 SHA256SUMS 3095b5e5: c32.py OK cpread.py OK evalpool.py OK poolgen.py OK poolguard.py OK poolread.py OK README.md OK
judge-c32-5 SHA256SUMS 78ea7ade: README.md OK c32.py OK cpread.py OK evalpool.py OK poolgen.py OK poolguard.py OK poolread.py OK
__pycache__ in either judge dir: 0
judge-c32-4.DEFECTS.md 3fe1ea90 (defect D1: band depends on control list order)
```

- judge-c32-4 (c32.py 53bf1a26) is the judge of record for checkpoint 6.
- judge-c32-5 (c32.py facb8bce) was accepted after checkpoint 6. It fixes D1: the bootstrap draws from the sorted
  readings. It also adds the KEYS INFO lines. This audit uses it only for validation and INFO.

## 2. Re-run of read 1 under judge-c32-4 (verbatim) [m]

Spec: controls `out-c32aa1,out-c32aa22,out-c32aa3,out-c32p0a`; candidates `out-c32q1c3,out-c32q1c2`;
arms `c32-s1+c32-s2+c32-s3+c32-s4+c32-stock@-` vs `c32-l9q1-a+c32-l9q1-b+c32-l9q1-c`; key `bdcc573b`.

```
BAND conc32 official   controls n 8 [55.72, 59.46, 57.17, 59.14, 57.07, 59.02, 58.5, 56.01] candidates n 3 [54.52, 53.24, 55.53] band 1.847 gain -3.331 verdict GAIN
BAND conc32 consistent controls n 8 [58.013, 62.308, 58.356, 59.773, 57.186, 61.449, 59.318, 57.062] candidates n 3 [55.673, 54.977, 56.825] band 2.460 gain -3.358 verdict GAIN
SOLID-AB conc32 CLEAR NO_REGRESS
POOLGUARD ttft_p90_ms controls n 8 [3243.57, 3042.82, 3663.1, 3014.59, 3208.48, 3447.36, 3497.34, 3179.63] candidates n 3 [3298.4, 4239.29, 3496.94] d +391.099 bound 648.510 -> OK
AA conc32 official   n 8 [...] mean 57.761 spread 3.740 band(n_cand 1) 2.394 band(n_cand 2) 2.028 band(n_cand 3) 1.847
AA conc32 consistent n 8 [...] mean 59.183 spread 5.246 band(n_cand 1) 3.251 band(n_cand 2) 2.692 band(n_cand 3) 2.460
```

Live gate rows (each candidate and control arm):

```
LIVE c32-l9q1-a PERFBOT-SWITCH-CHECK set=12 live=12 dead=- | PERFBOT-ATKEY key=bdcc573b record=bdcc573b salt=- hit=35 tuned=0 intact=yes | gate_failures [] -> VALID
LIVE c32-l9q1-b PERFBOT-SWITCH-CHECK set=12 live=12 dead=- | PERFBOT-ATKEY key=bdcc573b record=bdcc573b salt=- hit=35 tuned=0 intact=yes | gate_failures [] -> VALID
LIVE c32-l9q1-c PERFBOT-SWITCH-CHECK set=11 live=11 dead=- | PERFBOT-ATKEY key=bdcc573b record=bdcc573b salt=- hit=35 tuned=0 intact=yes | gate_failures [] -> VALID
LIVE c32-s1..s4  PERFBOT-SWITCH-CHECK set=1 live=1 dead=- | PERFBOT-ATKEY key=bdcc573b ... intact=yes | gate_failures [] -> VALID
REF  c32-stock  reference arm, no live gate | gate_failures [] -> KEPT
```

- [c] q1-c has 11 switches, and q1-a/b have 12. The only difference is `PERFBOT_EXP_CUTE_PRETRACE=1` in a/b.
  This switch traces the CuTe kernels at start-up. It changes ready time, not the served kernels. It also prints
  the `cubins=` line, so q1-c has no cubin line.
- Full outputs: `judge-out/aa4.out`, `judge-out/c32_4.out` (efe28f18, same as read 1), `judge-out/guard4.out` (2eb7bd70, same as read 1).

## 3. Validation under judge-c32-5 [m]

```
BAND conc32 official   controls n 8 [55.72, 56.01, 57.07, 57.17, 58.5, 59.02, 59.14, 59.46] candidates n 3 [53.24, 54.52, 55.53] band 1.845 gain -3.331 verdict GAIN
BAND conc32 consistent controls n 8 [57.062, 57.186, 58.013, 58.356, 59.318, 59.773, 61.449, 62.308] candidates n 3 [54.977, 55.673, 56.825] band 2.353 gain -3.358 verdict GAIN
SOLID-AB conc32 CLEAR NO_REGRESS
POOLGUARD ttft_p90_ms ... d +391.099 bound 648.510 -> OK
KEYS conc32 cand-only n 4 of 3 cand arms vs 8 ctl arms; isl sum 361503; top 0ede42538334:15 isl 127881 in 2/3, 1499c406c828:19 isl 79293 in 1/3, 1499c406c828:31 isl 77405 in 2/3, 1499c406c828:30 isl 76924 in 2/3
KEYS conc32 ctl-only n 5 of 8 ctl arms vs 3 cand arms; isl sum 1070683; top 085aacf44ac5:351 isl 659451 in 4/8, 0470d446a451:90 isl 136808 in 1/8, 006c98de37d8:24 isl 111642 in 4/8, 15ab8fdb8831:23 isl 85729 in 1/8, 14129d470e50:14 isl 77053 in 1/8
```

- The verdict does not change with the judge: CLEAR under both. The gain is the same. Only the band changes
  (1.847 -> 1.845, 2.460 -> 2.353), because judge-c32-5 does not depend on list order.
- KEYS: turn `085aacf44ac5:351` (ISL 659,451) was sent in 4 of 8 stock arms and in 0 of 3 q1 arms. The client
  sends it only if it starts before the profiling phase ends, so each run is a coin flip.
  Its cold prefill makes p90 ITL larger in the arms that send it.
- Full outputs: `judge-out/c32_5.out`, `judge-out/guard5.out`.

## 4. Key-matched INFO [m]

Controls are the 4 stock arms that did not send turn 351: c32aa1/s1, c32aa22/s2, c32aa3/s4, c32p0a/stock
(see `arms.md`). The band is the judge-c32-5 `band()` function, n_cand 3, the same seed.

```
KEYMATCHED-INFO conc32 official   ctl n 4 mean 56.825 cand n 3 mean 54.430 gain -2.395 band 1.574 -> GAIN ratio 1.0440x
KEYMATCHED-INFO conc32 consistent ctl n 4 mean 57.895 cand n 3 mean 55.825 gain -2.070 band 1.314 -> GAIN ratio 1.0371x
```

These are the same numbers as the 08:36Z INFO price. Output: `judge-out/keymatched.out`.

## 5. Per-arm identity [m, from arm files]

`arms.md` has one row per arm: official and consistent p90 ITL, TTFT p90, gate failures, the CuTe cubin set,
KV cache size, and whether the arm sent turn 351.

- All 8 stock arms: cubins 753:8ec85c07d66a, KV 42.3 GiB.
- q1 arms: cubins 757:aeb4f0841207 (a, b, eval), KV 36.95 GiB.
- The metric values in `arms.md` are equal to the judge's control and candidate lists.

## 6. What this audit did not check

- The publish commit itself (item 02's staging branch). Item 03 audits it separately, against this file,
  when item 02 posts PUSH-READY.
- Conc 1 and conc 8. Those points run the checkpoint 5 environment byte for byte. Their checkpoint 5 bands carry over.
