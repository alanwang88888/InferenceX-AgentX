# Checkpoint 6: land-9 with a per-point recipe (conc 1, 8 and 32)

Checkpoint 6 adds the conc 32 point. The overlay code is the checkpoint 5 land-9 r2 code at every point. Only the
server arguments and the switch file (`arm_env.json`) change per point. InferenceX lets the submitter set these
per-point fields. Tags: [m] measured (a file in this directory), [d] arithmetic, [c] code or log read, [u] not known.

## Per-point recipe

| point | switch file (sha256) | server_command.txt of record (sha256) | max seqs | gpu-memory-utilization | CUDA-graph captures | draft |
|---|---|---|---:|---:|---|---|
| conc 1 | `overlay/arm_env.json` 1242c82a | `results/checkpoint-5/conc1/*/c1-land9/` cf2e9c44 [m] | 2 | 0.92 | land-9 list 5..8192 | DSpark, 5 tokens, synthetic AL 3.62 |
| conc 8 | `overlay/arm_env.json` 1242c82a | `results/checkpoint-5/conc8/*/land9/` 73458bd2 [m] | 16 | 0.92 | land-9 list 5..8192 | DSpark, 5 tokens, synthetic AL 3.62 |
| conc 32 | `results/checkpoint-6/served-conc32/arm_env.json` 22877606 | `conc32/q1-*/c32-l9q1-*/` 7208739e [m] | 64 | 0.90 | recipe 1..64, 128..8192 | none (the recipe's conc 32 point has no draft) |

- All three points: `--max-num-batched-tokens 16384`, the same overlay code. The conc 1 and conc 8 servers are the
  checkpoint 5 servers, byte for byte (same switch file, same server command). Their numbers carry from checkpoint 5.
- Conc 32 switch file = the conc 1/8 file without `PERFBOT_SERVER_EXTRA_ARGS` (no land-9 capture list, no draft), and
  the measured arms add `--gpu-memory-utilization 0.90` (the last flag wins over the recipe's 0.92). The server
  arguments are the InferenceX recipe's conc 32 override (`override_c32`: 64 sequences, captures 1..64, no draft).
- Why 0.90 [m]: at 0.92 the land-9 conc 32 server died on a fatal CUDA out-of-memory (a 3.28 GiB MoE workspace request
  with 2.84-2.99 GiB free) in two runs. At 0.90 every land-9 conc 32 run had 0 fatal out-of-memory events.
- Served tree at conc 32: `python3 scripts/check_overlay_identity.py --point conc32` = `d77685c1…` (205 files), the
  digest every conc 32 land-9 arm recorded. It is the default tree with 3 files replaced: the switch file and the
  log-only STEPDEC record (`perfbot_kernels/reqlog/__init__.py` and its CPU test; two fields `pn`, `pnmax` added at
  measurement-harness revision `5b410c38`). The served path is the same.

## Conc 32: the runs [m]

Fast protocol (1200 s), 8x B300, the same client and dataset as checkpoints 1-5, `PERFBOT_CONC=32`.

| run directory | arms | content | official p90 ITL ms | consistent (full-decode) p90 ms | TTFT p90 ms |
|---|---|---|---|---|---|
| `conc32/stock-c32aa1/` | c32-s1, c32-s2, c32-s3 | stock | 55.72 / 59.46 / 57.17 | 58.01 / 62.31 / 58.36 | 3244 / 3043 / 3663 |
| `conc32/stock-c32aa22/` | c32-s1, c32-s2, c32-s3 | stock | 59.14 / 57.07 / 59.02 | 59.77 / 57.19 / 61.45 | 3015 / 3208 / 3447 |
| `conc32/stock-c32aa3/` | c32-s4 | stock | 58.50 | 59.32 | 3497 |
| `conc32/stock-c32p0a/` | c32-stock | stock (reference arm) | 56.01 | 57.06 | 3180 |
| `conc32/q1-c32q1c3/` | c32-l9q1-a, c32-l9q1-b | land-9 q1, 0.90 | 54.52 / 53.24 | 55.67 / 54.98 | 3298 / 4239 |
| `conc32/q1-c32q1c2/` | c32-l9q1-c | land-9 q1, 0.90 | 55.53 | 56.82 | 3497 |
| `conc32/eval-c32q1ev/` | c32-l9q1-ev | land-9 q1, 0.90, `eval_only` | gate arm | | |

- Stock = the public image with the measurement harness overlay (log-only) and no landing switch, at the recipe's conc
  32 point (`gpu-memory-utilization` 0.92).
- `stock-c32p0a/` ran a second arm, c32-l9nd (land-9 with the checkpoint 5 switch file at 0.92). It died on the fatal
  out-of-memory above before its point; it has no result, it is not copied, and `exit.json` reads 76. That run's
  stock arm finished and is a reference arm. It ran at harness revision `bd6a5f45` (checkpoint 5); every other run ran
  at `5b410c38` (the difference is the log-only STEPDEC fields and a start-up time-out knob, unset).
- One more stock run (three arms) ended on a hardware link fault before its point. It ran again with the same arms
  as `stock-c32aa22/`. It is not a result.
- Memory [m]: KV cache 36.95 GiB (20,618,315 tokens) on every land-9 q1 arm vs 42.3 GiB (23,609,516 tokens) on
  stock. Peak device memory on the q1 arms 253.76-258.74 GiB (stock 259.12-267.27 GiB; device 268.59 GiB).
  Non-fatal allocator retries: 1-2 events per arm in both configurations.
- Prefix-cache hit rate (prefix hits / prompt tokens) [m]: q1 0.835-0.840, stock 0.838-0.839. The smaller KV pool
  at 0.90 does not lower the hit rate at conc 32.

## Conc 32: the read of record (judge-c32-4) [m]

The judge was frozen before any candidate data. The spec (runs, arm groups, run order) was pre-registered.
Verbatim output (`judge/out/c32_4.out`, `judge/out/guard4.out`):

```
BAND conc32 official   controls n 8 [55.72, 59.46, 57.17, 59.14, 57.07, 59.02, 58.5, 56.01] candidates n 3 [54.52, 53.24, 55.53] band 1.847 gain -3.331 verdict GAIN
BAND conc32 consistent controls n 8 [58.013, 62.308, 58.356, 59.773, 57.186, 61.449, 59.318, 57.062] candidates n 3 [55.673, 54.977, 56.825] band 2.460 gain -3.358 verdict GAIN
SOLID-AB conc32 CLEAR NO_REGRESS
POOLGUARD ttft_p90_ms controls n 8 [3243.57, 3042.82, 3663.1, 3014.59, 3208.48, 3447.36, 3497.34, 3179.63] candidates n 3 [3298.4, 4239.29, 3496.94] d +391.099 bound 648.510 -> OK
```

- Band = 95 % half-width of a null bootstrap over the 8 control readings (2000 draws, seed 0) at 3 candidate arms.
  GAIN = the candidate mean is below the control mean by more than the band, on that statistic.
- CLEAR = GAIN on both statistics. The TTFT guard bound = the stock A/A spread (648.5 ms) over the stock mean 3287.1 ms
  (absolute 3935.6 ms), fixed in the same read before the candidate's TTFT was compared.
- Ratio of record: 57.761 / 54.430 = 1.061x [d]. tok/s/user = 1000 / p90 ITL: 18.4 vs 17.3 [d].

## Conc 32: key-matched INFO (not a verdict) [m]

The client ends the profiling phase at a fixed time. One very long turn (`085aacf44ac5:351`, input 659,451 tokens)
starts before that end in some runs and after it in others. It was sent in 4 of 8 stock arms and in 0 of 3 q1 arms.
Its cold prefill makes p90 ITL larger in the arms that send it. Against the 4 stock arms that did not send it
(`judge/out/keymatched.out`, judge-c32-5 `band()`, n_cand 3, same seed):

```
KEYMATCHED-INFO conc32 official   ctl n 4 mean 56.825 cand n 3 mean 54.430 gain -2.395 band 1.574 -> GAIN ratio 1.0440x
KEYMATCHED-INFO conc32 consistent ctl n 4 mean 57.895 cand n 3 mean 55.825 gain -2.070 band 1.314 -> GAIN ratio 1.0371x
```

Both numbers are reported: 1.061x of record and 1.044x key-matched. The key-matched read is INFO only and does not
replace the verdict. Why the q1 arms sent the turn in 0 of 3 runs is [u].

## Conc 32: correctness gate [m]

`conc32/eval-c32q1ev/` (`eval_only`, the conc 32 server): `PASS: gsm8k exact_match,strict-match = 0.9682 (>= 0.9 from default)`.
The conc 32 recipe has no speculative drafts, so the accept-length floor does not apply.

## Judges

- `judge-c32-4` is the judge of record (SHA256SUMS 3095b5e5, c32.py 53bf1a26). Its code is in
  `internal/judge-checkpoint-6/judge-c32-4/` (its default paths name internal hosts; the bytes are unchanged).
- Known defect D1 (`judge/judge-c32-4.DEFECTS.md`): its band depends on the order of the control runs in the spec.
  The pre-registered order gives 1.847 / 2.460; other orders gave 1.762-1.847 / 2.324-2.460. The verdict is GAIN on
  both statistics at every order seen.
- `judge-c32-5` (SHA256SUMS 78ea7ade, c32.py facb8bce, `internal/judge-checkpoint-6/judge-c32-5/`) fixes D1: the
  bootstrap draws from the sorted readings. It was accepted after checkpoint 6 and is the judge for later conc 32
  reads. On the same arms it gives the same gain and verdict, bands 1.845 / 2.353 (`judge/out/c32_5.out`).
- KEYS rule: judge-c32-5 also prints KEYS lines (turns sent only by the candidate or only by the control arms). They
  are an INFO stratifier, not a verdict. A key-matched comparison is INFO only.
- `AUDIT.md`: an independent re-judge of this checkpoint from the raw arm files with read-only copies of both judges
  (`python3 -I -B`). The judge-c32-4 re-run is byte-equal to the read of record. `AUDIT-arms.md`: one row per arm.

## Conc 32 varsplit: HELD (not in this checkpoint)

A conc 32 variant of the MLA decode split (varsplit: per-row split counts for 23-32 decode rows at conc 32) passed its
kernel gate (saved 144.1 / 121.6 us per layer at 26 / 32 rows; no change at 4-16 rows). Its e2e read (3 arms
vs the same 8 stock arms, judge-c32-4) was VOID: official GAIN but consistent INSIDE, and the TTFT guard FAILED
(d +1993.5 vs bound 648.5). The varsplit arms sent turn 351 in 3 of 3 runs, the q1 arms in 0 of 3, so that read was
not key-matched. Status: HELD KEY-UNMATCHED. It needs new arms under judge-c32-5 before any re-read.

## Conc 1 and conc 8

Unchanged from checkpoint 5 (`results/checkpoint-5/`): conc 8 4.818 vs 5.385 ms (GAIN, band 0.510), conc 1 2.845 vs
3.019 ms (GAIN, band 0.125); 1.368x and 1.083x vs the stock readings. The servers are the same bytes, so there is no
regression by identity.
