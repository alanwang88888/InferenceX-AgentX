# checkpoint-4: land-8 (land-5 + KdaSplit La + B4 occ2), conc 8 pair + conc 1 ABA + eval

<!-- Public copy of the item 02 package README. Seat names are labels (seat A / B / C); the
     internal publish steps are dropped. Paths under items/ and requests/ name internal records (internal/README.md). -->

Status: COMPLETE 2026-10-21 (all slots filled from landed, read results). Not published: the push needs the user's
approval. Tags: [m] measured, [d] arithmetic, [e] estimate,
[u] unverified.

User rule (coordinator 04:06Z): a checkpoint reports conc 8 AND conc 1 p90 tok/s/user. It shows no regression on either
vs the previous checkpoint. tok/s/user = 1000 / official p90 ITL (= p10 of per-request output_token_throughput_per_user;
checked [m] on cp4v2, CP4-NOTES.md). M1 is judged on official only.

## The set: land-8 = land-5 + La + occ2

- Bundle `items/item-02-p11-pdl-edges/bundle/land-8`, frozen 04:37Z, 0 writable files, `land-8.SHA256SUMS` (109 lines, fd390d61).
- vs land-5, 3 files change [m]:
  - `arm_env.json`: + PERFBOT_EXP_KDA_LA=1, + PERFBOT_EXP_KDA_B4OCC=1.
  - `perfbot_kernels/kdasplit_la/__init__.py`: item 04 5310ad8, blob 3f703e0d (+1 tagged PERFBOT-SWITCH line).
  - `perfbot_kernels/kdasplit_la/prepared.py`: item 04 e455608 = 5310ad8, blob b97afab3 (occ2: B4 verify row 1.8.2.4).
- Root record = land-5's `image-replaced-lines.json` (9128be3a).
- tree_digest a7777c742f02a7751b10897af38087e659cb78d966e6843e271ee486ec82978f (108 files).
- Served digest 6503addb23b2e1eaeab78d1db05b2674bf407d01f978c527e60674921ea69376 (203 files, held []).
  It is the same at 4b255e4 and f4b127e [m]. The cross-owner check by item 04 is a MATCH.
- Expected gain at conc 8 (item 04 price of record, [e]): La + occ2 = 0.0256-0.0260 ms official / 0.0288-0.0292 consistent.
  The occ2 part is 0.0112-0.0115 / 0.0118-0.0122, capped at the B4 exposed-KDA cap.
  This replaces the first figure, 0.032. At conc 1: 0.003. Both are far below the bars (0.66 / 0.625), so this pair cannot resolve the gain.
- CKPT-COW in land-8: the proof is blob identity with land-5. item 04 `items/item-04-p07-kda-ckpt/audit/ckpt_cow_land8_blobs.sh`
  (sha256 833a6fd0) gives PASS. The cp3 audit limit 3 still applies (`items/audit/land5-rejudge-ckpt-cow.md` §4.3, sha256 eaf5d648):
  the perf of record is bundle-r2 (2 member files), not the 5-file land-5 form, and cp3 measures the 5-file form.
- Prior art (links from item 04; each fetched HTTP 200 at 14:00:36Z by item 02 [m]; PR titles read [m]):
  - La lane and kda_b4occ (occupancy schedule of the same kernel): base = upstream vLLM CuTe DSL KDA decode,
    https://github.com/vllm-project/vllm/blob/49cb2bbf71cfd36ee5aa33d1e184b1cddff55294/vllm/models/kimi_k3/nvidia/specialized_decode/cutedsl/kda/kernel.py ;
    multi-token CuTe DSL KDA decode, FlashInfer `fused_kda_decode_multitoken`,
    https://raw.githubusercontent.com/flashinfer-ai/flashinfer/main/flashinfer/kda_kernels/fused_kda_decode_multitoken.py
  - KDA op: Kimi Linear, https://arxiv.org/abs/2510.26692 (https://github.com/MoonshotAI/Kimi-Linear); chunked delta-rule
    form: Gated Delta Networks, https://arxiv.org/abs/2412.06464; FLA, https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/kda
  - CKPT-COW (one-pass state checkpoints): https://github.com/vllm-project/vllm/pull/52789 ("Support internal prefill
    checkpoints for Mamba prefix caching") and https://github.com/vllm-project/vllm/pull/60888 ("Export prefill checkpoints
    at or before the initial-state column"; the per-row fresh gate is from its review, per item 04 [u]).

## Run identity

- land5land8: seat A, submitted 06:04:23Z, measure-2 @48c6a26, timeout 5200 / min 4160, arms file 8b46e63c (land8 first).
- conc1land8: seat B, 05:06:01Z, measure-2 @f4b127e, 2865 / 2290, arms 1aa57d9a.
- evland8: seat A, measure-2 @b42333d, exit 0 at 08:32:41Z, arms cbe96919, 3910 / 3130, eval_only fast.
- Dry-run sha vs run sha [m]: the dry runs were VALID at f4b127e. land5land8 ran at 48c6a26.
  f4b127e..48c6a26 = 8db5bd3 + 48c6a26, which touch driver/dispatch.py, driver/run_frontier.py and tests only.
  overlay/ and probes/ are unchanged, so the served digest is unchanged.

## Liveness (read first; an arm without it is VOID)

| arm | SWITCH-CHECK | kda_la=on | kda_b4occ=on B4=1.8.2.4 | ckpt_cow=on | la_liveness.py (5310ad8, ec57b682) |
|---|---|---|---|---|---|
| land8 (conc 8) | set=15 live=15 dead=- [m] | x8 [m] | x8 [m] | x10 [m] | LIVE rc 0 [m] |
| land5 (conc 8) | set=13 live=13 dead=- [m] | x0 [m] | x0 [m] | x10 [m] | DEAD rc 0 (--expect off) [m] |
| c1-land8 | set=16 live=16 dead=- [m] | x8 [m] | x8 [m] | x10 [m] | LIVE rc 0 [m] |
| ev-land8 | set=12 live=12 dead=- [m] | x8 [m] | x8 [m] | x10 [m] | LIVE rc 0 [m] |

## Autotune key per arm (W5 tuning of record; anything else = VOID)

| arm | PERFBOT-ATKEY |
|---|---|
| land8 | key=a4b725cd record=a4b725cd salt=- hit=35 tuned=0 intact=yes [m] |
| land5 | key=a4b725cd record=a4b725cd salt=- hit=35 tuned=0 intact=yes [m] |
| c1-land8 | key=a4b725cd record=a4b725cd salt=- hit=35 tuned=0 intact=yes [m] |
| ev-land8 | key=a4b725cd record=a4b725cd salt=- hit=35 tuned=0 intact=yes [m] |

SWITCH-CHECK counts [d]: land8 15 = land5 13 + KDA_LA + KDA_B4OCC; c1-land8 16 adds PERFBOT_CONC and LOADTIME.

## The number: conc 8 pair (land5land8, cp4judge.py dfe1ad11 on (land8, land5))

Bars (coordinator ruling): official 0.66 = pooled ctl A/A; consistent 0.625 = land5land7 same-run A/A.
Enum per statistic: GAIN (delta < -bar), LOSS (delta > +bar), INSIDE. Judged at the 3rd digit.

| statistic | land5 | land8 | land8 - land5 | bar | enum |
|---|---|---|---|---|---|
| official p90 ITL (ms) | 5.290 [m] | 5.240 [m] | -0.050 | 0.66 | INSIDE |
| tok/s/user (= 1000 / official) | 189.0 [d] | 190.8 [d] | 1.010x | | |
| consistent (ms) | 6.014 [m] | 5.863 [m] | -0.151 | 0.625 | INSIDE |

Run: out-land5land8, seat A, measure-2 @48c6a26 (clean), exit 0 at 07:16:50Z; read8.py 6aead87d.

land-5 content A/A, official [m]: 5.120 (cp4v2), 5.390 and 5.460 (land5land7, land-7 inert), 5.290 (land5land8). The cross-run spread is 0.34.

Reference rows (coordinator ruling):

| row | official p90 ITL (ms) | tok/s/user | consistent (ms) | note |
|---|---|---|---|---|
| stock of record (cp4v2 c0) | 6.59 [m] | 151.7 [d] | 7.508 [m] | the fixed M1 / 1.5x basis |
| checkpoint 3 of record (land-5, cp4v2) | 5.120 [m] | 195.3 [d] | 6.179 [m] | previous checkpoint |

No stock arm at the W5 key exists. The key confound is closed: the a4b725cd tactics are byte-identical to 7fa8e43d
(dev-46 [m]; `items/GAP-M1.md` §2.3 rev 15, sha256 da434426).

## The number: conc 1 (single arm; no same-run pair)

| row | official p90 ITL (ms) | tok/s/user | source |
|---|---|---|---|
| c1-land8 | 3.02 [m] | 331.1 [d] | out-conc1land8, f4b127e |
| stock c1-c0 of record | 3.080 [m] | 324.7 [d] | conc1pair, da2526f (cross-pin); the basis |
| checkpoint 3 c1 (c1-land5) | 2.754 [m] (consistent 3.046) | 363.1 [d] | out-conc1land5a, 4b255e4, seat A, 07:52:49Z |
| c1-land6 (reference, other content) | 2.924 [m] (consistent 3.066) | 342.0 [d] | out-conc1land6, 261d0a1 |
| conc1aba c1-land5a / c1-land8 / c1-land5b | 2.994 / 3.017 / 2.995 [m] (consistent 3.069 / 3.056 / 3.049) | 334.0 / 331.5 / 333.9 [d] | out-conc1aba2, seat C, 95ef4af, exit 0 at 13:52:45Z |

### conc1aba: same-request ABA (pre-registered `bundle/c1aba.py` f8f9fc15; NOTE column `bundle/c1quart.py` 7944cf37)

- Liveness [m]: c1-land5a DEAD (x0), SWITCH-CHECK 14/14; c1-land8 LIVE (kda_la x8, kda_b4occ x8), 16/16; c1-land5b DEAD, 14/14.
  gates_failed empty. ATKEY on all three: key=a4b725cd record=a4b725cd salt=- hit=35 tuned=0 intact=yes. All VALID.
- Official: A/A |2.994 - 2.995| = 0.000 < floor 0.171 -> band 0.171. delta = 3.017 - 2.9945 = +0.023 -> NO_REGRESSION.
- Consistent (note): A/A 0.021 < floor 0.062 -> band 0.062. delta -0.003 -> NO_REGRESSION.
- NOTE (c1quart): paired land8 - land5a n 152, p50 -0.003 ms, arrival-quartile medians -0.008 / -0.019 / -0.001 / +0.030;
  land8 - land5b n 153, p50 +0.000, quartiles +0.003 / -0.003 / -0.002 / +0.006. Mean vs land5b +0.123 with p50 0: a few tail requests.
- [d] In this run the p90 has no cliff (ranks 14-17 within 0.04 ms on every arm). The earlier c1-land5 2.754 (seat A) is
  0.24 below both same-run land-5 arms; the cross-run 0.266 was run variance, not land-8.
- Pin [m]: 95ef4af vs the b42333d dry run = W7 (eval AL window, sampling record) + W8 (seat query 60 s). Both serve.sh lines act
  outside the throughput serve path (a file copy after the point; an eval_only snapshot). All three arms share the pin.

Both c1-land5 and c1-land8: key a4b725cd hit=35 tuned=0; probes STEPLOG + CUTE_PRETRACE + LOADTIME; same arm form [m].
They differ in run, seat (seat A vs seat B) and pin (4b255e4 vs f4b127e, driver-only diff).

- vs stock: 3.08 / 3.02 = 1.020x [d, cross-pin].
- vs checkpoint 3 (c1-land5), cross-run [m, `bundle/c1pair.py`, paired by (conversation, turn), n 152, OSL equal 152/152]:
  - official p90 2.754 -> 3.021 (+0.266); consistent p90 3.047 -> 3.052 (+0.005); ITL p50 2.420 -> 2.417.
  - Per-request delta: p50 +0.017, mean +0.073 ms.
  - [d] The official p90 sits on a cliff: in c1-land5 the 16th-slowest request is 2.75 ms and the 15 above it are 2.95-7.37 ms.
    A few requests that cross the cliff move the p90 by about 0.2. c1-land6 vs c1-land5 shows the same cliff:
    paired p90 2.750 -> 2.945 (+0.195).
  - Verdict: UNRESOLVED [u]. There is no same-request A/A at conc 1. The consistent statistic and the median are equal.
    The official +0.266 is inside the cross-run cliff of about 0.2-0.27, which is not an A/A of record.

## Two-row table (README template, user rule)

| conc | official p90 ITL (ms) | tok/s/user (= 1000 / ITL) | ratio vs stock | ratio vs previous checkpoint |
|---|---|---|---|---|
| 8 | 5.240 | 190.8 | 1.258x (vs 6.59) | 0.977x cross-run (vs 5.120); same-run land8 vs land5 1.010x, INSIDE |
| 1 | 3.017 (conc1aba) | 331.5 | 1.021x (cross-pin) | same-run ABA vs land-5 mean 2.9945: 0.993x, NO_REGRESSION (band 0.171) |

## Guards (reported per arm)

| arm | TTFT p90 (ms) | out tok/s/GPU |
|---|---|---|
| land5 | 969.7 [m] | 45.111 [m] |
| land8 | 1188.2 [m] (+218.5, +22.5 %; outside the land-5 TTFT samples 872.6-969.7) | 45.062 [m] |
| c1-land8 | 1624.9 [m] | 17.740 [m] |
| conc1aba c1-land5a / c1-land8 / c1-land5b | 1658.0 / 1666.9 / 1650.5 [m] | 17.753 / 17.765 / 17.765 [m] |

conc-1 guard (k3_inferenceX `scripts/ingest_results.py --guard`, bound = control spread) [m]: TTFT p90 c1-land8 - mean(ctl)
+12.6 ms vs spread 7.5 -> `GUARD FAIL`, so the ingested enum is `NO_REGRESSION-GUARD` (+0.8 % TTFT). tok/s/GPU +0.006 vs 0.012: ok.
Split (`bundle/ttftsplit.py`) [m]: prefill mean 753.4 / 752.3 ms and queue mean 63.7 / 62.9 ms (land5a / land8); TTFT p50
equal. Paired by (conversation, turn), equal ISL: land8 - land5a n 35 p50 0 p90 +8; land8 - land5b n 33 p50 +2 p90 +9;
land5b - land5a (A/A) n 43 p50 -2 p90 +32 ms. [d] The breach is inside the same-request A/A; the bound of two p90 values
(7.5 ms) is narrower than that A/A. Ruling (coordinator): the guard bar per statistic = max(script control spread,
same-run same-content A/A of that arm shape) -> conc-1 TTFT guard PASS-WITH-NOTE. Disclosed to the user in the push request.

## Eval of record (ev-land8, eval_only, real acceptance; evalread.py 6a5fbcd)

- Rule: FAIL if GSM8K < 0.90, gate_failures is not empty, AL < ref - max(0.03, AA_SHIFT 0.042), or AL < stock - 0.06.
  AUDIT if AL > ref + the same bound. NOT_LIVE if a switch has no PERFBOT-SWITCH line.
- Coordinator ruling (from checkpoint 4 on): the primary eval quality statistic is token parity of the greedy GSM8K run
  vs the previous checkpoint (`bundle/evparity.py`, sha256 0bb4c8cc). Bit-identical = PASS. A tolerance-class landing
  passes on GSM8K >= 0.90 and names its differing docs. Replay AL is a NOTE, not a verdict, until the eval arm bounds
  its window by request count on a fixed set (dev-46). No eval A/A arm for AL.
- Parity row [m]: ev-land8 vs ev-land5 1319/1319 identical -> PASS. vs stock: 436 docs differ, the same 436 since land-2.
- AL reference (NOTE only) = mean of the content-equivalent predecessors' eval readings: 1.5590, 1.6007, 1.5568 -> 1.5722 [d].
  The FAIL floor is 1.5302 (one-sided), the AUDIT edge is 1.6142, and stock is 1.5898.

| GSM8K strict | AL | AL - ref | verdict |
|---|---|---|---|
| 0.9697 [m] | 1.5328 [m] | -0.0394 (floor 1.5302: margin +0.0026) | PASS [m] |

- Run: out-evland8, measure-2 @b42333d (clean), exit 0 at 08:32:41Z. evalread.py at f5e4646, whose only diff from 6a5fbcd is
  one docstring line.
- Pool: land-3 1.5590, land-4 1.6007, land-5 1.5568. These are the evals of the bundles whose content land-8 builds on
  (land-8 = land-5 + La + occ2). land-6 1.6039 is out: it carries F2 + P-SCALE B, which land-8 does not.
- Stock drift: 1.5328 - 1.5898 = -0.0570 (cap -0.06: margin +0.0030). vs land-5 alone: -0.0240.
- Both margins are about 0.003 AL. A next landing that lowers AL by about 0.003 fails the rule.
- Liveness [m]: see the liveness table; evalread: 24 liveness items seen, missing none.
- Requests [m]: 315 = 15 warmup (11 with InvalidInferenceResultError) + 300 profiling (1 error). 299 profiled;
  the 16 dropped = 15 warmup + 1 profiling error.
- AL window [m] (`bundle/evpoint.py`): AL = 1 + accepted/drafts over the after-point prom window, which is the agentx
  replay point (conc 8, 1200 s duration bound), not lm-eval. Each run profiles a different request set (285-307 requests).
  Replay AL: stock 1.5898, land-2 1.5597, land-3 1.5590, land-4 1.6007, land-5 1.5568, land-6 1.6039, land-8 1.5328
  (range 0.071; not monotone).
- Greedy parity [m] (`bundle/evparity.py`): GSM8K (temperature 0) responses of land-2/3/4/5/6/8 are identical,
  1319/1319 docs. All differ from stock on the same 436 docs. No landing after land-2 changed greedy output.
- [d] The AL bounds (0.042, 0.06) are inside the replay's own spread. There is no same-bundle eval A/A.
  The 0.003 margins carry no content signal.
- Second TTFT sample: the eval arm runs lm-eval at another concurrency and request mix, so its TTFT does not compare
  with the conc 8 pair [d].

## M1 and ratio vs stock

| row | conc 8 | conc 1 |
|---|---|---|
| M1 at conc 8 (official <= 5.050) | NOT MET: 5.240 [m] (expected NOT MET [e]: 5.094) | n/a |
| ratio vs stock (stock official / land-8 official) | 1.258x (vs 6.59) | 1.020x (vs 3.080; cross-pin) |

## Push criteria (user decision, relayed by the coordinator: structural builds open at the checkpoint-4 push regardless of M1)

| criterion | rule | status |
|---|---|---|
| no regression, conc 8 | land8 - land5 is not LOSS on official (bar 0.66) or on consistent (bar 0.625) | PASS: -0.050 / -0.151, INSIDE / INSIDE [m] |
| no regression, conc 1 | coordinator ruling: c1-land8 vs stock c1-c0 3.080 (checkpoint 3 has no c1 arm of record). c1-land8 3.020 < 3.080, delta -0.060 [m/d]: not a regression. A/A band at conc 1 [u] until c1-land5 a/b lands; c1-land5 is pending and does not block. Confounds: the comparison is cross-pin (da2526f vs f4b127e) and cross-key (c1-c0 ran before the W5 record). User ruling (relayed by the coordinator): the push is HELD until conc1aba resolves this slot against its own same-request band (c1-land5a / c1-land8 / c1-land5b) | PASS: conc1aba NO_REGRESSION, delta +0.023 inside band 0.171 (consistent -0.003 inside 0.062) [m]; vs stock 1.021x (3.080 / 3.017, cross-pin) |
| liveness | SWITCH-CHECK dead=- on every serve arm, kda_la=on and kda_b4occ=on marks, la_liveness LIVE; ATKEY hit=35 tuned=0 on every arm | PASS for land8, land5, c1-land8, ev-land8 [m] |
| eval | greedy GSM8K token parity vs checkpoint 3 (land-5), evparity.py 0bb4c8cc; GSM8K >= 0.90 | PASS: 1319/1319 identical, GSM8K 0.9697 [m]. NOTE: replay AL 1.5328 (old rule PASS, margins 0.0026 / 0.0030; not a verdict) |
| guards | TTFT p90 and out tok/s/GPU inside the spread | tok/s/GPU PASS; TTFT p90 +218 ms: PASS-WITH-NOTE (tail-only, < 2 % of requests; paired median equal; no prefill path in the landing); split below. conc 1: TTFT +12.6 ms vs spread 7.5 = mechanical `GUARD FAIL` [m], PASS-WITH-NOTE (same-run A/A band +32 ms; script spread 7.5 ms too narrow) |
| audit | item 08 section clean, or its diffs quoted | REPRODUCED (`items/item-08-experts-stage-endstate/audit/CP4-REJUDGE-RESULT.md`, sha256 1e87f63d): Tier A conc1land8, evland8 (0.9), conc1aba (0.95) REPRODUCED; land5land8 REPRODUCED_WITH_DIFFS (0.9), the one diff = TTFT p90 +218.5 ms, ruled PASS-WITH-NOTE; Tier B 0.9; Tier C 0.85 (no bit-equality proof of the merged files) |

TTFT guard split (`bundle/ttftsplit.py`; all lanes, profiling requests; vLLM histograms = after-point minus after-load) [m]:
1. conc 8, land5 -> land8 (n 467 / 469): TTFT p50 is equal (head 10 % 692 / 699, rest 335 / 334 ms). The p90 moves:
   head 10 % 2002 -> 2818, rest 912 -> 1067 ms.
2. p90 per run decile: land5 2002 920 499 837 582 958 1059 1158 1002 543; land8 2818 981 402 871 683 1166 1123 1240 1571 544.
3. Server: queue mean 334.7 -> 338.6 ms; prefill mean 696.0 -> 722.6 ms (+3.8 %), n 486 each.
4. Paired by (conversation, turn), equal ISL, n 114: delta p50 0, p90 +191, mean +22 ms.
5. conc 1, c1-land6 (no La) vs c1-land8 (La): paired n 35, p50 +0, p90 +4 ms. Prefill mean 743.5 / 741.7 ms. No La TTFT cost.
6. [d] Tail-only: about 5-10 requests, in decile 1 and decile 9. Medians are equal, so there is no per-request prefill-path cost.
7. Cause class NEITHER (item 04, from the server logs): no compile after ready; La is decode-plan only (no prefill path).
   Item 04's pairing (465 requests) gives median diff -1.0 ms; quartile medians -1.0 / +0.2 / -2.1 / +0.3. The p90 moved on
   about 8 queue / collision tail events. My pairing with an equal-ISL filter is the n 114 above.
   Ruling: PASS-WITH-NOTE. Second TTFT sample: none; the eval arm's TTFT does not compare with the conc 8 pair [d].

CARD-LENS / anti-hacking note for item 08 (checkpoint-5 audit): check that every eval verdict is computed on the window
it names (AL counters = the replay point, not lm-eval), and that parity compares greedy responses of the same 1319 docs.

M1 is reported, not a push criterion.
Deviation from the user rule, for the user at push approval: the rule says "no regression vs the previous checkpoint" at
conc 1. Checkpoint 3 has no conc 1 arm of record, so the coordinator ruled that conc 1 is judged vs stock 3.080 for this
checkpoint.

## Verdict

checkpoint-4 = land-8: conc 8 pair NO_REGRESSION vs checkpoint-3, conc 1 ABA NO_REGRESSION, eval parity 1319/1319 GSM8K 0.9697, audits REPRODUCED (land5land8 with the TTFT note); official conc 8 5.240 (1.258x vs stock 6.59), conc 1 3.017 (1.021x vs 3.080)

## Re-judge audit (item 08, cross-owner auditor)

Item 08, read-only, run 2026-10-21T07:18Z-07:21Z after the land5land8 landing (exit 0, 07:16:50Z). No GPU script ran.
Cut rule as checkpoint 3: every row not reproduced as a plain "yes", every claim not shown by the artifact, the verdict.
Full report with all rows: `items/item-08-experts-stage-endstate/audit/CP4-REJUDGE-RESULT.md`; plan: `.../audit/CP4-REJUDGE-PLAN.md`.

Script versions run (cited copy from `git show` or file copy, read-only; on-disk sha256 beside it) [m]:

| script | cited | sha256 run | on disk |
|---|---|---|---|
| la_liveness.py | 5310ad8 | ec57b682... | same |
| cp4judge.py | dfe1ad11, as cited: BAR official 0.66 / consistent 0.19 (not changed) | dfe1ad11... | same |
| split.py + cp2io.py | 3c7cb5d6 + 72c0fccd | same | same |
| cp4read.py + purediff.py (item 03) | 1e4e64f4 + ac936b49 | same | same |
| evalread.py | 6a5fbcd, staged only (evland8 pending) | 803f4ac3... | 55fdad48... (f5e4646, one docstring line) |
| ckpt_cow_land8_blobs.sh (item 04) | 833a6fd0 | 833a6fd0... | same |

### Tier A: arms

land5land8: liveness land8 `PERFBOT-LA-LIVENESS LIVE expect=on | server.log kda_la=on x8 kda_b4occ=on x8`, land5 `DEAD expect=off`;
SWITCH-CHECK 15/15 and 13/13; ATKEY `key=a4b725cd record=a4b725cd salt=- hit=35 tuned=0 intact=yes` on both; trees a7777c74 / b470f7bf;
served land8 6503addb; gate_failures [] [m]. cp4judge dfe1ad11 (script bars) [m]: official 5.290 -> 5.240, -0.050 (bar 0.66) INSIDE;
consistent 6.014 -> 5.863, -0.151 (bar 0.19) INSIDE. **Ruled land-8 bars (ST:1204): official 0.66 / consistent 0.625: INSIDE / INSIDE [d].**
conc1land8: every booked value and line reproduces (LIVE, 16/16, ATKEY, 3.02 / 3.051, 1624.9 ms, 17.740) [m]. evland8: PENDING (no out dir at 07:20:26Z).

Rows not reproduced as a plain "yes":

| check | booked / expected | reproduced | yes |
|---|---|---|---|
| guard TTFT p90 (land5land8) | no guard enum with one ctl (CN:17) | land5 969.7, land8 1188.2 ms, +218.5 (+22.5 %) [m]; cp4read prints `GUARD land8 ... FAIL` vs the cp3 bound 5.8 [m]; land5land7 same-content A/A 86.8 ms (CN:129) | **NO by p90; ruling of record PASS-WITH-NOTE (STATUS:1434, guard rule addendum: p90 breach on < 2 % of requests is judged on the paired per-request distribution; 465 paired, median -1.0 ms, ~8 tail events; second sample = evland8) [d]** |
| guard out tok/s/GPU | as above | 45.111 / 45.062, -0.049 [m]; cp4read bound 0.012: FAIL; land5land7 A/A 0.203 | inside the A/A, NO by the cp3 bound |
| cp4judge DECISION line | not applicable (cp4v2 F2 rule) | `DECISION land8-land5 CANDIDATE` [m] | n/a, ignore |
| La + occ2 price (ruling 8) | 0.026 official | -0.050, inside +-0.66 [d] | not resolvable at this bar |

### Tier B: La + occ2

All booked rows reproduce [m]: La kernel.py 9346b193 = timed bundle-la-ko; land-8 `__init__.py` minus the line tagged `PERFBOT P-07 La: serve liveness mark`
= 7b465725 = timed blob; land-8 `prepared.py` minus the 6 lines tagged `PERFBOT P-07 b4occ` = 4910a731 = timed blob = land-5 = land-1;
La check.log 14/14 bit_equal_inc; B5 14.26 -> 10.48 us; occ2 prepared b97afab3 = frozen land-7; i04b4occ / i04b4delta / i04b4member gains, bars and BUILD lines equal the booked values.

| check | booked | reproduced | yes |
|---|---|---|---|
| occ2 i04b4occ b4 gain | 3.283 (ST:994) | 3.28399986 -> 3.284 [m] | yes (STATUS truncates; minor) |
| La price 0.012; occ2 price of record (ruling 8) | as booked | not re-derived | [u] |

### Tier C: blob checks (13 members = the 14 land-5 members minus 08L, which item 05 audits below)

Generator [m]: land-8 vs land-5 sums differ only at arm_env.json and the two kdasplit_la files (+ the record file, same bytes 9128be3a),
so every Tier C member serves its land-5 bytes. Equal to the priced bundle [m]: draft chain (29/29), prefill graph (14/14), ctx segments (4/4),
cs35 (EXTRA_ARGS value), CKPT-COW (ckpt_cow_land8_blobs.sh 833a6fd0 rc 0 `VERDICT PASS`, output byte-identical to .out b3326cc1).

| member | priced bundle | land-8 vs priced [m] | yes |
|---|---|---|---|
| D + G | bundles/i02-tail-dg | glue.py a90255d1 vs d9edf1be: 5 priced lines absent (lever docstring, `LEVERS = frozenset(...)`), 15 untagged extra lines; other 14 files equal | **MERGE-EQUIVALENT (item 02 [m]: docstring rewrite + LEVERS / EXCLUSIVE supersets; ACTIVE sets identical under GLUE "D,G" / "D" / "G" / ""; status kept)** |
| X2 | bundles/i08-land-r2 | glue.py a90255d1 vs a5dee244: superset (0 absent, 4 extra, 2 untagged) | **NO (superset)** |
| S | bundles/i06-s-r2 | glue.py as X2; front.py b9442cc2 vs 9055ad2e: superset (1 tagged extra) | **NO (superset)** |
| pin | item-17 bundle | front.py b9442cc2 vs 49ce8556: superset (5 tagged extra); 11 files equal | **NO (superset)** |
| L6 | bundles/i01-v2 | mla/prepared.py 83b8aa65 vs bd356357: 1 priced line absent (`cp.update(mla_varsplit.decode_args(...))`); providers.py superset; 7 files equal | **MERGE-EQUIVALENT (item 01 [m]: same statement, trailing comment changed; rejudge_lines.py reads SUPERSET; status kept)** |
| E2b | bundles/i03-b5fill (timed 243a08d) | mla_core kernel.py 32d176c4 vs 457cb4b0, __init__.py 98c62bf8 vs 190915ce | **NO (known, CP3 E2b claim 1)** |
| MLA-LOOP | bundles/i03-loopb4 | kernel.py equal; __init__.py 98c62bf8 vs 70c427be (default flip of SPLIT_MMA at 41013eb) | **NO (known)** |

### Claims not shown by the artifact

1. TTFT p90 on land8 is +218.5 ms (+22.5 %), 2.5x the land5land7 same-content A/A (86.8 ms) [d]. No guard enum is pre-registered for a one-ctl run. I did not re-run the TTFT split (ST:1432) [u].
2. cp4judge dfe1ad11 carries the consistent bar 0.19, not the ruled 0.625. The enum is INSIDE at both, so the ruling does not change the result [d].
3. The La + occ2 price (0.026 official) is 25x below the bar 0.66 [d]; this pair cannot confirm or reject it.
4. split.py, official band: land8 pure +0.102, collision -0.216, jitter +0.004 vs land5 [d]; purediff `NO_CONTENT_PURE_GAIN (fit +0.006, slope -0.017, drop -0.102)` [m] names request mix. A collision-only change is not shown to come from La + occ2 [u].
5. evland8 has not run: the eval gate and ev-land8 liveness are open.
6. occ2 bits are a safety result only (C == S on all 5 cases, LAND7-DELTA.md:31-32); the serve proof is the switch line + the B4 timing.
7. CKPT-COW, cp3 audit limit 3: scheduler.py and mamba/checkpoint.py are not in the priced bundle-r2 (script: `=0431162=land-5 (not in priced bundle)`); they are priced only through the 5-file reads [m].
8. X2, S, pin: the merged glue.py / front.py hold every priced line plus other members' lines; no bit-equality proof of a merged file exists [u].
9. D + G and L6: land-8 drops priced lines (glue.py 5, L6 prepared.py 1) [m]; cause not traced by me [u]. The checkpoint-2 audit covers those land-2 bytes. Sent to item 02 (D + G) and item 01 (L6); their answers are appended here.
   Answers (coordinator, closed): both MERGE-EQUIVALENT, status kept (rows above). Fix for checkpoint 5 [m]: `items/item-08-experts-stage-endstate/audit/rejudge_lines.py`
   (bd189d80, CPU) compares code lines after ast.parse + docstring drop + ast.unparse (.json by priced keys). On land-8: L6 SUPERSET; D + G DIFFER on the 2 real
   statements only (LEVERS, EXCLUSIVE; docstring noise gone); X2 / S / pin / cs35 SUPERSET; CKPT-COW 2 equal + 2 not in bundle-r2 (= limit 3); E2b, MLA-LOOP DIFFER (known).
   Self-test [m]: comment + docstring change gives AST_EQUAL; a removed statement gives DIFFER.
10. E2b: E2BBITS (70cc7a6 vs 243a08d) does not cover the land-8 mla_core bytes [m blobs]. No GPU script was run.

### Verdict

> **Tier A land5land8: REPRODUCED_WITH_DIFFS, confidence 0.9** (liveness, ATKEY, digests and both enums reproduce; TTFT p90 +218.5 ms, guard PASS-WITH-NOTE by the coordinator ruling STATUS:1434). **conc1land8: REPRODUCED, 0.95. evland8: PENDING.**
> **Tier B La + occ2: REPRODUCED, confidence 0.9** (tagged-line strip gives the timed blobs exactly; kernel-only numbers reproduce; prices not re-derived).
> **Tier C: REPRODUCED, confidence 0.85** (land-8 = land-5 bytes for all 13; 5 equal their priced bundle; D + G and L6 MERGE-EQUIVALENT by owner answers (coordinator ruling); X2 / S / pin superset; E2b and MLA-LOOP known at checkpoint 3; no bit-equality proof of the merged files).

### 08L (item 05)

1. Blobs: the 5 `perfbot_kernels/experts_stage/` files of land-8 (6b1397b5 / bf5ec7f0 / 73b3e1ce / 0bdfbc45 / c2024177) equal, by sha256, the binary the 08L price was measured on (item 08 i08-r2dl5, whose tree still hashes to the out-i08L submit record b5c3cef7) and the l2b admission bundle l1b-l6-pin-08l [m audit_08l.py 9a5a45bc: AUDIT_08L PASS].
2. Served: the land-8 tree hashes today to the tree out-land5land8 served (a7777c74, driver tree_digest); land-8 arm_env holds `PERFBOT_EXP_STAGE=L`, and land8/server.log has 8 `PERFBOT-SWITCH PERFBOT_EXP_STAGE=L` lines (experts_stage/__init__.py:56) [m].
3. Checkpoint-3 row: README-checkpoint-3.md:41 (liveness, STAGE=L x8). The checkpoint-3 re-judge audit (item 04) covered E2b and MLA-LOOP only, so this is 08L's first cross-owner blob check. Verdict: PASS; the 08L price of record stands on the binary it was measured on.

## Open items

- None for this checkpoint. The conc 1 A/A band is closed by conc1aba (NO_REGRESSION).
- For checkpoint 5: replay AL becomes a fixed request-count window (dev-46); the ingest guard takes the same-run A/A as its
  bar when one exists (dev-46); the clean-vs-probe split ABA is open (GAP-M1 §2.3).
