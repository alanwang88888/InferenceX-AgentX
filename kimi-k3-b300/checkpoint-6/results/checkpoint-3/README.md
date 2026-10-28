# checkpoint-3: the real-point ABA of land-5 (throughput, synthetic acceptance 3.84, fast protocol, conc 8)

Checkpoint of record: **land-5, official p90 ITL 5.150 ms / consistent 6.202 ms, CONFIRMED-ITL, guards ok; eval PASS (evland5, rule v2); audits done; gate 0 v2 PASS** (the land-5 eval_only pass has landed: see Eval of record below; `eval/`).

Three arms on one request, same image, same node, same flags, run in this order: `c0` = control (harness overlay only),
`land5` = land-5 (the checkpoint overlay), `c1` = control (harness overlay only). Per-arm files as in checkpoint 2
(`metrics.json`, `identity.json`, `server_command.txt`, `phases.json`, `server-ready-seconds.txt`, `point/`, `server.log.gz` sanitized).
Request-level: `arms.json`, `bundles.json`, `workload-revision.json` (head `9e593b8`, clean), `exit.json` (exit 0, finished 2026-10-20T21:34:36Z; arms c0 / land5 / c1 rc 0).
`eval/` is the eval_only request `evland5` (one arm `ev-land5`, real acceptance; exit 0, 2026-10-20T19:34:17Z).

## The set: land-5 = land-2 + E2b + MLA-LOOP + CKPT-COW

| member | switch (`land5/identity.json` `env_perfbot`) | what it is | class |
|---|---|---|---|
| land-2 (11 members) | as checkpoint 2 (`results/checkpoint-2/README.md`) | see checkpoint 2 | as checkpoint 2 |
| E2b (item 03), in land-3 | `PERFBOT_EXP_MLACORE=1` | MLA decode core fork with the variable-split port | fp64 tolerance 1.02 (judge v5.3); eval needed |
| MLA-LOOP (item 03), in land-4 | `PERFBOT_MLA_CORE_SPLITMMA=1` (= code default) | MLA core PV MMA split, no wait on P; the mark is at `mla_core/__init__.py:33-35` | bit-exact vs E2b [u item 03]; token-parity AUDIT open |
| CKPT-COW (item 04), in land-5 | `PERFBOT_EXP_CKPT_COW=1` | mamba checkpoint copy-on-write when the prefix-hit checkpoint column collides with the initial-state column (EQ, EQ_ALIGNED, LT); all other rows take the served path | tolerance (different chunking of the same history); bar-1 PASS B200 |

The harness pair `PERFBOT_EXP_CUTE_CACHE=1` / `PERFBOT_EXP_INITTIME=1` is set in every arm (`arms.json`).
Bundle tree sha256 `b470f7bfb7f9e8ff41056e3e54139907f839c0161e7c7eb5fb1859483e49b2e4` (108 files, class R2; = the evland5 tree).
Control overlay tree `44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a` (= checkpoints 1 and 2).
Served tree: `e471ad7a9bcaf96aba04e4cd7f1ff3d11463b053b3a35169140fc37bd8d34ae5` over 196 files (`land5/metrics.json` `overlay_tree_sha256`).
Served tree of the eval arm (`eval/ev-land5/metrics.json`, workload `b7aa262`): `d4cbef57176f69c1833c4d7004f264c58a418d4dfe52f09cfd758cd202f55c93`.
NOT EQUAL [m]. The cause is the pin, not land-5: the bundle tree is b470f7bf in both requests (`bundles.json`); base `overlay/` at `9e593b8` adds 7 files over `b7aa262` (perfbot_kernels/nullarm x3, pretrace x4) and one off-by-default switch line pair (PRETRACE, model.py + vllm_base.json); land-5 shadows none of them. Recomputed with the identity_gate.py walk (git archive overlay at the pin + land-5): `b7aa262` gives d4cbef57 (189 files), `9e593b8` gives e471ad7a (196 files), base `9e593b8` gives 41bbf877 = c0 = c1. Gate 0 as first worded (served digests equal) is not met.

**Gate 0 v2 (coordinator ruling 2026-10-20): PASS [m].** Rule: the bundle tree is equal in both requests AND the served digest recomputed at each pin by the identity_gate.py walk reproduces the recorded one.

| pin | arm | recomputed served digest | files | recorded in |
|---|---|---|---:|---|
| `b7aa262e4e967328981169cbf5858e91874976ae` | base + land-5 | `d4cbef57176f69c1833c4d7004f264c58a418d4dfe52f09cfd758cd202f55c93` | 189 | `eval/ev-land5/metrics.json` (equal) |
| `9e593b8c5289f5826bfe8283799003d923c06ed0` | base + land-5 | `e471ad7a9bcaf96aba04e4cd7f1ff3d11463b053b3a35169140fc37bd8d34ae5` | 196 | `land5/metrics.json` (equal) |
| `9e593b8c5289f5826bfe8283799003d923c06ed0` | base only (c0) | `41bbf8774e4b58270ae400f4c12c5dec5bbec6c4be8fc0eee61f4c78ae984c1c` | 95 | `c0/metrics.json` (equal) |
| `9e593b8c5289f5826bfe8283799003d923c06ed0` | base + ctl (c1) | `41bbf8774e4b58270ae400f4c12c5dec5bbec6c4be8fc0eee61f4c78ae984c1c` | 95 | `c1/metrics.json` (equal) |

Bundle tree b470f7bfb7f9e8ff41056e3e54139907f839c0161e7c7eb5fb1859483e49b2e4 in both `bundles.json` files [m].
Liveness: every land-5 switch has a `PERFBOT-SWITCH` line in `land5/server.log` (closes checkpoint-2 audit finding 5): 189 PERFBOT-SWITCH lines; CKPT_COW=1 x10, MLACORE=1 x8, MLA_CORE_SPLITMMA=1 x8, plus every land-2 switch (GLUE C/D/E/G/R/S/X2 x9, DRAFT_CHAIN x9, FRONTPIN x8, MLA_CTX_SEGMENTS=auto x8, PREFILL_GRAPH x8, STAGE=L x8) [m].
CKPT-COW engage count in `land5/server.log` (EQ / EQ_ALIGNED / LT): not available: the served code prints no EQ / EQ_ALIGNED / LT counter (server.log, metrics-after-point.prom) [m]; not required, official is below -0.557.

### Price lines per member (pre-registered [e]; roadmap price.py R() as is)

Transfer factors (price.py, decode share 0.74 / 0.71, p(b) UP2b): 1 ms/step at b5 = 0.0504 official / 0.0612 consistent ms per p90 token;
b4 0.0729 / 0.0771; b3 0.0851 / 0.0554.

| member | measured | transfer | price official / consistent ms/token |
|---|---|---|---|
| E2b | dl5 d5 -0.096 ms/step [m l3]; equals the in-request A/A (audit claim 6) | b5 factor, d5 only | 0.005 / 0.006; bit-equality proven on the timed path (splits=None) only (E2BBITS EQUAL) |
| MLA-LOOP | dl5 d5 -0.040 ms/step [m l4], inside the A/A band (neutral) | b5 factor | 0.002 / 0.002 |
| CKPT-COW | R_off / R_con [m i20cow] | 1.0 (already ms per p90 token) [e, point transfer] | 0.491 / 0.626 |
| land-2 (cp2) | land2 - mean(ctl) [m cp2] | none | -0.550 / -0.244 |
| **land-5 prediction** | | | **-1.048 / -0.878** (without CKPT-COW -0.557 / -0.252) |

## Protocol

Same as checkpoint 2: official InferenceX fast protocol (`AIPERF_EXPERIMENTAL_FAST`, 1200 s window, one warmup request per lane).
**Fast vs full caveat:** the leaderboard rows ran the 3600 s protocol (`results/checkpoint-1/LEADERBOARD-CONTEXT.md`); this checkpoint
did not. Throughput mode: synthetic acceptance AL 3.84 (measured 3.840 / 3.839 / 3.838 (c0 / land5 / c1) per arm), so CKPT-COW and the MLA members act on
step time only here; their acceptance effect is in the eval.
GPU clock/power sampler (workload `9e593b8`): `PERFBOT-GPU-SAMPLER` present in every arm (`arm.log`: on + stop rows=51000 / 50928 / 50688; `gpu-clock-power.csv`); median power / SM clock per arm c0 822 W / 2032 MHz, land5 851 W / 2032 MHz, c1 822 W / 2032 MHz (median over samples with utilization >= 50 %, 8 GPUs; limit 1100 W; SM clock at 2032 in every arm, no clock drop).

## The number

| arm | official p90 ITL ms | p90 tok/s/user | source |
|---|---:|---:|---|
| land5 | 5.150 | 194.2 | `land5/metrics.json` |
| c0 / c1 (stock controls, this request) | 6.450 / 7.110 | 154.9 / 140.6 | `c0/metrics.json`, `c1/metrics.json` |
| checkpoint 2 (arm `land2`) | 5.880 | 170.1 | `results/checkpoint-2/VERDICT.md` |

## Verdict

Rule, pre-registered before the request ran (cp1itl.py 7c89402): CONFIRMED-ITL if land-5 beats both controls by more than the control
spread on both p90 statistics; FAIL if the 95 % interval of land5 - mean(controls) is above 0 on either; NOT_RESOLVED otherwise.
Guards: TTFT p90 and output tok/s per GPU (all lanes) must not regress by more than the control spread.
Pre-registered overall enum: **NOT_RESOLVED** (official alone is expected to resolve). Result: **CONFIRMED-ITL**.

**M1 bar (official p90 ITL <= 5.05 ms): MISSED by 0.10 ms** (land5 5.150) [m].

| statistic | c0 | land5 | c1 | ctl spread | land5 - mean(ctl) [95 %] | pre-registered prediction |
|---|---:|---:|---:|---:|---|---|
| official p90 ITL ms | 6.450 | 5.150 | 7.110 | 0.660 | -1.630 [-2.211, -0.896] | -1.048 +- 0.83 |
| consistent (full-decode) p90 ms | 7.555 | 6.202 | 7.741 | 0.186 | -1.446 [-2.413, -0.834] | -0.878 +- 0.91 |
| guard TTFT p90 ms | 1287.6 | 1006.4 | 1293.4 | 5.8 | -284.1 ok | no regression |
| guard output tok/s/GPU | 42.860 | 45.322 | 42.848 | 0.012 | +2.468 ok | no regression |

Reading rule (pre-registered): inside prediction +- resolution = as predicted. Official above -0.557 = CKPT-COW did not carry to the
real point; the engage count above is then the first fact, not a re-run. Result [m]: official -1.630 (inside -1.048 +- 0.83), consistent -1.446 (inside -0.878 +- 0.91): as predicted on both. The pre-registered overall enum NOT_RESOLVED was wrong; land5 beats both controls on both statistics. STEPDEC (item 20 split.py, official band) land5 - mean(ctl): pure -0.639, collision -0.787, statistic -1.427 ms/token; consistent band: pure -0.723, collision -0.699, statistic -1.423. Control collision 1.383 / 1.880 (mean 1.632; of record 1.590).
Empty-completion requests per arm (known 0-0.7 % class): cp1itl failed fraction 0.0045 / 0.0000 / 0.0045 (c0 / land5 / c1); successful / total 442 / 459, 472 / 487, 445 / 462.

## Eval of record (`eval/ev-land5`, eval_only, real acceptance; eval rule v2, evalread.py 6a5fbcd)

| gate | floor / band | land-5 | result |
|---|---|---:|---|
| GSM8K exact_match strict | >= 0.90 | 0.9697 | PASS |
| AL vs ref = mean(land-3 1.5590, land-4 1.6007) = 1.5798 (one-sided) | >= ref - max(0.03, 0.042) = 1.5378 | 1.5568 (-0.0231) | PASS |
| AL cumulative vs stock 1.5898 | >= -0.06 | -0.0330 | PASS |
| gate failures | none | none | PASS |
| liveness (evalread NOT_LIVE check) | every arm_env item | 19 / 19; CKPT_COW=1 x10 and `ckpt_cow=on` x10 | PASS |
| land-4 AL rise (+0.0417 vs land-3, MLA-LOOP) | AUDIT: token parity | item-03 parity logs: 0 mismatches over 410,531,328 elements | AUDIT closed [m] by kernel parity |

Per-position rates p0..p6: 0.2719 0.1267 0.0712 0.0416 0.0242 0.0140 0.0071 (land-4: 0.2789 0.1371 0.0795 0.0491 0.0295 0.0173 0.0092).
**Rule-after-data caveat:** eval rule v2 (mean of content-equivalent predecessors as ref) was set by the coordinator after the evland5
data. Under v1 (ref = land-4's single reading 1.6007) evland5 reads FAIL by 0.0019. v2 re-judges evland3 and evland4 unchanged (PASS).
land-3 and land-4 differ by 0.042 on content item 03 calls bit-exact: that is the eval A/A noise of record.

## Audit (item 04 read-only re-judge, gate 4; coordinator ruling 2026-10-20)

Every diff of both reports, quoted verbatim (cut mechanically: every table row not reproduced as a plain "yes", every "Claims not shown by the artifact" item, the Verdict; generator item-02 scratch cp3/audit_section.py). Report paths are relative to the K3-UP work tree (internal; the reports themselves go under internal/ at publish).

E2b follow-up: E2BBITS (B300 kernel-only, frozen 70cc7a6 core vs timed 243a08d core at splits=None, out and lse bit for bit): **E2BBITS EQUAL rows=140 draws=2 gpus=8/8** [m, filler i02e2bbits, 8x B300, exit 0 2026-10-20T20:34:54Z; file blobs checked
before any GPU work]. Scope (coordinator ruling): EQUAL proves that the timed price applies to the frozen bytes on the timed path
(splits=None) only; the in-situ var-split path has no timed counterpart. On EQUAL the E2b price row returns to its pre-audit status with
this caveat, and E2b stays a member (cp3 measures the served bundle whole). On DIFFER, E2b leaves the member list (listed as run, not as a
member of record) and its switch is off in LAND-6.

### `items/audit/land5-rejudge-E2b.md` (sha256 9b1d4f3d0533; quoted verbatim)


Enum reproduced (per reading): rows not reproduced as a plain "yes"

| reading | booked enum | script output | reproduced |
|---|---|---|---|
| i03b5r4b, E2B_KERNEL | STABLE_GAIN | not emitted by any script. Rule ">= 3/4 GPUs STABLE, ratio + band < 1" appears first in CARD.md at 31187cd, the same commit that books the result [m]. Applied by hand: 3/4 stable, max ratio+band 0.9379 < 1 [d] | yes (by hand; rule written after the data) |
| l3, e2b in-situ enum | READMIT_ALL | not emitted by admit.py. Rule in LAND-2.md 5e: d5 <= -0.05 and admit PASS at d1-d4. -0.096 <= -0.05 and PASS [d] | yes (by hand) |

Numeric diffs: rows not reproduced as a plain "yes"

| booked claim | booked | reproduced | equal |
|---|---|---|---|
| gain "6.2-6.7 %" | 6.2-6.7 % | 1 - ratio_of_record = 6.37 / 6.47 / 6.71 %; 1 - (ratio+band) = 6.21 / 6.36 / 6.34 % [d] | yes, but mixed basis (see below) |
| g0 diagnostic in-block ratio | 0.930 (STATUS 10:4xZ) / 0.930-0.931 (CARD) | block range 0.9305-0.9312, median 0.9307 [m] | CARD yes; STATUS 0.930 rounds 0.9307 down (minor) |
| d5 interval | [-0.112, -0.079] | [-0.111..-0.112, -0.079..-0.080] over 3 seeds [m] | yes (within seed noise) |

Claims not shown by the artifact:

1. **Measured kernel is not the frozen E2b.** i03b5r4b timed the bundle `i03-b5fill` mla_core: `__init__.py` 2536827e and `kernel.py` 2d674046 = item-03 243a08d (E2b, 2026-10-19) [m]. The freeze 70cc7a6 (in l3-e2b, land-3) has 9185637d / a9548c0f = 2286b4b, which adds item 01's var-split carry (+24 lines kernel, +16/-4 init) [m]. The b5fill case declares no splits [c], but l3-e2b runs with `PERFBOT_EXP_B35=F,O` (var split on) [m]. So the kernel result of record did not time the frozen bytes or the var-split path. Same behavior at splits=None is [u] (no bit-equality check found on file for 2d674046 vs a9548c0f).
2. **"6.2-6.7 %" mixes two bases.** 6.2 % comes from 1 - (ratio + band) of g1; 6.7 % comes from 1 - ratio_of_record of g3 [d]. On one basis: 6.4-6.7 % (ratio of record) or 6.2-6.4 % (ratio + band).
3. **E2B_KERNEL enum was not pre-registered.** Its rule is first written in CARD.md at 31187cd, together with the result [m]. The rule is lenient (ratio + band < 1); the data clear it by a wide margin, so the enum does not depend on this.
4. **Primary R4b guard did not run.** `clock_arm_delta` is null in all 20 timed blocks [m]; every void/valid call came from the fallback (clock-spread 2x A/A). STATUS 08:30Z states this; the 10:4xZ booking does not.
5. **Seat name.** Session rows carry node `ef780a3c5f0b` (container id) and gpu "NVIDIA B300 SXM6 AC" [m]. <seat> (name redacted at publish) is shown only by dispatch.json / driver.log [m]. Session throttle reason 0x4 (power cap) on all 4 GPUs [m]; clocks 1590-1635 MHz in timed blocks [m].
6. **The in-situ gain is not resolved from noise.** e2b d5 -0.096 equals the in-request A/A (+0.096) and is inside admit's own d5 band (FLOOR 0.100) [m]. The bootstrap interval covers only sampling noise inside one window. Per the 12:06-12:07Z retraction, ctl-0 and ctl-end ran back to back before every candidate (l3 10:54:56-11:07:37; land2 11:16:52; e2b 11:32:43) [m, phases.json], so cross-server drift is unmeasured. READMIT_ALL needs only "no regression + d5 <= -0.05"; that holds, but the -0.05 threshold is below the measured A/A scale. LAND-2.md 5e says the A/A is "as large as the e2b effect"; the STATUS line does not.
7. **d4 raw regression.** e2b d4 delta is +1.135 ms/step (> band 0.557), admitted only by the mix-fixed column +0.443 [m]. n at d4: e2b 87 vs land2 477 [m]. "d1-d4 no FAIL" is true by the rule, but the raw d4 number is 12x the d5 gain [d] and is not in the STATUS line (it is in LAND-2.md 5e).
8. **Liveness.** l3-e2b server.log has no MLA-core / E2b marker [m]; only identity.json env has `PERFBOT_EXP_MLACORE=1` [m]. Booked as "liveness not instrumented"; STATUS 12:07Z "l3-e2b identity carries MLACORE=1 [m]" is an env claim, not a code-path proof.
9. **d5 interval tool.** admit.py 070adb2 prints no interval [c]. The booked interval is from an unnamed bootstrap (LAND-2.md 5e header). I reproduced it with my own bootstrap; the original script is [u].
10. **LAND-2.md 5e says "Reading order: ctl-0 / land2 / e2b / kcf / ctl-end".** phases.json shows ctl-end before land2 [m]. The retraction in the same section corrects this, but the first line was not struck.

Verdict:

> **REPRODUCED_WITH_DIFFS, confidence 0.85.** Every booked number and judge state reproduces from raw artifacts with the booked scripts; diffs are in basis and provenance (measured blob 243a08d is not the 70cc7a6 freeze; mixed-basis 6.2-6.7 %; post-hoc kernel enum; in-situ -0.096 not resolved from A/A scale; d4 +1.135 raw; land-5 runs E2b only under SPLITMMA=1 composition).

### `items/audit/land5-rejudge-MLA-LOOP.md` (sha256 0884e84c3b82; quoted verbatim)


Numeric diffs: rows not reproduced as a plain "yes"

| claim (source) | booked | reproduced | equal |
|---|---|---|---|
| power (15:24Z line "1049-1066 W") | 1049-1066 W | 1049.6-1071.7 W | NO: upper end 1071.7, not 1066 (null GPU g7) [m] |
| "cycles = wall within 0.004" (14:39Z, card) | <= 0.004 | per-GPU median wall-cycles up to 0.0041 (g0); per block up to 0.0042 | NO, small: 0.0041-0.0042 > 0.004 [m] |
| gain input to price (card s9, loopb_price.txt) | 1 - 0.8885 = 0.1115 ("median R4e wall") | median of 4 R4e walls = 0.88832 -> 0.11168 | NO, small: 0.0002 [d]. Price effect about +0.2 % (0.052 stays 0.052 at 3 digits) [e] |
| price 0.052 official / 0.039 b3-b5 | 0.0522 / 0.0387 | not re-run: the PB=UP2b price.py of record is not on a path I read; law T per b is not in the file | [u] |
| evland4 vs stock | +0.011 / +0.031 | +0.0109 vs 1.5898; the second stock value (+0.031) not checked | yes / [u] |

Claims not shown by the artifact:

1. Freeze smoke "default split, bit-equal to E2b path at b1/b3/b5" (14:47Z, 028 dev 2). I found no log file for this smoke. It is in the commit message of 41013eb and in the card only [u]. Indirect support [m]: item-03 bench/audit/parity_s1.log + parity_s2.log show land-3 core vs land-4 core: 300 calls, 800,256 rows, 410,531,328 elements, 0 mismatches, graph replay B1-B5 0 mismatches. The control lines (162.9 vs 147.5 us) show two different binaries. The sums match the card [d].
2. The fork-null arm is a real recompile (cute.compile no_cache). The session rows on GPUs 4-7 carry the candidate label "pkg:mla_core_smma" (a constant of loopb.py), and the judge log has no per-arm marker. Evidence is code only [c]: loopb.py child() asserts m.SPLIT_MMA and m.NO_CACHE per instance, and all 112 child steps exit rc 0 [m]. The card already marks "binary diff of the two compiles not taken" [u].
3. "The faster kernel lowers the clock at the power cap" is a correlation [m], not a cause. On the split GPUs the A/A blocks ran 1524-1575 MHz and the timed blocks 1466-1513 MHz. Both arms in one timed block read the same probe clock (difference 0-0.0002, except one g3 block +0.0069). The cause is [e].
4. loopb.json top-level field `state` and `ratio_of_record` use rule r4c, not R4e. In that field g3 reads "unstable" and its ratio is null. The kernel-only.log line "state=stable,unstable" is r4c. This is not a defect of the booked line (R4e was ruled of record at 13:37:38Z, before data at 14:31:44Z [m git ac50a31]). But a reader of loopb.json alone gets 3/4, not 4/4.
5. evalread band: AA_SHIFT 0.042 was set at 16:58Z after the evland4 data (16:42Z), and it equals the evland4 shift itself (+0.0417 < 0.042 by 0.0003). Under the 0.03 band pre-registered in LAND-2.md 5g, evland4 is "AL HIGH" (AUDIT), as item 02 first booked. The PASS of record therefore rests on a post-hoc band. The open AUDIT was closed by the parity logs (claim 1), which I confirm as [m].
6. Price 0.052 / 0.039 not re-derived [u] (see table).

Verdict:

> REPRODUCED_WITH_DIFFS, confidence 0.9. All enums (LAND_READING R4e 4/4, l4 PASS/NEUTRAL, evland4 PASS) and all booked kernel and l4 numbers reproduce to the printed digit; land-5 carries the frozen MLA-LOOP content and switch. The diffs are small text errors (power upper bound 1066 vs 1071.7 W; cycles-wall 0.0041-0.0042 vs "within 0.004"; price gain input 0.1115 vs 0.1117), one unlogged freeze smoke, and a post-hoc evalread band.


### Audit diff of the publish: CKPT-COW replaces 4 image lines (found 2026-10-20 at this publish) [m]

`scripts/make_image_diff.py --check` failed on 2 files of the served land-5 tree. CKPT-COW does not only insert tagged lines:
- `vllm/v1/kv_cache_interface.py`: the base lines `initial_state_col = ...`, `checkpoint_col = ...` and
  `and checkpoint_col > initial_state_col` become one tagged `perfbot_ckpt_case(...) is not None` condition.
- `vllm/v1/core/single_type_kv_cache_manager.py`: the line `physical_block_cap = 1 + int(has_partial_hit) + checkpoint_block` gets
  `+ cow_extra` (tagged).

With the switch off, `perfbot_ckpt_case` returns `GT` iff `checkpoint_col > initial_state_col`, and `cow_extra` is 0, so the off path
is the base [d, by reading]. The served bytes are what this checkpoint measured, so they stay.
- Ruling: the overlay rule now reads "image file + tagged additions + recorded tagged replacements, each listed".
- The replacements are in `patches/image-replaced-lines.json` (exact text).
- `make_image_diff.py --check` reverses each one, strips the tagged lines, and reproduces the image hashes 39ea5be3 / 40cdcd41 (RESULT PASS).
- Negative tests fail as they should: no record; a typo in a recorded base line; an untagged edit; drift in a recorded block.

Which gate skipped the strip test on the land-5 bundle: none ran it.
- The workload's `driver/tests/test_overlay_base.py` checks only the workload tree's own `overlay/`, never an arm bundle.
- The bundle composer drops each part's `vllm_base.json` as provenance.
- The measurement driver's dry run runs no strip test on arm overlays.
- At the CKPT-COW source commit (`0431162`, item 04 branch, never merged), `test_overlay_base.py` fails 2 [m]. Its commit message cites
  only the member's own probes (T1-T7), not the driver suite.

**Rule going forward (member entry):** a member's source commit must pass the driver suite, including `test_overlay_base`, or carry a recorded
replaced-lines entry. The driver dry run gains the strip / replaced-lines check as a mechanical gate on every arm overlay.

## Open items

- MLA-LOOP token-parity AUDIT (land-4, item 03): CLOSED (coordinator ruling 2026-10-20) [m] by kernel parity: item-03 parity logs, 0 mismatches over 410,531,328 elements, confirmed by item 04 (audit claim 1).
- CKPT-COW is a tolerance member; its bar-1 is on B200; no B300 numerics run of record [u].
- Fast protocol only (see Protocol).
- CKPT-COW has no engage counter in the served code. Its counter was written after the land-5 freeze and is not served.
- Attribution: land5 - mean(ctl), split by the step decomposition into ms/token [m]:

  | band | pure step | collision |
  |---|---:|---:|
  | official | -0.639 | -0.787 |
  | consistent | -0.723 | -0.699 |

  - The CKPT-COW share is consistent with its price but not resolved across runs [e].
  - About 0.46 ms of the pure-step drop has no priced member [u].
  - The next checkpoint request (c0 / land4 / land5 / land6, same run) separates them.
- CKPT-COW is insert-only in a later set (owner item 04); then its replaced-lines entries go away.

## Reproduce the verdict

    python3 scripts/ingest_results.py <out-cp3> checkpoint-3 --arms <arms-cp3.json> --guard land5,c0,c1 --verdict "<line>"
    python3 scripts/extract_metrics.py results/checkpoint-3/land5
    python3 scripts/check_overlay_identity.py      # e471ad7a..., 196 files
    python3 scripts/make_image_diff.py --check     # base hashes, with the recorded replacements reversed
