# Item 05 ride-along reading (nsys evidence arm, B300; coordinator schedules; never a measured arm)

Question: at n 1-4k / ctx < 100k, does the prefill step run in the piecewise graph, and are its launch gaps + host tail
< 10 ms per step (roadmap clause 1 — predicted TRUE, STEP0.md)? Does the class-A prologue hole follow 0.6 + 0.037 x ctx_k ms?

Window: >= 10 prefill steps with n 1,024-4,095 at ctx < 100k (stock arm, switch off). Export the rank-0 sqlite
(`nsys export --type sqlite`), and the step bounds as families.csv from `family-gap-prefill/families_prefill.py`
(it supplies the step / n / ctx / wall columns these readers take).

| # | reader (CPU) | what it prints per step | decision |
|---|---|---|---|
| 1 | `graphsplit2.py <sqlite> <families.csv> <steps>` | kernels in graph vs eager; cudaGraphLaunch count; gaps charged to the kernel before them | graph live iff cudaGraphLaunch = 95 (92 eager breaks + 3) for every n <= 8,192 step |
| 2 | `gapnames.py <sqlite> <families.csv> <steps>` | gaps > 50 us by (kernel before -> kernel after) and the host API inside them | class A = `_apply_write_kernel -> _apply_write_kernel`; class B = per-KDA-layer pairs |
| 3 | `hostspan.py <sqlite> <families.csv> <steps>` | prologue hole (first `_apply_write` -> first `_prepare_prefill_inputs`), host launch span vs device busy | the hole column is valid only on the first step of a turn (a back-to-back step's prologue falls in the previous window: huge values) |
| 4 | `prologue.py <sqlite> <families.csv> <step>` | the first 80 device events with their launch times | names the host hole by hand on one step |

Verdicts (pre-registered 2026-10-19):
- V1 graph: cudaGraphLaunch = 95 on every window step with n <= 8,192 (else the STEP0 premise is wrong: report).
- V2 clause 1: median (inner gaps + host tail) < 10 ms over the window steps -> the roadmap's graph-coverage item is void.
- V3 form R basis: class-A hole on first-of-turn steps within +-30 % of 0.6 + 0.037 x ctx_k ms (ctx in thousands);
  form R stays useful iff the hole is >= 2 ms at the bucket median ctx (64k: 3.0 ms predicted).
- With PERFBOT_EXP_PREFILL_GRAPH=1 in a second window (optional): the class-A hole falls to <= 1 ms on first-of-turn steps.

Origin: these readers were written on the B0 trace R-UP-2b b0-spec-nsys (STEP0.md tables); graphsplit2 and gapnames
bound a step by families.csv's mark + wall; host threads are matched by globalPid (correlationId repeats across ranks).
