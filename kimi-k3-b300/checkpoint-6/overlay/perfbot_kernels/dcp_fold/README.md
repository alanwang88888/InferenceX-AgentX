# dcp_fold

## Kernel-gate history (bar: save >= 120 us/step = 24 x target T48 + 5 x draft T42; LOG.md rows D-fold-01..03)

- D-fold-01 (v1: tpb 1, fence in every thread, PDL): correctness PASS; 48.4 us/step, FAIL. The dispatch arrival cost
  ate the fold (dispatch + combine alone: -0.6 us/pair).
- D-fold-02 (tpb 4 + fence1, PDL): correctness PASS; 153.1 us/step, PASS. Alt PDL off: 236.6.
- D-fold-03 (memory-model fix below, same config): correctness PASS; 98.1 us/step, FAIL. Alt PDL off: 172.3.
- Noise: the same config moved 2-4 us/pair between passes (default 153 -> 98; tpb 2 at T48 +4.6 -> +0.2 -> -2.8),
  while every rank agreed within 0.15 us/step. The gate ranks configs only coarsely.
- PDL off is the default (PERFBOT_DCPFOLD_PDL=0; "1" = PDL on; both in the compile key). It was the only config above
  the bar in both D-fold-02 and -03. The isolated chain has no neighbour kernels, but in the server PDL changes the
  overlap with qkv/attention/W_UV, so the e2e decides PDL on vs off.

# Memory-model pattern check (fence1 publish)

This is a pattern check, not a formal proof, and it does not use the stress tests as evidence. Line numbers refer to
`kernel.py` (sha in `source_sha256()`, 2026-10-17 after the D-fold-02 fix).

PTX ISA "Memory Consistency Model" terms used here (PTX ISA 9.4):
- **8.8 release pattern**: a release fence, or an acquire-release fence, followed by a strong write; or a release
  operation.
- **8.8 acquire pattern**: a strong read followed by an acquire fence; or an acquire operation.
- **8.9.4 synchronizes-with**: release pattern X synchronizes with acquire pattern Y when a write in X precedes a read in
  Y in observation order, and the first operation of X and the last operation of Y are morally strong (8.7: both strong,
  each one's scope includes the other one's thread).
- **8.9.4 barriers**: "A `bar{.cta}.sync` ... operation *synchronizes* with a `bar{.cta}.sync` ... operation executed on
  the same barrier." `bar.warp.sync` is **not** in this list, so a warp barrier gives no causality edge.
- **8.9.2 observation order**: a write W reaches a read R either directly or "through an optional sequence of atomic
  read-modify-write operations".
- **8.9.5 base causality order**: program order and synchronizes-with, chained transitively. A release pattern orders
  every operation in its causality prefix ("cumulativity").

## Shared arrival `_Fold.arrive` (QGather and Dispatch, fence1 = default)

| step | line | operation | pattern role |
|---|---|---|---|
| A1 | 297 / 356, 361 | every thread stores data (multimem.st.relaxed.sys / st.global, weak) | payload |
| A2 | 252 | `bar.sync` by all threads of the CTA | 8.9.4: thread j's bar synchronizes with thread 0's bar, so A1 of every thread precedes A3 in base causality |
| A3 | 254 | thread 0: `fence.sc.sys` | release fence, scope sys (multicast / NVLink peers need sys) |
| A4 | 261 | thread 0: `atom.add.relaxed.gpu` on ctr | strong write. A3 + A4 is a release pattern. The reader is on the same GPU, so gpu scope makes A4 morally strong with A5 |
| A5 | 261 | last CTA, thread 0: the same RMW reads the final count | strong read. Observation order chains through the other CTAs' RMWs (8.9.2), so every CTA's A4 reaches it |
| A6 | 263 | last CTA, thread 0: `fence.acq_rel.sys` | A5 + A6 is an acquire pattern. A3 (sys) and A6 (sys) are morally strong, so each CTA's A3/A4 synchronizes with A6, and every CTA's A1 precedes A6 |
| A7 | 262, 264 | thread 0: ctr := 0, epoch := e+1 (`st.relaxed.gpu`) | after A6 in program order (see edge 5) |

The non-fence1 path (lines 256-257: every thread fences, then bar.sync) is the image's order and also a release
pattern for each thread.

## Edge 1: QGather data → peer signal → peer reads q

| # | line | check |
|---|---|---|
| 1 | 297, 252, 254 | All threads' multicast stores precede thread 0's sys fence: rows A1-A3 |
| 2 | 254, 261 | The fence is `.sys`; the following write (atom) is strong: rows A3-A4 |
| 3 | 261, 263, 300 | Last CTA: RMW read, then `fence.acq_rel.sys` (acquire), then **the same thread** issues `multimem.st.release.sys` (a release operation, 8.8) on the signal. Cumulative over all CTAs' data |
| 4 | 302 → `poll` 165/169 | Peer: `ld.acquire.sys` spin on the signal (acquire operation, sys, morally strong with line 300). The q data is read by the **next kernel** (attention). That edge is the end of the grid, or `griddepcontrol.wait` under PDL. This is CUDA's stream-order rule, not a rule of the PTX memory model. The image has the same edge (one thread polls, then the kernel ends) |
| 5 | 263-264 | Epoch: see Edge 3 |

## Edge 2: Dispatch payload → counter → last CTA → 8 signals → Combine reads

| # | line | check |
|---|---|---|
| 1 | 356, 361, 252, 254 | All threads' peer stores (st.global to peer symmetric memory) precede thread 0's sys fence: rows A1-A3 |
| 2 | 254, 261 | The fence is `.sys`; the atom after it is strong: rows A3-A4 |
| 3 | 261, 263, 365 | Last CTA, thread 0: RMW read, then `fence.acq_rel.sys` (acquire, and the release fence for what follows), then **the same thread** issues 8 × `st.relaxed.sys` signal stores. Fence plus strong sys write is a release pattern (8.8), cumulative over the other CTAs' payloads acquired at A6. **Changed** from v2: lanes 0-7 had issued `st.release.sys` after `bar.warp.sync`, an edge the model does not define |
| 4 | 399 → `poll` 165/169; 400; 405; 428 | Combine, lane r: `ld.acquire.sys` spin on signal r, then lane r's own LSE read (program order). Every thread's data reads (line 428) follow a CTA `bar.sync` at line 405, which synchronizes with lane r's `bar.sync` (8.9.4). **Changed** from v2: there it was `bar.warp.sync` |

## Edge 3: epoch write-back (both kernels)

Every CTA reads e at entry (lines 288 / 330) and arrives later (line 261). The last CTA writes e+1 (line 264) after A6.
So each read precedes the write in causality order and cannot observe it (coherence). The next call reads the epoch
(lines 288 / 330 / 394) in a later grid on the same stream, after `griddepcontrol.wait` (line 246) or the end of this
grid. That is stream order, not the PTX model. In stream order, all of this grid's data and signals are performed
first. The epoch and ctr are local, so no peer reads them.

## Multimem stores (lines 67, 73): residual risk

PTX 8.2.3 says: "A multimem address is a virtual address which points to multiple distinct memory locations across
devices." It also says: "Only multimem.* operations are valid on multimem addresses." The memory-model chapter says
nothing more: no other section mentions multimem addresses. So the model does **not** say:
- that a `multimem.st` counts as one strong write to each replica location;
- that a unicast `ld.acquire.sys` on the replica "overlaps completely" (8.7, condition 3) with the multimem release
  store at line 300.

Edge 1 rests on that reading. It is a **residual risk**, the same one the image op carries (image
`dcp_direct_q_gather.cu` uses the same `multimem.st.relaxed.sys` + `__threadfence_system` + `multimem.st.release.sys`).
I could not read the text of the multimem instruction section (9.7.10.15) itself.
