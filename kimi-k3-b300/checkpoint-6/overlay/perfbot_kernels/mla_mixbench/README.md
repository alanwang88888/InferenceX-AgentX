# perfbot_kernels.mla_mixbench

Kernel-only bench of K3 MLA prefill attention for single-request mixed steps (expedition line M).
It replays the 12 M-03 shapes (`shapes.json` = `runs/M-03/analysis/bench_shapes.json`) on 8 GPUs and times one
MLA layer call per variant. Nothing here has run on a GPU yet.

## What it measures

One request: `n` new tokens after `ctx` cached tokens. DCP 8, strided layout (global token `p` on rank `p % 8`,
local row `p // 8`), 12 local heads, latent 512 + rope 64, q nope 128, v 128, fp8 e4m3 latent cache, page 64,
scales 1.0. Random data. Every rank builds the same global inputs from one seed.

| variant | what runs | M-03 wall |
|---|---|---|
| `image` | the server's prefill path: overlay `MultiHeadLatentAttention._perfbot_prefill_image` on a minimal layer object, pinned `MLACommonBaseImpl._context_parallel_compute_prefill_context`, real `TrtllmRaggedPrefillBackend`, real `build_mla_chunked_context_metadata` (65536 context tokens per chunk), overlay `DirectDCPKVGatherWorkspace` | `m03_image_ms` |
| `image_split` | `image`, but each context chunk's KV is cut into S slices that run as S batch entries of one `trtllm_ragged_attention_deepseek` call, then merged with `merge_attn_states` (ln base) | none |
| `fa4_split` | `image`, but the context chunk uses the pinned FA4 CuTe varlen kernel (fp8, unit descales, `num_splits=0`) | none |
| `absorbed` | the `mla_absorb` gate chain (`selftest.Rank.chain`) with the server's 8192-token DCP workspaces | `m03_absorbed_ms` |

Default `--variants image,image_split,absorbed`. `fa4_split` is opt-in.

Per shape and variant (pass 1, eager):
- `first_call_s`: the untimed warm-up call. JIT compiles land here (tokenspeed per q bucket, FA4 per compile key).
- `eager_synced_us`: 5 reps x 20 calls, sync before each call, events around it, median of the rep means.
  This copies the in-server shadow timing (M-03 synced after every layer call).
- `eager_pipelined_us`: 5 reps x 20 back-to-back calls between two events.
- correctness: rel-L2 vs `image` over all ranks (and per rank), against the variant's bound
  (`image_split` <= 1e-2, `fa4_split` <= 2e-2); rel-L2 vs an fp64 reference on <= 8 rows.

Pass 2 (after all eager records are saved): CUDA graph of one call, 5 x 20 replays, only where every rank captured.

`us_per_layer x --layers (24)` is compared with the M-03 column.

## Calibration rule (the bench's own validity gate)

For `image` and `absorbed`: `ratio = eager_synced_us x 24 / 1000 / m03_ms`.
A variant is calibrated when `|ratio - 1| <= 0.25` on at least 10 of the 12 shapes (a missing shape counts as a miss).
The bench prints `PERFBOT-MIXBENCH-CALIBRATION ok=<bool>` and exits 0 only if both are calibrated.
`--calib-mode pipelined` switches the basis; the default is fixed before the first run.

## Modes

- default: the bench above. Record `$PERFBOT_WORKSPACE/$PERFBOT_RESULTS/mixbench.json`.
  Lines: `MIXBENCH cls= n= ctx= variant= us_per_layer= x24_ms= m03_ms= ratio= rel_vs_image= ...`.
- `--split-sweep`: `image_split` at S in {1, 2, 4, 8, 12, 16, 24} on (n, ctx) = (128, 123776), (256, 471296),
  (768, 171264), (2560, 115200). Per S: us (synced, pipelined), grid CTAs `ceil(n/256) x 12 x S`, rel-L2 and
  bit-equality vs `image` (S = 1 is the same call). `MIXBENCH-SWEEP-KNEE` gives the smallest S within 5 % of the
  best. If the persistent tile scheduler caps the grid, us will not drop with S. Record `mixbench-sweep.json`.
- `--gate` (landing gate for `image_split`): n in {120, 256, 512, 2048} x ctx in {30000, 100000, 300000},
  `image_split` at S = default and S = 1. Reference: the cache shards all-gathered and placed with
  `mla_absorb.global_positions`, then `mla_absorb.ref_attention` in fp64 on <= 16 rows.
  PASS iff on every case and every rank `rel(image_split) <= 1.02 x rel(image)`, the gathered context equals the
  generated latent bit for bit, and a CUDA graph of the `image_split` chain at (120, 30000) gives 2 replays on new q
  bit-equal to eager. Prints `PERFBOT-MIXBENCH-GATE pass=<bool>`; exit 0 iff pass. Record `mixbench-gate.json`.

## image_split

S = `max(1, round(148 / (12 x ceil(n / 256))))` (n = the chunk's max query length; 256 = mStepQ of the SM103
context cubins). `PERFBOT_MIXBENCH_SPLITS=<S>` overrides it. Slice bounds sit on 128-token tiles when possible.
Batch entries are request-major (r, s), so the KV slices tile `k` in order; q rows repeat S times.
The kernel's log2 LSE goes through the same `log2_lse_to_ln` before the merge, so all merges use ln, as in the server.
The S partials merge in place into block 0 (S - 1 `merge_attn_states` launches, contiguous lse).
The plan is cached per (q lens, kv lens, S) on the warm-up call; with more than one request in a chunk it reads
`query_start_loc` to the host once (the server's builder could emit it on the CPU).

## fa4_split

Unproven: FA4 fp8 on SM103 has not run in this expedition. The first call JIT-compiles (seen in `first_call_s`,
excluded from timing). Before the first call every rank checks the imports, `arch // 10 == 10` and the head dims;
on failure the bench prints `MIXBENCH variant=fa4_split unavailable: <reason>` and runs the others.
FA4 returns ln LSE `[H, total_q]` (`cute/softmax.py:229`), so no conversion is needed.

## Adding a variant

```python
# in variants.py (or any module imported before the bench runs)
from perfbot_kernels.mla_mixbench import register

@register("mine", rel_bound=1e-2)          # m03_key=None: no calibration; available=fn(env) -> reason | None
def mine(case):
    st = case.memo.get("mine")             # one-time setup on the untimed warm-up call
    ...
    return out                              # [n, 12, 128] bf16, this rank's heads
```

`case` fields: `n, ctx, G, q [n,12,192] bf16 (local heads), q_nope [n,96,128], q_pe [n,96,64], kv_c [n,512],
k_pe [n,1,64], slots, cache_u8 [pages,64,576] (fp8 bits), block_table, local, ctx_lat [ctx,576] fp8 (global),
memo, env`. `env` holds `kv_b_proj`, `absorb` (W_UK_T / W_UV and the absorbed workspaces), `chunk_ws`,
`dcp_manager`, `backends`, `one`, `rank`, `dev`. To reuse the image path with another context-chunk kernel,
subclass the trtllm backend as `backend_classes()` does and call `run_image_family(case, name, backend)`.
Then run `--variants image,mine`.

## Arms (modelled on arms/M-05g.json; the expedition session submits them)

```json
[
 {"name": "mixbench", "env": {
   "PERFBOT_KERNEL_ONLY": "1",
   "PERFBOT_PRE_EXEC": "timeout 1700 python3 -B -m perfbot_kernels.mla_mixbench.bench --deadline-s 1600",
   "PERFBOT_PRE_EXEC_TIMEOUT_S": "1750",
   "PERFBOT_RESULTS": "results/mixbench",
   "PYTHONDONTWRITEBYTECODE": "1"}},
 {"name": "split-sweep", "env": {
   "PERFBOT_KERNEL_ONLY": "1",
   "PERFBOT_PRE_EXEC": "timeout 1200 python3 -B -m perfbot_kernels.mla_mixbench.bench --split-sweep --deadline-s 1100",
   "PERFBOT_PRE_EXEC_TIMEOUT_S": "1250",
   "PERFBOT_RESULTS": "results/mixbench-sweep",
   "PYTHONDONTWRITEBYTECODE": "1"}},
 {"name": "split-gate", "env": {
   "PERFBOT_KERNEL_ONLY": "1",
   "PERFBOT_KERNEL_ONLY_GATE": "1",
   "PERFBOT_PRE_EXEC": "timeout 660 python3 -B -m perfbot_kernels.mla_mixbench.bench --gate --deadline-s 560",
   "PERFBOT_PRE_EXEC_TIMEOUT_S": "700",
   "PERFBOT_RESULTS": "results/mixbench-gate",
   "PYTHONDONTWRITEBYTECODE": "1"}}
]
```

Add `--variants image,image_split,fa4_split,absorbed` to the first arm to include FA4.
The run times are estimates, not measurements.

## Fidelity notes (what is not the server's object)

- `kv_b_proj` is a `torch.nn.functional.linear` module, the call `UnquantizedLinearMethod` makes. The checkpoint
  weight is assumed unquantized bf16 (AG-02n showed a bf16 nvjet GEMM).
- The layer and the impl are minimal objects with the real methods bound; `attn_metadata` is
  `SimpleNamespace(prefill=MLACommonPrefillMetadata(...))` (only `.prefill` is read).
- `MLADCPManager` is created with `object.__new__` and given the `DirectDCPKVGatherWorkspace.gather` that
  `init_kv_gather` would pick; the workspace is built on the WORLD group.
- The workspace manager and the trtllm workspace are real (`init_workspace_manager`, 394 MiB).
- Shapes are single-request; in the server, decode rows of other requests run in the same step but not in this path.

## CPU checks

`python -I overlay/perfbot_kernels/mla_mixbench/test_cpu.py` (torch only): the chunk planner, the workspace rule and
the DCP local-length rule lifted from the pinned source text; shard layout vs `mla_absorb.global_positions`;
`ref_rows` vs `mla_absorb.ref_attention`; the split plan emulated in fp64 (S slices + LSE merge = full attention);
the registry and CLI.
