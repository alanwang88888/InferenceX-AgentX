# InferenceX leaderboard context for the Kimi-K3 AgentX p90 point

Written 2026-10-20 from sources read that day. Tags: [m] seen in the source, [e] inferred, [u] not found.
Repo snapshot: SemiAnalysisAI/InferenceX at commit 418d8d48a3bc98ca715c091033199603c9c5a9b9 (2026-10-20).

## Our point

8x B300, vLLM nightly 49cb2bbf, `latency_specialized` (CuTe DSL whole-decode executor, vLLM PR #61867), conc 8.
Official per-request p90 ITL 6.61 ms = 151.3 tok/s/user.
Speculation of our run: dspark, num_speculative_tokens 7, draft probabilistic, rejection synthetic at AL 3.84 [m: results/checkpoint-1/c0/server_command.txt line 1; vLLM resolved it as SpeculativeConfig(method=dspark, num_spec_tokens=7) in the raw server log]. This matches the leaderboard rows' effective setting (see Speculation).

## Leaderboard entries

The leaderboard metric `p90_intvty` is 1 / p90 of the per-request `inter_token_latency` values. Same statistic as ours [m].

| Entry | p90 tok/s/user | p90 ITL | GPUs | vLLM commit / image | Framework | Layout | Spec decoding | Decode executor | Row id, date | Source |
|---|---|---|---|---|---|---|---|---|---|---|
| GB300 conc 4 (agg, no offload) | 210.1 [m] | 4.76 ms [m] | 8 (2 nodes x 4) [m] | nightly-49cb2bbf71cf... [m] | dynamo-vllm [m] | TP8, DCP8, EP1, 1 worker [m] | DSpark 7 drafts, probabilistic, block rejection [m] | `latency_specialized` [m] | 445223, 2026-10-18 [m] | master config `kimik3-fp4-gb300-dynamo-vllm-agentic-dspark-mooncake-dcp8-agg` + recipe `agg-megakernel-variants.yaml:override_tp8_dspark7` |
| GB300 conc 4 (agg, Mooncake) | 200.0 [m] | 5.00 ms [m] | 8 [m] | same image [m] | dynamo-vllm [m] | TP8, DCP8, EP1, max-num-seqs 2 [m] | DSpark 7 [m] | stock (no `decode_backend`) [m] | 445224, 2026-10-18 [m] | same key + `agg-variants.yaml:override_dcp8_dspark7_maxseq2` |
| GB300 conc 48 (disagg 1P3D) | 106.4 [m] | 9.40 ms [m] | 32 = 8 prefill + 3 x 8 decode [m] | same image [m] | dynamo-vllm, NIXL [m] | prefill TP8/DCP8; decode 3 workers x TP8/DCP8 [m] | DSpark 4 [m] | stock [m] | 445232, 2026-10-18 [m] | `kimik3-fp4-gb300-dynamo-vllm-agentic-dspark-mooncake-dcp8-disagg` + `disagg-variants.yaml:override_1p3d_dcp8_dcp8_dspark4` |
| GB300 conc 1 / 2 (agg TP16) | 423.7 / 349.7 [m] | 2.36 / 2.86 ms [m] | 16 (4 nodes) [m] | same image [m] | dynamo-vllm [m] | TP16, DCP16, EP1 [m] | DSpark 5 [m] | `latency_specialized` [m] | 445218 / 445219, 2026-10-18 [m] | same agg key + `agg-megakernel-variants.yaml:override_tp16_dspark5` |
| B300 conc 8 (single node) | 63.9 [m] | 15.65 ms [m] | 8 [m] | vllm/vllm-openai:nightly-dev-x86_64-cu13.0.1-5894fdf [m] | vllm [m] | TP8, DCP8, EP1, max-num-seqs 16, Mooncake DRAM [m] | DSpark 7 drafts (conc <= 8) [m] | stock [m] (recipe has no `decode_backend`; only the GB300 megakernel recipe sets it) | 439935, 2026-08-16 [m] | `kimik3-fp4-b300-vllm-agentic-dspark` + `single_node/.../b300-fp4-mtp/agentic.yaml:override_c8` |
| B300 other conc (same run) | c1 202.8, c2 132.5, c4 121.1, c16 32.1 [m] | 4.93 / 7.55 / 8.26 / 31.1 ms [m] | 8 | same | vllm | TP8, DCP8 | DSpark 7 (c16: 3 drafts) [m] | stock [m] | 439932..439941 [m] | same recipe |
| Other B300 / GB300 at conc 8 | none [m] | | | | | | | | | GB300 recipes list conc 1, 2, 4, 48, 52, 56, 64 only |

Notes on the B300 row: the master config at HEAD pins `nightly-dev-x86_64-cu13-3696c77` (PR 3040, "replace unavailable 5894fdf"), but the published row is still the 5894fdf run of 2026-08-16 [m]. No newer B300 row exists [m].
GB300 curve moved to 49cb2bbf and gained the megakernel points in PR 4208 [m].

## AgentX measurement protocol (what the leaderboard ran)

- Client: `agentx profile` (AIPerf-based) on dataset `semianalysis_cc_traces_weka_062126`, seed 42, one server restart per concurrency [m].
- Duration: `DURATION` = 3600 s by default (`DEFAULT_AGENTIC_DURATION_SECONDS = 3600`); `MIN_SCENARIO_DURATION_S = 900` below which `--unsafe-override` is set [m]. Rows above show `duration_seconds` 3586-3630 s [m].
- Warmup: the `agentx` preset supplies 10 warmup requests per lane [m].
- p90 tok/s/user: `p90_intvty = 1 / p90_itl`, where `p90_itl` is the 90th percentile over per-request `inter_token_latency` (ms, converted to s) [m].
- Fast mode: `AIPERF_EXPERIMENTAL_FAST=1` sets duration 1200 s and 1 warmup request per lane; it is a workflow input `agentx-fast`, off by default [m]. Published rows above ran ~3600 s [m].

## Comparison to our 151.3 at conc 8 on 8 GPUs

1. Versus B300 conc 8 (63.9): same hardware count, same statistic, same TP8/DCP8 layout, but older commit (5894fdf) and stock decode; ours is 2.37x. Not like-for-like on the decode path; like-for-like only if our run also drafts with DSpark 7.
2. Versus GB300 conc 4 megakernel (210.1, 8 GPUs, same commit, same executor): same path, but conc 4 and GB300; the nearest same-path reference, half the concurrency.
3. Versus GB300 conc 48 (106.4): 32 GPUs, disaggregated, stock decode; not comparable to a single-node conc 8 point.

## Source URLs (all read 2026-10-20)

- https://github.com/SemiAnalysisAI/InferenceX/blob/418d8d48a3bc98ca715c091033199603c9c5a9b9/inferencex-e2e/configs/nvidia-master.yaml (keys `kimik3-fp4-b300-vllm-agentic-dspark`, `kimik3-fp4-gb300-dynamo-vllm-agentic-dspark-mooncake-dcp8-agg`, `...-dcp8-disagg`)
- https://github.com/SemiAnalysisAI/InferenceX/blob/418d8d48a3bc98ca715c091033199603c9c5a9b9/inferencex-e2e/benchmarks/single_node/srt-slurm-recipes/kimik3/vllm/b300-fp4-mtp/agentic.yaml
- https://github.com/SemiAnalysisAI/InferenceX/blob/418d8d48a3bc98ca715c091033199603c9c5a9b9/inferencex-e2e/benchmarks/multi_node/srt-slurm-recipes/kimik3/vllm/gb300-fp4/agentx/agg-megakernel-variants.yaml
- https://github.com/SemiAnalysisAI/InferenceX/blob/418d8d48a3bc98ca715c091033199603c9c5a9b9/inferencex-e2e/benchmarks/multi_node/srt-slurm-recipes/kimik3/vllm/gb300-fp4/agentx/agg-variants.yaml
- https://github.com/SemiAnalysisAI/InferenceX/blob/418d8d48a3bc98ca715c091033199603c9c5a9b9/inferencex-e2e/benchmarks/multi_node/srt-slurm-recipes/kimik3/vllm/gb300-fp4/agentx/disagg-variants.yaml
- https://github.com/SemiAnalysisAI/InferenceX/blob/418d8d48a3bc98ca715c091033199603c9c5a9b9/inferencex-e2e/perf-changelog.yaml (PRs 3040, 4208)
- https://github.com/SemiAnalysisAI/InferenceX/blob/418d8d48a3bc98ca715c091033199603c9c5a9b9/inferencex-e2e/infx/bench/agentic/replay.py
- https://github.com/SemiAnalysisAI/InferenceX/blob/418d8d48a3bc98ca715c091033199603c9c5a9b9/inferencex-e2e/infx/matrix/validation.py
- https://github.com/SemiAnalysisAI/InferenceX/blob/418d8d48a3bc98ca715c091033199603c9c5a9b9/inferencex-e2e/infx/results/agentic/request_metrics.py
- https://github.com/SemiAnalysisAI/InferenceX/blob/418d8d48a3bc98ca715c091033199603c9c5a9b9/inferencex-e2e/docs/agentx-standalone.md
- https://github.com/SemiAnalysisAI/InferenceX/blob/418d8d48a3bc98ca715c091033199603c9c5a9b9/.github/workflows/benchmark-tmpl.yml (`agentx-fast` input)
- https://inferencex.semianalysis.com/api/v1/benchmarks?model=Kimi-K3 (raw rows; ids above)
- https://inferencex.semianalysis.com/api/v1/views/options (model key `kimik3`)
- Run logs: https://github.com/SemiAnalysisAI/InferenceX/actions/runs/31893747354/attempts/2 (B300), https://github.com/SemiAnalysisAI/InferenceX/actions/runs/39055199395/attempts/1 (GB300)
- https://github.com/vllm-project/vllm/pull/61867 ("[Perf] Kimi K3 Specialized Decode: CuTeDSL Optimized GB300 (1.5-2x)")

## Speculation

Recipe strings, verbatim [m]:

- B300 conc 8 (`single_node/.../kimik3/vllm/b300-fp4-mtp/agentic.yaml`, `override_c8`):
  `speculative-config: '{"method":"dspark","model":"Inferact/Kimi-K3-DSpark","num_speculative_tokens":7,"attention_backend":"TOKENSPEED_MLA","draft_sample_method":"probabilistic","rejection_sample_method":"block"}'`
- GB300 conc 4 megakernel (`multi_node/.../kimik3/vllm/gb300-fp4/agentx/agg-megakernel-variants.yaml`, `override_tp8_dspark7`):
  `speculative-config: '{"model":"Inferact/Kimi-K3-DSpark","attention_backend":"TOKENSPEED_MLA","method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic","rejection_sample_method":"block"}'`

Neither recipe string names `synthetic_acceptance_length` [m]. The harness rewrites it at submit time [m]:
`infx/srt_slurm/synthetic_acceptance.py::build_overrides` — for engine `vllm` (frameworks `vllm`, `dynamo-vllm`), when `EVAL_ONLY != true`, `IS_AGENTIC` set and `SPEC_DECODING != none`, it sets
`rejection_sample_method="synthetic", synthetic_acceptance_length=<golden AL>`; eval-only runs keep `block`.
Golden AL for probabilistic/block DSpark on Kimi-K3 (`infx/golden_al_distribution/kimik3_dspark_probabilistic_sample_method_block_rejection_sample_method.yaml`): 4 -> 3.36, 5 -> 3.62, 7 -> 3.84 [m]; the master config comment for the B300 key and PR 4208 for GB300 state the same values [m].

Effective leaderboard setting for both rows: draft `probabilistic`, rejection `synthetic`, `synthetic_acceptance_length` 3.84 (7 drafts) [e, derived from the rewrite above; the submitted override was not read].
Ours: draft probabilistic, rejection synthetic, AL 3.84 -> rejection method and AL match both rows.

Sources: https://github.com/SemiAnalysisAI/InferenceX/blob/418d8d48a3bc98ca715c091033199603c9c5a9b9/inferencex-e2e/infx/srt_slurm/synthetic_acceptance.py ,
https://github.com/SemiAnalysisAI/InferenceX/blob/418d8d48a3bc98ca715c091033199603c9c5a9b9/inferencex-e2e/infx/golden_al_distribution/kimik3_dspark_probabilistic_sample_method_block_rejection_sample_method.yaml
