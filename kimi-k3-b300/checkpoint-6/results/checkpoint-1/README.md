# checkpoint-1: the real-point ABA (throughput, synthetic acceptance 3.84, fast protocol, conc 8)

Three arms on one request, same image, same node, same flags, run in this order: `c0` = control (harness overlay only),
`lb` = land-1b (the checkpoint overlay), `c1` = control (harness overlay only). Per arm: `metrics.json` (summary),
`identity.json`, `server_command.txt`, `phases.json`, `server-ready-seconds.txt`, `point/` (InferenceX aggregate
`kimik3_…_conc8.json`, `benchmark_command.txt`, timings, `aiperf_artifacts/` summary export, csv and per-request jsonl),
`server.log.gz` (gzip) with host names and internal paths replaced. Request-level: `arms.json`, `bundles.json` (internal paths
replaced), `workload-revision.json`, `exit.json`.

`VERDICT.md` holds the verdict line and the per-arm table (written by `scripts/ingest_results.py` from
`scripts/extract_metrics.py`). Rule, pre-registered before the request ran: CONFIRMED if land-1b beats both controls by
more than the control-to-control spread on both p90 statistics; FAIL if the 95 % interval of lb - mean(controls) is above
0 on either; NOT_RESOLVED otherwise. Result: NOT_RESOLVED (favourable direction on both statistics, interval contains 0).
lb and c1 each had 2 of about 450 requests end with an empty completion (aiperf InvalidInferenceResultError); this class
occurs at 0-0.7 % in every earlier real-point arm, including the stock image, and is not caused by the landing set.
Read one arm with `python3 scripts/extract_metrics.py results/checkpoint-1/lb`.
