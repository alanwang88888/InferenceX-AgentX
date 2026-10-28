# eval-gate-land-1b: correctness gate of land-1b at the real point (eval_only, real acceptance)

Two arms on one request, same image, same node, same flags; the only difference is the overlay tree:
`l1-eval-off` = control (harness overlay, tree `1d1f6567…`), `l1-eval-on` = land-1b (tree `2afe2488…`).
Per arm: `metrics.json` (summary), `identity.json` (image, model, drafter, node), `server_command.txt`,
`phases.json`, `server-ready-seconds.txt`, `point/` (InferenceX aggregate `kimik3_…_conc8.json`, `benchmark_command.txt`,
aiperf summary export, timings), `eval/` (lm-eval GSM8K result, timing). Request-level: `arms.json`, `bundles.json`
(bundle source paths replaced by `<internal path>`), `workload-revision.json`.
Read with `python3 scripts/extract_metrics.py results/eval-gate-land-1b/l1-eval-on`.
