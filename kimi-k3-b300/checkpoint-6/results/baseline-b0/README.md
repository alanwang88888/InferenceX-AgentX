# baseline-b0: the public-image baseline at the real point (throughput, synthetic acceptance 3.84, fast, conc 8)

`b0-spec/`: official p90 ITL 6.57 ms = 152 tok/s/user; full-decode p90 7.53 ms; 43.24 output tok/s/GPU; AL 3.839; no gate failure.
Server: the recipe flags plus `--enable-flashinfer-autotune --additional-config {"decode_backend":"latency_specialized"} --mamba-ssm-cache-dtype float32`
(`server_command.txt`). Its overlay (tree `0c98b4e8…`, 65 files) held harness instrumentation only, no kernel change.
Read with `python3 scripts/extract_metrics.py results/baseline-b0/b0-spec`.
