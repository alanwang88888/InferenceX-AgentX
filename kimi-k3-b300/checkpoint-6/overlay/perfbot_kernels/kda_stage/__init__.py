"""KDA stage of Kimi-K3 decode as one persistent kernel per layer (expedition stage 1, docs/expedition/PLAN.md).

Replaces the image's six launches per KDA layer on the spec-verify decode path
(vllm/models/kimi_k3/nvidia/kda.py @3696c772, KimiK3DeltaAttention.forward + _forward):
  in_proj_qkvgfab (cuBLAS) -> f_b_proj (cuBLAS) -> causal_conv1d_update (spec) -> fused_recurrent_kda (spec,
  one state slot stored per token) -> o_norm (gated RMSNorm) -> o_proj (cuBLAS).
The all-reduce after o_proj stays as it is.

Modules:
  common.py   real-weight loader (/model, TP rank slice), served-state builder, the image control chain, the fp64
              reference, graph timing. One reference for every kernel of the stage.
  control.py  kernel-only arm: timings of the image chain at served shapes (the gate's baseline).
  core.py     phase B: f_b_proj + conv update + recurrence + o_norm in one CuTe DSL launch (selftest_core.py).
  serve.py    server adapter the overlay kda.py calls on a pure spec-decode batch (selftest_serve.py).
in_proj and o_proj stay cuBLAS: the phase-C streaming GEMM failed its gate (docs/expedition/attempts/A-gemm/).

Switch: PERFBOT_EXP_KDA=0 restores the image path (default on once the call site lands).
"""

import os

ENV = "PERFBOT_EXP_KDA"


def enabled() -> bool:
    return os.environ.get(ENV, "1") != "0"
