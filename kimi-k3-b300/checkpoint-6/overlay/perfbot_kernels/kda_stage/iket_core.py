"""IKET phase driver of the fused KDA core (core.py variant "iket": push/pop ranges per phase, minctasm 3).

  run-iket --output-dir "$PERFBOT_IKET_DIR" --clobber profile --postprocess json -- \
      python3 -B -m perfbot_kernels.kda_stage.iket_core

Kernel-only IKET arm (docs/iket-arm.md). The served variant "full" carries no ranges (its PTX is unchanged).
Per N in IKET_N, eager launches in this order (the order of the core launches in the trace):
  warm          core iket, core iket_nostore, no predecessor (first calls compile)
  inproj        F.linear in_proj (cuBLAS, as the server) then core iket, PDL launch          x REPS
  spin          a CuTe predecessor that calls launch_dependents first, spins SPIN clocks, then writes proj,
                accepted, spec_idx and the start state slots; then core iket, PDL launch     x REPS
  alone         synchronize, then core iket (nothing to wait for)                              x REPS
  spin_nostore  spin predecessor, then core iket_nostore (state stores compiled out)          x REPS
The manifest (results_path("iket-manifest.json")) lists the core launches in that order.
Phase ranges (per warp): cta > prologue, cluster_sync, gdc_wait, meta_fb, conv, qk_norm, state_wait, rec_t0, rec_t1_7,
rec_barrier, norm_send, mbar_wait, out_store. IKET timings are evidence of where time goes, never a speed number.
"""

import json
import sys

import torch
import torch.nn.functional as F

from . import common as c
from . import core as k
from .selftest_core import predecessor

IKET_N = (6, 16)
REPS = 3
SPIN = 8000  # clocks, ~4 us: the predecessor triggers at its start, so the core's prologue can overlap it
IDX_RS = 33  # spec_idx as the server's block-table slice


def main() -> int:
    lw = c.load_layer(c.kda_layers()[0])
    pred = predecessor()
    manifest = []
    for n in IKET_N:
        st = c.make_state(n, seed=300 + n, idx_row_stride=IDX_RS)
        proj = F.linear(st.x, lw.w_in)
        out = torch.empty(1, st.t, c.H, c.D, dtype=torch.bfloat16, device="cuda")
        acc, sidx = st.accepted.cpu(), st.spec_idx.cpu()
        starts = [int(sidx[i, int(acc[i]) - 1]) for i in range(n)]
        stage = torch.stack([st.ssm[s].clone() if s > 0 else torch.zeros_like(st.ssm[0]) for s in starts]).contiguous()
        a_src, i_src, pdst = st.accepted.clone(), st.spec_idx.contiguous(), torch.empty_like(proj)

        def core(case, variant, p):
            k.core(lw, st, p, out, variant=variant)
            manifest.append({"n": n, "case": case, "variant": variant, "grid": [k.CFG[0], c.H, n]})

        def spin():
            pred(SPIN, proj, pdst, a_src, st.accepted, i_src, st.spec_idx, stage, st)

        for variant in ("iket", "iket_nostore"):
            core("warm", variant, proj)
            torch.cuda.synchronize()
        for _ in range(REPS):
            core("inproj", "iket", F.linear(st.x, lw.w_in))
        torch.cuda.synchronize()
        for _ in range(REPS):
            spin()
            core("spin", "iket", pdst)
        torch.cuda.synchronize()
        for _ in range(REPS):
            torch.cuda.synchronize()
            core("alone", "iket", proj)
        torch.cuda.synchronize()
        for _ in range(REPS):
            spin()
            core("spin_nostore", "iket_nostore", pdst)
        torch.cuda.synchronize()
    c.results_path("iket-manifest.json").write_text(json.dumps(
        {"cfg": list(k.CFG), "spin_clocks": SPIN, "idx_row_stride": IDX_RS, "core_launches": manifest}, indent=1))
    print(f"PERFBOT-KDA-CORE-IKET ok core_launches={len(manifest)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
