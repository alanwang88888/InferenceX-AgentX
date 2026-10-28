# SPDX-License-Identifier: Apache-2.0
"""Entry for the kernel-validation skill's verify_kernel.py (independent tolerance check, one GPU stand-in).
run_kernel(xs [8, M, 7168], residual [M, 7168], gamma [7168]) -> normed of emulated rank 0, or residual_out
when DRAFT_AR_OUT=rout. eps = 1e-5."""

import os

import torch

from perfbot_kernels.draft_ar import StandInBox

_BOX = {}


def run_kernel(xs, residual, gamma):
    if "b" not in _BOX:
        _BOX["b"] = StandInBox(xs.device)
    normed, rout = _BOX["b"].run(0, xs.contiguous(), residual.contiguous(), gamma.contiguous(), 1e-5)
    torch.cuda.synchronize()
    return rout[0] if os.environ.get("DRAFT_AR_OUT") == "rout" else normed[0]
