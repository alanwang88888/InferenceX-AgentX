# SPDX-License-Identifier: Apache-2.0
"""Draft AR consumer (item 15 form F-B, kernel 1): TP8 two-shot Lamport all-reduce + residual + RMSNorm, one launch.

    normed, residual_out = box.run(parity, x, residual, gamma, eps)

x [M, 7168] bf16 is this rank's un-reduced row-parallel output, residual [M, 7168] bf16, gamma [7168] bf16.
Outputs match the served `fused_allreduce_rms_norm` (flashinfer oneshot Lamport) bit for bit; see kernel.py for the
arithmetic and selftest/judge.py for its torch emulation. Kernel in kernel.py.

Boxes:
  TwoShotBox(group, device, rank)  real TP8: vLLM common/twoshot.TwoShot buffers (symmetric memory + multicast).
  StandInBox(device)               one GPU: x is [8, M, 7168] (the 8 ranks' partials); outputs are [8, M, 7168],
                                   one copy per emulated rank.
Parity: consecutive calls must alternate parity (twoshot's re-arm race); the draft graph's call sites take
call_index % 2 (10 calls per step, an even count, so the sequence alternates across steps too).
"""

from functools import cache

import torch

TP, H, CAP, PARITIES = 8, 7168, 64, 2
SENTINEL = -0x80000000


def served_warps_per_block(m, sm_count=148, regs=72):
    """WB of the served launch (allreduce_fusion_kernel_launcher, cuh:1720-1800): cluster 8 -> 4 at 224
    threads, then halve the cluster while M * cluster > SMs and threads <= min(65536 / regs, 1024) / 2.
    regs = 72 is cuobjdump -res-usage of the served bf16 TP8 pattern-1 kernel."""
    tpb, cluster = 224, 4
    max_threads = min(65536 // regs, 1024)
    while m * cluster > sm_count and cluster > 1 and tpb <= max_threads // 2:
        tpb, cluster = tpb * 2, cluster // 2
    return tpb // 32


@cache
def compiled(owners, ranks, copies, wb, mutant="", iket=False):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import dynamic_fake, fake

    from .kernel import DraftAR

    kernel = DraftAR(owners, ranks, copies, wb, mutant, iket)
    x, res, out = dynamic_fake(3), dynamic_fake(2), dynamic_fake(3)
    recv = fake((TP, CAP, H // owners))
    peers = fake((owners,), None, torch.int64)
    gather = fake((copies, CAP, H))
    gamma = fake((H,), (1,))
    scalars = (cutlass.Int64(0), cutlass.Int32(0), cutlass.Float32(1e-5), cutlass.Float32(float(H)))
    return cute.compile(kernel, x, res, gamma, out, out, recv, peers, gather, *scalars, make_fake_stream())


def _views(tensors):
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import dynamic_view

    return tuple(dynamic_view(t) for t in tensors)


class _Box:
    owners = ranks = copies = 1
    rank = 0

    def views(self, parity):  # (recv, peers, gather, gather multicast address)
        raise NotImplementedError

    def launch(self, parity, x, residual, gamma, eps, normed, rout, mutant="", iket=False):
        import cutlass

        from vllm.models.kimi_k3.nvidia.ops.cute_dsl.latent_moe_tail.primitives import to_cute
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import stream

        m = x.shape[-2]
        assert 1 <= m <= CAP and x.shape[-1] == H and x.dtype == torch.bfloat16
        wb = served_warps_per_block(m)
        code = compiled(self.owners, self.ranks, self.copies, wb, mutant, iket)
        recv, peers, gather, mc = self.views(parity)
        xs = x.view(self.ranks, m, H)
        rest = (to_cute(recv), to_cute(peers), to_cute(gather), cutlass.Int64(mc), cutlass.Int32(self.rank))
        scal = (cutlass.Float32(eps), cutlass.Float32(float(H)))
        code(*_views((xs, residual)), to_cute(gamma), *_views((normed, rout)), *rest, *scal, stream(x.device))

    def run(self, parity, x, residual, gamma, eps, **kw):
        m = x.shape[-2]
        normed = torch.empty(self.copies, m, H, dtype=torch.bfloat16, device=x.device)
        rout = torch.empty_like(normed)
        self.launch(parity, x, residual, gamma, eps, normed, rout, **kw)
        if self.copies == 1:
            return normed[0], rout[0]
        return normed, rout


class StandInBox(_Box):
    """One-GPU stand-in: rank 0 owns all columns; recv [parity, TP, cap, 7168], gather [parity, 8, cap, 7168]."""

    owners, ranks, copies = 1, TP, TP

    def __init__(self, device):
        kw = dict(dtype=torch.bfloat16, device=device)
        self.recv = torch.empty(PARITIES, TP, CAP, H, **kw)
        self.gather = torch.empty(PARITIES, self.copies, CAP, H, **kw)
        self.rearm_all()
        # [parity, 2]: rows 16-byte aligned (to_cute asserts it); column 1 is padding
        self.peers = torch.tensor([[self.recv[p].data_ptr(), 0] for p in range(PARITIES)], dtype=torch.int64,
                                  device=device)

    def rearm_all(self):
        for t in (self.recv, self.gather):
            t.view(torch.int32).fill_(SENTINEL)

    def views(self, parity):
        return self.recv[parity], self.peers[parity, :1], self.gather[parity], 0


class TwoShotBox(_Box):
    """Real TP8 (not run on one GPU): vLLM's TwoShot buffers, recv [TP, cap, 896] per parity, multicast gather."""

    owners, ranks, copies = TP, 1, 1

    def __init__(self, group, device, rank):
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.twoshot import TwoShot

        self.rank = rank
        self.box = TwoShot(group, device, TP, rank, H, CAP, PARITIES)

    def views(self, parity):
        return self.box.views(parity)
