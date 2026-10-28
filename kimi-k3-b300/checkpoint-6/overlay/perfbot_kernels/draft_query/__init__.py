# SPDX-License-Identifier: Apache-2.0
"""DraftQuery (item 15 form F-B, kernel 2): the draft layer's q_a/kv_a norms, q_b, W_UK absorb, RoPE + FP8 latent
KV insert and DCP8 query publish in one PDL launch. Kernel and arithmetic: kernel.py. Judge: selftest/judge.py.

    box.run(qkv, weights, cos_sin, positions, cache, slots, q_scale_inv, k_scale_inv, eps)

qkv [M, 2112] bf16 = the fused_qkv_a output (q_c 1536 | kv_c 512 | k_pe 64); weights = (q_a gamma [1536],
kv_a gamma [512], q_b weight [E x 1536, 1536], W_UK_T [E x 8, 128, 512]); cos_sin [max_pos, 64] fp32 (cos | sin);
positions [M] int64; cache [E, blocks, block_size, 576] uint8 (FP8 E4M3 bytes); slots [E, M] int64 (-1: not this
rank's slot); scales [1] fp32 (1 / q scale, 1 / k scale). E = emulated ranks (stand-in 8, real 1).

Boxes:
  StandInBox(device)        one GPU, 8 emulated ranks in one launch: the gathered query lands in 8 replicas of the
                            served final buffer ([8, cap x 64 x 576] bytes), signals [8, 2, 8], epoch / completion [8].
  LamportBox(dcp)           real DCP8, Lamport form (constexpr variant; the epoch form stays the default): the q
                            mailbox of a draft mla.dcp.DcpBox, multimem stores + a poll of this CTA's share of the
                            gathered rows; kernel 3 (lse_combine) re-arms it after tokenspeed has read it.
  WorkspaceBox(ws)          real DCP8 (not run on one GPU): the served DirectDCPQGatherWorkspace (dcp.py:1176) -
                            its final_query, received_signal, completion, epoch and multicast pointers - so the
                            published bytes sit where tokenspeed reads them (dcp.py:1257-1284) and the epoch stays
                            shared with any served q-gather caller.
"""

from functools import cache

import torch

HEADS, WORLD, ENTRY, Q_LORA, KV_LORA = 8, 8, 576, 1536, 512
CAP = 64


def layout(t):
    return tuple(t.shape), tuple(t.stride()), t.dtype


@cache
def compiled(specs, batch, ranks, copies, world, waiters, mutant="", iket=False, lamport=False):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import fake

    from .kernel import DraftQuery

    fakes = [fake(shape, stride, dtype, 4 if dtype in (torch.float32, torch.int32) else 8 if dtype == torch.int64
                  else 16) for shape, stride, dtype in specs]
    kernel = DraftQuery(batch, ranks, copies, world, waiters, mutant, iket, lamport)
    scalars = (cutlass.Int64(0), cutlass.Int64(0), cutlass.Int32(0), cutlass.Float32(1e-5), cutlass.Float32(1536.0),
               cutlass.Float32(512.0))
    return cute.compile(kernel, *fakes, *scalars, make_fake_stream())


class _Box:
    ranks = copies = 1
    world = WORLD
    rank = 0
    waiters = None
    mc_query = mc_signal = 0
    lamport = False

    def target(self, parity):
        """(final buffer view, multicast address) the query is published to: the epoch form's final buffer, or the
        Lamport q mailbox of ``parity``."""
        return self.final, self.mc_query

    def launch(self, qkv, weights, cos_sin, positions, cache_bytes, slots, q_scale_inv, k_scale_inv, eps,
               mutant="", iket=False, parity=0):
        import cutlass

        from vllm.models.kimi_k3.nvidia.ops.cute_dsl.latent_moe_tail.primitives import to_cute
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import stream

        m = qkv.shape[0]
        assert 1 <= m <= CAP and qkv.shape[1] >= Q_LORA + KV_LORA + 64 and qkv.dtype == torch.bfloat16
        gq, gkv, w_qb, w_uk = weights
        final, mc_query = self.target(parity)
        tensors = (qkv, gq, gkv, w_qb, w_uk, cos_sin, positions[:m], cache_bytes, slots[:, :m], q_scale_inv,
                   k_scale_inv, final, self.signal, self.epoch, self.completion)
        specs = tuple(layout(t) for t in tensors)
        waiters = self.world if self.waiters is None else self.waiters
        code = compiled(specs, m, self.ranks, self.copies, self.world, waiters, mutant, iket, self.lamport)
        views = [to_cute(t, 4 if t.dtype in (torch.float32, torch.int32) else 8 if t.dtype == torch.int64 else 16)
                 for t in tensors]
        scal = (cutlass.Int64(mc_query), cutlass.Int64(self.mc_signal), cutlass.Int32(self.rank),
                cutlass.Float32(eps), cutlass.Float32(float(Q_LORA)), cutlass.Float32(float(KV_LORA)))
        code(*views, *scal, stream(qkv.device))

    run = launch


class StandInBox(_Box):
    """One GPU: ``ranks`` emulated ranks per launch; the final query buffer and the signals have ``copies``
    replicas (replica e = emulated rank e's local view). ``waiters``: signals waited on (None = world)."""

    def __init__(self, device, ranks=WORLD, copies=WORLD, waiters=None, cap=CAP, mailbox=None, shift=False):
        """``mailbox`` (Lamport form): a [copies, 2, >= cap x 64 x 576] uint8 q mailbox (sentinel 0xFF), e.g. kernel 3's
        StandInBox.qbox. ``shift`` (timing only, Lamport, copies 8, one rank): replica c is head block c of replica
        0, so one rank's 8 copies fill all 8 head blocks of its own mailbox."""
        self.ranks, self.copies, self.waiters, self.cap = ranks, copies, waiters, cap
        self.lamport, self.mailbox, self.shift = mailbox is not None, mailbox, shift
        self.final = torch.zeros(copies, cap * WORLD * HEADS * ENTRY, dtype=torch.uint8, device=device)
        self.signal = torch.zeros(copies, 2, WORLD, dtype=torch.int32, device=device)
        self.epoch = torch.zeros(ranks, dtype=torch.int64, device=device)
        self.completion = torch.zeros(ranks, dtype=torch.int32, device=device)

    def target(self, parity):
        if not self.lamport:
            return self.final, 0
        size = self.cap * WORLD * HEADS * ENTRY
        if self.shift:
            base = self.mailbox[0, parity]
            return base.as_strided((self.copies, size), (HEADS * ENTRY, 1)), 0
        return self.mailbox[:, parity, :size], 0

    def gathered(self, copy, m, parity=0):
        """Replica ``copy``'s published query [m, 64, 576] (FP8 bytes): what tokenspeed reads."""
        buf = self.final[copy] if not self.lamport else self.mailbox[copy, parity]
        return buf[: m * WORLD * HEADS * ENTRY].view(m, WORLD * HEADS, ENTRY)


class LamportBox(_Box):
    """Real DCP8, Lamport form: the q mailbox of a draft mla.dcp.DcpBox (multicast stores, parity alternating by
    layer; kernel 3's LamportBox re-arms it). Not exercised on the one-GPU stand-in (no multicast object)."""

    copies = 0
    lamport = True

    def __init__(self, dcp):
        assert dcp.heads == HEADS and dcp.size == WORLD
        self.rank, self.dcp = dcp.rank, dcp
        device = dcp.q.device
        self.signal = torch.zeros(1, 2, WORLD, dtype=torch.int32, device=device)  # unused in the Lamport form
        self.epoch = torch.zeros(1, dtype=torch.int64, device=device)
        self.completion = torch.zeros(1, dtype=torch.int32, device=device)

    def target(self, parity):
        return self.dcp.q[parity].view(torch.uint8).view(1, -1), self.dcp.q_mc + parity * self.dcp.q[0].numel()


class WorkspaceBox(_Box):
    """Real DCP8 on the served q-gather workspace (one per (group, heads, dtype), shared by the 5 draft layers).
    Not exercised on the one-GPU stand-in (no multicast object on one GPU: probe5)."""

    copies = 0

    def __init__(self, ws, ubatch=0):
        assert ws.heads_per_rank == HEADS and ws.world_size == WORLD and ws.head_dim == ENTRY
        self.rank = ws.rank
        self.final = ws.final_query[ubatch].view(torch.uint8).view(1, -1)
        self.signal = ws.received_signal[ubatch].view(1, 2, WORLD)
        self.epoch = ws.epoch[ubatch : ubatch + 1]
        self.completion = ws.completion[ubatch]
        self.mc_query, self.mc_signal = ws.multicast_ptrs[ubatch]
