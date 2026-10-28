# SPDX-License-Identifier: Apache-2.0
"""DraftValue (item 15 form F-B, kernel 3): the draft layer's DCP8 exchange of the tokenspeed rows, the served LSE
combine (bit for bit) and W_UV in one PDL launch. Kernel and arithmetic: kernel.py. Judge: selftest/judge.py.

    box.run(rows, rows_lse, seq_lens, qsl, w_uv, out, parity=0, dump=None)

rows [E, M, 64, 512] bf16 and rows_lse [E, M, 64] fp32 (log2) = each rank's tokenspeed output; seq_lens [E, S] int32
(local KV lengths per request; 0 = empty shard) and qsl [S + 1] int32 (query_start_loc); w_uv [E x 8, 512, 128] bf16
with a contiguous latent axis (the absorbed kv_b_proj view, mla.py:443-452); out [E, M, 1024] bf16; dump [E, M, 8, 512]
bf16: the merged latent (judge only). E = emulated ranks (stand-in 8 or 1, real 1). parity: Lamport mailbox parity
(alternates by layer; the epoch form derives it from the epoch).

Boxes:
  StandInBox(device)        one GPU, 8 emulated ranks in one launch (8 mailbox replicas; peer tables = replica bases);
                            solo=True: one rank whose peer tables are shifted so its 8 pushes land in the 8 source slots
                            of its own mailbox (one rank's real store and read volume, all local).
  WorkspaceBox(ws)          real DCP8, epoch form, on the served DirectDCPA2AWorkspace (dcp.py:1046): its received_*
                            buffers, peer tables, signal and epoch (+ one completion counter of ours).
  LamportBox(dcp)           real DCP8, Lamport form, on a draft mla.dcp.DcpBox: partial / LSE mailboxes and the q
                            mailbox that kernel 2's Lamport publish writes (re-armed here).
"""

from functools import cache

import torch

HEADS, WORLD, LATENT, CAP = 8, 8, 512, 64
QROW = WORLD * HEADS * 576  # q mailbox bytes per token
QPAD = 7 * HEADS * 576  # tail room for kernel 2's head-block-shifted timing view


def layout(t):
    return tuple(t.shape), tuple(t.stride()), t.dtype


def _align(dtype):
    return 4 if dtype in (torch.float32, torch.int32) else 8 if dtype == torch.int64 else 16


@cache
def compiled(specs, batch, ranks, copies, lamport, pushers, mutant="", dump=False, iket=False):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import fake

    from .kernel import DraftValue

    fakes = [fake(shape, stride, dtype, _align(dtype)) for shape, stride, dtype in specs]
    kernel = DraftValue(batch, ranks, copies, lamport, pushers, mutant, dump, iket)
    return cute.compile(kernel, *fakes, cutlass.Int32(0), cutlass.Int32(0), make_fake_stream())


class _Box:
    ranks = copies = 1
    rank = 0
    lamport = False
    pushers = 4

    def run(self, rows, rows_lse, seq_lens, qsl, w_uv, out, parity=0, dump=None, mutant="", iket=False):
        import cutlass

        from vllm.models.kimi_k3.nvidia.ops.cute_dsl.latent_moe_tail.primitives import to_cute
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import stream

        e, m = rows.shape[:2]
        assert e == self.ranks and 1 <= m <= self.box.shape[3] and rows.dtype == torch.bfloat16
        assert w_uv.stride(1) == 1, "W_UV needs a contiguous latent axis"
        if dump is None:
            dump = self.no_dump
        tensors = (rows, rows_lse, seq_lens, qsl, w_uv, out, dump, self.box, self.lbox, self.peers, self.lpeers,
                   self.signal, self.speers, self.epoch, self.completion, self.qbox)
        specs = tuple(layout(t) for t in tensors)
        code = compiled(specs, m, self.ranks, self.copies, self.lamport, self.pushers, mutant,
                        dump is not self.no_dump, iket)
        views = [to_cute(t, _align(t.dtype)) for t in tensors]
        code(*views, cutlass.Int32(self.rank), cutlass.Int32(parity), stream(rows.device))

    def _common(self, device):
        self.no_dump = torch.empty(1, 1, 1, 8, dtype=torch.bfloat16, device=device)


class StandInBox(_Box):
    """One GPU. ``ranks`` emulated ranks per launch, ``copies`` mailbox replicas (replica d = rank d's local
    mailboxes). solo: one rank, one replica, its 8 destinations shifted onto the 8 source slots."""

    def __init__(self, device, ranks=WORLD, copies=None, lamport=False, solo=False, cap=CAP, pushers=4):
        copies = (1 if solo else ranks) if copies is None else copies
        assert not solo or (ranks == 1 and copies == 1)
        assert solo or copies == WORLD
        self.ranks, self.copies, self.lamport, self.pushers = ranks, copies, lamport, pushers
        self._common(device)
        self.box = torch.empty(copies, 2, WORLD, cap, HEADS, LATENT, dtype=torch.bfloat16, device=device)
        self.lbox = torch.empty(copies, 2, WORLD, cap, HEADS, dtype=torch.float32, device=device)
        self.box.view(torch.int32).fill_(-0x80000000)  # PAIR_SENT
        self.lbox.view(torch.int32).fill_(-1)  # LSE_SENT
        self.signal = torch.zeros(copies, 2, WORLD, dtype=torch.int32, device=device)
        self.epoch = torch.zeros(ranks, dtype=torch.int64, device=device)
        self.completion = torch.zeros(ranks, dtype=torch.int32, device=device)
        self.qbox = torch.full((copies, 2, cap * QROW + QPAD), 0xFF, dtype=torch.uint8, device=device)
        slot_b, slot_l = cap * HEADS * LATENT * 2, cap * HEADS * 4
        if solo:
            dests = [(self.box[0].data_ptr() + d * slot_b, self.lbox[0].data_ptr() + d * slot_l,
                      self.signal[0].data_ptr() + d * 4) for d in range(WORLD)]
        else:
            dests = [(self.box[d].data_ptr(), self.lbox[d].data_ptr(), self.signal[d].data_ptr())
                     for d in range(WORLD)]
        table = torch.tensor([dests] * ranks, dtype=torch.int64)  # [E, W, 3]
        self.peers, self.lpeers, self.speers = (table[..., i].contiguous().to(device) for i in range(3))

    def mailbox(self, parity):
        """[copies, cap x 64 x 576] q mailbox bytes of ``parity`` (kernel 2's Lamport publish target)."""
        return self.qbox[:, parity, : self.qbox.shape[2] - QPAD]


class WorkspaceBox(_Box):
    """Real DCP8, served epoch protocol: the DirectDCPA2AWorkspace of the draft (heads 8, head_dim 512)."""

    def __init__(self, ws, ubatch=0):
        assert ws.heads_per_rank == HEADS and ws.world_size == WORLD and ws.head_dim == LATENT
        device = ws.received_output.device
        self._common(device)
        self.rank = ws.rank
        self.box = ws.received_output[ubatch][None]
        self.lbox = ws.received_lse[ubatch][None]
        self.signal = ws.received_signal[ubatch][None]
        self.peers = ws.peer_output_ptrs[ubatch].view(1, WORLD)
        self.lpeers = ws.peer_lse_ptrs[ubatch].view(1, WORLD)
        self.speers = ws.peer_signal_ptrs[ubatch].view(1, WORLD)
        self.epoch = ws.epoch[ubatch : ubatch + 1]
        self.completion = torch.zeros(1, dtype=torch.int32, device=device)
        self.qbox = torch.full((1, 2, 16), 0xFF, dtype=torch.uint8, device=device)  # unused in the epoch form


class LamportBox(_Box):
    """Real DCP8, Lamport form, on a draft DcpBox (one per draft group, parity alternating by layer)."""

    lamport = True

    def __init__(self, dcp):
        assert dcp.heads == HEADS and dcp.size == WORLD
        device = dcp.partial.device
        self._common(device)
        self.rank = dcp.rank
        self.box, self.lbox = dcp.partial[None], dcp.lse[None]
        self.peers, self.lpeers = dcp.peers.view(1, WORLD), dcp.lse_peers.view(1, WORLD)
        self.qbox = dcp.q.view(1, 2, -1)
        self.signal = torch.zeros(1, 2, WORLD, dtype=torch.int32, device=device)  # unused in the Lamport form
        self.speers = torch.zeros(1, WORLD, dtype=torch.int64, device=device)
        self.epoch = torch.zeros(1, dtype=torch.int64, device=device)
        self.completion = torch.zeros(1, dtype=torch.int32, device=device)
