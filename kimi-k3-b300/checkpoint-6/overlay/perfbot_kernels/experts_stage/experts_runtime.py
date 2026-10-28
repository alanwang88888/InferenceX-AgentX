# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent routed experts for B<=64 (B>8: grouped units of <=8 tokens per expert)
over one shared core and model-owned workspace. The entry wait completes the
predecessor (incl. workspace rearm/counters) before shared state is read; output rows
keep the caller's top16 order."""

import functools

import cuda.bindings.driver as cuda
import cutlass as c
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import check_residency, fake, stream, tensor_maps, to_cute
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.policy import batch_table
from .experts import (
    PLAN_BYTES,
    SWS_BYTES,
    SWS_H,
    ExpertsGeometry,
    ExpertsGroupKernel,
    ExpertsKernel,
    ExpertsPlanKernel,
    fc2_pairs,
)


def tensor(value, align=16):
    return to_cute(value.reshape(-1), align)


def streamed(batch):
    """B<=8 on the incremental producer: ~1-1.5% at B4-8, neutral at B1-2, which keep
    the previous producer."""
    return bool(batch_table("4-8:1").get(batch, 0))


def pregrouped(tp, batch):
    """B>8 experts read the unit table a one-CTA kernel builds after routing, beside
    the fused shared expert (hidden at B9-32; at B64 its ~2.3 us after routing outlast
    the shared expert)."""
    table = batch_table(PREGROUP[tp])
    return batch > 8 and bool(table.get(batch, 0))


PREGROUP = {8: "9-32:1", 16: "9-32:1"}
# With the shared join skipped (moe.prepared.JOIN), TP8 B16 releases at entry.
EARLY = {8: "9-16:1", 16: ""}


def early_group(tp, batch):
    """Pregrouped batches whose unit-table kernel releases the experts at its entry."""
    table = batch_table(EARLY[tp])
    return pregrouped(tp, batch) and bool(table.get(batch, 0))


def stealing(batch):
    """B>8 experts balance the FC2(B) tail by claims."""
    return batch > 8 and bool(batch_table("9-64:1").get(batch, 0))


@functools.cache
def compile_experts(tp, experts, batch, grid, device, publish=False, *knobs):
    """Only code and geometry are cached; no owner tensors or descriptors. ``knobs``:
    (B<=8 stream, pregrouped, steal, ExpertsStage, stage probes, stage P0, stage S2c order,
    late-h stress ns, R1 dose: MMA-loop spin cycles per stage, no post-stage sync_warp)."""
    defaults = (streamed(batch), False, False, False, False, False, False, 0, 0, False, False, False)
    defaults += (False, False, 0, 0)  # R2: timeline stamps, FC1 dose (-), drain pool grains, drain stress ns
    stream, grouped, steal, staged, probes, p0, s2c, hdelay, spin, nosync, lean, lprobe, tl, fdose, drain, ddelay = (
        knobs + defaults[len(knobs) :]
    )
    capacity = 1 << (batch - 1).bit_length()
    shared = batch if staged else None
    moe = ExpertsKernel(
        tp, capacity, experts, grid, publish, stream, grouped, steal, shared, probes, p0,
        s2c, hdelay, spin, nosync, lean, lprobe, tl, fdose, drain, ddelay,
    )
    flat = lambda n, dtype=c.Uint8, align=16: fake((n,), (1,), dtype, align)
    box = (c.Int64(0), c.Int32(0), c.Int32(0)) if publish else ()
    if publish == "twoshot":  # receive slots, owners' bases, parity x TP + rank
        box = (fake((3 * tp, 64, 3584 // tp)), fake((tp,), None, c.Int64), c.Int32(0))
    x, ids = fake((batch, 3584), (3584, 1)), flat(batch * 16, c.Int32)
    x = flat(PLAN_BYTES) if moe.grouped else x  # grouped: the MXFP8 plan
    scales = (flat(experts * n) for n in (moe.w13_scale_bytes, moe.w2_scale_bytes))
    out = flat(batch * 16 * (1 if publish else 3584), c.BFloat16)
    maps = flat(768 if moe.paired else 256, align=64)
    tensors = (x, ids, maps, *scales, flat(moe.workspace_bytes), out)
    scalars = (c.Int32(batch), *map(c.Float32, (100, 0.25, 0.04)), *box)
    if staged:  # x [batch, 7168], maps, workspace, RS (peers, parity, rank, capacity), SiTU
        stage = (flat(batch * 7168, c.BFloat16), flat(384, align=64), flat(SWS_BYTES))
        stage += (fake((tp,), None, c.Int64), c.Int32(0), c.Int32(0), c.Int32(64))
        scalars += (*stage, c.Float32(4), c.Float32(25))
    with torch.accelerator.device_index(device):
        return cute.compile(moe, *tensors, *scalars, make_fake_stream())


@functools.cache
def compile_group(tp, experts, batch, grid, device, early=False):
    """The pregrouped experts' unit-table kernel: (ids, workspace, batch); ``early``:
    it releases the experts at entry."""
    capacity = 1 << (batch - 1).bit_length()
    kernel = ExpertsGroupKernel(tp, capacity, experts, grid, early=early)
    flat = lambda n, dtype=c.Uint8: fake((n,), (1,), dtype, 16)
    args = (flat(batch * 16, c.Int32), flat(kernel.workspace_bytes), c.Int32(batch))
    with torch.accelerator.device_index(device):
        return cute.compile(kernel, *args, make_fake_stream())


@functools.cache
def compile_plan(batch, device):
    """Plan writer for callers without the routing kernel's copy role in front."""
    args = (fake((batch, 3584)), fake((PLAN_BYTES,), (1,), c.Uint8), make_fake_stream())
    with torch.accelerator.device_index(device):
        return cute.compile(ExpertsPlanKernel(batch), *args)


def scratch(tp, device):
    """Model-owned expert buffers: the workspace and the grouped plan (MXFP8 x, written
    by the routing kernel or ``compile_plan``)."""
    size, u8 = (ExpertsGeometry(tp).workspace_bytes, torch.uint8)
    workspace = torch.full((size,), 255, device=device, dtype=u8)
    plan = torch.empty(PLAN_BYTES, device=device, dtype=u8)
    return {"workspace": workspace, "plan": plan}


@functools.cache
def stage_scratch(device):
    """ExpertsStage workspace (counters zero; ``experts.SWS_*``), one per device: stage
    calls are serialized on the stream."""
    return torch.zeros(SWS_BYTES, device=device, dtype=torch.uint8)


class PersistentExperts:
    def __init__(self, w13, s13, w2, s2, *, beta=4.0, linear_beta=25.0):
        experts, rows, _ = w13.shape
        self.tp, self.grid = (8, 136) if rows == 768 else (16, 152)
        self.device, self.experts = w13.device, experts
        self.weights = (w13, s13, w2, s2)
        # TP8: B3-4 on 148 CTAs; B5-8 publish on 128-136 CTAs, leaving SMs for the
        # one-wave tail; grouped B9-64 on 152.
        grids = "3-4:148,5-6:128,7:132,9-64:152" if self.tp == 8 else ""
        self.grids = batch_table(grids)
        scalars = (beta * linear_beta, 1 / beta, 1 / linear_beta)
        self.scalars = tuple(map(c.Float32, scalars))
        # The experts kernel uses maps 0-1 (L2 promotion on w13); the TP8 B1 cluster
        # experts use maps 2-4 (w13, w2 two-chunk and one-chunk boxes, promoted); TP8
        # FC2 pairs (B4-64) use map 5 (w2 chunk 2 of 256 rows).
        w2_tile = (192, 3, 64) if self.tp == 8 else (128, 2, 32)
        specs = [(w13, rows, 1792, 28, 64, 2, True), (w2, 3584, *w2_tile, 2, False)]
        if self.tp == 8:
            specs += [specs[0]] + [(w2, 3584, 192, 3, 64, box, True) for box in (2, 1)]
        dtype = cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B
        swizzle = cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B
        l2 = cuda.CUtensorMapL2promotion
        l2 = (l2.CU_TENSOR_MAP_L2_PROMOTION_NONE, l2.CU_TENSOR_MAP_L2_PROMOTION_L2_256B)
        maps = [
            (weight.data_ptr(), dtype, (128, experts * n, chunks), (row_bytes, stride))
            + ((128, 128, box), swizzle, l2[promote])
            for weight, n, row_bytes, chunks, stride, box, promote in specs
        ]
        if self.tp == 8:
            shape = (128, experts * 3584, 3), (192, 64), (128, 256, 1)
            maps.append((w2.data_ptr(), dtype, *shape, swizzle, l2[False]))
        self.maps = tensor_maps(maps, self.device)
        scales = (tensor(s.view(torch.uint8)) for s in (s13, s2))
        self.views = (to_cute(self.maps[:256], 64), *scales)
        self.all_maps = to_cute(self.maps, 64)  # TP8 FC2 pairs: maps 0-5
        self.prepared = {}
        self.stage, self.probes, self.p0 = None, False, False  # p0: S2 P0 (a measured loss; off)
        self.s2c, self.hdelay = False, 0  # s2c: the one-stream order; hdelay: late-h stress (ns)
        self.mma_spin, self.mma_nosync = 0, False  # R1 dose harness knobs (off)
        self.lean = False  # R1 lean MMA loop (stage knob, off): 1 = (a), 2 = (a) + (b)
        self.loop_probe = False  # R1 harness: MMA-loop timer row past the workspace end
        self.tl, self.fc1_dose = False, False  # R2 step 0 harness: timeline stamps, FC1 dose (-)
        self.drain, self.drain_delay = 0, 0  # R2 (ii): pool grains per CTA (int, or {batch: grains}; lean=2 only), stress ns

    def attach(self, maps, scalars, mailbox, rows, grid):
        """ExpertsStage at ``rows``: the shared expert's gate/up and down TMA maps
        (``shared.weight_maps``), SiTU (beta, linear beta), the tail's RS mailbox (peers,
        parity, rank, capacity), on ``grid`` CTAs."""
        self.stage = (maps[:256], scalars, mailbox, frozenset(rows), grid)

    def prepare(
        self, batch, buffers, out, publish=(), planned=False, twoshot=False, early=False
    ):
        """``publish``: (multicast, parity, rank) of the WaveMoeTail latent slot to
        publish (``twoshot``: the ``common.twoshot`` push box); ``planned``: B>8 calls
        find the plan written (by the routing kernel), else run writes it from x first;
        ``early``: the caller runs ``group`` (pregrouped batches) before ``run``."""
        mode = "twoshot" if twoshot else bool(publish)
        grouped = pregrouped(self.tp, batch)
        staged = self.stage is not None and batch in self.stage[3]
        grid = self.stage[4] if staged else self.grid_for(batch)
        args: tuple = (batch, grid, self.device.index, mode)
        args += (streamed(batch), grouped, stealing(batch))
        args += (staged, self.probes, self.p0, self.s2c, self.hdelay) if staged else ()
        lean = self.lean if self.tp == 8 and stealing(batch) and not staged else False  # B>8
        if self.mma_spin or self.mma_nosync or lean or self.loop_probe:  # R1 knobs (off: same key)
            args += () if staged else (False, False, False, False, 0)
            args += (self.mma_spin, self.mma_nosync, lean, self.loop_probe)
        drain = self.drain.get(batch, 0) if isinstance(self.drain, dict) else self.drain
        drain = drain if lean == 2 else 0  # the pool rides the lean=2 rows only
        if self.tl or self.fc1_dose or drain:  # R2 knobs (off: same key)
            args += (False, False, False, False, 0) if len(args) == 7 else ()  # staged .. hdelay
            args += (0, False, False, False) if len(args) == 12 else ()  # spin, nosync, lean, loop probe
            args += (self.tl, self.fc1_dose)
            args += (drain, self.drain_delay if drain else 0) if drain else ()
        code = compile_experts(self.tp, self.experts, *args)
        # CTAs wait on other clusters' partials: the whole grid must be resident.
        check_residency(code, self.device.index, args[1], 384, "Kimi K3 MoE experts")
        ws, maps = tensor(buffers["workspace"]), self.views[0]
        plan: tuple | None = None
        capacity = 1 << (batch - 1).bit_length()
        if fc2_pairs(self.tp, capacity, batch > 8 or streamed(batch)):
            maps = self.all_maps
        group = None
        if batch > 8:
            plan = (tensor(buffers["plan"]), compile_plan(batch, self.device.index))
            plan = plan[:1] if planned else plan
        if grouped:
            release = early_group(self.tp, batch)
            grouping = compile_group(self.tp, self.experts, *args[:3], release)
            group = grouping, ws, early
        if staged:  # gate/up, down and h [64, 768] (box: the padded batch) maps
            assert len(publish) == 3 and mode is True, "the stage publishes the latent"
            sws, sn = stage_scratch(self.device), (batch + 7) // 8 * 8
            dtype = cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
            swizzle = cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B
            l2 = cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B
            h = (sws.data_ptr() + SWS_H, dtype, (768, 64), (1536,), (64, sn), swizzle, l2)
            smaps = torch.cat([self.stage[0], tensor_maps([h], self.device)])
            weights, mailbox = self.stage[1:3]
            publish = (*publish, smaps, sws, to_cute(smaps, 64), tensor(sws))
            publish += (*mailbox, *weights)
        self.prepared[batch] = (out, tensor(out), ws, code, publish, plan, maps, group)

    def group(self, ids):
        """Pregrouped batches prepared ``early``: the unit-table kernel (run it on the
        stream after routing; ``run`` then skips it)."""
        group = self.prepared[ids.shape[0]][7]
        if group is not None and group[2]:
            self.launch_group(ids, group)

    def launch_group(self, ids, group):
        code, ws, _ = group
        code(tensor(ids), ws, c.Int32(ids.shape[0]), stream(self.device))

    def grid_for(self, batch):
        return self.grids.get(batch, self.grid)

    def run(self, x, ids, weights=None, shared=None):
        """``shared``: the MoE input [batch, 7168] of an ExpertsStage batch."""
        batch = x.shape[0]
        out, output, workspace, code, box, plan, maps, group = self.prepared[batch]
        output = tensor(weights) if box else output
        assert (shared is not None) == (len(box) > 3), "ExpertsStage batches take x"
        if shared is not None:  # (publish box, kept tensors, stage args)
            box = (*box[:3], tensor(shared), *box[5:])
        source = to_cute(x, 16)
        if plan is not None:
            if len(plan) > 1:
                plan[1](source, plan[0], stream(self.device))
            source = plan[0]
        if group is not None and not group[2]:
            self.launch_group(ids, group)
        args = (source, tensor(ids), maps, *self.views[1:], workspace, output)
        code(*args, c.Int32(batch), *self.scalars, *box, stream(self.device))
        return out
