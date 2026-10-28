"""Item 08 lever A: ExpertsStage (bundle/DESIGN.md s3, s9), the K3 MoE routed experts at 17-24 rows in one grid.

experts.py, experts_runtime.py and shared.py fork the vLLM 49cb2bbf image's cutedsl/moe modules. Stage S0: the
relative imports of the image's common / gemm packages are made absolute; no other byte differs, so the forked
kernels do the image's work (gate: bit-equal to the image ExpertsKernel at 16 / 24 / 40 rows, P-04 judge single.py
arm cand:stage).

Switch PERFBOT_EXP_STAGE, read once at import. Unset or empty = the image path: `wrap` returns its argument and the
fork never loads. "A" = the forked experts serve the batches in ROWS; other batches, and the B1 small-batch
chain (built from the image object in decode.py), stay on the image experts. An unknown value raises, so a typo
cannot run a control arm by accident. Call site: the overlay copy of cutedsl/moe/prepared.py, "# PERFBOT P-04E".

Stage S1 (bundle/S1_PLAN.md): at the "A" rows the front splits and the tail publishes (`Stage.rows`), the fork runs
the ExpertsStage kernel on every SM: the routed experts plus the shared expert (gate/up, SiTU, down rows peer-stored
into the tail's RS mailbox), with no side stream (`run`). PERFBOT_EXP_STAGE_PROBES=1 adds the stage probes.

ROWS is the measured set (bundle/evidence/s2c/rows_*.log, timed_s2c.log; B200, 600 samples per arm): S1 beats the
image path by more than 1.0 us at the median and its p90 does not regress at 17-25 rows (-16.5 at 17 down to -2.5 us at
25). It loses or ties at 16 (+4.4), 26 (-0.3), 28 (+3.7), 30 (+5.6), 32 (+5.4) and 40 (0.0). Rows 9-15, 27, 29,
31 and 33-39 are unmeasured and stay on the image path. Item 06 X2 (JOIN off) is priced as landed: against the image
with X2 the gain is 2.0-2.4 us smaller (x2_*.log: -12.6 / -7.8 / -2.3 us at 18 / 20 / 24), so 25 rows (about -0.3 [e])
fails the 1.0 us rule and ROWS stops at 24.

"L" = item 08 R1, the lean MMA-warp loop (bundle/evidence/r1, commits 285cd2a / 5886dbb): the routed experts of the
batches in LEAN_ROWS run on the fork with lean=2 (consume_lean2: the same MMAs, order, accumulators and signals as
the image, latent bit-equal on 32 cases). Nothing else changes: no ExpertsStage kernel, no shared-expert change; the
side stream, PERFBOT_EXP_GLUE X2 / S and the grid table stay as the image path has them (the fork shares the image's
grids, so S's SharedDown SMs give both the same experts grid). Every other batch, and B1, stay on the image object.
LEAN_ROWS is the measured set (judge_timed_b200.jsonl, B200, 600 samples per arm vs today + X2 + S): window
-2.27 / -1.95 / -5.92 us at 24 / 32 / 40 rows; 16 rows +0.13 (no gain: the window end is not the experts kernel).
Rows 17-23, 25-31 and 33-39 are unmeasured and stay on the image path.

DRAIN is item 08 R2 (ii), the dynamic FC2 drain on the lean fork (bundle/evidence/r2, build 39e640a, evidence
64f8c82): the last 8 FC2(B) grains of every CTA are claimable by any CTA (latent bit-equal on 31 cases). Served only
where measured to win (drain_b200.jsonl, 600 samples per arm): window -2.82 us vs L at 40 rows; 24 / 32 rows are
inside the A/A band (0.00 / -0.13 us), so they keep drain 0."""

import os

ENV = "PERFBOT_EXP_STAGE"
PROBES = os.environ.get("PERFBOT_EXP_STAGE_PROBES", "") == "1"  # stage %globaltimer / IKET probes
MODES = frozenset({"A", "L"})
ROWS = frozenset(range(17, 25))  # "A": measured, see the docstring
LEAN_ROWS = __import__("perfbot_kernels.k5q", fromlist=["k5q"]).LEAN_ROWS  # PERFBOT K5 item 6 (land-8: {24, 32, 40}, measured)
LEAN = 2  # PersistentExperts.lean of the "L" fork (consume_lean2)
DRAIN = dict(__import__("perfbot_kernels.k5q", fromlist=["k5q"]).DRAIN)  # PERFBOT K5 item 6 (land-8: {40: 8}) # "L": PersistentExperts.drain per row (R2 (ii) pool grains per CTA); measured, see the docstring


def parse(value):
    value = value.strip()
    if value and value not in MODES:
        raise ValueError(f"{ENV}: unknown mode {value!r}; known {sorted(MODES)}")
    return value


MODE = parse(os.environ.get(ENV, ""))


def wrap(experts, front, tail):
    """The PreparedMoE experts object: `experts` itself when the switch is off, else a Stage over it."""
    if not MODE:
        return experts
    from .experts_runtime import PersistentExperts

    if MODE == "L":
        return Stage(experts, PersistentExperts, serve=LEAN_ROWS, lean=LEAN, drain=DRAIN)
    return Stage(experts, PersistentExperts, front, tail)


def staged(moe, batch):
    """PreparedMoE.run hook: this batch runs the ExpertsStage chain."""
    return isinstance(moe.experts, Stage) and batch in moe.experts.rows


def run(moe, x, out, consume):
    """The S1 chain on the main stream: route, the stage (routed experts + shared expert
    into the RS mailbox), the next layer's prefetch fork, the tail. No side stream."""
    b = x.shape[0]
    ids, weights, latent = moe.front.route(x)
    moe.experts.group(ids)
    expanded = moe.experts.fork.run(latent, ids, weights, shared=x)
    if moe.prefetch is not None:
        moe.prefetch.fork("next_attention", moe.layer_idx, b)
    shared = moe.front.plans[b][0][5]  # the fused shared expert's unwritten rows
    return moe.tail.run(expanded, weights, shared, out, consume=consume)


class Stage:
    """Sends each batch to the fork (rows in ROWS) or to the image object. The fork shares the image's weights,
    SiTU scalars and grid table (PreparedMoE caps `grids` in place). Stage S1: rows in ROWS that the front runs
    as a split fused shared expert and the tail publishes are ExpertsStage batches on every SM."""

    def __init__(self, image, cls, front=None, tail=None, serve=ROWS, lean=False, drain=None):
        self.image, self.fork = image, cls(*image.weights)
        self.fork.scalars, self.fork.grids = image.scalars, image.grids
        self.serve = frozenset(serve)  # the batches the fork runs
        self.fork.lean = lean  # "L": the R1 lean loop (the fork applies it at its stealing rows)
        self.fork.drain = dict(drain or {})  # "L": R2 drain grains per row (the fork applies it at lean=2 rows)
        self.rows = frozenset()  # without front / tail: the S0 fork (routed experts only)
        if front is None:
            return
        import torch
        from cutlass import Int32

        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe.tail_runtime import CAP

        self.rows = frozenset(b for b in ROWS if b in front.split and b in tail.publish)
        self.rows -= frozenset(getattr(tail, "twoshots", ()))
        sms = torch.cuda.get_device_properties(image.device).multi_processor_count
        mailbox = (tail.peers, *tail.ring, Int32(CAP))
        self.fork.attach(front.maps, front.scalars[1:], mailbox, self.rows, sms)
        self.fork.probes = PROBES

    def __getattr__(self, name):
        return getattr(self.image, name)

    def pick(self, batch):
        return self.fork if batch in self.serve else self.image

    def prepare(self, batch, *args, **kwargs):
        return self.pick(batch).prepare(batch, *args, **kwargs)

    def grid_for(self, batch):
        return self.pick(batch).grid_for(batch)

    def group(self, ids):
        return self.pick(ids.shape[0]).group(ids)

    def run(self, x, ids, weights=None):
        return self.pick(x.shape[0]).run(x, ids, weights)
