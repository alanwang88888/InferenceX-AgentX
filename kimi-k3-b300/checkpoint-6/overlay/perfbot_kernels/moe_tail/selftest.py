"""Kernel-only gate of the one-launch latent-MoE tail (expedition line B). Run in the server env (PERFBOT_PRE_EXEC):

  python3 -B -m perfbot_kernels.moe_tail.selftest [--check] [--graph] [--pdl-order] [--time]

8 processes (one per GPU, NCCL + symmetric memory) build the image KimiK3LatentMoETailOp (overlay collective,
prefilled mode) and run every mode on every rank. Weights: routed_expert_up_proj / routed_expert_norm of 4 MoE
layers of /model. A self-test predecessor (kernel.Feed) plays fc2 + the aux shared GEMM: PDL, optional per-rank
delay, then this rank's routed rows -> slot (token, rank) of every rank's routed buffer (multicast) and the
shared partial -> the tail's input. control = op(...) (3 image launches); candidate = moe_tail.run(...).
  --check      M 1/8/16/48/64 x 3 seeds, image and candidate interleaved on one op. Per call: flags rotate, the
               next generation and the mailbox are all sentinel. Candidate error vs fp64 (same bf16 inputs):
               mean <= control, max <= control + 1/2 bf16 ulp of max|ref|; two calls bit-equal; all ranks equal.
               Declines: M 128, switch off, not prefilled -> None.
  --graph      capture [feed -> tail], 2 replays on new inputs, bit-equal to eager. M 8 / 48 / 64.
  --pdl-order  feed with a random per-rank delay (0..60 x 500 ns): bit-equal to the undelayed call; 76 calls,
               60 at M 48. Info: the tail on an unordered stream must see a stale shared input.
  --time       CUDA-graph replay of a 4-layer chain per M 8/16/48/64: real moe_stage fc2 (distinct w2 per layer,
               multicasts the routed rows) -> tail -> attn_res_one (delta = tail out); up weights distinct per
               layer. tail us = (chain - chain without tail) / 4. Stamps of the candidate at M 8/48/64. Then (info)
               a second pass per M for each ALT_TIME config: probe (error vs fp64 <= the default's, protocol),
               baseline chains again, alt chain, stamps; an alt failure keeps the first pass (time_partial).
GATE: check, graph, pdl-order pass on all 8 ranks; control - candidate >= 10 us at M 48 and >= 0 at every M (rank 0;
all ranks recorded). M 128: the candidate declines (= image path).
Record: $PERFBOT_WORKSPACE/$PERFBOT_RESULTS/tail-gate.json (+ moetail-rank<r>.json).
"""

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch

TP, H, L, SH, TOPK = 8, 7168, 3584, 896, 16
CHECK_M = [1, 8, 16, 48, 64]
GRAPH_M = [8, 48, 64]
TIME_M = [8, 16, 48, 64]
GAIN_US = 10.0
MODEL = os.environ.get("PERFBOT_MODEL_DIR", "/model")
OPMOD = "vllm.models.kimi_k3.nvidia.ops.latent_moe_tail"
KBANK, KUSED = 10, 8
SENT = -0x80000000
PORT = 29531
ALT_CHECK = ()  # configs checked bit-equal to the default in --check
ALT_TIME = ({"postfence": True},)  # info, after every default timing: probe (error vs fp64 <= default's, protocol), time


def results_path(name="tail-gate.json") -> Path:
    root = Path(os.environ.get("PERFBOT_WORKSPACE", "/workspace"))
    out = root / os.environ.get("PERFBOT_RESULTS", "results/measure")
    out.mkdir(parents=True, exist_ok=True)
    return out / name


def log(tag, obj):
    print(f"{tag} {json.dumps(obj, default=str)}", flush=True)


def same(a, b) -> bool:
    return a.shape == b.shape and bool(torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16)))


def err(got, ref) -> dict:
    d = (got.double() - ref).abs()
    return {"max_abs": d.max().item(), "mean_abs": d.mean().item()}


def bar(c, k, absmax) -> bool:
    half_ulp = 2.0 ** (math.floor(math.log2(max(absmax, 1e-30))) - 8)
    return c["mean_abs"] <= k["mean_abs"] and c["max_abs"] <= k["max_abs"] + half_ulp


def _find(d, key):
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for v in d.values():
            r = _find(v, key)
            if r is not None:
                return r
    return None


def load_layers(dev, n=4):
    """n MoE layers over the depth: [(up [7168, 3584] bf16, gamma [3584] bf16)], meta, eps."""
    from safetensors import safe_open

    wm = json.loads((Path(MODEL) / "model.safetensors.index.json").read_text())["weight_map"]
    suf = "routed_expert_up_proj.weight"
    ups = sorted((k for k in wm if k.endswith(suf)), key=lambda k: int(k.split(".layers.")[1].split(".")[0]))
    if not ups:
        raise RuntimeError(f"no *{suf}; sample keys {list(wm)[:20]}")
    handles, layers, meta = {}, [], []

    def get(key):
        f = wm[key]
        if f not in handles:
            handles[f] = safe_open(str(Path(MODEL) / f), framework="pt", device="cpu")
        return handles[f].get_tensor(key)

    for ku in [ups[(i * len(ups)) // n] for i in range(n)]:
        kn = ku[: -len(suf)] + "routed_expert_norm.weight"
        if kn not in wm:
            raise RuntimeError(f"{kn} not in the index; near: {[k for k in wm if k.startswith(ku[:-len(suf)])][:20]}")
        up, gm = get(ku), get(kn)
        meta.append({"up": ku, "norm": kn, "dtypes": [str(up.dtype), str(gm.dtype)],
                     "shapes": [list(up.shape), list(gm.shape)]})
        assert tuple(up.shape) == (H, L) and tuple(gm.shape) == (L,), meta[-1]
        layers.append((up.to(dev, torch.bfloat16).contiguous(), gm.to(dev, torch.bfloat16).contiguous()))
    cfg = json.loads((Path(MODEL) / "config.json").read_text())
    eps = float(_find(cfg, "rms_norm_eps") or 1e-6)
    return layers, meta, eps


STAMPS = ["entry", "wait_done", "ar_done", "cluster_shared_sent", "cta_b_done", "partial_stored", "epi_partials_ready",
          "epi_done", "gather_polled", "end"]


def stamp_summary(d) -> dict:
    """d [98, 16] int64 globaltimer ns (slot 15 = SM id): per stamp [min, median, max] us from the first entry, the
    per-CTA phase durations [median, max], and SM placement."""
    t = d[:, :10].double()
    rel = (t - t[:, 0].min()) / 1000.0
    q = lambda v: [round(float(v.min()), 2), round(float(v.median()), 2), round(float(v.max()), 2)]
    rec = {"unit": "us from the first CTA entry [min, median, max]"}
    for c, n in enumerate(STAMPS):
        rec[n] = q(rel[:, c])
    rec["phase_us_median_max"] = {f"{STAMPS[c - 1]}->{STAMPS[c]}": [round(float((rel[:, c] - rel[:, c - 1]).median()), 2),
                                                                   round(float((rel[:, c] - rel[:, c - 1]).max()), 2)]
                                  for c in range(1, 10)}
    sm = d[:, 15]
    rec["distinct_sms"] = int(sm.unique().numel())
    rec["max_ctas_per_sm"] = int(torch.bincount(sm.clamp(min=0)).max())
    return rec


# ---------------------------------------------------------------- one rank


class Rank:
    def __init__(self, rank, dev, rec, save):
        import torch.distributed as dist

        import perfbot_kernels.moe_tail as mt

        self.rank, self.dev, self.rec, self.save, self.dist, self.mt = rank, dev, rec, save, dist, mt
        group = dist.group.WORLD
        try:
            import torch.distributed._symmetric_memory as symm_mem

            symm_mem.enable_symm_mem_for_group(group.group_name)
        except Exception:  # noqa: BLE001
            pass
        t0 = time.time()
        self.layers, rec["weights"], self.eps = load_layers(dev)
        rec["load_s"] = round(time.time() - t0, 1)
        import importlib

        opm = importlib.import_module(OPMOD)
        c = opm.KimiK3LatentMoETailContract(tp_group_id=id(group), tp_size=TP, device=dev, dtype=torch.bfloat16,
                                            hidden_size=H, latent_size=L, max_num_tokens=128, rms_eps=self.eps,
                                            experts_per_token=TOPK)
        t0 = time.time()
        self.op = opm.KimiK3LatentMoETailOp(c, group)
        rec["op_build_s"] = round(time.time() - t0, 1)
        co = self.op._collective
        rec["collective_file"] = sys.modules[type(co).__module__].__file__
        rec["prefilled_ready"] = bool(getattr(co, "perfbot_prefilled_ready", False))
        rec["eps"] = self.eps
        rec["max_active_clusters_7"] = mt.max_active_clusters()
        rec["kernel_sha256"] = mt.source_sha256()
        self.co, self.up = co, self.op._up_projection
        from perfbot_kernels.moe_stage import PrefilledMoEOutput

        self.PO = PrefilledMoEOutput
        self.g2 = torch.zeros((1, L), dtype=torch.bfloat16, device=dev)
        self._feed = None

    # -------- plumbing

    def wait(self, what, limit=400.0):
        ev = torch.cuda.Event()
        ev.record()
        t0 = time.time()
        while not ev.query():
            if time.time() - t0 > limit:
                self.rec["hang"] = what
                self.save()
                log("HANG", {"rank": self.rank, "what": what})
                os._exit(3)
            time.sleep(0.0005)

    def barrier(self, what):
        self.wait(what)
        self.dist.barrier(device_ids=[self.dev.index])

    def prefilled(self, m):
        return self.PO(self.g2, torch.zeros((m, TOPK), dtype=torch.bfloat16, device=self.dev),
                       torch.zeros((m, TOPK), dtype=torch.int32, device=self.dev))

    def feed(self, rows, ssrc, sdst, sleeps=0):
        import cutlass
        import cutlass.cute as cute
        from cuda.bindings.driver import CUstream
        from cutlass.cute.runtime import make_ptr

        gm = cute.AddressSpace.gmem
        P = lambda t, dt: make_ptr(dt, t.data_ptr(), gm, assumed_align=16)
        stream = CUstream(torch.cuda.current_stream().cuda_stream)
        if self._feed is None:
            from perfbot_kernels.moe_tail.kernel import Feed

            self._feed = cute.compile(Feed(), P(rows, cutlass.BFloat16), cutlass.Int64(self.co._routed_multicast_ptr),
                                      P(self.co._routed_flags, cutlass.Uint32), P(ssrc, cutlass.BFloat16),
                                      P(sdst, cutlass.BFloat16), cutlass.Int32(0), cutlass.Int32(1),
                                      cutlass.Int32(0), stream)
        self._feed(P(rows, cutlass.BFloat16), cutlass.Int64(self.co._routed_multicast_ptr),
                   P(self.co._routed_flags, cutlass.Uint32), P(ssrc, cutlass.BFloat16), P(sdst, cutlass.BFloat16),
                   cutlass.Int32(self.rank), cutlass.Int32(rows.shape[0]), cutlass.Int32(sleeps), stream)

    def inputs(self, m, seed):
        """All ranks' routed rows [8, m, L] and shared partials [8, m, H] (same on every rank), own copies."""
        rows, shs = [], []
        for r in range(TP):
            g = torch.Generator(device=self.dev).manual_seed(seed * 1009 + r)
            rows.append((torch.randn(m, L, device=self.dev, generator=g) * 0.25).to(torch.bfloat16))
            sh = (torch.randn(m, H, device=self.dev, generator=g) * 0.05).to(torch.bfloat16)
            sh[0, :16] = -0.0  # sanitized by the shared send
            shs.append(sh)
        rows, shs = torch.stack(rows), torch.stack(shs)
        return rows, shs, rows[self.rank].contiguous(), shs[self.rank].contiguous()

    def reference(self, rows, shs, li):
        up, gamma = self.layers[li]
        xs = rows.double().sum(0)
        inv = torch.rsqrt((xs * xs).mean(-1, keepdim=True) + self.eps)
        return (xs * inv * gamma.double()) @ up.double().T + shs.double().sum(0)

    def ctl(self, m, sdst, li):
        up, gamma = self.layers[li]
        return self.op(self.prefilled(m), sdst, gamma, up)

    def cand(self, m, sdst, li, stamps=False, cfg=None):
        up, gamma = self.layers[li]
        out = self.mt.run(self.op, self.prefilled(m), sdst, gamma, up, stamps=stamps, **(cfg or {}))
        if out is None:
            raise RuntimeError("moe_tail declined a served shape")
        return out

    def protocol(self, gr, gs) -> dict:
        """After a call (all GPU work done): both flag sets rotated, counters zero, the next generation of both
        Lamport buffers, the whole mailbox and the candidate's split-K workspace (part, sums) are sentinel."""
        rf, sf = self.co._routed_flags.view(torch.int32).cpu(), self.co._shared_flags.cpu()
        nr, ns = int(rf[0]), int(sf[0])
        rbpb = int(rf[2]) // 4
        rws = self.co._routed_workspace.view(torch.int32).view(-1)
        sws = self.co._shared_workspace.view(torch.int32).view(3, -1)
        return {"routed_rotated": nr == (gr + 1) % 3 and int(rf[1]) == gr and int(rf[8]) == 0,
                "shared_rotated": ns == (gs + 1) % 3 and int(sf[1]) == gs and int(sf[8]) == 0,
                "routed_next_clean": bool((rws[nr * rbpb: nr * rbpb + 128 * TP * L // 2] == SENT).all()),
                "shared_next_clean": bool((sws[ns] == SENT).all()),
                "mailbox_clean": bool((self.up._mailbox.view(torch.int32) == SENT).all())}

    def call(self, which, m, rows_own, sh_own, li, sleeps=0, check=False, cfg=None):
        sdst = torch.empty((m, H), dtype=torch.bfloat16, device=self.dev)
        gr, gs = int(self.co._routed_flags.view(torch.int32)[0]), int(self.co._shared_flags[0])
        self.feed(rows_own, sh_own, sdst, sleeps)
        out = self.ctl(m, sdst, li) if which == "ctl" else self.cand(m, sdst, li, cfg=cfg)
        self.wait(f"{which} M{m}")
        pr = self.protocol(gr, gs) if check else None
        self.barrier(f"after {which} M{m}")
        return out, pr

    # -------- modes

    def check(self) -> dict:
        shapes, ok = {}, True
        for m in CHECK_M:
            for seed in range(3):
                li = (m + seed) % len(self.layers)
                rows, shs, ro, so = self.inputs(m, 100 + seed)
                ref = self.reference(rows, shs, li)
                ctl, p0 = self.call("ctl", m, ro, so, li, check=True)
                cand, p1 = self.call("cand", m, ro, so, li, check=True)
                cand2, p2 = self.call("cand", m, ro, so, li, check=True)
                ctl2, p3 = self.call("ctl", m, ro, so, li, check=True)  # image after candidate
                alt = {json.dumps(c): self.call("cand", m, ro, so, li, check=True, cfg=c) for c in ALT_CHECK}
                allr = torch.empty((TP, m, H), dtype=torch.bfloat16, device=self.dev)
                self.dist.all_gather_into_tensor(allr, cand.contiguous())
                ec, ek = err(cand, ref), err(ctl, ref)
                absmax = ref.abs().max().item()
                rec = {"layer": li, "cand": ec, "ctl": ek, "ref_absmax": absmax, "numerics_ok": bar(ec, ek, absmax),
                       "strict_le_info": ec["max_abs"] <= ek["max_abs"] and ec["mean_abs"] <= ek["mean_abs"],
                       "deterministic": same(cand, cand2), "ctl_repeat_eq": same(ctl, ctl2),
                       "ranks_equal": all(same(allr[r], allr[0]) for r in range(TP)),
                       "cand_eq_ctl_frac_info": float((cand.view(torch.int16) == ctl.view(torch.int16)).float().mean()),
                       "protocol": {k: all(p[k] for p in [p0, p1, p2, p3] + [a[1] for a in alt.values()])
                                    for k in p0},
                       "alt_cfg_eq": {k: same(a[0], cand) for k, a in alt.items()}}
                rec["ok"] = bool(rec["numerics_ok"] and rec["deterministic"] and rec["ranks_equal"]
                                 and all(rec["protocol"].values()) and all(rec["alt_cfg_eq"].values()))
                ok &= rec["ok"]
                shapes[f"M{m}-s{seed}"] = rec
                if self.rank == 0:
                    log(f"CHECK M={m} seed={seed}", rec)
        up, gamma = self.layers[0]
        sd = torch.zeros((128, H), dtype=torch.bfloat16, device=self.dev)
        dec = {"M128": self.mt.run(self.op, self.prefilled(128), sd, gamma, up) is None}
        os.environ["PERFBOT_EXP_MOETAIL"] = "0"
        dec["switch_off"] = self.mt.run(self.op, self.prefilled(8), sd[:8], gamma, up) is None
        os.environ["PERFBOT_EXP_MOETAIL"] = "1"
        dec["not_prefilled"] = self.mt.run(self.op, object(), sd[:8], gamma, up) is None
        ok &= all(dec.values())
        return {"ok": bool(ok), "declines": dec, "shapes": shapes}

    def graph(self) -> dict:
        shapes, ok = {}, True
        for m in GRAPH_M:
            li = m % len(self.layers)
            up, gamma = self.layers[li]
            rows_b = torch.empty((m, L), dtype=torch.bfloat16, device=self.dev)
            sh_b = torch.empty((m, H), dtype=torch.bfloat16, device=self.dev)
            sdst = torch.empty((m, H), dtype=torch.bfloat16, device=self.dev)
            _, _, ro, so = self.inputs(m, 300)
            rows_b.copy_(ro)
            sh_b.copy_(so)
            self.feed(rows_b, sh_b, sdst)
            self.cand(m, sdst, li)
            self.barrier("graph warm")
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self.feed(rows_b, sh_b, sdst)
                out = self.cand(m, sdst, li)
            self.barrier("graph captured")
            rec = {}
            for rep in range(2):
                rows, shs, ro, so = self.inputs(m, 310 + rep)
                rows_b.copy_(ro)
                sh_b.copy_(so)
                self.barrier("graph inputs")
                graph.replay()
                self.wait(f"graph replay M{m}")
                got = out.clone()
                self.barrier("graph replayed")
                want, _ = self.call("cand", m, ro, so, li)
                rec[f"replay{rep}_eq_eager"] = same(got, want)
                e = err(got, self.reference(rows, shs, li))
                rec[f"replay{rep}_err_info"] = e
            rec["ok"] = all(v for k, v in rec.items() if k.endswith("eager"))
            ok &= rec["ok"]
            shapes[f"M{m}"] = rec
            if self.rank == 0:
                log(f"GRAPH M={m}", rec)
            del graph
            self.barrier("graph freed")
        return {"ok": bool(ok), "shapes": shapes}

    def pdl_order(self) -> dict:
        plan_rng, rng = random.Random(5), random.Random(1000 + self.rank)
        plan = [48] * 60 + [8] * 8 + [64] * 8
        plan_rng.shuffle(plan)
        rows_out, ok = [], True
        for i, m in enumerate(plan):
            li = i % len(self.layers)
            _, _, ro, so = self.inputs(m, 500 + i)
            want, _ = self.call("cand", m, ro, so, li, 0)
            sleeps = rng.randint(0, 60)
            got, _ = self.call("cand", m, ro, so, li, sleeps)
            eq = same(got, want)
            ok &= eq
            rows_out.append({"m": m, "sleeps_x500ns": sleeps, "eq": eq})
        sens = []
        for i in range(6):  # info: the tail on an unordered stream reads the shared input before the feed wrote it
            m = 48
            _, _, ro, so = self.inputs(m, 900 + i)
            want, _ = self.call("cand", m, ro, so, 0, 0)
            sdst = torch.zeros((m, H), dtype=torch.bfloat16, device=self.dev)
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            self.feed(ro, so, sdst, 200)
            with torch.cuda.stream(side):
                bad = self.cand(m, sdst, 0)
            torch.cuda.current_stream().wait_stream(side)
            self.wait("pdl sensitivity")
            sens.append(not same(bad, want))
            self.barrier("pdl sensitivity")
        rec = {"ok": bool(ok), "n": len(rows_out), "n_m48": sum(r["m"] == 48 for r in rows_out),
               "mismatches": [r for r in rows_out if not r["eq"]],
               "unordered_stream_mismatch_frac_info": sum(sens) / len(sens)}
        if self.rank == 0:
            log("PDL-ORDER", rec)
        return rec

    def time(self) -> dict:
        import perfbot_kernels.attn_res_one as ar
        import perfbot_kernels.moe_stage as ms

        dev, nl = self.dev, len(self.layers)
        g = torch.Generator(device=dev).manual_seed(77 + self.rank)
        E = ms.E
        u8 = lambda *s: torch.randint(0, 256, s, dtype=torch.uint8, device=dev, generator=g)
        sf = lambda *s: torch.randint(118, 124, s, dtype=torch.uint8, device=dev, generator=g)
        w13, s13 = u8(E, 2 * ms.I, ms.H // 2), sf(E, 2 * ms.I, ms.H // 32)
        w2s = [(u8(E, ms.H, ms.I // 2), sf(E, ms.H, ms.I // 32)) for _ in range(nl)]
        ms.set_target(dev, self.co._routed_multicast_ptr, self.co._routed_flags, self.rank, 128)
        rec = {"unit": "us per layer", "method": "CUDA-graph replay of a 4-layer chain, median of 5 x 100, ranks "
               "aligned by a barrier before each batch; tail = chain/4 - chain-without-tail/4", "layer":
               "moe_stage fc2 (distinct w2) -> tail -> attn_res_one (delta = tail out, PDL)", "per_m": {}}

        def graph_timed(fn, reps=100):
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    fn()
            torch.cuda.current_stream().wait_stream(s)
            self.barrier("time warm")
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fn()
            self.barrier("time captured")
            for _ in range(10):
                graph.replay()
            times = []
            for _ in range(5):
                self.barrier("time batch")
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                for _ in range(reps):
                    graph.replay()
                b.record()
                self.wait("time batch", 120)
                times.append(a.elapsed_time(b) * 1000.0 / reps)
            del graph
            self.barrier("time done")
            return statistics.median(times)

        def reset():  # after a chain without tail: fc2 wrote a generation nobody consumed
            self.wait("reset")
            self.co._routed_workspace.fill_(-0.0)
            self.wait("reset fill")
            self.barrier("reset")

        for m, alt in [(m, False) for m in TIME_M] + [(m, True) for m in TIME_M]:
            try:
                # pass 1: the default at every M (gate numbers, saved first); pass 2: the alt configs (info only)
                if alt and not ALT_TIME:
                    break
                x = (torch.randn(m, ms.H, device=dev, generator=g) * 0.5).to(torch.bfloat16)
                ids = torch.stack([torch.randperm(E, device=dev, generator=g)[:TOPK] for _ in range(m)]).int()
                wts = torch.rand(m, TOPK, device=dev, generator=g) + 0.1
                wts = (wts / wts.sum(-1, keepdim=True)).to(torch.bfloat16)
                packed = ((ids << 16) | (wts.view(torch.int16).int() & 0xFFFF)).int().contiguous()
                sdst = (torch.randn(m, H, device=dev, generator=g) * 0.05).to(torch.bfloat16)
                bank = torch.randn(m, KBANK, H, device=dev, generator=g).to(torch.bfloat16)
                nw = (1.0 + 0.1 * torch.randn(H, device=dev, generator=g)).to(torch.bfloat16)
                qk = (0.05 * torch.randn(H, device=dev, generator=g)).to(torch.bfloat16)
                prefix = [torch.randn(m, H, device=dev, generator=g).to(torch.bfloat16) for _ in range(nl)]
                fixed = torch.randn(m, H, device=dev, generator=g).to(torch.bfloat16)
                xs = [torch.empty(m, H, device=dev, dtype=torch.bfloat16) for _ in range(nl)]
                po = ms.run(x, packed, w13, s13, w2s[0][0], w2s[0][1], 0, 4.0, 25.0)  # full stage once: workspace
                if po is None:
                    raise RuntimeError("moe_stage declined")
                self.cand(m, sdst, 0)  # consumes the generation fc2 wrote
                self.barrier("time stage warm")

                def fc2(i):
                    ms.run(x, packed, w13, s13, w2s[i][0], w2s[i][1], 0, 4.0, 25.0, only="fc2")

                def attn(i, delta):
                    ar.launch(prefix[i], delta, bank, nw, qk, nw, KUSED, -1, 1e-6, 1e-6, out=xs[i])

                def chain(tail):
                    def fn():
                        for i in range(nl):
                            fc2(i)
                            attn(i, fixed if tail is None else tail(i))
                    return graph_timed(fn) / nl

                if not alt:
                    row = {"pred_us": chain(None)}
                    reset()
                    row["ctl_chain_us"] = chain(lambda i: self.ctl(m, sdst, i))
                    row["cand_chain_us"] = chain(lambda i: self.cand(m, sdst, i))
                    row["ctl_tail_us"] = row["ctl_chain_us"] - row["pred_us"]
                    row["cand_tail_us"] = row["cand_chain_us"] - row["pred_us"]
                    row["gain_us"] = row["ctl_tail_us"] - row["cand_tail_us"]
                    rec["per_m"][f"M{m}"] = row
                    self.rec["time_partial"] = rec
                    self.save()
                    stamp_cfgs = ({},)
                else:  # same inputs for the alt pass baseline: predecessor chain and the default again
                    row = rec["per_m"][f"M{m}"]
                    row["alt_pass_pred_us"] = chain(None)
                    reset()
                    row["alt_pass_default_tail_us"] = chain(lambda i: self.cand(m, sdst, i)) - row["alt_pass_pred_us"]
                    stamp_cfgs = ()
                for c in (ALT_TIME if alt else ()):  # probe first: error vs fp64 <= the default's, protocol clean
                    key = json.dumps(c, sort_keys=True)
                    rows, shs, ro, so = self.inputs(m, 700 + m)
                    ref = self.reference(rows, shs, 0)
                    want, _ = self.call("cand", m, ro, so, 0, check=True)
                    got, pr = self.call("cand", m, ro, so, 0, check=True, cfg=c)
                    e_alt, e_def = err(got, ref), err(want, ref)
                    ok_n = bar(e_alt, e_def, ref.abs().max().item())
                    row[f"alt {key} probe"] = {"eq_default": same(got, want), "err": e_alt, "default_err": e_def,
                                               "numerics_ok": ok_n, "protocol": pr}
                    self.save()
                    if not (ok_n and all(pr.values())):
                        continue
                    row[f"alt {key} tail_us"] = chain(lambda i: self.cand(m, sdst, i, cfg=c)) - row["alt_pass_pred_us"]
                    row[f"alt {key} gain_us"] = row["ctl_tail_us"] - row[f"alt {key} tail_us"]
                    self.save()
                    stamp_cfgs = stamp_cfgs + (c,)
                if m in (8, 48, 64):  # steady-state per-CTA stamps: last layer of the last replay of a stamped chain
                    for c in stamp_cfgs:
                        key = json.dumps(c, sort_keys=True)
                        self.mt.compiled(dev, True, **c)
                        self.barrier("stamps compiled")
                        row[f"{key} stamped_chain_us_info"] = chain(lambda i: self.cand(m, sdst, i, stamps=True, cfg=c))
                        d = self.mt._WS[dev.index]["dbg"].view(self.mt.CTAS, 16).cpu()
                        row[f"{key} stamps"] = stamp_summary(d)
                        self.barrier("stamps")
                rec["per_m"][f"M{m}"] = row
                if self.rank == 0:
                    log(f"TIME M={m}{' alt' if alt else ''}", row)
                if not alt and m == TIME_M[-1]:  # the gate verdict before any alt runs
                    gains = {k: r["gain_us"] for k, r in rec["per_m"].items() if "gain_us" in r}
                    rec["gain_us"] = gains
                    rec["ok"] = bool(gains.get("M48", -1e9) >= GAIN_US and all(v >= 0 for v in gains.values()))
                self.save()
            except Exception:  # noqa: BLE001  an alt failure keeps the default's gate numbers
                if not alt:
                    raise
                rec["alt_error"] = traceback.format_exc()[-3000:]
                self.save()
                break
        rec["per_m"]["M128"] = {"info": "moe_stage and moe_tail serve M <= 64; the candidate declines (image path)"}
        return rec


def _rank_main(rank: int, port: int, out_dir: str, modes: list) -> None:
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=TP, device_id=dev)
    rec = {"rank": rank}
    path = Path(out_dir, f"moetail-rank{rank}.json")
    save = lambda: path.write_text(json.dumps(rec, indent=1, default=str))
    try:
        rk = Rank(rank, dev, rec, save)
        save()
        for name in modes:
            t0 = time.time()
            try:
                rec[name] = getattr(rk, name)()
            except Exception:  # noqa: BLE001
                rec[name] = {"ok": False, "error": traceback.format_exc()}
                traceback.print_exc()
            rec[name]["seconds"] = round(time.time() - t0, 1)
            save()
    except Exception:  # noqa: BLE001
        rec["error"] = traceback.format_exc()
        traceback.print_exc()
    save()
    print(f"MOETAIL rank={rank} done {[(n, (rec.get(n) or {}).get('ok')) for n in modes]}", flush=True)
    try:
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001
        pass


def main() -> int:
    os.environ.setdefault("PERFBOT_EXP_MOETAIL", "1")  # the gate measures the candidate; the switch defaults off
    ap = argparse.ArgumentParser()
    for f in ("check", "graph", "pdl-order", "time"):
        ap.add_argument(f"--{f}", action="store_true")
    ap.add_argument("--deadline-s", type=float, default=1380.0)
    args = ap.parse_args()
    modes = [n for n in ("check", "graph", "pdl_order", "time") if getattr(args, n)]
    path = results_path()
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_rank_main, args=(r, PORT, str(path.parent), modes)) for r in range(TP)]
    t0 = time.time()
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=max(1.0, args.deadline_s - (time.time() - t0)))
    for p in procs:
        if p.is_alive():
            p.terminate()
    ranks = []
    for r in range(TP):
        f = path.parent / f"moetail-rank{r}.json"
        ranks.append(json.loads(f.read_text()) if f.is_file() else {"rank": r, "error": "no record"})
    record = {"item": "expedition-B-tail", "modes": modes, "exitcodes": [p.exitcode for p in procs],
              "seconds": round(time.time() - t0, 1), "kernel_sha256": ranks[0].get("kernel_sha256"),
              "hangs": {r["rank"]: r["hang"] for r in ranks if "hang" in r},
              "errors": {r["rank"]: r["error"][-1500:] for r in ranks if "error" in r}}
    for k in ("weights", "eps", "collective_file", "prefilled_ready", "max_active_clusters_7", "load_s",
              "op_build_s"):
        record[k] = ranks[0].get(k)
    parts = {}
    for n in modes:
        recs = [r.get(n) or (r.get("time_partial") if n == "time" else None) or {} for r in ranks]
        parts[n] = all(x.get("ok") is True for x in recs)
        record[n] = {"ok": parts[n], "rank0": recs[0],
                     "failed_ranks": [r["rank"] for r, x in zip(ranks, recs) if x.get("ok") is not True]}
        if n == "time":
            record[n]["gain_us_all_ranks"] = {r["rank"]: x.get("gain_us") for r, x in zip(ranks, recs)}
            parts[n] = recs[0].get("ok") is True
            record[n]["ok"] = parts[n]
    record["parts_ok"] = parts
    record["gate_pass"] = bool(set(parts) == {"check", "graph", "pdl_order", "time"} and all(parts.values()))
    path.write_text(json.dumps(record, indent=1, default=str))
    print(f"PERFBOT-MOE-TAIL-GATE parts={parts} gate_pass={record['gate_pass']} record={path}", flush=True)
    return 0 if record["gate_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
