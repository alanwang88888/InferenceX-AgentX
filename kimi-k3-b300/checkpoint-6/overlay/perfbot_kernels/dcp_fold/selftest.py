"""Kernel-only gate of the direct-DCP epoch + signal fold (expedition line D). Run in the server env (PERFBOT_PRE_EXEC):

  python3 -B -m perfbot_kernels.dcp_fold.selftest [--check] [--graph] [--tail] [--time]

8 processes (one per GPU, NCCL + symmetric memory) build the image workspaces (DirectDCPQGatherWorkspace,
DirectDCPA2AWorkspace of vllm/v1/attention/ops/dcp.py) for two shapes: target (12 heads/rank) and draft (8 heads/rank),
q fp8 [T, H, 576], partial output bf16 [T, 8H, 512], LSE fp32 [T, 8H], max_num_tokens 64. control = the image ops
(ws.gather, ws.lse_reduce: 2 + 4 launches); candidate = dcp_fold.q_gather / dcp_fold.combine (1 + 2 launches).
  --check  T 1/7/8/42/48/64 x shapes x seq variants (none; 6-token seqs with empty shards) x lse base (2, e); LSE with
           -inf, NaN, +inf; one padded-token-stride q. Each case runs the call pair (q-gather, combine) as
           image/image, candidate/candidate, image/candidate and rank-mixed (even ranks candidate): every output
           bit-equal to the image's, q-gather also equal to the reference concatenation; after each case the epochs
           equal the call counts, the completion and dispatch counters are 0, the newest signal slot holds the epoch
           on every rank. Declines: switch off, T 65, fp16 output, bf16 LSE -> None.
  --graph  capture [candidate pair, image pair] per shape at T 48 / 42 / 8; 2 replays on new inputs: candidate bit-equal
           to image, epochs advance 2 per replay.
  --tail   3000 call pairs, shape / T / input set from a shared seed, path (image or candidate) chosen per rank and per
           op, random per-rank GPU delays; every output bit-equal to the precomputed image output; then a captured
           graph of 8 rank-mixed pairs replayed 300 times. No hang (watchdog 300 s), counters consistent.
  --time   CUDA-graph replay of a 24-pair chain per shape, median of 5 x 50 replays, ranks aligned by a barrier:
           image/image, candidate/candidate, candidate-q/image-a2a, image-q/candidate-a2a; then (info) each ALT config
           after a bit-equality probe against the default, saving vs a same-pass image chain.
GATE: check, graph, tail pass on all 8 ranks; saving per step = 24 x save(target T 48) + 5 x save(draft T 42) >= 120 us
(rank 0; all ranks recorded). Record: $PERFBOT_WORKSPACE/$PERFBOT_RESULTS/dcpfold-gate.json (+ dcpfold-rank<r>.json).
"""

import argparse
import json
import os
import random
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch

W, MAXT, DQ, DO = 8, 64, 576, 512
SHAPES = {"target": 12, "draft": 8}
SERVED = {"target": (48, 24), "draft": (42, 5)}  # T, MLA calls per step
CHECK_T = [1, 7, 8, 42, 48, 64]
GAIN_US = 120.0
PORT = 29541
ALT = ({"tpb": 1, "fence1": False}, {"tpb": 6}, {"fence1": False}, {"pdl": True}, {"tpb": 6, "pdl": True})


def results_path(name="dcpfold-gate.json") -> Path:
    root = Path(os.environ.get("PERFBOT_WORKSPACE", "/workspace"))
    out = root / os.environ.get("PERFBOT_RESULTS", "results/measure")
    out.mkdir(parents=True, exist_ok=True)
    return out / name


def log(tag, obj):
    print(f"{tag} {json.dumps(obj, default=str)}", flush=True)


def same(a, b) -> bool:
    return a.shape == b.shape and bool(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))


class Case:
    """Inputs of one call pair for this rank; q_ref (all ranks' q, concatenated) is the same on every rank."""

    def __init__(self, dev, rank, shape, t, seed, seqs=True, base_e=False, pad=0):
        h = SHAPES[shape]
        self.shape, self.t, self.base_e = shape, t, base_e
        g = lambda r: torch.Generator(device=dev).manual_seed(seed * 7919 + t * 131 + h * 17 + r)
        qs = [torch.randint(0, 256, (t, h * DQ), dtype=torch.uint8, device=dev, generator=g(r)) for r in range(W)]
        self.q_ref = torch.stack(qs, 1).view(t, W * h, DQ).view(torch.float8_e4m3fn)
        store = torch.zeros((t, h * DQ + pad), dtype=torch.uint8, device=dev)
        store[:, : h * DQ] = qs[rank]
        self.q = torch.as_strided(store.view(torch.float8_e4m3fn), (t, h, DQ), (h * DQ + pad, DQ, 1))
        gr = g(100 + rank)
        self.po = (torch.randn(t, W * h, DO, device=dev, generator=gr) * 2).to(torch.bfloat16)
        lse = torch.randn(t, W * h, device=dev, generator=gr) * 3
        lse.view(-1)[::37] = float("-inf")
        lse.view(-1)[5::53] = float("nan")
        lse.view(-1)[11::71] = float("inf")
        self.lse = lse
        self.seq = self.qsl = None
        if seqs:
            n = max(1, t // 6)
            bounds = [min(t, 6 * i) for i in range(n)] + [t]
            self.qsl = torch.tensor(bounds, dtype=torch.int32, device=dev)
            self.seq = torch.randint(0, 3, (n,), dtype=torch.int32, device=dev, generator=gr)


class Rank:
    def __init__(self, rank, dev, rec, save):
        import torch.distributed as dist

        self.rank, self.dev, self.rec, self.save, self.dist = rank, dev, rec, save, dist
        group = dist.group.WORLD
        try:
            import torch.distributed._symmetric_memory as symm_mem

            symm_mem.enable_symm_mem_for_group(group.group_name)
        except Exception:  # noqa: BLE001
            pass
        import vllm._C_stable_libtorch  # noqa: F401  (registers torch.ops._C.direct_dcp_*)
        from vllm.v1.attention.ops import dcp

        import perfbot_kernels.dcp_fold as fold

        self.fold = fold
        rec["dcp_file"] = dcp.__file__
        rec["kernel_sha256"] = fold.source_sha256()
        self.qws = {s: dcp.DirectDCPQGatherWorkspace(group, dev, MAXT, h, DQ, torch.float8_e4m3fn, 1, None)
                    for s, h in SHAPES.items()}
        self.aws = {s: dcp.DirectDCPA2AWorkspace(group, dev, MAXT, h, DO, torch.bfloat16, 1) for s, h in SHAPES.items()}
        self.calls = {s: [0, 0] for s in SHAPES}  # q-gather, combine calls per shape (all ranks issue the same count)

    # -------- plumbing

    def wait(self, what, limit=300.0):
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

    def qg(self, path, c, cfg=None):
        ws = self.qws[c.shape]
        self.calls[c.shape][0] += 1
        if path == "img":
            return ws.gather(c.q)
        out = self.fold.q_gather(ws, c.q, **(cfg or {}))
        assert out is not None, "candidate q_gather declined"
        return out

    def a2(self, path, c, cfg=None):
        ws = self.aws[c.shape]
        self.calls[c.shape][1] += 1
        if path == "img":
            return ws.lse_reduce(c.po, c.lse, c.base_e, c.seq, c.qsl)
        out = self.fold.combine(ws, c.po, c.lse, c.base_e, c.seq, c.qsl, **(cfg or {}))
        assert out is not None, "candidate combine declined"
        return out

    def pair(self, pq, pa, c, cfg=None):
        q = self.qg(pq, c, cfg).clone()
        return q, self.a2(pa, c, cfg)

    def protocol(self) -> dict:
        """Epochs = call counts, counters 0, newest signal slot = epoch on every rank (after a sync)."""
        self.wait("protocol")
        bad = {}
        for s in SHAPES:
            nq, na = self.calls[s]
            qw, aw = self.qws[s], self.aws[s]
            eq, ea = int(qw.epoch[0]), int(aw.epoch[0])
            sq = qw.received_signal[0].view(2, W)[nq & 1].tolist()
            sa = aw.received_signal[0].view(2, W)[na & 1].tolist()
            ctr = getattr(aw, "_perfbot_dcpfold_ctr", None)
            got = {"epoch_q": eq, "epoch_a": ea, "completion": int(qw.completion[0, 0]),
                   "ctr": 0 if ctr is None else int(ctr[0]), "sig_q": sq, "sig_a": sa}
            want = {"epoch_q": nq, "epoch_a": na, "completion": 0, "ctr": 0, "sig_q": [nq & 0xFFFFFFFF] * W,
                    "sig_a": [na & 0xFFFFFFFF] * W}
            if nq == 0:
                del got["sig_q"], want["sig_q"]
            if na == 0:
                del got["sig_a"], want["sig_a"]
            if got != want:
                bad[s] = {"got": got, "want": want}
        return bad

    # -------- modes

    def check(self) -> dict:
        rec = {"cases": 0, "fail": [], "protocol": []}
        mixed = "cand" if self.rank % 2 == 0 else "img"
        combos = [("img", "img"), ("cand", "cand"), ("img", "cand"), (mixed, mixed)]
        cases = []
        for s in SHAPES:
            for t in CHECK_T:
                for seqs in (False, True):
                    cases.append(dict(shape=s, t=t, seed=1, seqs=seqs, base_e=False))
                cases.append(dict(shape=s, t=t, seed=2, seqs=True, base_e=True))
            cases.append(dict(shape=s, t=48, seed=3, seqs=True, base_e=False, pad=64))
        for kw in cases:
            c = Case(self.dev, self.rank, **kw)
            ref_q, ref_o = self.pair("img", "img", c)
            if not same(ref_q, c.q_ref):
                rec["fail"].append({**kw, "what": "image q-gather != reference"})
            for pq, pa in combos[1:]:
                q, o = self.pair(pq, pa, c)
                if not (same(q, ref_q) and same(o, ref_o)):
                    rec["fail"].append({**kw, "paths": [pq, pa], "q_equal": same(q, ref_q), "out_equal": same(o, ref_o),
                                        "max_abs": (o.float() - ref_o.float()).abs().nan_to_num(9e9).max().item()})
            rec["cases"] += 1
            bad = self.protocol()
            if bad:
                rec["protocol"].append({**kw, **bad})
            self.barrier("check case")
        c = Case(self.dev, self.rank, "target", 8, 4)
        os.environ[self.fold.ENV] = "0"
        off = (self.fold.q_gather(self.qws["target"], c.q), self.fold.combine(self.aws["target"], c.po, c.lse, False))
        os.environ[self.fold.ENV] = "1"
        big = Case(self.dev, self.rank, "target", MAXT + 1, 4, seqs=False)
        dec = {"switch_off": [x is None for x in off],
               "t65": [self.fold.q_gather(self.qws["target"], big.q) is None,
                       self.fold.combine(self.aws["target"], big.po, big.lse, False) is None],
               "fp16_out": self.fold.combine(self.aws["target"], c.po.half(), c.lse, False) is None,
               "bf16_lse": self.fold.combine(self.aws["target"], c.po, c.lse.bfloat16(), False) is None}
        rec["declines"] = dec
        rec["ok"] = (not rec["fail"] and not rec["protocol"] and all(dec["switch_off"]) and all(dec["t65"])
                     and dec["fp16_out"] and dec["bf16_lse"])
        return rec

    def graph(self) -> dict:
        rec = {"per_shape": {}}
        ok = True
        for s, t in (("target", 48), ("draft", 42), ("target", 8)):
            c = Case(self.dev, self.rank, s, t, 10)
            self.pair("cand", "cand", c)  # eager warm-up (compile, counter)
            self.barrier("graph warm")
            g = torch.cuda.CUDAGraph()
            n0 = list(self.calls[s])
            with torch.cuda.graph(g):
                cq, co = self.pair("cand", "cand", c)
                iq, io = self.pair("img", "img", c)
            self.calls[s] = n0  # capture issued nothing
            self.barrier("graph captured")
            reps = []
            for k in range(2):
                n = Case(self.dev, self.rank, s, t, 11 + k)
                for dst, src in ((c.q, n.q), (c.po, n.po), (c.lse, n.lse), (c.seq, n.seq), (c.qsl, n.qsl)):
                    dst.copy_(src)
                g.replay()
                self.calls[s][0] += 2
                self.calls[s][1] += 2
                self.wait("graph replay")
                reps.append({"q_equal": same(cq, iq), "q_ref": same(iq, n.q_ref), "out_equal": same(co, io)})
            bad = self.protocol()
            del g
            self.barrier("graph done")
            r_ok = all(all(r.values()) for r in reps) and not bad
            ok &= r_ok
            rec["per_shape"][f"{s}-T{t}"] = {"replays": reps, "protocol": bad, "ok": r_ok}
        rec["ok"] = ok
        return rec

    def tail(self) -> dict:
        shared = random.Random(1234)
        mine = random.Random(5678 + self.rank)
        sets = {}
        for s in SHAPES:
            for t in (8, 42, 48):
                for k in range(3):
                    c = Case(self.dev, self.rank, s, t, 100 + k, seqs=k > 0)
                    sets[(s, t, k)] = (c, *self.pair("img", "img", c))
        self.barrier("tail precompute")
        keys = sorted(sets)
        bad = torch.zeros((), dtype=torch.int64, device=self.dev)
        n_iter, mix = 3000, {"img": 0, "cand": 0}
        t0 = time.time()
        for i in range(n_iter):
            c, rq, ro = sets[keys[shared.randrange(len(keys))]]
            pq, pa = mine.choice(("img", "cand")), mine.choice(("img", "cand"))
            mix[pq] += 1
            mix[pa] += 1
            if mine.random() < 0.3:
                torch.cuda._sleep(mine.randrange(1, 40000))
            q, o = self.pair(pq, pa, c)
            bad += (~torch.eq(q.view(torch.uint8), rq.view(torch.uint8))).sum()
            bad += (~torch.eq(o.view(torch.int16), ro.view(torch.int16))).sum()
            if i % 200 == 199:
                self.wait(f"tail iter {i}")
        self.wait("tail end")
        rec = {"iters": n_iter, "paths": mix, "mismatch_elems": int(bad), "seconds": round(time.time() - t0, 1),
               "protocol": self.protocol()}
        self.barrier("tail eager done")
        # graph: 8 rank-mixed pairs replayed 300 times
        plan = [(mine.choice(("img", "cand")), mine.choice(("img", "cand")), keys[shared.randrange(len(keys))])
                for _ in range(8)]
        for pq, pa, key in plan:
            self.pair(pq, pa, sets[key][0])  # eager warm-up of every path
        self.barrier("tail graph warm")
        g = torch.cuda.CUDAGraph()
        n0 = {s: list(v) for s, v in self.calls.items()}
        with torch.cuda.graph(g):
            outs = [self.pair(pq, pa, sets[key][0]) for pq, pa, key in plan]
        self.calls = n0
        self.barrier("tail graph captured")
        for k in range(300):
            g.replay()
            for pq, pa, key in plan:
                self.calls[key[0]][0] += 1
                self.calls[key[0]][1] += 1
            if k % 50 == 49:
                self.wait(f"tail graph replay {k}")
        self.wait("tail graph end")
        gbad = sum(int(not (same(q, sets[key][1]) and same(o, sets[key][2]))) for (q, o), (_, _, key) in zip(outs, plan))
        rec["graph"] = {"pairs": 8, "replays": 300, "plan": [p[:2] for p in plan], "bad_pairs": gbad,
                        "protocol": self.protocol()}
        del g
        self.barrier("tail done")
        rec["ok"] = (rec["mismatch_elems"] == 0 and not rec["protocol"] and gbad == 0
                     and not rec["graph"]["protocol"])
        return rec

    def time(self) -> dict:
        n_chain = 24

        def timed(c, pq, pa, cfg=None, reps=50):
            self.pair(pq, pa, c, cfg)
            self.barrier("time warm")
            n0 = {s: list(v) for s, v in self.calls.items()}
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(n_chain):
                    self.pair(pq, pa, c, cfg)
            self.calls = n0
            self.barrier("time captured")

            def replay(k):
                for _ in range(k):
                    g.replay()
                    self.calls[c.shape][0] += n_chain
                    self.calls[c.shape][1] += n_chain

            replay(5)
            times = []
            for _ in range(5):
                self.barrier("time batch")
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                replay(reps)
                b.record()
                self.wait("time batch", 120)
                times.append(a.elapsed_time(b) * 1000.0 / reps / n_chain)
            del g
            self.barrier("time done")
            return round(statistics.median(times), 3)

        rec = {"unit": "us per call pair (q-gather + combine)", "method": "CUDA-graph replay of a 24-pair chain, "
               "median of 5 x 50 replays, ranks aligned by a barrier before each batch", "per_shape": {}}
        for s, (t, _) in SERVED.items():
            c = Case(self.dev, self.rank, s, t, 20)
            r = {p: timed(c, *p.split("/")) for p in ("img/img", "cand/cand", "cand/img", "img/cand")}
            r["save"] = round(r["img/img"] - r["cand/cand"], 3)
            rec["per_shape"][s] = r
            self.rec["time_partial"] = rec
            self.save()
        rec["save_per_step_us"] = round(sum(rec["per_shape"][s]["save"] * n for s, (_, n) in SERVED.items()), 2)
        rec["ok"] = rec["save_per_step_us"] >= GAIN_US
        self.rec["time_partial"] = rec
        self.save()
        alts = []
        for cfg in ALT:  # info: probe bit-equality vs default, then time
            a = {"cfg": cfg}
            try:
                for s, (t, n) in SERVED.items():
                    c = Case(self.dev, self.rank, s, t, 21)
                    dq, do = self.pair("cand", "cand", c)
                    aq, ao = self.pair("cand", "cand", c, cfg)
                    a[s] = {"bit_equal": same(dq, aq) and same(do, ao), "protocol": self.protocol(),
                            "img": timed(c, "img", "img"), "cand": timed(c, "cand", "cand", cfg)}
                    a[s]["save"] = round(a[s]["img"] - a[s]["cand"], 3)  # same-pass control
                a["save_per_step_us"] = round(sum(a[s]["save"] * n for s, (_, n) in SERVED.items()), 2)
            except Exception:  # noqa: BLE001
                a["error"] = traceback.format_exc()[-800:]
            alts.append(a)
            rec["alt"] = alts
            self.save()
        return rec


def _rank_main(rank: int, port: int, out_dir: str, modes: list) -> None:
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=W, device_id=dev)
    rec = {"rank": rank}
    path = Path(out_dir, f"dcpfold-rank{rank}.json")
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
    print(f"DCPFOLD rank={rank} done {[(n, (rec.get(n) or {}).get('ok')) for n in modes]}", flush=True)
    try:
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001
        pass


def main() -> int:
    os.environ["PERFBOT_EXP_DCPFOLD"] = "1"  # the gate measures the candidate; the switch defaults off
    ap = argparse.ArgumentParser()
    for f in ("check", "graph", "tail", "time"):
        ap.add_argument(f"--{f}", action="store_true")
    ap.add_argument("--deadline-s", type=float, default=1380.0)
    args = ap.parse_args()
    modes = [n for n in ("check", "graph", "tail", "time") if getattr(args, n)]
    path = results_path()
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_rank_main, args=(r, PORT, str(path.parent), modes)) for r in range(W)]
    t0 = time.time()
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=max(1.0, args.deadline_s - (time.time() - t0)))
    for p in procs:
        if p.is_alive():
            p.terminate()
    ranks = []
    for r in range(W):
        f = path.parent / f"dcpfold-rank{r}.json"
        ranks.append(json.loads(f.read_text()) if f.is_file() else {"rank": r, "error": "no record"})
    record = {"item": "expedition-D-fold", "modes": modes, "exitcodes": [p.exitcode for p in procs],
              "seconds": round(time.time() - t0, 1), "kernel_sha256": ranks[0].get("kernel_sha256"),
              "dcp_file": ranks[0].get("dcp_file"), "hangs": {r["rank"]: r["hang"] for r in ranks if "hang" in r},
              "errors": {r["rank"]: r["error"][-1500:] for r in ranks if "error" in r}}
    parts = {}
    for n in modes:
        recs = [r.get(n) or (r.get("time_partial") if n == "time" else None) or {} for r in ranks]
        parts[n] = all(x.get("ok") is True for x in recs)
        record[n] = {"ok": parts[n], "rank0": recs[0],
                     "failed_ranks": [r["rank"] for r, x in zip(ranks, recs) if x.get("ok") is not True]}
        if n == "time":
            record[n]["save_per_step_us_all_ranks"] = {r["rank"]: x.get("save_per_step_us") for r, x in zip(ranks, recs)}
            parts[n] = recs[0].get("ok") is True
            record[n]["ok"] = parts[n]
    record["parts_ok"] = parts
    record["gate_pass"] = bool(set(parts) == {"check", "graph", "tail", "time"} and all(parts.values()))
    path.write_text(json.dumps(record, indent=1, default=str))
    print(f"PERFBOT-DCPFOLD-GATE parts={parts} gate_pass={record['gate_pass']} record={path}", flush=True)
    return 0 if record["gate_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
