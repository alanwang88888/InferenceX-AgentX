"""Item 17 rung "pin": kernel-only arm, 1 GPU. Run from the bundle root, no PYTHONPATH:

    python3 -B -m perfbot_kernels.moe_front.b300_pin --batches 16,24,32,40

Bracket (item 08 / item 17 r0): StandIn (3 CTAs, release at entry, 6 us spin; c = its end) -> WsGemm (publish)
-> RouteSitu (served: route + copy, polled, plan). Mailbox template: the other 7 ranks published, rank 0 armed.
Publish = unicast store (pin.unicast_words) in BOTH arms (multimem needs >= 2 GPUs).

1. Bit check per rows: served vs pin from the same inputs, ``--draws`` plain draws + needle (x row 0 x 64) +
   tie (17 experts +10). Every output byte: published logit words and latent, ids, weights, copied latent,
   plan, mailbox after the route. Negative control: pin with x[0, 0] + 1 must change the published words.
2. Graph replay: 3 layers per CUDA graph, served vs pin; every layer's outputs bit-equal after one replay.
3. Timing (rows in --time): arms served / pin / served2 (A/A), ABBA blocks, CUPTI kernel times (torch.profiler):
   route end - c and WsGemm end - c, mean and p50 (us).
Pass line: bit-equal everywhere AND at 24 and 40 rows: gain = served - pin mean route end >= 0.3 us and >= 5 x
the A/A gap (|served - served2| mean route end).
Why the mean: the route end is bimodal (Lamport polling), so its p50 can jump between modes; on B200 the
served-vs-served p50 differed by 0.19 us at 24 rows while the means agreed (docs/item-17/pin).
Why bit checks before all timing: interleaving them raised the 24-row A/A gap from 0.06 to 0.13 us on B200, and
the A/A gap is part of the pass line. Both rules were set during the B200 build, before the B300 arm, and disclosed.
Exit 0 = pass, 1 = bit-equal but gain below the line, 2 = not bit-equal or crash. Last line:
PIN PASS|FAIL rows=... equal=... d24=... d40=... exit=N"""
import argparse
import json
import sys
import time
import traceback

import torch

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common import host as H
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm import ws_gemm as G
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe import front as F

from . import pin as PIN

G.multicast_words = PIN.unicast_words  # before any compile: both arms trace the unicast store

import cutlass.cute as cute  # noqa: E402
from cutlass import Float32, Int32, Int64  # noqa: E402
from cutlass.cute.runtime import make_fake_stream  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

from . import STANDIN_NS, StandIn  # noqa: E402

N, SHARD, VALID, SENT = 560, 448, 112, -0x80000000
POOL = {16: 119, 17: 125, 24: 169, 32: 200, 33: 205, 40: 234}  # item 08 timed.py POOL; others 100
KINDS = ("StandIn", "WsGemm", "RouteSitu")


class Layer:
    def __init__(self, seed, b, dev, case="plain"):
        g = torch.Generator(device=dev).manual_seed(500 + seed)
        rnd = lambda *s, k=0.02: (torch.randn(*s, device=dev, generator=g) * k).bfloat16()
        self.b, self.dev = b, dev
        self.x, self.packed = rnd(b, 7168, k=1.0), rnd(N, 7168)
        if case == "needle":
            self.x[0] *= 64
        count = 17 if case == "tie" else POOL.get(b, 100)
        pool = torch.randperm(896, generator=torch.Generator().manual_seed(seed))[:count]
        self.bias = torch.zeros(896, dtype=torch.float32, device=dev)
        self.bias[pool.to(dev)] = 10.0
        t = torch.randn(1, 64, F.WIDTH, device=dev, generator=g).bfloat16()
        words = t.view(torch.int32)
        words[..., 1792:] = torch.randn(1, 64, 896, device=dev, generator=g).view(torch.int32)
        words[..., : SHARD // 2] = SENT
        words[..., 1792: 1792 + VALID] = SENT
        self.template, self.box = t, t.clone()
        z = lambda *s, dt: torch.zeros(*s, dtype=dt, device=dev)
        self.logits, self.ids = z(b, 896, dt=torch.float32), z(b, 16, dt=torch.int32)
        self.wts, self.latent = z(b, 16, dt=torch.bfloat16), z(b, 3584, dt=torch.bfloat16)
        self.gate, self.h = z(b, 1536, dt=torch.float32), z(b, 768, dt=torch.bfloat16)
        self.plan, self.sb = z(F.PLAN_BYTES, dt=torch.uint8), z(STANDIN_NS, dt=torch.int64)
        ks = F.front_ks(b, N)
        route = F.compile_route(b, 768, True, (True, True, False), dev.index, True, True)
        self.k = {"served": (F.compile_publish(N, N - SHARD, VALID, b, dev.index, ks), route),
                  "pin": (PIN.compile_publish(N, N - SHARD, VALID, b, dev.index, ks), route)}
        self.k["served2"] = self.k["served"]

    def reset(self):
        self.box.copy_(self.template)
        for t in (self.logits, self.ids, self.wts, self.latent):
            t.zero_()
        self.plan.fill_(0x5A)

    def front(self, arm, s):
        self.k[arm][0](H.to_cute(self.x), H.to_cute(self.packed), H.to_cute(self.box), Int64(self.box.data_ptr()),
                       Int32(0), Int32(0), H.to_cute(self.logits), s)

    def route(self, arm, s):
        args = (self.logits, self.bias, self.ids, self.wts, self.box, self.latent, self.gate, self.h)
        self.k[arm][1](*map(H.to_cute, args), Float32(2.5), Float32(4.0), Float32(25.0), H.to_cute(self.plan), s)

    def chain(self, arm, standin, ns):
        s = H.stream(self.dev)
        self.box.copy_(self.template)
        standin(Int32(ns), Int64(self.sb.data_ptr()), s)
        self.front(arm, s)
        self.route(arm, s)

    def outputs(self, published=None):
        """Byte copies (a view would alias the live buffers)."""
        out = {k: getattr(self, k).contiguous().view(torch.uint8).clone()
               for k in ("ids", "wts", "latent", "plan", "box")}
        if published is not None:
            w = published.view(torch.int32)[0, : self.b]
            out["pub_logits"] = w[:, 1792: 1792 + VALID].contiguous().view(torch.uint8).clone()
            out["pub_latent"] = published[0, : self.b, :SHARD].contiguous().view(torch.uint8).clone()
        return out


def eager(L, arm, s):
    L.reset()
    L.front(arm, s)
    torch.cuda.synchronize()
    published = L.box.clone()
    L.route(arm, s)
    torch.cuda.synchronize()
    return L.outputs(published)


def check(b, dev, draws):
    """Served vs pin, eager, per case; negative control. Returns (equal, row)."""
    s = H.stream(dev)
    row, equal = {"batch": b, "cases": {}}, True
    cases = [("plain", 1000 * d + b) for d in range(draws)] + [("needle", 7000 + b), ("tie", 7100 + b)]
    for case, seed in cases:
        L = Layer(seed, b, dev, case)
        a, p = eager(L, "served", s), eager(L, "pin", s)
        diff = sorted(k for k in a if not torch.equal(a[k], p[k]))
        row["cases"][f"{case}{seed}"] = diff or "equal"
        equal &= not diff
    L.x[0, 0] += 1.0  # negative control on the last case
    n = eager(L, "pin", s)
    row["negative_control_differs"] = not torch.equal(n["pub_logits"], p["pub_logits"]) and not torch.equal(
        n["pub_latent"], p["pub_latent"])
    equal &= row["negative_control_differs"]
    return equal, row


def capture(layers, arm, standin, ns):
    def body():
        for L in layers:
            L.chain(arm, standin, ns)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            body()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        body()
    torch.cuda.synchronize()
    return g


def segments(events):
    """Per stand-in: start / end (us after its end c) of each kernel after it (item 08 e0_rate.segments)."""
    ks = sorted((s, e, k) for s, e, n in events for k in KINDS if k in n)
    stand = [k for k in ks if k[2] == "StandIn"]
    out = []
    for j, st in enumerate(stand):
        hi = stand[j + 1][0] if j + 1 < len(stand) else float("inf")
        seg = [k for k in ks if st[1] <= k[1] and k[0] < hi and k[2] != "StandIn"]
        r = {}
        for k in seg:
            if f"{k[2]}_end" not in r:
                r[f"{k[2]}_end"] = k[1] - st[1]
        if "RouteSitu_end" in r and "WsGemm_end" in r:
            out.append(r)
    return out


def median(v):
    return sorted(v)[len(v) // 2] if v else float("nan")


def graph_and_time(b, dev, standin, a, timed):
    layers = [Layer(l, b, dev) for l in range(3)]
    arms = ("served", "pin", "served2") if timed else ("served", "pin")
    graphs = {arm: capture(layers, arm, standin, a.consumer_ns) for arm in arms}
    snap = {}
    for arm in ("served", "pin"):
        for L in layers:
            L.reset()
        graphs[arm].replay()
        torch.cuda.synchronize()
        snap[arm] = [L.outputs() for L in layers]
    row = {"batch": b, "graph_equal": all(torch.equal(x[k], y[k]) for x, y in zip(snap["served"], snap["pin"])
                                          for k in x)}
    if not timed:
        return row
    seg = {arm: [] for arm in arms}
    per = a.replays // a.blocks
    for blk in range(a.blocks):
        for arm in (arms if blk % 2 == 0 else arms[::-1]):
            for _ in range(3):
                graphs[arm].replay()
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(per):
                    torch.cuda._sleep(40000)
                    graphs[arm].replay()
                torch.cuda.synchronize()
            seg[arm] += segments([(e.time_range.start, e.time_range.end, e.name) for e in prof.events()
                                  if e.device_type.name == "CUDA"])
    mean = lambda v: sum(v) / len(v) if v else float("nan")
    for arm in arms:
        fe, re = [r["WsGemm_end"] for r in seg[arm]], [r["RouteSitu_end"] for r in seg[arm]]
        row[arm] = {"n": len(seg[arm]), "front_end": round(median(fe), 3), "route_end": round(median(re), 3),
                    "front_mean": round(mean(fe), 3), "route_mean": round(mean(re), 3)}
    # The gate uses means: the route end is bimodal (polling), so its p50 can flip between modes; on B200 the
    # served-vs-served p50 differed by 0.19 us at 24 rows in one layout while the means agreed (scratch/pin).
    d = lambda arm, key: round(row["served"][key] - row[arm][key], 3)
    row["gain_route"], row["gain_front"] = d("pin", "route_mean"), d("pin", "front_mean")
    row["aa_route"] = abs(d("served2", "route_mean"))
    row["gain_route_p50"], row["aa_route_p50"] = d("pin", "route_end"), abs(d("served2", "route_end"))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="16,24,32,40")
    ap.add_argument("--time", default="16,24,32,40", help="rows timed (subset of --batches)")
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--replays", type=int, default=360)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--consumer-ns", type=int, default=6000)
    a = ap.parse_args()
    rows = [int(v) for v in a.batches.split(",")]
    timed = {int(v) for v in a.time.split(",")} & set(rows)
    equal, gains, code = False, {}, 2
    t0 = time.time()
    try:
        dev = torch.device("cuda", 0)
        torch.cuda.set_device(dev)
        print(json.dumps({"device": torch.cuda.get_device_name(dev), "rows": rows,
                          "unicast_patch": G.multicast_words is PIN.unicast_words}), flush=True)
        standin = cute.compile(StandIn(), Int32(0), Int64(0), make_fake_stream())
        equal = True
        for b in rows:  # phase 1: bits (eager cases, negative control, graph replay)
            ok, row = check(b, dev, a.draws)
            row.update(graph_and_time(b, dev, standin, a, False))
            equal &= ok and row["graph_equal"]
            row["wall_s"] = round(time.time() - t0, 1)
            print(json.dumps(row), flush=True)
            torch.cuda.empty_cache()
        for b in sorted(timed):  # phase 2: timing, rows in a fixed order after every bit check
            row = graph_and_time(b, dev, standin, a, True)
            equal &= row["graph_equal"]
            gains[b] = (row["gain_route"], row["aa_route"])
            row["wall_s"] = round(time.time() - t0, 1)
            print(json.dumps(row), flush=True)
            torch.cuda.empty_cache()
        line = all(b in gains and gains[b][0] >= 0.3 and gains[b][0] >= 5 * gains[b][1] for b in (24, 40))
        code = 0 if equal and line else (1 if equal else 2)
    except Exception:
        traceback.print_exc()
        code, equal = 2, False
    d = {b: (f"{gains[b][0]:.3f}" if b in gains else "na") for b in (24, 40)}
    verdict = "PASS" if code == 0 else "FAIL"
    print(f"PIN {verdict} rows={','.join(map(str, rows))} equal={equal} d24={d[24]} d40={d[40]} exit={code}",
          flush=True)
    sys.exit(code)


if __name__ == "__main__":
    main()
