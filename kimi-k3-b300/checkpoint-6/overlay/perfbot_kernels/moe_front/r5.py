"""Item 17 R5: where does WsGemm's wake after c (c -> griddepcontrol_wait return, 1.0-1.16 us in R0) go?

Same bracket as r0.py (stand-in -> kernel(s), 3 layers per graph, ABBA, CUPTI + stamps). Arms:
  served   the served WsGemm -> RouteSitu (no stamps; CUPTI anchor)
  pin      fork WsGemm, no stamps, prologue drained + pinned above the wait (ready=True) -> served RouteSitu
  on       fork WsGemm (stamps) -> RouteSitu (stamps): R0's chain, plus the "pre" stamp before the wait
  ready    as on, but the WsGemm prologue is drained before "pre" (cp.async wait + register weights consumed)
  floor16  WakeFloor: WsGemm's launch shape (112 CTAs x 320, cluster 16) doing only stamps + griddepcontrol_wait
  floor1   WakeFloor without a cluster (112 CTAs x 320)
Per sample: c0 = stand-in thread 0 exit read (R0's c), clast = last warp's last instruction of the stand-in.
Per CTA (us after c0): pre and wait; min / p50 / max over CTAs, then p50 over samples. CUPTI: kernel start and end
after the CUPTI stand-in end; first-CTA entry stamp vs CUPTI start = the offset of c0 against the CUPTI c."""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r0  # noqa: E402
from cutlass import Int32, Int64  # noqa: E402
import cutlass.cute as cute  # noqa: E402
from cutlass.cute.runtime import make_fake_stream  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

MF, H, F = r0.MF, r0.H, r0.F
r0.KINDS = ("StandIn", "WsGemm", "RouteSitu", "WakeFloor")
FLOOR = {}


def chain(L, arm, standin, ns):
    s = H.stream(L.dev)
    L.box.copy_(L.template)
    standin(Int32(ns), Int64(L.stamps.data_ptr()), s)
    if arm.startswith("floor"):
        FLOOR[arm](Int64(L.stamps.data_ptr() + 8 * L.ws_off), s)
        return
    if arm in ("served", "pin"):
        L.front(arm, s)
        L.route(arm, s)
        return
    if arm.startswith("S_"):  # bisect: WsGemm with a stamp subset, served RouteSitu
        L.k[arm][0](H.to_cute(L.x), H.to_cute(L.packed), H.to_cute(L.box), Int64(L.box.data_ptr()), Int32(0),
                    Int32(0), H.to_cute(L.logits), Int64(L.stamps.data_ptr() + 8 * L.ws_off), s)
        L.route("served", s)
        return
    k = L.k[arm]
    sb = L.stamps.data_ptr()
    k[0](H.to_cute(L.x), H.to_cute(L.packed), H.to_cute(L.box), Int64(L.box.data_ptr()), Int32(0), Int32(0),
         H.to_cute(L.logits), Int64(sb + 8 * L.ws_off), s)
    args = (L.logits, L.bias, L.ids, L.wts, L.box, L.latent, L.gate, L.h)
    k[1](*map(H.to_cute, args), r0.Float32(2.5), r0.Float32(4.0), r0.Float32(25.0), H.to_cute(L.plan),
         Int64(sb + 8 * L.rs_off), s)


def analyse(rec, L, arm):
    r = rec.reshape(-1, rec.shape[-1]).numpy()
    st = r[:, :12].reshape(len(r), 3, 4)
    c0 = st[:, :, 2].max(axis=1)
    clast = r[:, 12:15].max(axis=1)
    f = np.median((st[:, :, 3] - st[:, :, 1]) / (st[:, :, 2] - st[:, :, 0]), axis=1)
    if arm.startswith("floor"):
        ns, names, edge = MF.WF_NS, MF.WF_SLOTS, len(MF.WF_SLOTS)
    else:
        ns, names, edge = MF.WS_NS, MF.WS_SLOTS, MF.WS_EDGE
    a = r[:, L.ws_off: L.ws_off + 112 * ns].reshape(len(r), 112, ns)
    ge, ce = a[:, :, edge], a[:, :, edge + 1]
    out = {"samples": len(r), "clast_minus_c0_us": round(float(np.median((clast - c0) / 1000.0)), 3),
           "clk_per_ns": round(float(np.median(f)), 4)}
    for n in ("entry", "pre", "wait"):
        i = names.index(n)
        v = a[:, :, i]
        t = np.where((v > 0) & (ge > 0), ((ge - c0[:, None]).astype(np.float64) + (v - ce) / f[:, None]) / 1000.0,
                     np.nan)
        out[n] = {q: round(float(np.nanmedian(fn(t, axis=1))), 3) for q, fn in
                  (("min", np.nanmin), ("p50", np.nanmedian), ("max", np.nanmax))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="16,40")
    ap.add_argument("--arms", default="served,pin,on,ready,floor16,floor1")
    ap.add_argument("--replays", type=int, default=200)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--blocks", type=int, default=4)
    a = ap.parse_args()
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    standin = cute.compile(MF.StandIn(), Int32(0), Int64(0), make_fake_stream())
    FLOOR["floor16"] = cute.compile(MF.WakeFloor(cluster=16), Int64(0), make_fake_stream())
    FLOOR["floor1"] = cute.compile(MF.WakeFloor(cluster=1), Int64(0), make_fake_stream())
    r0.Layer.chain = lambda L, arm, standin, ns: chain(L, arm, standin, ns)
    print(json.dumps({"gpu": r0.gpu_state()}), flush=True)
    for b in map(int, a.batches.split(",")):
        layers = [r0.Layer(l, b, dev) for l in range(a.layers)]
        for L in layers:
            ks = F.front_ks(b, r0.N)
            L.k["ready"] = (MF.compile_publish(r0.N, r0.N - r0.SHARD, r0.VALID, b, dev.index, ks, stamps=True,
                                               ready=True), L.k["on"][1])
            for arm in a.arms.split(","):
                if arm.startswith("S_"):
                    only = None if arm == "S_all" else tuple(arm[2:].split("+"))
                    L.k[arm] = (MF.compile_publish(r0.N, r0.N - r0.SHARD, r0.VALID, b, dev.index, ks, stamps=True,
                                                   only=only), None)
        arms = a.arms.split(",")
        graphs = {arm: r0.capture(layers, arm, standin, 6000) for arm in arms}
        per = a.replays // a.blocks
        n = layers[0].stamps.numel()
        rec = {arm: torch.zeros(a.replays, a.layers, n, dtype=torch.int64, device=dev) for arm in arms}
        cnt = {arm: 0 for arm in arms}
        cupti = {arm: [] for arm in arms}
        for blk in range(a.blocks):
            for arm in (arms if blk % 2 == 0 else arms[::-1]):
                g = graphs[arm]
                for _ in range(3):
                    g.replay()
                torch.cuda.synchronize()
                for L in layers:
                    L.stamps.zero_()
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    for _ in range(per):
                        torch.cuda._sleep(40000)
                        g.replay()
                        for li, L in enumerate(layers):
                            rec[arm][cnt[arm], li].copy_(L.stamps)
                            L.stamps.zero_()
                        cnt[arm] += 1
                    torch.cuda.synchronize()
                cupti[arm] += r0.segments([(e.time_range.start, e.time_range.end, e.name) for e in prof.events()
                                           if e.device_type.name == "CUDA"])
        row = {"batch": b}
        for arm in arms:
            v = cupti[arm]
            keys = sorted({k for x in v for k in x})
            row[arm] = {"cupti": {k: round(r0.q([x[k] for x in v if k in x], 0.5), 3) for k in keys},
                        "stamps": None if arm in ("served", "pin") or arm.startswith("S_") else analyse(rec[arm][: cnt[arm]].cpu(), layers[0], arm)}
        print(json.dumps(row), flush=True)
        del layers, graphs
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
