"""Item 17 R2: judge and timing of the tcgen05 front (tc_front.TcFront) vs the served WsGemm, served RouteSitu.

judge (coordinator ruling, tolerance not bit-equal), per rows b and draw d:
  truth  y64 = x W^T in FP64. Router logits (rows < 112, FP32 words) and latent (rows 112.., BF16) separately:
         R2 max abs and rms error <= 1.02 x the served kernel's on the same draw.
  ids    equal to the served route's, except rows whose FP64 16th/17th key gap (sigmoid + bias) < 2 x the key
         error bound (0.25 x max router logit error of the two kernels); there both must be a valid top-16.
  wts    |R2 - served| <= 1 BF16 ulp per (row, id) held by both.
  copy   copied latent == R2's published latent (bits); the route on the R2 mailbox with the served latent
         shard gives the served copied latent and plan (bits): copy and plan depend on the latent only.
  cases  rows 9,16,17,24,32,33,40 x draws; "needle" (x row 0 x 64); "tie" (17 experts biased +10, so the
         16th/17th choice is by logits). Mutants that must FAIL: drop_k, skip_n, column, bias (route with
         zero bias), silent (no publish, mailbox not re-armed: the route reads the last draw's values).
time: same bracket as r0 (StandIn -> front -> RouteSitu, 3 layers / graph, ABBA, CUPTI); arms served / r2."""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r0  # noqa: E402  (corun unicast patch, Layer data, capture, segments)
from cutlass import Float32, Int32, Int64  # noqa: E402
import cutlass.cute as cute  # noqa: E402
from cutlass.cute.runtime import make_fake_stream  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

import perfbot_kernels.moe_front as MF  # noqa: E402
from perfbot_kernels.moe_front import tc_front as T  # noqa: E402

H, F, G = r0.H, r0.F, r0.G
N, SHARD, VALID, SENT = r0.N, r0.SHARD, r0.VALID, r0.SENT
r0.KINDS = ("StandIn", "WsGemm", "TcFront", "RouteSitu")
CFG = {"wide": False, "pairs": 2, "spread": False, "kahan": False, "owner64": True, "tree": True, "tile": 64, "ks": 8}  # TcFront variant (--wide, --pairs)


class Layer(r0.Layer):
    """r0.Layer's data (seed ``seed``), kernels: served WsGemm / R2 TcFront (+ mutants), the served route."""

    def __init__(self, seed, b, dev, case="plain", mutants=(), stamped=False):
        self.__dict__.update(_data(seed, b, dev, case))
        self.case = case
        route = F.compile_route(b, 768, True, (True, True, False), dev.index, True, True)
        self.k = {"served": (F.compile_publish(N, N - SHARD, VALID, b, dev.index, F.front_ks(b, N)), route)}
        for m in (None, *mutants):
            self.k["r2" if m is None else m] = (T.compile_tc(b, N - SHARD, VALID, dev.index, mutant=m, **CFG), route)
        if stamped:
            self.k["r2s"] = (T.compile_tc(b, N - SHARD, VALID, dev.index, stamps=True, **CFG), route)
        self.maps = T.front_maps(self.packed, self.x, (b + 7) // 8 * 8, CFG["tile"])

    def front(self, arm, s):
        if arm == "served":
            return super().front(arm, s)
        extra = (Int64(self.stamps.data_ptr() + 8 * MF.STANDIN_NS),) if arm == "r2s" else ()
        self.k[arm][0](H.to_cute(self.maps), H.to_cute(self.box), Int64(self.box.data_ptr()), Int32(0), Int32(0),
                       *extra, s)


def _data(seed, b, dev, case):
    """r0.Layer.__init__'s tensors with seed ``seed`` (no compile); stamps unused (zero length)."""
    g = torch.Generator(device=dev).manual_seed(500 + seed)
    rnd = lambda *s, k=0.02: (torch.randn(*s, device=dev, generator=g) * k).bfloat16()
    d = {"b": b, "dev": dev}
    d["x"], d["packed"] = rnd(b, 7168, k=1.0), rnd(N, 7168)
    if case == "needle":
        d["x"][0] *= 64
    d["bias"] = torch.zeros(896, dtype=torch.float32, device=dev)
    count = 17 if case == "tie" else r0.POOL.get(b, 100)
    pool = torch.randperm(896, generator=torch.Generator().manual_seed(seed))[:count]
    d["bias"][pool.to(dev)] = 10.0
    t = torch.randn(1, 64, F.WIDTH, device=dev, generator=g).bfloat16()
    words = t.view(torch.int32)
    words[..., 1792:] = torch.randn(1, 64, 896, device=dev, generator=g).view(torch.int32)
    words[..., : SHARD // 2] = SENT
    words[..., 1792: 1792 + VALID] = SENT
    d["template"], d["box"] = t, t.clone()
    z = lambda *s, dt: torch.zeros(*s, dtype=dt, device=dev)
    d.update(logits=z(b, 896, dt=torch.float32), ids=z(b, 16, dt=torch.int32), wts=z(b, 16, dt=torch.bfloat16),
             latent=z(b, 3584, dt=torch.bfloat16), gate=z(b, 1536, dt=torch.float32), h=z(b, 768, dt=torch.bfloat16),
             plan=z(F.PLAN_BYTES, dt=torch.uint8), stamps=z(MF.STANDIN_NS + len(T.TC_SLOTS), dt=torch.int64), ws_off=0, rs_off=0)
    return d


def run(L, arm, s, box=None, bias=None):
    """Front (unless box is given: the route starts from that mailbox), then route. Returns outputs."""
    L.reset()
    if box is not None:
        L.box.copy_(box)
    if arm is not None:
        L.front(arm, s)
    torch.cuda.synchronize()
    published = L.box.clone()
    w = published.view(torch.int32)[0, : L.b]
    left = int((w[:, 1792: 1792 + VALID] == SENT).sum() + (w[:, : SHARD // 2] == SENT).sum())
    out = {"box": published, "sentinel_left": left}
    if left:
        return out  # the route would poll forever: the judge fails on the publish
    saved = L.bias.clone()
    if bias is not None:
        L.bias.copy_(bias)
    L.route("served", s)
    torch.cuda.synchronize()
    L.bias.copy_(saved)
    out.update(ids=L.ids.clone(), wts=L.wts.clone(), latent=L.latent.clone(), plan=L.plan.clone())
    return out


def errors(L, box, y64):
    b = L.b
    logit = box.view(torch.int32)[0, :b, 1792: 1792 + VALID].view(torch.float32).double()
    latent = box[0, :b, :SHARD].double()
    out = {}
    for name, got, ref in (("router", logit, y64[:, :VALID]), ("latent", latent, y64[:, VALID:])):
        e = (got - ref).abs()
        out[name] = {"max": float(e.max()), "rms": float(e.pow(2).mean().sqrt())}
    return out


def keys64(L, y64):
    logit = L.template.view(torch.int32)[0, : L.b, 1792:].view(torch.float32).double().clone()
    logit[:, :VALID] = y64[:, :VALID]
    return torch.sigmoid(logit) + L.bias.double()


def judge(L, s, mutant=None, prev_box=None):
    """One (rows, draw): served vs R2 (or a mutant). Returns the row and a pass flag."""
    y64 = L.x.double() @ L.packed.double().T
    served = run(L, "served", s)
    if mutant == "bias":
        cand = run(L, "r2", s, bias=torch.zeros_like(L.bias))
    elif mutant == "silent":
        cand = run(L, "silent", s, box=prev_box)
    else:
        cand = run(L, mutant or "r2", s)
    row = {"batch": L.b, "case": L.case, "mutant": mutant}
    if cand["sentinel_left"]:
        row.update(ok=False, why=f"{cand['sentinel_left']} sentinel words left")
        return row, cand["box"]
    es, ec = errors(L, served["box"], y64), errors(L, cand["box"], y64)
    ratio = {f"{k}_{m}": ec[k][m] / max(es[k][m], 1e-30) for k in es for m in ("max", "rms")}
    row["err_served"], row["err_r2"], row["ratio"] = es, ec, {k: round(v, 4) for k, v in ratio.items()}
    ok_err = all(v <= 1.02 for v in ratio.values())
    # ids
    keys = keys64(L, y64)
    bound = 2 * 0.25 * max(es["router"]["max"], ec["router"]["max"])
    top = keys.sort(dim=1, descending=True).values
    gap = top[:, 15] - top[:, 16]
    ids_ok, exempt, mism = True, 0, 0
    wts_ok, worst_ulp = True, 0.0
    for t in range(L.b):
        a, c_ = served["ids"][t].tolist(), cand["ids"][t].tolist()
        if a != c_:
            mism += 1
            if gap[t] < bound:
                exempt += 1
                for chosen in (a, c_):  # a valid top-16: none outside beats one inside by more than the bound
                    inside = keys[t, chosen].min()
                    mask = torch.ones(896, dtype=torch.bool, device=keys.device)
                    mask[chosen] = False
                    if keys[t][mask].max() > inside + bound or len(set(chosen)) != 16:
                        ids_ok = False
            else:
                ids_ok = False
        ws = dict(zip(a, served["wts"][t].float().tolist()))
        for i, w in zip(c_, cand["wts"][t].float().tolist()):
            if i in ws:
                ulp = torch.finfo(torch.bfloat16).eps * 2.0 ** torch.floor(torch.log2(torch.tensor(abs(ws[i]) + 1e-30)))
                d = abs(w - ws[i]) / float(ulp)
                worst_ulp = max(worst_ulp, d)
                wts_ok &= d <= 1.0
    row.update(ids_mismatch_rows=mism, ids_exempt_rows=exempt, ids_ok=ids_ok, min_gap=float(gap.min()),
               key_bound=bound, wts_ok=wts_ok, wts_worst_ulp=round(worst_ulp, 3))
    # copy and plan
    copy_ok = torch.equal(cand["latent"][:, :SHARD].view(torch.int16), cand["box"][0, : L.b, :SHARD].view(torch.int16))
    mixed = cand["box"].clone()
    mixed[0, : L.b, :SHARD] = served["box"][0, : L.b, :SHARD]
    again = run(L, None, s, box=mixed)
    plan_ok = torch.equal(again["latent"].view(torch.int16), served["latent"].view(torch.int16)) and torch.equal(
        again["plan"], served["plan"])
    row.update(copy_ok=bool(copy_ok), plan_ok=bool(plan_ok))
    row["ok"] = bool(ok_err and ids_ok and wts_ok and copy_ok and plan_ok)
    return row, cand["box"]


def main_judge(a, dev, s):
    mutants = ("drop_k", "skip_n", "column", "silent")
    for b in map(int, a.batches.split(",")):
        prev = None
        for d in range(a.draws):
            L = Layer(1000 * d + b, b, dev, mutants=mutants if d == 1 else ())
            row, box = judge(L, s)
            print(json.dumps({"draw": d, **row}), flush=True)
            if d == 1:  # mutants on draw 1; "silent" reads draw 0's R2 mailbox
                for m in (*mutants, "bias"):
                    r, _ = judge(L, s, m, prev_box=prev)
                    print(json.dumps({"draw": d, **{k: v for k, v in r.items() if k in
                                      ("batch", "mutant", "ok", "why", "ratio", "ids_ok", "ids_mismatch_rows")}}),
                          flush=True)
            prev = box
        for case in ("needle", "tie"):
            for d in range(2):
                L = Layer(7000 + 100 * d + b, b, dev, case=case)
                row, _ = judge(L, s)
                print(json.dumps({"draw": d, **row}), flush=True)


def main_time(a, dev):
    standin = cute.compile(MF.StandIn(), Int32(0), Int64(0), make_fake_stream())
    for b in map(int, a.batches.split(",")):
        layers = [Layer(l, b, dev) for l in range(a.layers)]
        arms = ("served", "r2")
        graphs = {arm: r0.capture(layers, arm, standin, a.consumer_ns) for arm in arms}
        cupti = {arm: [] for arm in arms}
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
                cupti[arm] += r0.segments([(e.time_range.start, e.time_range.end, e.name) for e in prof.events()
                                           if e.device_type.name == "CUDA"])
        row = {"batch": b, "gpu": r0.gpu_state(), "cupti": {}}
        for arm, v in cupti.items():
            front = [r.get("WsGemm_end", r.get("TcFront_end")) for r in v]
            row["cupti"][arm] = {"n": len(v), "front_end": [round(r0.q(front, f), 3) for f in (0.1, 0.5, 0.9)],
                                 "route_end": [round(r0.q([r["RouteSitu_end"] for r in v], f), 3)
                                               for f in (0.1, 0.5, 0.9)]}
        row["stamps_r2"] = stamped(layers, standin, a, b, dev)
        print(json.dumps(row), flush=True)
        torch.cuda.empty_cache()


def stamped(layers, standin, a, b, dev):
    """A separate pass of the stamped R2 (own graph, after the timed arms): per slot, max over CTAs - c (us),
    p10 / p50 / p90 over replays x layers. c = the stand-in's exit globaltimer read (max over its CTAs)."""
    for L in layers:
        L.k["r2s"] = (T.compile_tc(b, N - SHARD, VALID, dev.index, stamps=True, **CFG), L.k["r2"][1])
    g = r0.capture(layers, "r2s", standin, a.consumer_ns)
    rows = []
    for _ in range(a.replays // 2):
        for L in layers:
            L.stamps.zero_()
        torch.cuda._sleep(40000)
        g.replay()
        torch.cuda.synchronize()
        rows += [L.stamps.cpu().clone() for L in layers]
    out = {}
    for i, name in enumerate(("c", *T.TC_SLOTS)):
        v = []
        for r in rows:
            c0 = int(r[2:12:4].max())
            t = c0 if i == 0 else int(r[MF.STANDIN_NS + i - 1])
            if t:
                v.append((t - c0) / 1000.0)
        out[name] = [round(r0.q(v, f), 3) for f in (0.1, 0.5, 0.9)] if v else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="judge", choices=("judge", "time", "smoke"))
    ap.add_argument("--batches", default="9,16,17,24,32,33,40")
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--replays", type=int, default=200)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--consumer-ns", type=int, default=6000)
    ap.add_argument("--wide", type=int, default=0)
    ap.add_argument("--pairs", type=int, default=2)
    ap.add_argument("--spread", type=int, default=0)
    ap.add_argument("--kahan", type=int, default=0)
    ap.add_argument("--owner64", type=int, default=1)
    ap.add_argument("--tree", type=int, default=1)
    ap.add_argument("--tile", type=int, default=64)
    ap.add_argument("--ks", type=int, default=8)
    a = ap.parse_args()
    CFG.update(wide=bool(a.wide), pairs=a.pairs, spread=bool(a.spread), kahan=bool(a.kahan), owner64=bool(a.owner64), tree=bool(a.tree), tile=a.tile, ks=a.ks)
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    s = H.stream(dev)
    print(json.dumps({"device": torch.cuda.get_device_name(dev), "provenance": r0.provenance(), "cfg": CFG}),
          flush=True)
    if a.mode == "smoke":
        for b in map(int, a.batches.split(",")):
            row, _ = judge(Layer(1, b, dev), s)
            print(json.dumps(row), flush=True)
    elif a.mode == "judge":
        main_judge(a, dev, s)
    else:
        main_time(a, dev)


if __name__ == "__main__":
    main()
