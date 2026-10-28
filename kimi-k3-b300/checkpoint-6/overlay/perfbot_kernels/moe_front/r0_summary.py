"""Item 17 R0: fold r0.py --mode time repeats (time_r*.jsonl) into STAMPS.md tables + stamps_summary.json.

Every number is the median over repeats of a per-repeat p50; `spread` = max - min over repeats.
    python3 r0_summary.py <dir with time_r*.jsonl and a0 baseline dir>"""
import glob
import json
import statistics
import sys

D = sys.argv[1]
A0 = sys.argv[2] if len(sys.argv) > 2 else None
ROWS = (16, 24, 32, 40)


def load(pattern):
    reps = []
    for f in sorted(glob.glob(pattern)):
        rows = {}
        for line in open(f):
            d = json.loads(line)
            if "batch" in d:
                rows[d["batch"]] = d
        reps.append(rows)
    return reps


def med(vals):
    vals = [v for v in vals if v is not None]
    return (round(statistics.median(vals), 2), round(max(vals) - min(vals), 2)) if vals else (None, None)


def get(d, *path):
    for p in path:
        if d is None:
            return None
        if isinstance(p, int):
            d = d[p] if isinstance(d, list) and p < len(d) else None
        else:
            d = d.get(p) if isinstance(d, dict) else None
    return d


reps = load(f"{D}/time_r*.jsonl")
base = load(f"{A0}/rate_r*.jsonl") if A0 else []
out = {"repeats": len(reps), "rows": {}}
md = []
md.append("## Overhead (CUPTI, us after c, p50; median of repeats [spread])\n")
md.append("| rows | route end - c: served / on / off | on - served (same process) | a0 baseline (e0_rate chain) | "
          "on - a0 | WsGemm end - c: served / on | gate 0.15 |")
md.append("|---|---|---|---|---|---|---|")
for b in ROWS:
    o = out["rows"].setdefault(b, {})
    r = {arm: med([get(x.get(b), "cupti", arm, "RouteSitu_end", 1) for x in reps]) for arm in ("served", "on", "off")}
    w = {arm: med([get(x.get(b), "cupti", arm, "WsGemm_end", 1) for x in reps]) for arm in ("served", "on")}
    d = med([get(x.get(b), "cupti", "on", "RouteSitu_end", 1) - get(x.get(b), "cupti", "served", "RouteSitu_end", 1)
             for x in reps if x.get(b)])
    a0 = med([get(x.get(b), "chain", "RouteSitu_end", 1) for x in base]) if base else (None, None)
    don = round(r["on"][0] - a0[0], 2) if a0[0] is not None else None
    ok = abs(d[0]) <= 0.15 and (don is None or abs(don) <= 0.15)
    o["overhead"] = {"route_end": r, "wsgemm_end": w, "on_minus_served": d, "a0": a0, "on_minus_a0": don, "pass": ok}
    md.append(f"| {b} | {r['served'][0]} / {r['on'][0]} / {r['off'][0]} [{r['on'][1]}] | {d[0]} [{d[1]}] | {a0[0]} | "
              f"{don} | {w['served'][0]} / {w['on'][0]} | {'pass' if ok else 'FAIL'} |")


def table(title, key, dkey, order_key=None, default_order=None):
    md.append(f"\n## {title}\n")
    md.append("Start = event time after c, p50 over CTAs x samples (last CTA: p50 over samples of the slowest CTA). "
              "Dur = per CTA, event minus the previous event, p50.\n")
    md.append("| event | " + " | ".join(f"{b}: start / last CTA / dur" for b in ROWS) + " |")
    md.append("|---|" + "---|" * len(ROWS))
    names = []
    for b in ROWS:
        for x in reps:
            if x.get(b):
                for n in (x[b]["stamps"][order_key] if order_key else default_order):
                    if n not in names:
                        names.append(n)
    for n in names:
        cells = []
        for b in ROWS:
            s = med([get(x.get(b), "stamps", key, n, "p50") for x in reps])
            last = med([get(x.get(b), "stamps", key, n, "last_cta_p50") for x in reps])
            du = med([get(x.get(b), "stamps", dkey, n) for x in reps])
            out["rows"][b].setdefault(key, {})[n] = {"start": s, "last": last, "dur": du}
            cells.append("-" if s[0] is None else f"{s[0]} / {last[0]} / {du[0] if du[0] is not None else '-'}")
        md.append(f"| {n} | " + " | ".join(cells) + " |")


table("WsGemm (112 CTAs)", "ws", "ws_dur_p50", order_key="ws_order")
table("RouteSitu route role (CTAs 0..r-1)", "route", "route_dur_p50",
      default_order=["entry", "poll", "keys", "cutoff", "top16", "out", "release", "exit"])
table("RouteSitu copy role (CTAs r..2r-1)", "copy", "copy_dur_p50",
      default_order=["entry", "poll", "out", "release", "exit"])
md.append("\n## Stamp vs CUPTI (us after c, p50, on arm)\n")
md.append("| rows | WsGemm last-CTA exit (stamp) / CUPTI end | route last-CTA exit (stamp) / CUPTI end | samples per repeat |")
md.append("|---|---|---|---|")
for b in ROWS:
    ws = med([get(x.get(b), "stamps", "ws", "exit", "last_cta_p50") for x in reps])
    rt = med([max(get(x.get(b), "stamps", "route", "exit", "last_cta_p50"),
                  get(x.get(b), "stamps", "copy", "exit", "last_cta_p50")) for x in reps if x.get(b)])
    n = [get(x.get(b), "stamps", "samples") for x in reps]
    out["rows"][b]["stamp_vs_cupti"] = {"ws_exit": ws, "route_exit": rt}
    md.append(f"| {b} | {ws[0]} / {out['rows'][b]['overhead']['wsgemm_end']['on'][0]} | {rt[0]} / "
              f"{out['rows'][b]['overhead']['route_end']['on'][0]} | {n} |")
gb = [get(x.get(b), "graph_bit_equal_on_vs_served") for x in reps for b in ROWS]
md.append(f"\nGraph path, every repeat and row: stamps-on replay outputs bit-equal to the served replay: {all(gb)} ({len(gb)} cases).")
out["graph_bit_equal"] = all(gb)
json.dump(out, open(f"{D}/stamps_summary.json", "w"), indent=1)
open(f"{D}/stamps_tables.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))
