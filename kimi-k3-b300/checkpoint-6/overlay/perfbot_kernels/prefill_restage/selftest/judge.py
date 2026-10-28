# PERFBOT P-07F: judge of the form R equivalence check (CPU; reads the recorder's JSON lines and the arms' server logs).
"""Compare the off and on arms of the frozen prefill case. Verdict: one value of PASS | FAIL | INCOMPLETE.

  python3 -B judge.py --off off.jsonl --on on.jsonl --on-log on/server.log --off-log off/server.log \
      [--off2 off2.jsonl] [--out verdict.json]

Class: same kernels (form R changes which host tokens are copied into all_token_ids, no kernel), so the bar is bit-equal.
  G1 bits: on every request of the case (plain and penalty chains), the greedy ids and the first token's top-k logprobs
     are equal off vs on. If --off2 differs from off on a request, the engine is not deterministic at conc 1 there and
     that request cannot decide: G1 is INCOMPLETE (not FAIL) unless another request fails.
  G2 fired: the on log carries PERFBOT-P07F lines, the largest rows_suffix >= 1 and tokens_skipped >= the smallest
     warm plain-turn hit column s of the case; the off log carries none (switch off = 49cb2bbf staging).
  G3 fallback: the penalty chain's requests are in G1 (same bits), and the on log's largest rows_full >= the case's
     whole rows (cold primes + penalty turns): the penalty rows were staged whole.
Reported, not gated: exact row counts vs make_case.expected_rows (prewarm requests and cache misses move them);
per-request hits off vs on; wall_s off vs on (form R removes 0.6 + 0.037 ms per 1k ctx of host time per new request [e]).
"""
import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_case  # noqa: E402

LINE = re.compile(r"PERFBOT-P07F rows_suffix=(\d+) rows_full=(\d+) tokens_skipped=(\d+)")


def load(path):
    return {(r["chain"], r["turn"]): r for r in map(json.loads, Path(path).read_text().splitlines())}


def counts(log_text):
    rows = [tuple(map(int, m.groups())) for m in LINE.finditer(log_text)]
    return [max(r[i] for r in rows) for i in range(3)] if rows else None


def judge(case, off, on, on_log, off_log, off2=None):
    keys = [(ch["chain"], t["turn"]) for ch in case["chains"] for t in ch["turns"]]
    reasons, notes = [], []
    missing = [k for k in keys if k not in off or k not in on]
    unstable = [k for k in keys if off2 is not None and k in off and k in off2
                and (off[k]["ids"], off[k]["top0"]) != (off2[k]["ids"], off2[k]["top0"])]
    diff = [k for k in keys if k in off and k in on and k not in unstable
            and (off[k]["ids"], off[k]["top0"]) != (on[k]["ids"], on[k]["top0"])]
    g1 = "FAIL" if diff else ("INCOMPLETE" if missing or unstable else "PASS")
    if diff:
        reasons.append(f"G1 bits differ off vs on on {diff}")
    if missing:
        reasons.append(f"G1 requests missing: {missing}")
    if unstable:
        reasons.append(f"G1 off vs off2 differ (not deterministic) on {unstable}")

    exp = make_case.expected_rows(case)
    c_on, c_off = counts(on_log), counts(off_log)
    min_s = min(t["s"] for ch in case["chains"] if ch["kind"] == "plain" for t in ch["turns"] if t["warm"])
    if c_off is not None:
        g2 = "FAIL"
        reasons.append("G2 the off log carries PERFBOT-P07F lines (switch leaked into the off arm)")
    elif c_on is None:
        g2 = "FAIL"
        reasons.append("G2 no PERFBOT-P07F line in the on log (form R never ran)")
    elif c_on[0] < 1 or c_on[2] < min_s:
        g2 = "FAIL"
        reasons.append(f"G2 on log rows_suffix {c_on[0]} / tokens_skipped {c_on[2]}: no warm turn staged suffix-only (min s {min_s})")
    else:
        g2 = "PASS"
    g3 = "PASS" if c_on is not None and c_on[1] >= exp["rows_full"] else "FAIL"
    if g3 == "FAIL":
        reasons.append(f"G3 on log rows_full {c_on and c_on[1]} < case whole rows {exp['rows_full']}")
    if c_on is not None and tuple(c_on) != (exp["rows_suffix"], exp["rows_full"], exp["tokens_skipped"]):
        notes.append(f"row counts {c_on} vs case {exp} (prewarm requests or cache misses; reported only)")
    hit_diff = [k for k in keys if k in off and k in on and off[k].get("hit") != on[k].get("hit")]
    if hit_diff:
        notes.append(f"prefix hits differ off vs on on {hit_diff}")
    walls = [(k, round(off[k]["wall_s"] - on[k]["wall_s"], 4)) for k in keys if k in off and k in on]
    gates = dict(G1=g1, G2=g2, G3=g3)
    verdict = "FAIL" if "FAIL" in gates.values() else ("INCOMPLETE" if "INCOMPLETE" in gates.values() else "PASS")
    return dict(verdict=verdict, gates=gates, reasons=reasons, notes=notes, counts_on=c_on, expected=exp,
                wall_off_minus_on_s=walls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--off", required=True)
    ap.add_argument("--on", required=True)
    ap.add_argument("--on-log", required=True)
    ap.add_argument("--off-log", required=True)
    ap.add_argument("--off2")
    ap.add_argument("--case", default=str(HERE / "frozen_case.json"))
    ap.add_argument("--out")
    a = ap.parse_args()
    case = json.loads(Path(a.case).read_text())
    v = judge(case, load(a.off), load(a.on), Path(a.on_log).read_text(errors="replace"),
              Path(a.off_log).read_text(errors="replace"), load(a.off2) if a.off2 else None)
    text = json.dumps(v, indent=1)
    if a.out:
        Path(a.out).write_text(text + "\n")
    print(text)
    raise SystemExit({"PASS": 0, "FAIL": 1, "INCOMPLETE": 2}[v["verdict"]])


if __name__ == "__main__":
    main()
