# PERFBOT P-07F: frozen prefill case of the form R equivalence check (CPU only, no vLLM import).
"""Build frozen_case.json: conversation chains in the P-07 bucket (re-prefill turns of 1,024-4,095 new tokens over a
cached ctx < 100k), sent serially to one server per arm.

  python3 -B make_case.py [--out frozen_case.json] [--check]

A turn's prompt is FROZEN: prompt_k = prompt_{k-1} + ASSIST frozen "assistant" ids + u_k new "user" ids, so the off and
on arms send byte-identical requests. Token ids come from random.Random(f"{SEED}:{chain}") over [ID_LO, ID_HI).
Turn 0 of a chain is cold (num_computed 0: form R stages the whole row, as stock). Every later turn hits the previous
turn's tail checkpoint (K3 hit rule [e], P-07 make_case: cp(P) = P//128*128 - 128), so num_computed > 0 and form R
stages the suffix only, unless the chain uses a history reader:
  kind "plain"   : greedy, no penalty -> suffix-only rows (the lever fires)
  kind "penalty" : repetition_penalty 1.05 -> the sampler's penalties state reads the prompt, the row stays whole
                   (the fallback the ruling asks to prove)
Request: temperature 0, MAX_TOKENS greedy ids + first-token top-LOGPROBS logprobs, ignore_eos.
Bucket medians (P-07 §2, 45 turns): n 2,063, ctx 64,128; frozen turn n 2,103 / ctx 80,000.
"""
import argparse
import json
import random
from pathlib import Path

SEED = "20261019:p07f"
ASSIST = 256
ID_LO, ID_HI = 1000, 100000  # ordinary-token id range (as P-07 form K's case) [u: K3 special ids not read]
MAX_TOKENS, LOGPROBS = 32, 20
CHAINS = [  # (kind, cold prime length, new user tokens per warm turn, sampling extras)
    ("plain", 60_032, [1_024, 2_063, 2_103, 2_600, 3_300, 3_500], {}),
    ("plain", 80_000, [1_100, 2_063, 3_000, 1_500], {}),
    ("penalty", 50_048, [2_063, 3_000], {"repetition_penalty": 1.05}),
]


def build():
    chains = []
    for c, (kind, prime, news, extra) in enumerate(CHAINS):
        turns, P = [dict(turn=0, s=0, P=prime, new=prime, warm=False)], prime
        for k, u in enumerate(news, start=1):
            s = P // 128 * 128 - 128  # hit at the previous turn's tail checkpoint [e]
            P = P + ASSIST + u
            turns.append(dict(turn=k, s=s, P=P, new=P - s, warm=True))
        assert all(t["P"] < 100_000 for t in turns), "bucket: ctx < 100k"
        chains.append(dict(chain=c, kind=kind, extra=extra, turns=turns))
    return dict(version=1, seed=SEED, assist=ASSIST, id_lo=ID_LO, id_hi=ID_HI, max_tokens=MAX_TOKENS,
                logprobs=LOGPROBS, chains=chains)


def token_ids(case, chain, n):
    rng = random.Random(f"{case['seed']}:{chain}")
    return [rng.randrange(case["id_lo"], case["id_hi"]) for _ in range(n)]


def expected_rows(case):
    """Rows form R should stage suffix-only / whole, if every warm turn hits the cache (the judge reports, not gates, equality)."""
    suffix = sum(t["warm"] for ch in case["chains"] if ch["kind"] == "plain" for t in ch["turns"])
    whole = sum(len(ch["turns"]) for ch in case["chains"]) - suffix
    skipped = sum(t["s"] for ch in case["chains"] if ch["kind"] == "plain" for t in ch["turns"])
    return dict(rows_suffix=suffix, rows_full=whole, tokens_skipped=skipped)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).with_name("frozen_case.json")))
    ap.add_argument("--check", action="store_true", help="exit 1 unless --out equals a fresh build")
    a = ap.parse_args()
    text = json.dumps(build(), indent=1) + "\n"
    if a.check:
        raise SystemExit(0 if Path(a.out).read_text() == text else 1)
    Path(a.out).write_text(text)


if __name__ == "__main__":
    main()
