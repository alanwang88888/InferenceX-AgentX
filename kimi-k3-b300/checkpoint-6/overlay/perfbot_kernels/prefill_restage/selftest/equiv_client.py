# PERFBOT P-07F: GPU-side recorder of the frozen prefill case (runs against one served arm; never run by the CPU tests).
"""Send frozen_case.json to one vLLM server, serially (conc 1), and write one JSON line per request.

  python3 -B equiv_client.py --url http://127.0.0.1:8000 --model <served name> --out <arm>.jsonl \
      [--metrics http://127.0.0.1:8000/metrics ...]

Arm order: one server per arm (the switch is read at import: restart per arm). off = arm env without the switch,
on = PERFBOT_EXP_PREFILL_GRAPH=1; optional off2 = a second off server (determinism check). eval_only mode in every arm
(the throughput mode's synthetic rejection sampler makes the token stream not the model's); same seat, same image.
Request: /v1/completions, prompt = token ids, temperature 0, max_tokens and logprobs from the case, ignore_eos,
return_tokens_as_token_ids, plus the chain's extras (penalty chain: repetition_penalty). Hits: the
vllm:prefix_cache_hits_total delta around the request (conc 1, so the delta is this request's), summed over --metrics.
ttft_s: time to the first streamed chunk is not taken (non-streaming); wall_s is reported as a diagnostic only.
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

import make_case

HITS = "vllm:prefix_cache_hits_total"


def post(url, body):
    req = urllib.request.Request(url + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.loads(r.read())


def hits(urls):
    tot = 0.0
    for u in urls:
        for line in urllib.request.urlopen(u, timeout=60).read().decode().splitlines():
            if line.split("{", 1)[0].split(" ", 1)[0] == HITS:
                tot += float(line.rsplit(" ", 1)[1])
    return tot


def tid(tok):
    return int(tok.split(":", 1)[1])  # return_tokens_as_token_ids: "token_id:<n>"


def one(a, case, ids, extra):
    body = dict(model=a.model, prompt=ids, max_tokens=case["max_tokens"], temperature=0, logprobs=case["logprobs"],
                ignore_eos=True, return_tokens_as_token_ids=True, **extra)
    h0, t0 = hits(a.metrics), time.time()
    r = post(a.url, body)
    lp = r["choices"][0]["logprobs"]
    return dict(ids=[tid(t) for t in lp["tokens"]], top0={str(tid(k)): v for k, v in lp["top_logprobs"][0].items()},
                hit=hits(a.metrics) - h0 if a.metrics else None, wall_s=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--metrics", nargs="*", default=[])
    ap.add_argument("--case", default=str(Path(__file__).with_name("frozen_case.json")))
    a = ap.parse_args()
    case = json.loads(Path(a.case).read_text())
    with open(a.out, "w") as f:
        for ch in case["chains"]:
            stream = make_case.token_ids(case, ch["chain"], ch["turns"][-1]["P"])
            for t in ch["turns"]:
                rec = dict(chain=ch["chain"], turn=t["turn"], kind=ch["kind"], warm=t["warm"],
                           **one(a, case, stream[:t["P"]], ch["extra"]))
                f.write(json.dumps(rec) + "\n")
                f.flush()


if __name__ == "__main__":
    main()
