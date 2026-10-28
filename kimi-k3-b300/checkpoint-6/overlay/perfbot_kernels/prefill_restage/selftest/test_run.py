# PERFBOT P-07F: CPU test of judge/run.py in the driver's form (stub server; stdlib only).
import http.server
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

RUN = Path(__file__).resolve().parents[3] / "judge" / "run.py"


def stub(bump=None, hit_scale=1.0, short=False):
    """A served API: /v1/models, /v1/completions (ids from the prompt length), /metrics (hits = cached prefix)."""
    state = dict(hits=0.0)

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def reply(self, body, ctype="application/json"):
            data = body.encode()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/v1/models":
                self.reply(json.dumps(dict(data=[dict(id="moonshotai/Kimi-K3")])))
            else:
                self.reply(f"# HELP x\nvllm:prefix_cache_hits_total{{engine=\"0\"}} {state['hits']}\n", "text/plain")

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            P = len(body["prompt"])
            state["hits"] += (P // 128 * 128 - 128 - 300) * hit_scale if P not in (60032, 80000, 50048) else 0
            n = body["max_tokens"] - (1 if short else 0)
            ids = [(P + i) % 1000 for i in range(n)]
            top = {f"token_id:{P % 97}": -0.25 if P != bump else -0.2500001, "token_id:9": -1.5}
            self.reply(json.dumps(dict(choices=[dict(logprobs=dict(tokens=[f"token_id:{i}" for i in ids],
                                                                    top_logprobs=[top]))])))

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def drive(root, arm, srv):
    out = Path(root) / "results" / arm / "judge"
    out.mkdir(parents=True)
    env = dict(os.environ, PYTHONSAFEPATH="1")
    p = subprocess.run([sys.executable, "-B", str(RUN), "--url", f"http://127.0.0.1:{srv.server_port}",
                        "--out-dir", str(out)], cwd=out, env=env, capture_output=True, text=True, timeout=300)
    return p.returncode, json.loads((out / "verdict.json").read_text()), p


class RunTest(unittest.TestCase):
    def test_arms(self):
        with tempfile.TemporaryDirectory() as root:
            srv = stub()
            rc, v, p = drive(root, "off", srv)
            self.assertEqual((rc, v["scope"], v["gates"]["G1"]), (0, "record", None), p.stderr)
            self.assertEqual(len((Path(root) / "results/off/judge/record.jsonl").read_text().splitlines()), 15)
            rc, v, _ = drive(root, "on", stub())
            self.assertEqual((rc, v["scope"], v["peers"]), (0, "bits", ["off"]))
            rc, v, _ = drive(root, "on-bump", stub(bump=61312))  # one bit of one logprob in chain 0 turn 1
            self.assertEqual(rc, 1)
            self.assertTrue(any("(0, 1)" in r for r in v["reasons"]), v["reasons"])

    def test_record_gates(self):
        with tempfile.TemporaryDirectory() as root:
            rc, v, _ = drive(root, "cold", stub(hit_scale=0.5))  # warm turns below their hit column
            self.assertEqual((rc, v["gates"]["R"]), (1, "FAIL"))
        with tempfile.TemporaryDirectory() as root:
            rc, v, _ = drive(root, "short", stub(short=True))
            self.assertEqual((rc, v["gates"]["R"]), (1, "FAIL"))


if __name__ == "__main__":
    unittest.main()
