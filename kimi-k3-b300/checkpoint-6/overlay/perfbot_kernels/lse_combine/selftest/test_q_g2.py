# SPDX-License-Identifier: Apache-2.0
"""CPU test of b300_pair's q bar (q_g2 = kernel 2's G2 over the 8 source head blocks). No GPU.
A served-like q (fp64 reference rounded to E4M3) must pass with 0 bytes differing; a +3-code mutant of one byte must
fail ours, keep served passing, and report 1 byte at code distance 3. q_g2 is a closure in run_rank, so its source is
taken from b300_pair.py and run on CPU tensors.
Run:  python3 -B -m perfbot_kernels.lse_combine.selftest.test_q_g2   Exit 0 = pass, 3 = the q judge is broken,
4 = the test itself crashed.
"""

import ast
import pathlib
import sys
import time

import torch

from perfbot_kernels.draft_query.selftest import judge as qjudge
from perfbot_kernels.lse_combine import b300_pair


def load_q_g2(env):
    src = pathlib.Path(b300_pair.__file__).read_text()
    fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "q_g2")
    lines = src.splitlines()[fn.lineno - 1 : fn.end_lineno]
    ind = len(lines[0]) - len(lines[0].lstrip())
    exec("\n".join(line[ind:] for line in lines), env)
    return env["q_g2"]


def main():
    t0 = time.process_time()
    heads, m = b300_pair.HEADS, 2
    g = torch.Generator().manual_seed(0)
    weights = (None, None, (torch.randn(8 * 1536, 1536, generator=g) * 0.03).bfloat16(),
               (torch.randn(8 * heads, 128, 512, generator=g) * 0.1).bfloat16())
    table = torch.randn(16, 64, generator=g).clamp(-1, 1)
    pos, one = torch.tensor([3, 7]), torch.ones(1)
    q_c = torch.randn(m, 1536, generator=g).bfloat16()
    q_g2 = load_q_g2(dict(torch=torch, qjudge=qjudge, weights=weights, table=table, one=one, HEADS=heads,
                          ENTRY=b300_pair.ENTRY, kind="cpu"))
    refs = [qjudge.query_reference(q_c, weights[2][s * 1536 : (s + 1) * 1536], weights[3][s * heads : (s + 1) * heads],
                                   table[pos], one)[0] for s in range(8)]
    served = torch.cat(refs, 1).clamp(-448, 448).float().to(torch.float8_e4m3fn).view(torch.uint8)
    ident = q_g2(m, q_c, pos, served.clone(), served)
    # one RoPE byte (cols 512-575: tight bound, no W_UK cancellation) of moderate magnitude (no NaN code, no sign
    # wrap), +3 E4M3 codes
    mag = served & 0x7F
    pick = (mag >= 0x30) & (mag <= 0x70)
    pick[..., :512] = False
    idx = tuple(int(i) for i in pick.nonzero()[0])
    mut = served.clone()
    mut[idx] += 3
    mutant = q_g2(m, q_c, pos, mut, served)
    rows = {"identity": [ident[0]["ok"], ident[1]["ok"], ident[2], ident[3]],
            "mutant_plus3": [mutant[0]["ok"], mutant[1]["ok"], mutant[2], mutant[3]]}
    ok = (ident[0]["ok"] and ident[1]["ok"] and ident[2] == 0 and not mutant[0]["ok"] and mutant[1]["ok"]
          and mutant[2] == 1 and mutant[3] == 3)
    print({"probe": "i15-pair-q-judge", "ok": ok, "rows [mine_ok, served_ok, bytes_diff, max_code_dist]": rows,
           "cpu_s": round(time.process_time() - t0, 2)}, flush=True)
    return 0 if ok else 3


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # a crash of the test: not a judge verdict
        print({"probe": "i15-pair-q-judge", "ok": False, "error": f"{type(e).__name__}: {e}"}, flush=True)
        sys.exit(4)
