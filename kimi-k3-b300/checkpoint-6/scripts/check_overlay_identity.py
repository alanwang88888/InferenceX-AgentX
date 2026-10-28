#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prove that overlay/ plus internal/overlay-redacted/ is the served tree, byte for byte.

The measurement harness records one digest per served overlay: sha256 over the sorted lines
"<relative path>\\0<sha256 of the file>\\n" (every file; .gitkeep, __pycache__ and .pyc excluded).
metrics.json carries it as overlay_tree_sha256. This script recomputes it over the union of
overlay/ and internal/overlay-redacted/ (one documentation file moved out of the public tree)
and compares with the recorded value. Exit 1 on a mismatch.
--point <name> checks a per-point served tree of the record's "points" map: the same union with that point's
"layers" laid on top in order (a file of a later layer replaces the file at the same relative path).

    python3 scripts/check_overlay_identity.py [--expect <sha256>] [--point <name>]
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARTS = (ROOT / "overlay", ROOT / "internal" / "overlay-redacted")
RECORD = ROOT / "patches" / "served-overlay-identity.json"


def digest(parts) -> tuple[int, str]:
    files = {}
    for root in parts:
        if not root.is_dir():
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.name != ".gitkeep" and "__pycache__" not in p.parts and p.suffix != ".pyc":
                files[p.relative_to(root).as_posix()] = p
    h = hashlib.sha256()
    for rel in sorted(files):
        h.update(f"{rel}\0{hashlib.sha256(files[rel].read_bytes()).hexdigest()}\n".encode())
    return len(files), h.hexdigest()


def main() -> int:
    record = json.loads(RECORD.read_text())
    if "--point" in sys.argv:
        record = record["points"][sys.argv[sys.argv.index("--point") + 1]]
    expect = sys.argv[sys.argv.index("--expect") + 1] if "--expect" in sys.argv else record["tree_sha256"]
    n, got = digest(PARTS + tuple(ROOT / p for p in record.get("layers", ())))
    ok = got == expect
    print(f"files {n}\ntree_sha256 {got}\nexpected    {expect}\nRESULT {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
