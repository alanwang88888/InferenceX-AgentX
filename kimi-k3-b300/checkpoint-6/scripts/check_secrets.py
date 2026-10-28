#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scan the tree for internal names and secrets. Run before every push.

Outside internal/ nothing may match:
  - a secret: access tokens (glpat-, hf_..., Bearer ..., token=<long value>), the seat-token file name;
  - the measurement infrastructure: broker receipts, the sync/carrier path (/tmp/k3-split-sync), unix sockets (.sock);
  - an NVIDIA-internal host name or NFS path (umb-b300-*, umbriel-*, dp-NNN, computelab*, frontend-N, *.nvidia.com,
    gitlab-master, /home/scratch.*).
The word "tokens" (model tokens) and the MoE "carrier" tensor are not matches. Exit 1 on any hit.

    python3 scripts/check_secrets.py [<root>]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", "internal", "__pycache__"}
SKIP_FILES = {"scripts/check_secrets.py", "scripts/ingest_results.py"}  # they spell the patterns
ALLOW = re.compile(r"gitlab-master\.nvidia\.com/alanw/k3_inferenceX")  # this repository's own home, cited in CITATION.cff
PATTERNS = {
    "access token": re.compile(r"(glpat-[A-Za-z0-9_-]{10,}|\bhf_[A-Za-z0-9]{20,}|Bearer\s+[A-Za-z0-9._-]{16,}|\btoken\s*[=:]\s*['\"]?[A-Za-z0-9_-]{16,})", re.I),
    "seat token file": re.compile(r"\.seat-token|seat[_-]token", re.I),
    "broker receipt": re.compile(r"broker[_ -]?receipt|receipts/", re.I),
    "sync path": re.compile(r"/tmp/k3-split-sync|k3-split-sync", re.I),
    "carrier path": re.compile(r"carrier\.txt|sync_source|/carrier/", re.I),
    "unix socket": re.compile(r"(?<!\S)\S+\.sock\b"),
    "internal host": re.compile(r"\b(umb-b300-\S*|umbriel-\S*|dp-\d{3}\b|computelab\S*|frontend-\d+|(?<!\S)\S+\.nvidia\.com|gitlab-master\S*)", re.I),
    "internal path": re.compile(r"/home/scratch\.[a-z_]+|/home/alanw"),
}


def main() -> int:
    hits = 0
    for p in sorted(ROOT.rglob("*")):
        rel = p.relative_to(ROOT)
        if not p.is_file() or SKIP_DIRS & set(rel.parts) or rel.as_posix() in SKIP_FILES:
            continue
        try:
            text = p.read_text(errors="replace")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            line = ALLOW.sub("<repo>", line)
            for kind, pat in PATTERNS.items():
                m = pat.search(line)
                if m:
                    hits += 1
                    print(f"{kind}: {p.relative_to(ROOT)}:{i}: {m.group(0)[:60]}")
    print("RESULT", "PASS (no hit outside internal/)" if not hits else f"FAIL {hits} hits")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
