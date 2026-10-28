"""PerfBot lever switch: PERFBOT_EXP_GLUE is a comma list of lever names, read once at import.

Each name gates exactly one seam. P-11 (item 02): T1 WaveMoeTail early release at b 8, T3 KDA in_proj SplitKGemm
trigger at weight-warp entry (b 8), T5 OprojProducer load-before-wait (no seam: already in the image), T11 RouteSitu
early release at b 8, R / C / E the fused rejection tail, draft prep and next-step prologue (D = R,C,E), G the dead
draft-token gather guard (lane Q #2). Item 06: X1 experts EARLY at 9-40 rows, X2 JOIN off at 9-40 rows, X3 PREGROUP
at 9-40 rows (P-14 L3b); L48 / L64 / L96 the NEXT_ATTENTION cap at b 8 in MiB (P-11 lever K; at most one). Item 09:
K24 / K48 / K72 the 9-40-row prefetch tables (oproj_down and attention_moe rows, the MLA attention_moe fork after
in_proj, the new core_oproj point capped at that many MiB; at most one). K is kept as a name so older arm strings
still parse; it gates nothing. Item 06 S: TP8 verify SharedDown SMs 5 at 17-24
rows, 4 at 33-40 rows (interim; item 08 ExpertsStage supersedes it). Unset or empty = every lever off = the image. An unknown name raises, so a typo
cannot run a control arm by accident."""

import os
import sys

LEVERS = frozenset({"T1", "T3", "T5", "T11", "K", "R", "C", "E", "G", "X1", "X2", "X3", "L48", "L64", "L96",
                    "K24", "K48", "K72", "S"})
ALIASES = {"D": ("R", "C", "E")}
EXCLUSIVE = (frozenset({"L48", "L64", "L96"}), frozenset({"K24", "K48", "K72"}))


def parse(value):
    names = set()
    for name in filter(None, (n.strip() for n in value.split(","))):
        expanded = ALIASES.get(name, (name,))
        unknown = set(expanded) - LEVERS
        if unknown:
            raise ValueError(f"PERFBOT_EXP_GLUE: unknown lever {sorted(unknown)}; known {sorted(LEVERS | set(ALIASES))}")
        names.update(expanded)
        mark = getattr(sys.modules.get("sitecustomize"), "mark", None)  # PERFBOT liveness: overlay 0dcc27a+; older = no-op
        if name in ALIASES and mark:  # an alias is invisible to the environ marker, so name its items
            mark("PERFBOT_EXP_GLUE", ",".join(expanded))
    for group in EXCLUSIVE:
        if len(names & group) > 1:
            raise ValueError(f"PERFBOT_EXP_GLUE: at most one of {sorted(group)}, got {sorted(names & group)}")
    return frozenset(names)


ACTIVE = parse(os.environ.get("PERFBOT_EXP_GLUE", ""))


def on(name):
    assert name in LEVERS, name
    return name in ACTIVE


def value(group):
    """The one active lever of an exclusive group (e.g. ("K24", "K48", "K72")), or None."""
    hit = sorted(ACTIVE & frozenset(group))
    return hit[0] if hit else None
