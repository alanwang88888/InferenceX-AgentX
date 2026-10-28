"""Item 14 La: the lane L P-03 KdaSplit candidate `qbatch` (C2 occ2b5 + C5 warp4 + C8 qbatch) as the served
PreparedKda, behind PERFBOT_EXP_KDA_LA=1 (read at import).

kernel.py and prepared.py are byte copies of k3-lane-l-p03 docs/expedition/problems/P-03/kernel/qbatch/ (LOG.md
change table C2, C5, C8: judge pass full / twostep / swap / verify, bit-equal to the 49cb2bbf incumbent on 37/37
cases, B200). Against the incumbent: B5 runs at 2 CTAs/SM (schedule row 5:1.8.2.4), the gate warp is folded into the
four compute warps at the occupancy-3 rows, and at T > 1 each token's q butterfly runs one token late. The fp order
per token is unchanged, so the class is bit-equal.

Site: runtime/providers.py rebinds its PreparedKda name through pick() (two tagged lines).
"""

import os

ENV = "PERFBOT_EXP_KDA_LA"


def enabled(env=os.environ):
    return env.get(ENV, "") == "1"


ON = enabled()


def pick(stock):
    """The PreparedKda class the providers build: lane L qbatch when the switch is on, else ``stock``."""
    if not ON:
        return stock
    from .prepared import PreparedKda

    print("PERFBOT-SWITCH kda_la=on PreparedKda=perfbot_kernels.kdasplit_la", flush=True)  # PERFBOT P-07 La: serve liveness mark (on path only)
    return PreparedKda
