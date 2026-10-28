"""P-14 lever L6: a per-request MLA split-KV vector for the tokenspeed decode under DCP (the one generator).

The served planner gives every request of a batch the same split, a function of B alone (12 / 6 / 4 / 3 / 2 at
B1-B5). The grid (2, 6B, split) runs one wave on 148 SMs, so the MLA wall is the longest request's CTA and the CTAs
of short requests idle (prior-art/P-14/MAXCTX.md section 1). Here a one-thread CuTe DSL kernel (kernel.py) computes,
once per captured step, the split of each request by the greedy rule above the served split (var_splits is its
host reference), and tokenspeed partitions request b by block_split_kvs[b] (is_var_split_kv; prior art:
tokenspeed-mla 2.0.2 mla_decode_fp8.py:2109-2126 and the reducer :1859-1876, https://pypi.org/project/tokenspeed-mla/;
the P-08 var-split arm, lane-n-p08 582bb15).

Sites (tagged "# PERFBOT P-14" in the overlay copies of vLLM 49cb2bbf; all inside the reload trees):
  owner     specialized_decode/runtime/providers.py: the first MLA layer's PreparedMla.splits_owner = True
  producer  specialized_decode/cutedsl/mla/prepared.py stock_core -> decode_args(): the owner launches the device
            greedy over the step's padded local lengths into the batch's persistent Int32 buffer (one graph node per
            step, no PDL: tokenspeed starts after it completes); the pre-capture warm call compiles it
  consumer  specialized_decode/cutedsl/mla/dcp.py stock_decode(splits=...): min_split() raises the grid split to
            s_grid(B); compiler() turns is_var_split_kv on; the vector takes the block_split_kvs slot
The host never sees the lengths: specialized_decode runs Python only at capture (runtime/decode.py:212).

Switch PERFBOT_EXP_B35 (P-14 section 5): a comma list of letters; F turns L6 on. Unset = off: decode_args()
returns {} and stock_decode runs the served call byte for byte. O (01d, only with F): the buffer also holds the unit
table (unit_table; the device kernel writes it), and the vector compiles through core.py's split kernel (LPT dispatch
order, empty units exit). Bit-equal to F alone.
Domain: the prepared verify batches (requests, q) in DOMAIN (B3 / B4 two-wave behind the gate, B5 one wave); every
other batch keeps the served split.
Torch-only at import; CPU tests in test_cpu.py.
"""

import os
import types
from functools import cache

import torch

ENV, LETTER, ORDER_LETTER = "PERFBOT_EXP_B35", "F", "O"
# One unit = one CTA pair per packed M tile of a request: q 8 x 96 heads = 6 tiles of 128 rows, so a unit is
# 2 x 6 = 12 CTAs and 12 units = 144 CTAs fill one wave of 148 SMs at 1 CTA per SM (P-14 gate pass 20). K5 (q 6):
# 5 tiles, a unit is 10 CTAs, 14 units = 140 CTAs (k5q.consts).
from perfbot_kernels.k5q import UNITS, RULES as _K5_RULES, Q as _K5_Q  # PERFBOT K5 items 2-3 (land-8: UNITS = 12, q 8)
# (requests, q): the verify batches that use the vector. B1 / B2 keep the served split (the floor-free rule gave
# 8 < 12 at B1, +0.4 ms per step, P-14 gate pass 20).
#
# Numerics floor [m, selftest run1 on B200, 2026-10-19]: the error of a request falls with its split count, so a
# vector that gives a request fewer splits than the served 12 // B fails the 1.02 envelope (B4 (6,2,2,2) 1.15x,
# B3 (8,2,2) 1.24x; B5 (4,2,2,2,2) 0.99x). Every non-empty request therefore keeps the served split (served()),
# and only the spare units above B x served move. In one wave B3 / B4 have none (4 x 3 = 3 x 4 = 12); B5 has 2.
# Two-wave rule at B4 (item-01b, coordinator 2026-10-19): 15 units (180 CTAs), cap 2 x served = 6; kernel-only
# B200 (386k, 3 x 88k) 0.85x per call, envelope pass [m, run2]. B3 (01c, coordinator 2026-10-19): the same rule,
# 16 units (192 CTAs), cap 2 x served = 8; (8, 4, 4) at item 10's band [334k, 115k, 60k] 0.91x per call [m, B200].
# Its combined error sits within +-2 % of the served split's (run2 1.022, run3 0.982 on two seeds): the seed sweep
# in selftest (MLA_VARSPLIT_SEED_SHIFT) is its evidence.
# Gate [m, run3]: with equal lengths the second wave is a full extra round (b4-equal 4 x 200k: 1.40x slower). The
# second-wave units start when the first short request frees its SMs, so the two-wave wall ~ min_other / 3 + L / 6
# against the served L / 3 [e]: two waves only when 2 x the shortest other live length < the longest (two_wave_pays);
# else one wave, which at B4 is the served vector.
RULES = dict(_K5_RULES)  # requests -> (units, cap = the grid split s_grid)  # PERFBOT K5 item 3 (land-8: {5: (12, 8), 4: (15, 6), 3: (16, 8)})
EMPTY = 1 << 16  # 01d: the unit-table flag of an empty unit (core.py)
GATED = frozenset({3, 4})  # batches whose units above one wave need two_wave_pays
DOMAIN = frozenset((reqs, _K5_Q) for reqs in RULES)  # PERFBOT K5 item 2 (land-8: q 8)


def enabled(env=os.environ, letter=LETTER) -> bool:
    return letter in env.get(ENV, "").replace(" ", "").split(",")


ON = enabled()  # read once at import, as the other overlay switches
# 01d (letter O, only with F): the unit table and the empty-unit exit (core.py). Bit-equal to F alone.
ORDER = ON and enabled(letter=ORDER_LETTER)


def served(reqs):
    """The served uniform split: UNITS // reqs = 12 / 6 / 4 / 3 / 2 at q 8, 14 / 7 / 4 / 3 / 2 at q 6 [m planner]."""
    return max(1, UNITS // reqs)


def rule(reqs):
    """(units, cap) of a batch of ``reqs`` requests: RULES, else one wave with no cap below s_grid."""
    return RULES.get(reqs, (UNITS, UNITS + 1 - reqs))


def var_splits(lens):
    """The host reference of the rule (P-14 gate pass 20; P-08 judge.py var_splits) with the numerics floor:
    greedy min-max. Every non-empty request starts at the served split, a zero length (pad row) at 1; each further
    unit, up to the batch's units, goes to the non-empty request below the cap with the largest len_i / split_i,
    the lowest index on a tie (a pad row never takes one: its CTAs are empty). The device kernel (kernel.py)
    computes the same vector in the captured step."""
    units, cap = rule(len(lens))
    if len(lens) in GATED and not two_wave_pays(lens):
        units = min(units, UNITS)
    sp = [served(len(lens)) if n > 0 else 1 for n in lens]
    for _ in range(units - sum(sp)):
        open_ = [i for i in range(len(lens)) if sp[i] < cap and lens[i] > 0]
        if not open_:
            break
        sp[max(open_, key=lambda i: (lens[i] / sp[i], -i))] += 1
    return tuple(sp)


def two_wave_pays(lens):
    """Whether the shortest live request other than the longest (the first longest on a tie) is under half of it."""
    top = max(range(len(lens)), key=lambda i: (lens[i], -i))
    others = [n for i, n in enumerate(lens) if i != top and n > 0]
    return 2 * min(others, default=0) < lens[top]


def unit_table(lens, splits):
    """01d: the dispatch order of the grid's units (host reference of kernel.py). Entry u (grid unit z * B + b) is
    the unit b' + B * z' it runs: the live units of the requests in descending len / split (the lowest index on a
    tie; z' ascending), then every empty unit (z' >= split_b', + EMPTY: its CTAs exit, core.py) in (z', b') order.
    LPT order when the live units need two waves (sum > UNITS): the longest CTAs go to the first SMs [pic-c,
    evidence/01c-picc]. In one wave every live CTA starts at once, and LPT only packs a request's units on adjacent SMs
    (B5 equal 50k: 1.067x uni vs z-major 1.0x [m B200, evidence/01d/gain]), so the live units keep the grid's z-major
    order there."""
    B, cap = len(lens), s_grid(len(lens))
    if sum(splits) > UNITS:
        order = sorted(range(B), key=lambda b: (-lens[b] / splits[b], b))
        live = [b + B * z for b in order for z in range(splits[b])]
    else:
        live = [b + B * z for z in range(cap) for b in range(B) if z < splits[b]]
    return live + [b + B * z + EMPTY for z in range(cap) for b in range(B) if z >= splits[b]]


def s_grid(reqs):
    """The static grid split of a batch: the largest split var_splits can give one request."""
    return rule(reqs)[1]


_BUFFERS: dict = {}  # (device, reqs) -> the persistent Int32[reqs] vector; captured graphs read it by address


def decode_args(reqs, q, lengths, owner):
    """stock_core's extra stock_decode keywords: {} (the served call) unless L6 is on for this batch. The owner
    (the first MLA layer, providers.py) launches the device greedy over this step's local lengths into the batch's
    buffer; the step's other MLA layers read the same buffer. Each batch has its own buffer, so a vector always has
    its batch's request count and no split above s_grid(reqs)."""
    if not ON or (reqs, q) not in DOMAIN:
        return {}
    key = lengths.device, reqs
    if key not in _BUFFERS:  # 01d: the unit table follows the vector (reqs x s_grid entries)
        size = reqs * (1 + s_grid(reqs)) if ORDER else reqs
        _BUFFERS[key] = torch.full((size,), served(reqs), dtype=torch.int32, device=lengths.device)
        _BUFFERS[key][reqs:] = torch.arange(size - reqs, dtype=torch.int32)  # the identity order until produced
    if owner:
        from .kernel import produce

        produce(lengths[:reqs], _BUFFERS[key], rule(reqs), served(reqs), reqs in GATED)
    return dict(splits=_BUFFERS[key][:reqs])  # tokenspeed's block_split_kvs has the batch's shape; core.py


def min_split(served, reqs, splits):
    """The grid split floor: the served one, raised to s_grid(reqs) when the vector is in use."""
    return served if splits is None else max(served, s_grid(reqs))


class _VarSplitCompiler:
    """tokenspeed_mla.mla_decode with _get_compiled_mla_kernel compiling is_var_split_kv=True (01d: through the
    split kernel of core.py)."""

    def __init__(self, md):
        self.md = md

    def __getattr__(self, name):
        return getattr(self.md, name)

    def _get_compiled_mla_kernel(self, **kwargs):
        get = _ordered(self.md._get_compiled_mla_kernel) if ORDER else self.md._get_compiled_mla_kernel
        assert not kwargs.get("is_persistent"), "01d remaps the non-persistent grid only"
        return get(**{**kwargs, "is_var_split_kv": True})


@cache
def _ordered(get):
    """A copy of tokenspeed's cached compile function whose FP8 kernel class is core.py's (its own cache)."""
    from .core import kernel_class

    f = get.__wrapped__
    scope = dict(f.__globals__, BlackwellMultiHeadLatentAttentionForwardFP8=kernel_class())
    copy = types.FunctionType(f.__code__, scope, f.__name__, f.__defaults__, f.__closure__)
    copy.__kwdefaults__ = f.__kwdefaults__
    return cache(copy)


def compiler(md, splits, split, reqs):
    """The module stock_decode compiles through: md itself unless the vector is in use. The kernel needs every
    split_i <= the grid split; the scalar must be exactly s_grid (the workspace and reducer follow it)."""
    if splits is None:
        return md
    assert split == s_grid(reqs), (split, reqs)
    return _VarSplitCompiler(md)
