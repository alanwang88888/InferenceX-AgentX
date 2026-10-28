"""Kernel-only bench of MLA prefill attention for single-request mixed steps (expedition line M; README.md).

This module holds only the CPU-importable parts (torch only, no vllm, no CUDA): the served shapes, the server's
chunked-context workspace rule, the strided DCP shard layout, the variant registry and an fp64 reference.
The GPU variants live in variants.py; the 8-rank driver in bench.py.
"""

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

# Served K3 MLA under DCP 8 (TP 8, one process per GPU): local heads, latent, rope, q nope, v, page, interleave.
W, HL, L, R, P, V, PAGE, I = 8, 12, 512, 64, 128, 128, 64, 1
H = W * HL
SCALE = (P + R) ** -0.5  # qk_head_dim ** -0.5 (NoPE target: no mscale)
MAX_NUM_SEQS = 16  # M-03 server_command.txt --max-num-seqs
MAX_NUM_BATCHED_TOKENS = 16384  # M-03 --max-num-batched-tokens
CUDAGRAPH_MAX = 8192  # largest M-03 cudagraph_capture_sizes entry
NUM_SPEC = 7  # dspark num_speculative_tokens


def dcp_workspace_max_tokens() -> int:
    """overlay dcp.py get_dcp_workspace_max_num_tokens with the M-03 flags (no parallel drafting assumed):
    min(max_num_batched_tokens, max(max_num_seqs * (1 + spec), max_cudagraph_capture_size)) = 8192."""
    return min(MAX_NUM_BATCHED_TOKENS, max(MAX_NUM_SEQS * (1 + NUM_SPEC), CUDAGRAPH_MAX))


def chunked_workspace_rows(max_model_len: int, max_num_seqs: int = MAX_NUM_SEQS, page: int = PAGE,
                           dcp: int = W, interleave: int = I) -> int:
    """MLACommonMetadataBuilder.determine_chunked_prefill_workspace_size + align_mla_chunked_context_workspace_size
    (pinned mla_attention.py:2066, :1772): global context rows one chunk may cover. test_cpu.py checks this against
    the pinned source text."""
    rows = min(max(8 * max_model_len, 4 * max_num_seqs * page), 64 * 1024)
    align = math.lcm(page, dcp * interleave) if dcp > 1 else page
    return -(-max(rows, align) // align) * align


WORKSPACE_ROWS = chunked_workspace_rows(1 << 19)  # 65536 for any max_model_len >= 8192


def expected_chunks(ctx: int, rows: int = WORKSPACE_ROWS) -> int:
    """Context chunks the server runs for one request with ctx cached tokens (I = 1, ctx a multiple of 64)."""
    return -(-ctx // rows)


def shard_positions(rank: int, length: int, world: int = W, interleave: int = I, device=None) -> torch.Tensor:
    """Global positions held by `rank` in local order: DCP deals blocks of `interleave` tokens round-robin over
    the ranks (vllm get_dcp_local_seq_lens); local row j of rank k is position (j // I) * I * W + k * I + j % I."""
    p = torch.arange(length, device=device)
    return p[(p // interleave) % world == rank]


def load_shapes(path: Path | None = None) -> list[dict]:
    """The 12 replay shapes of M-03 (runs/M-03/analysis/bench_shapes.json): cls, n, ctx, m03_image_ms,
    m03_absorbed_ms (in-server eager walls of one MLA call summed over 24 layers)."""
    return json.loads((path or Path(__file__).with_name("shapes.json")).read_text())


# ----------------------------------------------------------------------------------------------- variant registry
@dataclass
class Variant:
    name: str
    run: Callable  # run(case) -> Tensor [n, HL, V] (this rank's heads); may cache setup in case.memo[name]
    m03_key: str | None = None  # shapes.json field the calibration compares against (None: no calibration)
    rel_bound: float | None = None  # per-shape correctness bound: rel-L2 vs the image variant (None: report only)
    available: Callable | None = None  # available(env) -> None or a reason string (checked on every rank)


VARIANTS: dict[str, Variant] = {}


def register(name: str, m03_key: str | None = None, rel_bound: float | None = None, available=None):
    """Decorator: `@register("mine")` on `def mine(case) -> torch.Tensor` adds a variant to the bench.

    The function gets the prepared per-rank case (bench.make_case: inputs, weights, cache shard, shared workspaces
    in case.env) and returns this rank's attention output [n, HL, V] bf16. Setup that the server does once per step
    (not per layer) goes in case.memo[name] on the first call; the bench's first call is the untimed warm-up.
    `available(env)` returning a string marks the variant "unavailable: <reason>" and the bench runs the others."""
    def deco(fn):
        VARIANTS[name] = Variant(name, fn, m03_key, rel_bound, available)
        return fn
    return deco


# ------------------------------------------------------------------------------------- split-KV plan (image_split)
# The plan and the backend live in perfbot_kernels.mla_split (the server runs the same code); re-exported here.
from perfbot_kernels.mla_split import NUM_SM, SPLIT_ALIGN, STEP_Q, ctas, slice_bounds, split_plan  # noqa: E402,F401
from perfbot_kernels import mla_split  # noqa: E402

SPLITS_ENV = "PERFBOT_MIXBENCH_SPLITS"  # bench only: the server always uses the formula


def default_splits(max_q: int) -> int:
    """mla_split.default_splits(max_q), overridden by PERFBOT_MIXBENCH_SPLITS (bench only)."""
    forced = os.environ.get(SPLITS_ENV)
    if forced:
        return max(1, int(forced))
    return mla_split.default_splits(max_q)


# ---------------------------------------------------------------------------------------------------- reference
def ref_rows(ctx_lat: torch.Tensor, new_lat: torch.Tensor, qn: torch.Tensor, qp: torch.Tensor,
             w_uk_t: torch.Tensor, w_uv: torch.Tensor, rows: torch.Tensor, scale: float = SCALE,
             chunk: int = 65536) -> torch.Tensor:
    """fp64 MLA attention of new-token rows `rows` (indices into the n new tokens), in the absorbed form
    (q W_UK against the latent, then W_UV) which equals the un-absorbed form exactly in real arithmetic and costs
    rows * G * 576 per head instead of G * 512 * 256. test_cpu.py checks it against mla_absorb.ref_attention.

    ctx_lat [ctx, L + R] (stored context latent, any dtype), new_lat [n, L + R]; qn [r, HL, P], qp [r, HL, R];
    w_uk_t [HL, P, L]; w_uv [HL, L, V]. Row i sees global positions <= ctx + rows[i]. Returns [r, HL, V] fp64."""
    ctx, dev = ctx_lat.shape[0], qn.device
    lat_l = ctx_lat.shape[1] - qp.shape[-1]
    qa = torch.einsum("rhp,hpl->rhl", qn.double(), w_uk_t.double())
    qr = qp.double()
    limit = ctx + rows.to(dev)
    r, hl = qa.shape[:2]
    m = torch.full((r, hl), float("-inf"), dtype=torch.float64, device=dev)
    s_sum = torch.zeros_like(m)
    acc = torch.zeros((r, hl, lat_l), dtype=torch.float64, device=dev)
    for base, src in ((0, ctx_lat), (ctx, new_lat)):
        for a in range(0, src.shape[0], chunk):
            lat = src[a : a + chunk].double()
            cl, pe = lat[:, :lat_l], lat[:, lat_l:]
            s = (torch.einsum("rhl,cl->rhc", qa, cl) + torch.einsum("rhd,cd->rhc", qr, pe)) * scale
            pos = base + a + torch.arange(lat.shape[0], device=dev)
            s = s.masked_fill(pos[None, None, :] > limit[:, None, None], float("-inf"))
            m_new = torch.maximum(m, s.max(-1).values)
            c_old = torch.exp(m - m_new)
            p = torch.exp(s - m_new[..., None])
            p = torch.nan_to_num(p, nan=0.0)  # a chunk wholly beyond every row's limit: -inf - -inf
            c_old = torch.nan_to_num(c_old, nan=1.0)
            acc = acc * c_old[..., None] + torch.einsum("rhc,cl->rhl", p, cl)
            s_sum = s_sum * c_old + p.sum(-1)
            m = torch.where(torch.isfinite(m_new), m_new, m)
    return torch.einsum("rhl,hlv->rhv", acc / s_sum[..., None], w_uv.double())
