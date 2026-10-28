"""Router GEMM + top-K routing in one launch (PerfBot roadmap item r3-301). CuTe DSL kernel in kernel.py.

`route(x, weight, bias, top_k, route_scale)` -> carrier [M, top_k + E] fp32 (one contiguous buffer):
  carrier.view(-1)[:M*top_k] = packed top-K [M, top_k] int32 bits, (expert_id << 16) | bf16 weight, rank order
  carrier.view(-1)[M*top_k:] = router logits [M, E] fp32 (x @ weight^T)
  x [M, K] bf16, weight [E, K] bf16, bias [E] fp32 (e_score_correction_bias), top_k 16, route_scale 1.0.
For M <= MAX_M one launch of RouterTopK writes both. For M > MAX_M (prefill) the op writes the logits with the
image's own router GEMM (GateLinear cuBLAS tier: torch.mm(out_dtype=fp32)) and leaves packed unset;
`split_carrier` then returns packed None and flashinfer routes from the logits exactly as in the image.
Sample inputs (K3 decode, conc 6): x [48, 7168] bf16, weight [896, 7168] bf16, bias [896] fp32, top_k 16,
route_scale 1.0 -> carrier [48, 912] fp32.
Numerical contract: packed equals flashinfer 0.6.17 routingIndicesBlockScoresKernel<SigmoidBias,
ScaledSumNormalize> run on this op's own logits, bit for bit; the logits are an fp32-accumulated bf16 GEMM
(sum order differs from cuBLAS; the equivalence gate rules on the end-to-end effect).

Registered with torch.library (`torch.ops.perfbot_router_topk.route`) so torch.compile traces one opaque node
and the M branch runs at call time. The kernel compiles once per (E, K, top_k) on first call (M is dynamic).
"""

import hashlib
import os
import sys
import threading
import time
from pathlib import Path

import torch

_DIR = Path(__file__).resolve().parent
_SOURCES = (_DIR / "kernel.py",)
MAX_M = 256  # rows per call served by the fused kernel (decode, spec-decode capture sizes)
_CNT_SLOTS = (MAX_M + 15) // 16
_LOCK = threading.Lock()
_COMPILED: dict = {}
_COUNTERS: dict = {}


def source_sha256() -> str:
    h = hashlib.sha256()
    for p in _SOURCES:
        h.update(p.name.encode() + b"\0" + p.read_bytes() + b"\n")
    return h.hexdigest()


def _iket() -> bool:
    return os.environ.get("PERFBOT_ROUTER_TOPK_IKET", "0") == "1"


def compiled(e: int, k: int, top_k: int):
    """Compile (once per shape key) and return the TVM-FFI callable. Raises on failure: no fallback."""
    key = (e, k, top_k, _iket())
    fn = _COMPILED.get(key)
    if fn is not None:
        return fn
    with _LOCK:
        fn = _COMPILED.get(key)
        if fn is not None:
            return fn
        import cutlass
        import cutlass.cute as cute
        from cuda.bindings.driver import CUstream
        from cutlass.cute.runtime import make_fake_compact_tensor

        if __package__:
            from .kernel import RouterTopK
        else:  # pragma: no cover
            from kernel import RouterTopK

        t0 = time.time()
        gemm = RouterTopK(n=e, k=k, top_k=top_k, iket=key[3])
        m = cute.sym_int()

        def fake(dtype, shape):
            order = (1, 0) if len(shape) == 2 else (0,)
            return make_fake_compact_tensor(dtype, shape, stride_order=order, assumed_align=16)

        fn = cute.compile(
            gemm,
            fake(cutlass.BFloat16, (m, k)),
            fake(cutlass.BFloat16, (e, k)),
            fake(cutlass.Float32, (e,)),
            fake(cutlass.Float32, (m, e)),
            fake(cutlass.Int32, (m, top_k)),
            fake(cutlass.Uint32, (m, gemm.cand_width)),
            fake(cutlass.Int32, (_CNT_SLOTS,)),
            cutlass.Float32(1.0),
            CUstream(torch.cuda.current_stream().cuda_stream),
            options="--enable-tvm-ffi",
        )
        fn.cand_width = gemm.cand_width
        _COMPILED[key] = fn
        print(f"PERFBOT-ROUTER-TOPK compiled sha={source_sha256()[:16]} key={key} "
              f"seconds={time.time() - t0:.1f}", file=sys.stderr, flush=True)
        return fn


def _counter(device: torch.device) -> torch.Tensor:
    cnt = _COUNTERS.get(device)
    if cnt is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("router_topk: first call inside CUDA-graph capture; call once eagerly first")
        cnt = torch.zeros(_CNT_SLOTS, dtype=torch.int32, device=device)
        _COUNTERS[device] = cnt
    return cnt


def split_carrier(carrier: torch.Tensor, top_k: int):
    """(packed [M, top_k] int32 or None, logits [M, E] fp32) views of a route() carrier."""
    m, width = carrier.shape
    flat = carrier.view(-1)
    logits = flat[m * top_k:].view(m, width - top_k)
    packed = flat[: m * top_k].view(m, top_k).view(torch.int32) if m <= MAX_M else None
    return packed, logits


@torch.library.custom_op("perfbot_router_topk::route", mutates_args=(), device_types="cuda")
def route(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, top_k: int,
          route_scale: float) -> torch.Tensor:
    m, k = x.shape
    e = weight.shape[0]
    carrier = torch.empty((m, top_k + e), dtype=torch.float32, device=x.device)
    if m == 0:
        return carrier
    flat = carrier.view(-1)
    logits = flat[m * top_k:].view(m, e)
    if m > MAX_M:
        logits.copy_(torch.mm(x, weight.T, out_dtype=torch.float32))
        return carrier
    packed = flat[: m * top_k].view(m, top_k).view(torch.int32)
    fn = compiled(e, k, top_k)
    cand = torch.empty((m, fn.cand_width), dtype=torch.int32, device=x.device).view(torch.uint32)
    from cuda.bindings.driver import CUstream

    fn(x, weight, bias, logits, packed, cand, _counter(x.device), float(route_scale),
       CUstream(torch.cuda.current_stream().cuda_stream))
    return carrier


@route.register_fake
def _route_fake(x, weight, bias, top_k, route_scale):
    return x.new_empty((x.shape[0], top_k + weight.shape[0]), dtype=torch.float32)


def eligible(weight: torch.Tensor, bias, top_k: int) -> str | None:
    """None when route() serves this gate; else the reason it does not."""
    if bias is None or bias.dtype != torch.float32:
        return "bias is not fp32"
    if weight.dtype != torch.bfloat16 or weight.dim() != 2:
        return f"weight {weight.dtype} {tuple(weight.shape)}"
    e, k = weight.shape
    if e % 64 != 0 or k % 8 != 0 or not (1 <= top_k <= 16) or ((e // 64) * top_k) % 32 != 0:
        return f"shape E={e} K={k} top_k={top_k} outside the kernel's tiling"
    return None
