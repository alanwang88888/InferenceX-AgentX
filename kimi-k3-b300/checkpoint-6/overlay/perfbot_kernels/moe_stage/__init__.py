"""Decode MoE stage in three CuTe DSL launches (PerfBot roadmap item r3-318). Kernels in kernels.py.

`run(x, packed, w13, s13, w2, s2, e_off, alpha, beta)` -> PrefilledMoEOutput or None, replacing
MXFP8QuantizeLinearKernel + flashinfer trtllm_fp4_block_scale_routed_moe(do_finalize=False) + the latent tail's
top-K finalize on the decode path (M <= MAX_M = 64 tokens per rank):
  1. MoePermute  (M CTAs) packed top-K [M, 16] int32 ((local id + e_off) << 16 | bf16 weight) and the routed
                 latent x [M, 3584] bf16 -> 16-slot tiles of routed pairs grouped by expert, each slot the token's
                 MXFP8 row and SF (quantized here, bit-identical to MXFP8QuantizeLinearKernel), bf16 weights,
                 permute map, per-row (token, weight).
  2. MoeGemm fc1 MXFP4 w13 x MXFP8 slots, epilogue SiTU-GLU + MXFP8 quant -> inter e4m3 + SF.
  3. MoeGemm fc2 MXFP4 w2 x MXFP8 inter, epilogue weighted red.add per token; after one grid-wide arrival the
                 whole grid stores the finished bf16 latent rows into the latent tail's NVLS multicast Lamport
                 buffer (slot (token, rank), generation flags[0]), so the wire starts inside fc2.
The returned UnfinalizedMoEOutput subclass `PrefilledMoEOutput` tells the overlay's latent-tail collective
(vllm/models/kimi_k3/nvidia/ops/cute_dsl/latent_moe_tail/allreduce_rmsnorm_reduce_scatter_early_exit.py) that the
routed rows are already in its buffer: it polls all ranks and skips its own finalize and store.
Sample inputs (K3 decode, conc 6): x [48, 3584] bf16, packed [48, 16] int32, w13 [896, 768, 1792] u8,
s13 [896, 768, 112] e4m3-typed ue8m0 bytes, w2 [896, 3584, 192] u8, s2 [896, 3584, 12].

Target: the process's KimiK3LatentMoETailOp collective (multicast base, routed flags, rank), found on the first
eager call; `set_target` injects a plain buffer for the single-GPU self-test. No target -> None (image path).
`PERFBOT_R3_318_MOE_STAGE=0` restores the image path (the call site checks `enabled()`).
Compiles once per (device, alpha, beta, e_off); M is a runtime argument. The first call must be eager (vLLM's
warm-up runs before graph capture); a first call inside capture returns None and the caller keeps the image path.
Not a torch.library op: it runs inside vLLM's opaque MoE runner op, and its real output is a write into the tail's
buffer that a functional op schema cannot state.
"""

import hashlib
import os
import sys
import threading
import time
from pathlib import Path

import torch

_DIR = Path(__file__).resolve().parent
_SOURCES = (_DIR / "kernels.py", _DIR / "__init__.py")
E, TOPK, H, I, MAX_M, TN, TP = 896, 16, 3584, 384, 64, 16, 8
TMAX = E + MAX_M * TOPK // TN
_LOCK = threading.Lock()
_COMPILED: dict = {}
_WS: dict = {}
_TARGET: dict = {}
_LOGGED: set = set()

try:
    from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput as _Base
except Exception:  # pragma: no cover - outside the image
    _Base = object


class PrefilledMoEOutput(_Base):
    """UnfinalizedMoEOutput whose routed rows fc2 already stored in the latent tail's multicast buffer."""

    perfbot_prefilled = True


def enabled() -> bool:
    return os.environ.get("PERFBOT_R3_318_MOE_STAGE", "1") != "0"


def _iket() -> bool:
    return os.environ.get("PERFBOT_MOE_STAGE_IKET", "0") == "1"


def source_sha256() -> str:
    h = hashlib.sha256()
    for p in _SOURCES:
        h.update(p.name.encode() + b"\0" + p.read_bytes() + b"\n")
    return h.hexdigest()


def _log(msg: str) -> None:
    print(f"PERFBOT-MOE-STAGE {msg}", file=sys.stderr, flush=True)


def _log_once(key, msg: str) -> None:
    if key not in _LOGGED:
        _LOGGED.add(key)
        _log(msg)


def compiled(device: torch.device, alpha: float, beta: float, e_off: int):
    key = (device.index, float(alpha), float(beta), int(e_off), _iket())
    fns = _COMPILED.get(key)
    if fns is not None:
        return fns
    with _LOCK:
        fns = _COMPILED.get(key)
        if fns is not None:
            return fns
        import cutlass
        import cutlass.cute as cute
        from cuda.bindings.driver import CUstream
        from cutlass.cute.runtime import make_ptr

        from .kernels import A_DTYPE, B_DTYPE, SF_DTYPE, MoeGemm, MoePermute

        t0 = time.time()
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        gm = cute.AddressSpace.gmem
        fake = lambda t, i: make_ptr(t, 4096 * (i + 1), gm, assumed_align=16)
        stream = CUstream(torch.cuda.current_stream(device).cuda_stream)
        i32, u8, i16, f32 = cutlass.Int32, cutlass.Uint8, cutlass.Int16, cutlass.Float32
        permute = cute.compile(MoePermute(e_off, iket=key[4]), fake(i32, 0), fake(i32, 1), fake(i32, 2),
                               fake(i32, 3), fake(i16, 4), fake(i32, 5), fake(i32, 6), fake(u8, 7), fake(i32, 8),
                               fake(i32, 9), cutlass.Int32(1), stream)
        gemm_args = lambda out_t: [fake(A_DTYPE, 0), fake(SF_DTYPE, 1), fake(B_DTYPE, 2), fake(SF_DTYPE, 3),
                                   fake(i32, 4), fake(i32, 5), fake(out_t, 6), fake(u8, 7), fake(i32, 8),
                                   fake(i32, 9), fake(i32, 10), fake(i32, 11), cutlass.Int64(4096),
                                   cutlass.Int32(0), cutlass.Int32(1), stream]
        fc1 = cute.compile(MoeGemm(True, sms, alpha, beta, iket=key[4]), *gemm_args(u8))
        fc2 = cute.compile(MoeGemm(False, sms, alpha, beta, iket=key[4]), *gemm_args(f32))
        fns = (permute, fc1, fc2)
        _COMPILED[key] = fns
        _log(f"compiled sha={source_sha256()[:16]} key={key} sms={sms} seconds={time.time() - t0:.1f}")
        return fns


def workspace(device: torch.device):
    ws = _WS.get(device.index)
    if ws is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        z = lambda n, dt: torch.zeros(n, dtype=dt, device=device)
        ws = {"tmeta": z(TMAX * 4, torch.int32), "cnt": z(4, torch.int32),
              "x_perm": z(TMAX * TN * H, torch.uint8), "sfx_perm": z(TMAX * 128 * (H // 32), torch.uint8),
              "inter": z(TMAX * TN * I, torch.uint8), "sf2": z(TMAX * 128 * (I // 32), torch.uint8),
              "rtok": z(MAX_M * TOPK, torch.int32), "rw": z(MAX_M * TOPK, torch.int32),
              "acc": z(MAX_M * H, torch.float32), "ctr": z(4, torch.int64), "dummy": z(64, torch.int32),
              "g2": z((1, H), torch.bfloat16)}
        _WS[device.index] = ws
    return ws


def set_target(device: torch.device, mc_ptr: int, flags: torch.Tensor, rank: int, max_m: int = 128) -> None:
    """Point fc2's finalize at a Lamport buffer (self-test: plain device memory with the tail's layout)."""
    _TARGET[device.index] = (int(mc_ptr), flags, int(rank), int(max_m))


def target(device: torch.device):
    """(multicast base, routed flags, rank, max_m) of this process's latent tail collective, or None."""
    t = _TARGET.get(device.index)
    if t is not None:
        return t
    if torch.cuda.is_current_stream_capturing():
        return None
    try:
        from vllm.models.kimi_k3.nvidia.ops.latent_moe_tail import KimiK3LatentMoETailOp
    except Exception as exc:  # noqa: BLE001
        _log_once("no-tail-import", f"latent tail not importable ({exc!r}); image path kept")
        return None
    for op in KimiK3LatentMoETailOp._instances.values():
        c = getattr(op, "_collective", None)
        if (c is not None and op.contract.device.index == device.index
                and getattr(c, "perfbot_prefilled_ready", False) and c.tp_size == TP and c.latent_dim == H
                and c.top_k == TOPK):
            t = (int(c._routed_multicast_ptr), c._routed_flags, int(c.rank), int(c.max_m))
            _TARGET[device.index] = t
            _log(f"tail target rank={c.rank} tp={c.tp_size} max_m={c.max_m} (overlay collective, prefilled mode)")
            return t
    _log_once("no-tail", "no overlay latent-tail collective with prefilled mode on this device; image path kept")
    return None


def eligible(x, packed, w13, s13, w2, s2):
    """None when run() serves this call; else the reason it does not."""
    if packed is None:
        return "no packed top-K"
    m = x.shape[0]
    if not (1 <= m <= MAX_M):
        return f"M={m} outside 1..{MAX_M}"
    if x.dtype != torch.bfloat16 or tuple(x.shape) != (m, H) or not x.is_contiguous():
        return f"x {x.dtype} {tuple(x.shape)}"
    if tuple(packed.shape) != (m, TOPK) or packed.dtype != torch.int32 or not packed.is_contiguous():
        return f"packed {packed.dtype} {tuple(packed.shape)}"
    want = {"w13": (E, 2 * I, H // 2), "s13": (E, 2 * I, H // 32), "w2": (E, H, I // 2), "s2": (E, H, I // 32)}
    for name, t in (("w13", w13), ("s13", s13), ("w2", w2), ("s2", s2)):
        if tuple(t.shape) != want[name] or t.element_size() != 1 or not t.is_contiguous():
            return f"{name} {t.dtype} {tuple(t.shape)}"
    return None


def run(x, packed, w13, s13, w2, s2, e_off: int, alpha: float, beta: float, only: str = ""):
    """PrefilledMoEOutput(placeholder [1, H], expert_weights [M, 16] bf16, permute map [M, 16] int32) or None.
    The stage's result is the bf16 rows fc2 stored into the target buffer. `only` ("permute", "fc1", "fc2") runs
    that one launch on the workspace left by a full call (benchmarks)."""
    import cutlass
    import cutlass.cute as cute
    from cuda.bindings.driver import CUstream
    from cutlass.cute.runtime import make_ptr

    from .kernels import A_DTYPE, B_DTYPE, SF_DTYPE

    dev = x.device
    ws = workspace(dev)
    tgt = target(dev)
    if ws is None or tgt is None:
        _log_once("capture", "first call inside CUDA-graph capture or no tail target: image path kept for this call")
        return None
    mc, flags, rank, max_m = tgt
    m = x.shape[0]
    if m > max_m:
        return None
    fns = compiled(dev, alpha, beta, e_off)
    wts = torch.empty((m, TOPK), dtype=torch.bfloat16, device=dev)
    perm = torch.empty((m, TOPK), dtype=torch.int32, device=dev)
    gm = cute.AddressSpace.gmem
    P = lambda t, dt: make_ptr(dt, t.data_ptr(), gm, assumed_align=16)
    i32, u8, i16, f32 = cutlass.Int32, cutlass.Uint8, cutlass.Int16, cutlass.Float32
    stream = CUstream(torch.cuda.current_stream(dev).cuda_stream)
    mm = cutlass.Int32(m)
    common = [P(ws["tmeta"], i32), P(ws["cnt"], i32)]
    tail = [P(flags, i32), cutlass.Int64(mc), cutlass.Int32(rank), mm, stream]
    dummy = P(ws["dummy"], i32)
    permute, fc1, fc2 = fns
    if only in ("", "permute"):
        permute(P(packed, i32), P(x, i32), *common, P(wts, i16), P(perm, i32), P(ws["x_perm"], i32),
                P(ws["sfx_perm"], u8), P(ws["rtok"], i32), P(ws["rw"], i32), mm, stream)
    if only in ("", "fc1"):
        fc1(P(w13, A_DTYPE), P(s13, SF_DTYPE), P(ws["x_perm"], B_DTYPE), P(ws["sfx_perm"], SF_DTYPE), *common,
            P(ws["inter"], u8), P(ws["sf2"], u8), dummy, dummy, dummy, *tail)
    if only in ("", "fc2"):
        # osf: a dedicated dummy (fc2 writes no SF; review pass 1, advisory 4)
        fc2(P(w2, A_DTYPE), P(s2, SF_DTYPE), P(ws["inter"], B_DTYPE), P(ws["sf2"], SF_DTYPE), *common,
            P(ws["acc"], f32), P(ws["dummy"], u8), P(ws["rtok"], i32), P(ws["rw"], i32), P(ws["ctr"], i32), *tail)
    _log_once(("shape", m), f"run M={m} e_off={e_off} alpha={alpha} beta={beta} rank={rank}")
    return PrefilledMoEOutput(ws["g2"], wts, perm)
