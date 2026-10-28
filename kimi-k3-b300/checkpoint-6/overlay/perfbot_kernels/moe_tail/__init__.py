"""K3 latent-MoE tail in one launch (expedition line B; docs/expedition/PLAN-B-tail.md). Kernel in kernel.py.

`run(op, routed_output, shared_output, rms_weight, up_weight)` -> out [M, 7168] bf16, or None (decline: the caller
runs `op(...)`, the image path). `op` is the runner's KimiK3LatentMoETailOp; the call replaces its three launches
(collective, up GEMM, Lamport copy) on the same buffers, flags and generations, so image and candidate calls may
interleave on one op.
  routed_output  PrefilledMoEOutput of perfbot_kernels.moe_stage: fc2 already stored the rows in op's routed buffer
  shared_output  [M, 7168] bf16 shared-expert partial of this rank (contiguous)
  rms_weight     [3584] bf16 routed_expert_norm.weight
  up_weight      [7168, 3584] bf16 routed_expert_up_proj.weight (full; the kernel reads this rank's 896 rows)
Declines: PERFBOT_EXP_MOETAIL=0; routed_output not perfbot_prefilled; M outside 1..64; TP != 8 or other shapes;
max active 7-CTA clusters < 14 (the n-tile arrival needs all 98 CTAs resident); first call (compile, workspace)
inside CUDA-graph capture. Compile once per (device, stamps); M is a run-time argument.
Calls on one device must not overlap in time (one workspace and counter set per device).
"""

import hashlib
import os
import sys
import threading
import time
from pathlib import Path

import torch

_DIR = Path(__file__).resolve().parent
_SOURCES = (_DIR / "kernel.py", _DIR / "__init__.py")
TP, H, L, SH, MAX_M, CTAS = 8, 7168, 3584, 896, 64, 98
DEFAULT = {"postfence": False}  # postfence: extra fence after the n-tile poll (A/B knob, see MoeTail)
_LOCK = threading.Lock()
_COMPILED: dict = {}
_WS: dict = {}
_LOGGED: set = set()


def enabled() -> bool:
    return os.environ.get("PERFBOT_EXP_MOETAIL", "1") != "0"


def source_sha256() -> str:
    h = hashlib.sha256()
    for p in _SOURCES:
        h.update(p.name.encode() + b"\0" + p.read_bytes() + b"\n")
    return h.hexdigest()


def _log_once(key, msg: str) -> None:
    if key not in _LOGGED:
        _LOGGED.add(key)
        print(f"PERFBOT-MOE-TAIL {msg}", file=sys.stderr, flush=True)


def max_active_clusters() -> int:
    from cutlass.utils import HardwareInfo

    return int(HardwareInfo().get_max_active_clusters(7))


def compiled(device: torch.device, stamps: bool = False, **cfg):
    """The compiled launch, or None inside capture before the first eager compile. Raises on compile failure."""
    cfg = {**DEFAULT, **cfg}
    key = (device.index, bool(stamps), tuple(sorted(cfg.items())))
    fn = _COMPILED.get(key)
    if fn is not None:
        return fn
    if torch.cuda.is_current_stream_capturing():
        return None
    with _LOCK:
        fn = _COMPILED.get(key)
        if fn is not None:
            return fn
        import cutlass
        import cutlass.cute as cute
        from cuda.bindings.driver import CUstream
        from cutlass.cute.runtime import make_ptr

        from .kernel import MoeTail

        t0 = time.time()
        gm = cute.AddressSpace.gmem
        fake = lambda dt, i: make_ptr(dt, 4096 * (i + 1), gm, assumed_align=16)
        bf, u32, i32, f32, i64 = cutlass.BFloat16, cutlass.Uint32, cutlass.Int32, cutlass.Float32, cutlass.Int64
        fn = cute.compile(MoeTail(stamps=stamps, **cfg), fake(bf, 0), fake(bf, 1), fake(bf, 2), fake(u32, 3), fake(bf, 4),
                          fake(bf, 5), fake(i32, 6), fake(i64, 7), fake(bf, 8), cutlass.Int64(4096), fake(bf, 10),
                          fake(f32, 11), fake(f32, 12), fake(i64, 13), fake(i64, 14), cutlass.Int32(8),
                          cutlass.Int32(0), cutlass.Float32(1e-6), CUstream(torch.cuda.current_stream(device).cuda_stream))
        _COMPILED[key] = fn
        _log_once(("compiled", key), f"compiled sha={source_sha256()[:16]} key={key} seconds={time.time() - t0:.1f}")
        return fn


def workspace(device: torch.device):
    """fp32 split-K partials [7][14][64][128], sumsq [7][14][64], u64 n-tile arrival counters [8] (only grow),
    stamps [98][16] (slot 15 = SM id)."""
    ws = _WS.get(device.index)
    if ws is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        z = lambda n, dt: torch.zeros(n, dtype=dt, device=device)
        ws = {"part": z(CTAS * MAX_M * 128, torch.float32), "sums": z(CTAS * MAX_M, torch.float32),
              "ctr": z(8, torch.int64), "dbg": z(CTAS * 16, torch.int64)}
        _WS[device.index] = ws
    return ws


_FIT: dict = {}


def why_not(op, routed_output, shared_output, rms_weight, up_weight):
    """None when run() serves this call; else the decline reason."""
    if not enabled():
        return "PERFBOT_EXP_MOETAIL=0"
    if not getattr(routed_output, "perfbot_prefilled", False):
        return "routed output not prefilled by moe_stage"
    m = routed_output.expanded_idx_to_permuted_idx.shape[0]
    if not 1 <= m <= MAX_M:
        return f"M={m} outside 1..{MAX_M}"
    c = op.contract
    if (c.tp_size, c.hidden_size, c.latent_size, c.dtype) != (TP, H, L, torch.bfloat16):
        return f"contract {c.tp_size} {c.hidden_size} {c.latent_size} {c.dtype}"
    for name, t, shape in (("shared_output", shared_output, (m, H)), ("rms_weight", rms_weight, (L,)),
                           ("up_weight", up_weight, (H, L))):
        if (tuple(t.shape) != shape or t.dtype != torch.bfloat16 or not t.is_contiguous() or t.data_ptr() % 16
                or t.device != c.device):
            return f"{name} {t.dtype} {tuple(t.shape)}"
    dev = c.device
    fit = _FIT.get(dev.index)
    if fit is None:
        if torch.cuda.is_current_stream_capturing():
            return "first call inside CUDA-graph capture"
        with torch.cuda.device(dev):
            fit = _FIT[dev.index] = max_active_clusters()
    if fit < CTAS // 7:
        return f"max active 7-CTA clusters {fit} < {CTAS // 7}"
    return None


def run(op, routed_output, shared_output, rms_weight, up_weight, stamps: bool = False, **cfg):
    why = why_not(op, routed_output, shared_output, rms_weight, up_weight)
    if why is not None:
        _log_once(("decline", why.split(" ")[0]), f"decline: {why}")
        return None
    import cutlass
    import cutlass.cute as cute
    from cuda.bindings.driver import CUstream
    from cutlass.cute.runtime import make_ptr

    dev = op.contract.device
    fn = compiled(dev, stamps, **cfg)
    ws = workspace(dev)
    if fn is None or ws is None:
        _log_once("capture", "decline: first call inside CUDA-graph capture")
        return None
    m = routed_output.expanded_idx_to_permuted_idx.shape[0]
    co, up = op._collective, op._up_projection
    out = torch.empty((m, H), dtype=torch.bfloat16, device=dev)
    gm = cute.AddressSpace.gmem
    P = lambda t, dt: make_ptr(dt, t.data_ptr(), gm, assumed_align=16)
    bf, u32, i32, f32, i64 = cutlass.BFloat16, cutlass.Uint32, cutlass.Int32, cutlass.Float32, cutlass.Int64
    fn(P(up_weight, bf), P(rms_weight, bf), P(co._routed_workspace, bf), P(co._routed_flags, u32),
       P(shared_output, bf), P(co._shared_workspace, bf), P(co._shared_flags, i32), P(co._shared_peer_ptrs, i64),
       P(up._mailbox, bf), cutlass.Int64(up._mailbox_multicast_ptr), P(out, bf), P(ws["part"], f32),
       P(ws["sums"], f32), P(ws["ctr"], i64), P(ws["dbg"], i64), cutlass.Int32(m), cutlass.Int32(op.rank),
       cutlass.Float32(op.contract.rms_eps), CUstream(torch.cuda.current_stream(dev).cuda_stream))
    _log_once(("run", m), f"run M={m} rank={op.rank} sha={source_sha256()[:16]}")
    return out
