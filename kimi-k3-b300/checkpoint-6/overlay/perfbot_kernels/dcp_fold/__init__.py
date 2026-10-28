"""Direct-DCP epoch + signal fold (expedition line D; docs/expedition/ANATOMY-next.md section 2). Kernels in kernel.py.

Two entry points, called from the overlay vllm/v1/attention/ops/dcp.py (MLADCPManager), each returning None to
decline (the caller then runs the image op on the same workspace):
  q_gather(ws, query)                    ws: DirectDCPQGatherWorkspace. Replaces increment_epoch + q-gather (2 -> 1 launch).
  combine(ws, out, lse, base_e, seq_lens, query_start_loc)
                                         ws: DirectDCPA2AWorkspace. Replaces increment_epoch + dispatch + signal +
                                         combine (4 -> 2 launches).
Same symmetric buffers, signal slots, epoch tensors and values as the image ops (image @65b7662
csrc/libtorch_stable/attention/dcp_utils), so image and candidate calls interleave, also rank by rank. The q-gather
arrives on the workspace's own completion counter; the dispatch on one local int32 per ubatch (ws._perfbot_dcpfold_ctr,
returns to 0 after every call). Outputs are bit-equal to the image ops (same data path, same fp32 op sequence).
Declines: PERFBOT_EXP_DCPFOLD unset or 0 (default off until line D lands); world size != 8; shapes, dtypes, strides or
alignment the kernels do not cover (bf16 output, fp32 LSE, head_dim % 256 == 0); T outside 1..max_num_tokens; the first
call (compile, counter) inside CUDA-graph capture. Compile once per (device, kind, cfg); T is a run-time argument.
Call site: overlay vllm/v1/attention/ops/dcp.py = image module at 3696c772 (sha256 b45ca817ffad5179ce6668023a4a6168e92c
d4d7a1ba0b981d83d869137c1e98) + _perfbot_dcp_fold() and one hook in each of MLADCPManager._direct_workspace_combine and
_direct_workspace_query_gather (mla.py unchanged: it calls dcp_manager.query_gather / .combine).
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
ENV = "PERFBOT_EXP_DCPFOLD"
W = 8
# qg_blocks 0 = image grid min(T, W). tpb 4 + fence1: D-fold-01 alts (tpb2 T48 +4.6 us, fence1 D42 +4.8 us/pair)
# pdl: PERFBOT_DCPFOLD_PDL "0" (default: PDL off, above the bar in D-fold-02/03) | "1"; read once at import, and part
# of the compile key, so both variants can coexist. The e2e decides it (README.md).
DEFAULT = {"pdl": os.environ.get("PERFBOT_DCPFOLD_PDL", "0") != "0", "fence1": True, "qg_blocks": 0, "tpb": 4, "wpb": 4}
_LOCK = threading.Lock()
_COMPILED: dict = {}
_LOGGED: set = set()


def enabled() -> bool:
    return os.environ.get(ENV, "0") != "0"  # default off until line D lands


def source_sha256() -> str:
    h = hashlib.sha256()
    for p in _SOURCES:
        h.update(p.name.encode() + b"\0" + p.read_bytes() + b"\n")
    return h.hexdigest()


def _log_once(key, msg: str) -> None:
    if key not in _LOGGED:
        _LOGGED.add(key)
        print(f"PERFBOT-DCPFOLD {msg}", file=sys.stderr, flush=True)


def _compiled(kind: str, device: torch.device, cfg: dict, dh: int = 0):
    """The compiled launch of kernel `kind` (q, d, c), or None inside capture before the first eager compile."""
    key = (device.index, kind, dh, tuple(sorted(cfg.items())))
    fn = _COMPILED.get(key)
    if fn is not None or torch.cuda.is_current_stream_capturing():
        return fn
    with _LOCK:
        fn = _COMPILED.get(key)
        if fn is not None:
            return fn
        import cutlass
        import cutlass.cute as cute
        from cuda.bindings.driver import CUstream

        from . import kernel as k

        i64, i32 = cutlass.Int64(0), cutlass.Int32(1)
        stream = CUstream(torch.cuda.current_stream(device).cuda_stream)
        t0 = time.time()
        if kind == "q":
            fn = cute.compile(k.QGather(pdl=cfg["pdl"], fence1=cfg["fence1"]), *[i64] * 8, *[i32] * 4, stream)
        elif kind == "d":
            fn = cute.compile(k.Dispatch(pdl=cfg["pdl"], fence1=cfg["fence1"], tpb=cfg["tpb"]), *[i64] * 6, i32,
                              *[i64] * 5, *[i32] * 6, stream)
        else:
            fn = cute.compile(k.Combine(dh=dh, wpb=cfg["wpb"], pdl=cfg["pdl"]), *[i64] * 5, *[i32] * 5, stream)
        _COMPILED[key] = fn
        _log_once(("compiled", key), f"compiled sha={source_sha256()[:16]} key={key} seconds={time.time() - t0:.1f}")
        return fn


def _ubatch() -> int:
    from vllm.v1.worker.ubatching import dbo_current_ubatch_id

    return dbo_current_ubatch_id()


def _al(*xs) -> bool:
    return all(int(x) % 16 == 0 for x in xs)


def why_not_q(ws, q: torch.Tensor):
    """None when q_gather() serves this call; else the decline reason."""
    if not enabled():
        return f"{ENV}=0"
    if ws.world_size != W:
        return f"world size {ws.world_size}"
    if q.dim() != 3 or q.shape[1] != ws.heads_per_rank or q.shape[2] != ws.head_dim:
        return f"query shape {tuple(q.shape)}"
    t, h, d = q.shape
    es = q.element_size()
    if not 1 <= t <= ws.max_num_tokens:
        return f"T={t} outside 1..{ws.max_num_tokens}"
    if q.dtype != ws.final_query.dtype or q.device != ws.final_query.device:
        return f"query {q.dtype} {q.device}"
    if q.stride(2) != 1 or q.stride(1) != d or q.stride(0) < h * d:
        return f"query strides {q.stride()}"
    if not _al(q.data_ptr(), q.stride(0) * es, h * d * es, ws.final_query.data_ptr()):
        return "query alignment"
    if not all(p for pair in ws.multicast_ptrs for p in pair):
        return "no multicast pointers"
    return None


def q_gather(ws, q: torch.Tensor, **cfg):
    why = why_not_q(ws, q)
    if why is not None:
        _log_once(("decline-q", why.split(" ")[0]), f"decline q_gather: {why}")
        return None
    cfg = {**DEFAULT, **cfg}
    fn = _compiled("q", q.device, cfg)
    if fn is None:
        _log_once("capture-q", "decline q_gather: first call inside CUDA-graph capture")
        return None
    import cutlass
    from cuda.bindings.driver import CUstream

    ub = _ubatch()
    t, h, d = q.shape
    es = q.element_size()
    g = W * h
    out = torch.as_strided(ws.final_query[ub], size=(t, g, d), stride=(g * d, d, 1))
    q_mc, sig_mc = ws.multicast_ptrs[ub]
    blocks = cfg["qg_blocks"] or min(t, W)
    I64, I32 = cutlass.Int64, cutlass.Int32
    fn(I64(q.data_ptr()), I64(q.stride(0) * es), I64(q_mc), I64(g * d * es), I64(sig_mc),
       I64(ws.received_signal[ub].data_ptr()), I64(ws.epoch[ub:ub + 1].data_ptr()),
       I64(ws.completion[ub].data_ptr()), I32(ws.rank), I32(t), I32(h * d * es // 16), I32(min(blocks, t)),
       CUstream(torch.cuda.current_stream(q.device).cuda_stream))
    _log_once(("run-q", t), f"q_gather T={t} H={h} rank={ws.rank} sha={source_sha256()[:16]}")
    return out


def _counter(ws):
    """Local arrival counter of the dispatch, one int32 per ubatch (always returns to 0)."""
    ctr = getattr(ws, "_perfbot_dcpfold_ctr", None)
    if ctr is None and not torch.cuda.is_current_stream_capturing():
        ctr = ws._perfbot_dcpfold_ctr = torch.zeros(ws.num_ubatches, dtype=torch.int32, device=ws.epoch.device)
    return ctr


def why_not_c(ws, out: torch.Tensor, lse: torch.Tensor, seq_lens, qsl):
    if not enabled():
        return f"{ENV}=0"
    if ws.world_size != W:
        return f"world size {ws.world_size}"
    if out.dim() != 3 or lse.dim() != 2:
        return f"dims {out.dim()} {lse.dim()}"
    t, nh, d = out.shape
    if out.dtype != torch.bfloat16 or ws.received_output.dtype != torch.bfloat16 or lse.dtype != torch.float32:
        return f"dtypes {out.dtype} {lse.dtype}"
    if nh != W * ws.heads_per_rank or d != ws.head_dim or d % 256 or tuple(lse.shape) != (t, nh):
        return f"shapes {tuple(out.shape)} {tuple(lse.shape)}"
    if not 1 <= t <= ws.max_num_tokens:
        return f"T={t} outside 1..{ws.max_num_tokens}"
    if out.stride(2) != 1 or out.stride(1) != d or out.stride(0) < nh * d or lse.stride(1) != 1 or lse.stride(0) < nh:
        return f"strides {out.stride()} {lse.stride()}"
    if not _al(out.data_ptr(), out.stride(0) * 2):
        return "output alignment"
    if (seq_lens is None) != (qsl is None):
        return "seq_lens without query_start_loc"
    if seq_lens is not None:
        for name, x in (("seq_lens", seq_lens), ("query_start_loc", qsl)):
            if x.dim() != 1 or x.stride(0) != 1 or x.dtype != torch.int32 or not x.is_cuda:
                return f"{name} {x.dtype} {tuple(x.shape)}"
        if seq_lens.numel() < 1 or qsl.numel() != seq_lens.numel() + 1:
            return "query_start_loc length"
    if _counter(ws) is None:
        return "first call inside CUDA-graph capture"
    return None


def combine(ws, out: torch.Tensor, lse: torch.Tensor, is_lse_base_on_e: bool, seq_lens=None, query_start_loc=None,
            **cfg):
    """The image direct_dcp_a2a_lse_reduce on `ws`: -> [T, heads_per_rank, head_dim] bf16, or None (decline)."""
    why = why_not_c(ws, out, lse, seq_lens, query_start_loc)
    if why is not None:
        _log_once(("decline-c", why.split(" ")[0]), f"decline combine: {why}")
        return None
    cfg = {**DEFAULT, **cfg}
    t, nh, d = out.shape
    fd, fc = _compiled("d", out.device, cfg), _compiled("c", out.device, cfg, d)
    if fd is None or fc is None:
        _log_once("capture-c", "decline combine: first call inside CUDA-graph capture")
        return None
    import cutlass
    from cuda.bindings.driver import CUstream

    ub = _ubatch()
    h = ws.heads_per_rank
    res = out.new_empty((t, h, d))
    I64, I32 = cutlass.Int64, cutlass.Int32
    stream = CUstream(torch.cuda.current_stream(out.device).cuda_stream)
    epoch = I64(ws.epoch[ub:ub + 1].data_ptr())
    seq = I64(seq_lens.data_ptr()) if seq_lens is not None else I64(0)
    qsl = I64(query_start_loc.data_ptr()) if seq_lens is not None else I64(0)
    nseq = I32(seq_lens.numel() if seq_lens is not None else 0)
    tpb = cfg["tpb"]
    fd(I64(out.data_ptr()), I64(out.stride(0) * 2), I64(lse.data_ptr()), I64(lse.stride(0)), seq, qsl, nseq,
       I64(ws.peer_output_ptrs[ub].data_ptr()), I64(ws.peer_lse_ptrs[ub].data_ptr()),
       I64(ws.peer_signal_ptrs[ub].data_ptr()), epoch, I64(_counter(ws)[ub:ub + 1].data_ptr()), I32(ws.rank),
       I32(t), I32(ws.max_num_tokens), I32(h), I32(d), I32(W * ((t + tpb - 1) // tpb)), stream)
    wpb = cfg["wpb"]
    fc(I64(ws.received_output[ub].data_ptr()), I64(ws.received_lse[ub].data_ptr()),
       I64(ws.received_signal[ub].data_ptr()), epoch, I64(res.data_ptr()), I32(ws.max_num_tokens), I32(h),
       I32(int(bool(is_lse_base_on_e))), I32(t * h), I32((t * h + wpb - 1) // wpb), stream)
    _log_once(("run-c", t), f"combine T={t} H={h} rank={ws.rank} sha={source_sha256()[:16]}")
    return res
