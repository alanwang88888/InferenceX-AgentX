"""ROUTECAP: the routed expert ids (and, for a few steps, the experts' input rows) of every specialized decode
MoE layer, per step, rank 0 (R-UP-CAP, P-04 card; evidence only, never in a measured arm).

Probe point: the overlay copy of specialized_decode/cutedsl/moe/prepared.py calls `record(layer, ids, weights,
latent)` right after `self.front.route(x)` (both paths of PreparedMoE.run). `ids` [b, 16] int32 is written inside
the FULL CUDA graph, so the host never sees it per step. Off (PERFBOT_ROUTECAP unset): `record` returns before any
torch op, so every captured graph is the image's graph.

On (PERFBOT_ROUTECAP=1), rank 0 only, torch ops on static buffers, capture-safe:
  slot = cnt[layer] % STEPS           (a device counter: a replayed graph files each step in its own slot)
  ids_ring[layer, slot, :b] = ids;  rows b..T-1 = -1;  nb[layer, slot] = b;  cnt[layer] += 1
  weights_ring likewise (bf16); latent_ring[layer, slot % ACT_STEPS, :b] = latent (the experts' input rows)
Buffers are allocated by prepare() from PreparedMoE.__init__ (eager, before capture); record() never allocates and
never raises, so a missing prepare() costs data, not the server.
`b` is the padded graph batch: rows of padding tokens carry ids too. A uniform decode step at conc k with q tokens
per request fills b = k * q rows exactly; the reader keeps slots with nb == k * q.

Drain: a daemon thread copies the rings to the host on its own stream every PERFBOT_ROUTECAP_PERIOD_S seconds
(default 30) and at exit, and writes `<PERFBOT_ROUTECAP_OUT>.pt` atomically (tmp + rename). A copy can race a
replay: the reader drops the two slots after each layer's counter in a snapshot (cnt is copied first).
vLLM captures in "global" error mode, so a copy from another thread during a capture would invalidate that capture on
rank 0 only. The drain thread first sets its own capture mode to relaxed (driver cuThreadExchangeStreamCaptureMode);
if that fails, no thread runs and only the exit drain writes the file.
"""

import atexit
import os
import threading

ENV = "PERFBOT_ROUTECAP"
OUT_ENV = "PERFBOT_ROUTECAP_OUT"
STEPS = int(os.environ.get("PERFBOT_ROUTECAP_STEPS", "") or 512)
ACT_STEPS = int(os.environ.get("PERFBOT_ROUTECAP_ACT_STEPS", "") or 4)
PERIOD_S = float(os.environ.get("PERFBOT_ROUTECAP_PERIOD_S", "") or 30)
T = 64  # PREPARED_MAX: the largest specialized decode batch (tokens)
MAX_LAYERS = 128
ON = os.environ.get(ENV, "0") == "1"
OUT = os.environ.get(OUT_ENV, "")
_S: dict = {}


def _rank0() -> bool:
    import torch.distributed as dist
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


def prepare(device, k, weights_dtype, latent_dim, latent_dtype) -> None:
    """Allocate the rings eagerly. Called from PreparedMoE.__init__ (overlay prepared.py), which runs before any
    FULL-graph capture: the specialized MoE's first run() IS a capture (R-UP-CAP exit 75: a lazy first-call
    allocation met the capture guard, rank 0 raised and the other ranks waited 30 min on gloo). Idempotent."""
    if not ON or _S:
        return
    import torch
    if not _rank0() or (torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()):
        _S["off"] = True
        if _rank0():
            print("PERFBOT-ROUTECAP off: prepare() inside a CUDA graph capture", flush=True)
        return
    dev = torch.device(device)
    _S.update(
        layers={}, k=k,
        cnt=torch.zeros(MAX_LAYERS, dtype=torch.int64, device=dev),
        nb=torch.zeros(MAX_LAYERS, STEPS, dtype=torch.int32, device=dev),
        ids=torch.full((MAX_LAYERS, STEPS, T, k), -1, dtype=torch.int32, device=dev),
        weights=torch.zeros((MAX_LAYERS, STEPS, T, k), dtype=weights_dtype, device=dev),
        latent=torch.zeros((MAX_LAYERS, ACT_STEPS, T, latent_dim), dtype=latent_dtype, device=dev),
    )
    print(f"PERFBOT-ROUTECAP on: rank 0, {STEPS} steps x {MAX_LAYERS} layers, out {OUT or '(none)'}", flush=True)
    if OUT:
        _S["out"] = OUT
        if dev.type == "cuda":
            threading.Thread(target=_drain_loop, name="routecap-drain", daemon=True).start()
        atexit.register(drain)


def record(layer, ids, weights, latent) -> None:
    """Never allocates and never raises: without prepare() (or off / not rank 0) it returns before any torch op."""
    if not ON or not _S or _S.get("off"):
        return
    import torch
    li = _S["layers"].setdefault(layer, len(_S["layers"]))
    b = ids.shape[0]
    pad = (0, 0, 0, T - b)
    cnt = _S["cnt"][li : li + 1]
    slot = torch.remainder(cnt, STEPS)
    _S["ids"][li].index_copy_(0, slot, torch.nn.functional.pad(ids, pad, value=-1).unsqueeze(0))
    _S["weights"][li].index_copy_(0, slot, torch.nn.functional.pad(weights, pad).unsqueeze(0))
    _S["nb"][li].index_fill_(0, slot, b)
    _S["latent"][li].index_copy_(0, torch.remainder(cnt, ACT_STEPS), torch.nn.functional.pad(latent, pad).unsqueeze(0))
    cnt.add_(1)


def snapshot() -> dict:
    """Host copies, counter first (the reader trusts slots older than cnt - 2)."""
    import torch
    stream = _S.setdefault("stream", torch.cuda.Stream(device=_S["cnt"].device) if _S["cnt"].is_cuda else None)
    with torch.cuda.stream(stream) if stream is not None else _null():
        snap = {name: _S[name].to("cpu") for name in ("cnt", "nb", "ids", "weights", "latent")}
    if stream is not None:
        stream.synchronize()
    snap.update(layers=dict(_S["layers"]), steps=STEPS, act_steps=ACT_STEPS, rows=T)
    return snap


def drain() -> None:
    if not _S or _S.get("off") or "out" not in _S:
        return
    import torch
    tmp = _S["out"] + ".tmp.pt"
    os.makedirs(os.path.dirname(tmp) or ".", exist_ok=True)
    torch.save(snapshot(), tmp)
    os.replace(tmp, _S["out"] + ".pt")


def _relax_capture_mode() -> bool:
    """This thread's CUDA calls become legal while another thread captures in global mode (CU_STREAM_CAPTURE_MODE_RELAXED = 2)."""
    import ctypes
    try:
        mode = ctypes.c_int(2)
        return ctypes.CDLL("libcuda.so.1").cuThreadExchangeStreamCaptureMode(ctypes.byref(mode)) == 0
    except OSError:
        return False


def _drain_loop() -> None:
    import time
    if not _relax_capture_mode():
        print("PERFBOT-ROUTECAP no periodic drain: relaxed capture mode unavailable; exit drain only", flush=True)
        return
    while True:
        time.sleep(PERIOD_S)
        try:
            drain()
        except Exception as e:  # evidence only: a failed drain never stops the server
            print(f"PERFBOT-ROUTECAP drain failed: {e!r}", flush=True)


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
