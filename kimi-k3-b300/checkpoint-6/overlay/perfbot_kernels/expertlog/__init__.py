"""EXPERTLOG: distinct routed experts each MoE layer reads per engine step (evidence only, log only).

Coordinator ruling 2026-10-18 (EXPERTLOG): at batch 1 a decode step routes 8 tokens x top-16 = 128 slots over 896
experts; how many distinct experts each layer touches per step (and per mixed step) prices every weight-read model of
the MoE stage. Behind PERFBOT_EXP_EXPERTLOG (default "0"), never in a measured arm or overlay/server_extra_args.

Experts are TP-sharded, not EP: the recipe has no --enable-expert-parallel, every rank holds all 896 experts' slices
(moe_stage: w13 [896, 768, 1792] per rank, e_off = ep_rank * local_num_experts = 0) and routes all tokens, so the
distinct count on rank 0 is the count every rank reads (each its 1/8 slice of every distinct expert).

Two halves:
  device  `record(owner, packed, logits, bias, top_k)`, called from TrtLlmMxfp4ExpertsMonolithic.apply (overlay
          trtllm_mxfp4_moe.py) right after the router carrier split, under the module constant
          _PERFBOT_EXPERTLOG (read once at import; off = the r3-318 apply op for op). It runs inside vLLM's opaque
          moe_forward op, i.e. inside the full / breakable CUDA graph. Capture-safe: torch ops only on static
          buffers allocated on the first (eager, profile-run) call; no .item(), no host read of device data, no
          data-dependent shapes. The step slot and the live-row count come from a device tensor `ctl` the host
          writes before each step, so a replayed graph files each step's counts in that step's row:
            ids    = packed >> 16 (M <= 256, router_topk's packed top-K), else topk(sigmoid(logits) + bias)
            ids    = E where row >= ctl[1] (cudagraph padding rows past the scheduled tokens)
            seen   = 0; seen[ids] = 1; rec[ctl[0], layer] = sum(seen[:E])
  host    `ExpertLogWorker` (--worker-cls perfbot_kernels.expertlog.ExpertLogWorker), rank 0 only: before each
          step's Worker.execute_model it writes (slot, scheduled tokens) into ctl with a non-blocking H2D copy from
          a pinned row on the current stream (the model runner's main stream, v1/worker/gpu/model_runner.py:516-518);
          after PERFBOT_EXP_EXPERTLOG_STEPS steps (default 128) it reads rec on the host (one blocking copy, outside
          the forward, once per period), prints the summary, and resets rec to -1. Other ranks leave ctl at the
          scratch row: same kernels, no host work.

Line (worker stdout -> server.log), one per (kind, batch) per period:
  PERFBOT-EXPERTLOG steps=<n> kind=<decode|mixed> batch=<requests> tokens=<mean live tokens> slots=<tokens*top_k>
  layers=<L> experts=<E> distinct_mean=<x> distinct_p10= distinct_p50= distinct_p90= distinct_min= distinct_max=
  layer_mean_min=<min over layers of the per-layer mean> layer_mean_max= missing=<(step, layer) cells not written>
Samples are the (step, layer) cells of the group; per-step resolution is kept on device and summarized per period.
Lost: steps of a partial last period at shutdown; for M > 256 rows (prefill-sized mixed steps) the ids are
torch.topk of sigmoid + bias, not flashinfer's own selection (ties may differ). Any error disables the log.
"""

import os
import sys

ENV = "PERFBOT_EXP_EXPERTLOG"
STEPS_ENV = "PERFBOT_EXP_EXPERTLOG_STEPS"
TAG = "PERFBOT-EXPERTLOG"
MAX_LAYERS = 128
_STATE: dict = {}
_LAYERS: dict = {}
_LOGGED: set = set()


def enabled() -> bool:
    return os.environ.get(ENV, "0") == "1"


def ring_steps() -> int:
    try:
        return max(1, int(os.environ.get(STEPS_ENV, "") or 128))
    except ValueError:
        return 128


def _log_once(key, msg: str) -> None:
    if key not in _LOGGED:
        _LOGGED.add(key)
        print(f"{TAG} {msg}", file=sys.stderr, flush=True)


def _capturing() -> bool:
    import torch

    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


class State:
    """Device buffers of one device: ctl [2] (slot, live rows), rec [ring + 1, MAX_LAYERS] (row ring = scratch),
    seen [E + 1] (column E = masked rows), host_ctl [ring + 1, 2] (pinned on CUDA)."""

    def __init__(self, device, experts: int, top_k: int, ring: int):
        import torch

        self.experts, self.top_k, self.ring = experts, top_k, ring
        i32 = dict(dtype=torch.int32, device=device)
        self.ctl = torch.tensor([ring, 0], **i32)
        self.rec = torch.full((ring + 1, MAX_LAYERS), -1, **i32)
        self.seen = torch.zeros(experts + 1, **i32)
        self.host_ctl = torch.zeros((ring + 1, 2), dtype=torch.int32, pin_memory=torch.device(device).type == "cuda")

    def set_step(self, slot: int, live: int) -> None:
        """Point the next forward at rec row `slot`, rows >= `live` masked. Stream-ordered, no host wait; a pinned
        row is rewritten only after read() synced the stream."""
        self.host_ctl[slot, 0], self.host_ctl[slot, 1] = slot, live
        self.ctl.copy_(self.host_ctl[slot], non_blocking=True)

    def read(self, n: int, layers: int) -> list:
        """Rows 0..n-1 as lists (blocking copy: waits for the forwards already queued), then rows reset to -1."""
        rows = self.rec[:n, :layers].cpu().tolist()
        self.rec[:n].fill_(-1)
        return rows


def state(device=None):
    return _STATE.get(None if device is None else getattr(device, "index", device))


def record(owner, packed, logits, bias, top_k: int) -> None:
    """rec[ctl[0], layer(owner)] = distinct experts over the live rows of this call. Device ops only; never raises."""
    try:
        import torch

        m, e = logits.shape
        dev = logits.device
        st = _STATE.get(dev.index)
        if st is None:
            if _capturing():
                _log_once("capture", "first call inside CUDA-graph capture: not recorded")
                return
            st = _STATE[dev.index] = State(dev, e, top_k, ring_steps())
            _log_once("on", f"on device={dev} experts={e} top_k={top_k} ring={st.ring} max_layers={MAX_LAYERS}")
        layer = _LAYERS.setdefault(id(owner), len(_LAYERS))
        if layer >= MAX_LAYERS or e != st.experts or m == 0:
            return
        if packed is not None:
            ids = packed >> 16
        else:
            scores = logits.float().sigmoid()
            if bias is not None:
                scores = scores + bias
            ids = torch.topk(scores, top_k, dim=-1).indices.int()
        live = torch.arange(m, dtype=torch.int32, device=dev) < st.ctl[1]
        ids = torch.where(live[:, None], ids, e)
        st.seen.zero_()
        st.seen.index_fill_(0, ids.reshape(-1).long(), 1)
        cnt = st.seen[:e].sum(dtype=torch.int32).view(1)
        st.rec.view(-1).index_copy_(0, st.ctl[:1].long() * MAX_LAYERS + layer, cnt)
    except Exception as exc:  # noqa: BLE001 - a log must never stop the forward
        _log_once("record-error", f"record skipped: {type(exc).__name__}: {exc}")


def step_kind(num_scheduled_tokens: dict, num_spec: int):
    """None for an empty step; "mixed" when any request runs past 1 + K tokens (a prefill chunk); else "decode"."""
    if not num_scheduled_tokens:
        return None
    return "mixed" if max(num_scheduled_tokens.values()) > 1 + num_spec else "decode"


def _pct(v: list, p: float):
    return v[min(len(v) - 1, max(0, -(-len(v) * p // 100) - 1))]  # nearest rank


def summarize(rows: list, meta: list, top_k: int, experts: int) -> list:
    """Lines per (kind, batch) from rec rows [step][layer] and meta [(kind, batch, tokens)] of the same steps."""
    groups: dict = {}
    for row, (kind, batch, tokens) in zip(rows, meta):
        groups.setdefault((kind, batch), []).append((row, tokens))
    lines = []
    for (kind, batch), items in sorted(groups.items()):
        n_layers = len(items[0][0])
        cells = [c for row, _ in items for c in row if c >= 0]
        missing = sum(len(row) for row, _ in items) - len(cells)
        tokens = sum(t for _, t in items) / len(items)
        per_layer = [[row[j] for row, _ in items if row[j] >= 0] for j in range(n_layers)]
        lmeans = [sum(c) / len(c) for c in per_layer if c]
        v = sorted(cells)
        stats = (f"distinct_mean={sum(v) / len(v):.2f} distinct_p10={_pct(v, 10)} distinct_p50={_pct(v, 50)} "
                 f"distinct_p90={_pct(v, 90)} distinct_min={v[0]} distinct_max={v[-1]} "
                 f"layer_mean_min={min(lmeans):.2f} layer_mean_max={max(lmeans):.2f}") if v else "distinct_mean=nan"
        lines.append(f"{TAG} steps={len(items)} kind={kind} batch={batch} tokens={tokens:g} slots={tokens * top_k:g} "
                     f"layers={n_layers} experts={experts} {stats} missing={missing}")
    return lines


class Host:
    """Rank-0 step bookkeeping: one ctl write per step, one read + print per period. `st` is a State-like object."""

    def __init__(self, num_spec: int, out=print):
        self.num_spec, self.out, self.meta, self.on = num_spec, out, [], True

    def on_step(self, num_scheduled_tokens: dict, st) -> None:
        if not self.on:
            return
        try:
            kind = step_kind(num_scheduled_tokens, self.num_spec)
            if kind is None or st is None:
                return  # no forward runs (model_runner.py:1514), or no layer recorded yet
            if len(self.meta) >= st.ring:
                self.flush(st)
            tokens = sum(num_scheduled_tokens.values())
            st.set_step(len(self.meta), tokens)
            self.meta.append((kind, len(num_scheduled_tokens), tokens))
        except Exception as exc:  # noqa: BLE001
            self.on = False
            self.out(f"{TAG} disabled: {type(exc).__name__}: {exc}")

    def flush(self, st) -> None:
        rows = st.read(len(self.meta), min(len(_LAYERS), MAX_LAYERS))
        for line in summarize(rows, self.meta, st.top_k, st.experts):
            self.out(line)
        self.meta = []


def _worker_class():
    """Built lazily: vllm is imported only in the server process (tests import this module without it)."""
    import torch
    from vllm.v1.worker.gpu_worker import Worker

    class ExpertLogWorker(Worker):
        """Worker whose execute_model files each step's per-layer distinct-expert counts (rank 0 prints)."""

        def execute_model(self, scheduler_output):
            if enabled() and int(self.rank) == 0:
                host = self.__dict__.get("_expertlog")
                if host is None:
                    spec = getattr(self.vllm_config, "speculative_config", None)
                    host = self.__dict__["_expertlog"] = Host(int(getattr(spec, "num_speculative_tokens", 0) or 0),
                                                              out=lambda s: print(s, flush=True))
                with torch.inference_mode():  # rec / ctl are inference tensors (made in the profile run)
                    host.on_step(dict(scheduler_output.num_scheduled_tokens or {}), state(torch.cuda.current_device()))
            return super().execute_model(scheduler_output)

    return ExpertLogWorker


def __getattr__(name: str):  # `ExpertLogWorker` resolves on first use (resolve_obj_by_qualname)
    if name == "ExpertLogWorker":
        return _worker_class()
    raise AttributeError(name)
