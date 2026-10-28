"""Evidence logs on the engine scheduler (log only): REQLOG, STEPLOG and MAMBA-TAINT. Each has its own switch.

Wiring: the arm passes `--scheduler-cls perfbot_kernels.reqlog.scheduler` (PERFBOT_SERVER_EXTRA_ARGS) and sets one or
more of PERFBOT_EXP_REQLOG=1, PERFBOT_EXP_STEPLOG=1, PERFBOT_EXP_MAMBA_TAINT=1. vLLM resolves the qualname and calls it
with the scheduler's keyword arguments (v1/engine/core.py:149-170 in the image source); `scheduler` builds the class
vLLM would pick itself (config/scheduler.py:170-178: AsyncScheduler when async_scheduling, else Scheduler), wrapped
when any switch is "1". The wrappers record, then call the base methods unchanged: scheduling, caching, sampling and
outputs are the base class's. With every switch off the base class is returned as it is. Any bookkeeping error
disables that one log; it never reaches the engine. All lines go to engine-core stdout -> server.log.
Line numbers below are vLLM 49cb2bbf.

REQLOG (PERFBOT_EXP_REQLOG). Coordinator ruling 2026-10-18 (R-split): per-request accept length, joined to
per-request ITL and the nsys step classification, decides whether the agentic p90 tail is acceptance variance or slow
steps. The aiperf records carry no accept length, and the image's spec-decode metrics are aggregates.
Line, at the first update_from_output after which the request is finished:
  PERFBOT-REQLOG id=<request_id> prompt=<prompt tokens> out=<output tokens> steps=<steps scheduled>
  decode_steps=<steps with <= 1 + K tokens> drafts=<draft tokens scheduled> accepted=<draft tokens accepted>
  shared_prefill=<decode steps of this request in which another request prefilled> batch_sum=<sum of step batch
  sizes over its decode steps> t_first=<unix s, first decode step> t_last=<unix s, last step> reason=<finish reason>
Accept length (tokens per decode step) = (accepted + decode steps with drafts) / decode_steps; mean step time of the
request's decode = (t_last - t_first) / decode_steps (wall between update calls, includes host time).

STEPLOG (PERFBOT_EXP_STEPLOG). One line per engine step that schedules at least one prefill row (n > 1 + K tokens,
K = num_speculative_tokens), at that step's update_from_output:
  PERFBOT-STEPLOG t=<unix s at update_from_output> dt_ms=<ms since the previous update_from_output call, any step;
  -1 on the first> sched_dt_ms=<ms from this output's schedule() return to its update_from_output> batch=<rows>
  decode_since_last=<steps with rows but no prefill row, updated since the previous STEPLOG line>
  rows=<rid:n:ctx_before:prompt_len[,...]> (prefill rows only; split each from the right on ':')
ctx_before is read when schedule() returns: with async scheduling the next step can be scheduled before this step's
update, and schedule() already advanced num_computed_tokens by n (scheduler.py:1633 -> :1707). So ctx_before =
num_computed_tokens - n at that moment, kept per output object until its update_from_output.

STEPDEC (PERFBOT_EXP_STEPDEC). Every engine step with rows, packed 200 records per line (one flushed write per line):
  PERFBOT-STEPDEC t0=<unix s of the line's first update_from_output> recs=<t_ms:dt_ms:batch:prefill_rows:ctx_min:ctx_max:q/r:pn:pnmax;...>
  t_ms = ms after t0 at this step's update_from_output; dt_ms = ms since the previous update_from_output (-1 on the
  first) = the host-side step period under async scheduling; batch = rows; prefill_rows = rows with n > 1 + K;
  ctx_min / ctx_max = min / max ctx_before over the decode rows (n <= 1 + K; -1 when none), ctx_before as STEPLOG's.
  q/r (item 15 F-F, a vocab-shard target sampler): r = rows that reach the sampler = batch (model_runner.py:1311-1312
  maps every scheduled request into idx_mapping_np); q = rows the sampler state marks plain (sample/states.py:43-64):
  top_k <= 0 or >= vocab, top_p >= 1, min_p == 0, logprobs None (0 still computes logprobs), and not watermarked
  (watermark_config set and watermarking not False; sampling_params.py:309). q = 0 on every step under
  rejection_sample_method "block" (rejection_sampler.py:175). Penalties, logit_bias, bad_words, temperature do not count.
  pn (item 04, C32 coordinator 2026-10-24 16:56Z): prefill tokens of the step = sum of n over its prefill rows (0 on
  a pure decode step); pnmax (kda-spec, same ruling 17:02Z): max n over its prefill rows (0 on a pure decode step).
  Both appended last so the six leading fields and q/r keep their positions.
  The PX proxy arm's step-wall source (docs/expedition/GUARDRAILS.md "PX proxy arm"); a partial last line is not
  written (the window ends long before the server stops).

MAMBA-TAINT (PERFBOT_EXP_MAMBA_TAINT). GPU evidence for vllm#60888's mid-block hash defect. A prefill chunk that starts
mid-block (s % B != 0) runs past the next boundary nb in one step when the backend exports the checkpoint
(scheduler.py:546, :571-572 drop the boundary stop). The block at column s // B holds the state at s: the CoW copy of
a partial hit (single_type_kv_cache_manager.py:322-325, :389-398), or the previous chunk's running state. The step
writes only the running column cdiv(end, B) - 1 > s // B and the checkpoint column, which is > s // B
(kv_cache_interface.py:1138-1142). Yet MambaManager.cache_blocks (single_type_kv_cache_manager.py:1966-1981, base
:477) hashes that block at its block end nb. A later hit on nb restores the state at s. Read-only hooks, installed in
the scheduler's __init__ on the instances it owns: self.kv_cache_manager (scheduler.py:328) .allocate_slots
(kv_cache_manager.py:371) and .get_computed_blocks (:264, called at scheduler.py:596, :1092), and the first MambaManager
in .coordinator (kv_cache_manager.py:167) .single_type_managers (kv_cache_coordinator.py:157), whose cache_blocks the
coordinator calls per group (kv_cache_coordinator.py:369-377). Lines:
  PERFBOT-MAMBA-TAINT rid=<> block_id=<> hash=<hex8> holds_state_at=<s> keyed_at=<nb>
    when cache_blocks, inside allocate_slots of a prefill row (n > 1 + K) whose chunk starts at s (num_computed +
    hit + external tokens) with s % B != 0 and caches past nb, gives the block at column s // B a new hash whose
    position (block_hash_num_tokens) != s.
  PERFBOT-MAMBA-TAINT-HIT rid=<> hash=<hex8> hit_tokens=<> keyed_at=<> holds_state_at=<>
    when a lookup's last Mamba hit block is that same block with that same hash. Every lookup counts: a request
    retried while blocked, or re-looked-up after preemption, counts again (dedupe by rid offline if needed).
  PERFBOT-MAMBA-TAINT-SUM lookups=<> hits=<> tainted_published=<> tainted_hits=<>   (every 100 lookups)
The number of record is tainted_hits. All Mamba groups share one layout, so only the first is watched.
"""

import inspect
import os
import time

ENV = "PERFBOT_EXP_REQLOG"
TAG = "PERFBOT-REQLOG"
STEP_ENV = "PERFBOT_EXP_STEPLOG"
STEP_TAG = "PERFBOT-STEPLOG"
TAINT_ENV = "PERFBOT_EXP_MAMBA_TAINT"
TAINT_TAG = "PERFBOT-MAMBA-TAINT"
DEC_ENV = "PERFBOT_EXP_STEPDEC"
DEC_TAG = "PERFBOT-STEPDEC"
DEC_EVERY = 200
ENVS = (ENV, STEP_ENV, TAINT_ENV, DEC_ENV)


def enabled(env: str = ENV) -> bool:
    return os.environ.get(env, "0") == "1"


def _disabled(tag, e):
    print(f"{tag} disabled: {type(e).__name__}: {e}", flush=True)


def _num_spec(sched) -> int:
    spec_cfg = getattr(sched.vllm_config, "speculative_config", None)
    return getattr(spec_cfg, "num_speculative_tokens", 0) or 0


def _counts(state, scheduler_output, model_runner_output, num_spec, num_sampled, now):
    """Update per-request counters for one engine step; returns the ids scheduled in it."""
    sched = scheduler_output.num_scheduled_tokens
    batch = len(sched)
    prefilling = {r for r, n in sched.items() if n > 1 + num_spec}
    spec = scheduler_output.scheduled_spec_decode_tokens or {}
    index = getattr(model_runner_output, "req_id_to_index", None) or {}
    sampled = getattr(model_runner_output, "sampled_token_ids", None) or []
    for rid, n in sched.items():
        st = state.get(rid)
        if st is None:
            st = state[rid] = dict(steps=0, decode_steps=0, drafts=0, accepted=0, shared_prefill=0, batch_sum=0,
                                   t_first=None, t_last=now)
        st["steps"] += 1
        st["t_last"] = now
        if rid in prefilling:
            continue
        st["decode_steps"] += 1
        st["batch_sum"] += batch
        if st["t_first"] is None:
            st["t_first"] = now
        if prefilling:
            st["shared_prefill"] += 1
        drafts = spec.get(rid)
        i = index.get(rid)
        if drafts and i is not None and i < len(sampled):
            st["drafts"] += len(drafts)
            st["accepted"] += max(len(sampled[i]) - num_sampled, 0)
    return list(sched)


def _line(rid, st, request) -> str:
    reason = request.get_finished_reason() if request is not None else None
    prompt = getattr(request, "num_prompt_tokens", -1) if request is not None else -1
    out = getattr(request, "num_output_tokens", -1) if request is not None else -1
    t_first = st["t_first"] if st["t_first"] is not None else st["t_last"]
    return (f"{TAG} id={rid} prompt={prompt} out={out} steps={st['steps']} decode_steps={st['decode_steps']} "
            f"drafts={st['drafts']} accepted={st['accepted']} shared_prefill={st['shared_prefill']} "
            f"batch_sum={st['batch_sum']} t_first={t_first:.6f} t_last={st['t_last']:.6f} reason={reason}")


def _step_scheduled(st, out, requests, num_spec, now):
    """At schedule() return: keep (output, time, prefill rows with ctx_before) for this output's update."""
    rows = [(rid, n, requests[rid].num_computed_tokens - n, requests[rid].num_prompt_tokens)
            for rid, n in out.num_scheduled_tokens.items() if n > 1 + num_spec]
    if rows:
        if len(st["pending"]) > 64:  # outputs never updated (aborts): drop, do not grow
            st["pending"].clear()
        st["pending"][id(out)] = (out, now, rows)


def _step_line(st, so, now):
    """At update_from_output: the STEPLOG line, or None for a step without a prefill row."""
    last, st["last"] = st["last"], now
    entry = st["pending"].pop(id(so), None)
    if entry is None or entry[0] is not so:
        st["decode"] += bool(so.num_scheduled_tokens)
        return None
    _, t_sched, rows = entry
    decode, st["decode"] = st["decode"], 0
    dt = (now - last) * 1e3 if last is not None else -1.0
    return (f"{STEP_TAG} t={now:.6f} dt_ms={dt:.3f} sched_dt_ms={(now - t_sched) * 1e3:.3f} "
            f"batch={len(so.num_scheduled_tokens)} decode_since_last={decode} "
            f"rows={','.join(f'{r}:{n}:{c}:{p}' for r, n, c, p in rows)}")


def _dec_engine(cfg) -> dict:
    """The engine-wide terms of q: vocab size, watermark configured, block verification (q = 0)."""
    spec = getattr(cfg, "speculative_config", None)
    return dict(vocab=cfg.model_config.get_vocab_size(), watermark=getattr(cfg, "watermark_config", None) is not None,
                block=getattr(spec, "rejection_sample_method", None) == "block")


def _plain(sp, eng) -> bool:
    """The sampler state of this request needs no top-k, top-p, min-p, logprobs or watermark."""
    return ((sp.top_k <= 0 or sp.top_k >= eng["vocab"]) and sp.top_p >= 1.0 and sp.min_p == 0.0
            and sp.logprobs is None and not (eng["watermark"] and sp.watermarking is not False))


def _dec_scheduled(st, out, requests, num_spec):
    """At schedule() return: (batch, prefill rows, decode ctx_before min / max, q) for this output's update."""
    sched = out.num_scheduled_tokens
    if not sched:
        return
    ctx = [requests[rid].num_computed_tokens - n for rid, n in sched.items() if n <= 1 + num_spec]
    eng = st["engine"]
    q = 0 if eng["block"] else sum(_plain(requests[rid].sampling_params, eng) for rid in sched)
    pre = [n for n in sched.values() if n > 1 + num_spec]  # PERFBOT P-07
    if len(st["pending"]) > 64:  # outputs never updated (aborts): drop, do not grow
        st["pending"].clear()
    st["pending"][id(out)] = (out, len(sched), len(sched) - len(ctx), min(ctx, default=-1), max(ctx, default=-1), q, sum(pre), max(pre, default=0))  # PERFBOT P-07


def _dec_record(st, so, now):
    """At update_from_output: buffer this step's record; return a full line every DEC_EVERY records, else None."""
    last, st["last"] = st["last"], now
    entry = st["pending"].pop(id(so), None)
    if entry is None or entry[0] is not so:
        return None
    if st["t0"] is None:
        st["t0"] = now
    dt = (now - last) * 1e3 if last is not None else -1.0
    st["recs"].append(f"{(now - st['t0']) * 1e3:.3f}:{dt:.3f}:{entry[1]}:{entry[2]}:{entry[3]}:{entry[4]}:{entry[5]}/{entry[1]}:{entry[6]}:{entry[7]}")  # PERFBOT P-07
    if len(st["recs"]) < DEC_EVERY:
        return None
    line = f"{DEC_TAG} t0={st['t0']:.6f} recs={';'.join(st['recs'])}"
    st["recs"], st["t0"] = [], None
    return line


class MambaTaint:
    """Read-only hooks on one KVCacheManager and its first MambaManager (see the module docstring)."""

    EVERY = 100  # R-EVST: a point has fewer than 1,000 lookups (578 REQLOG lines in 22 min), so 1,000 never printed

    def __init__(self, kvm, num_spec):
        from vllm.v1.core.single_type_kv_cache_manager import MambaManager
        self.mgr = next(m for m in kvm.coordinator.single_type_managers if isinstance(m, MambaManager))
        self.num_spec, self.on = num_spec, True
        self.ctx = {}  # rid -> chunk start s, only inside allocate_slots of a prefill row
        self.tainted = {}  # block hash -> (block_id, s, keyed_at)
        self.lookups = self.hits = self.published = self.tainted_hits = 0
        self._alloc, self._lookup, self._cache = kvm.allocate_slots, kvm.get_computed_blocks, self.mgr.cache_blocks
        self._alloc_sig = inspect.signature(self._alloc)
        kvm.allocate_slots, kvm.get_computed_blocks = self.allocate_slots, self.get_computed_blocks
        self.mgr.cache_blocks = self.cache_blocks

    def _off(self, e):
        self.on = False
        _disabled(TAINT_TAG, e)

    def allocate_slots(self, *args, **kwargs):
        rid = None
        if self.on:
            try:
                a = self._alloc_sig.bind(*args, **kwargs).arguments
                if a["num_new_tokens"] > 1 + self.num_spec:
                    req = a["request"]
                    s = (req.num_computed_tokens + a.get("num_new_computed_tokens", 0)
                         + a.get("num_external_computed_tokens", 0))
                    self.ctx[req.request_id] = s
                    rid = req.request_id  # set only once the entry exists
            except Exception as e:  # noqa: BLE001 - a log must never stop the engine
                self._off(e)
        try:
            return self._alloc(*args, **kwargs)
        finally:
            if rid is not None:
                self.ctx.pop(rid, None)

    def cache_blocks(self, request, num_tokens, *args, **kwargs):
        probe = None
        if self.on:
            try:
                s, B = self.ctx.get(request.request_id), self.mgr.block_size
                if s is not None and s % B and num_tokens > (s // B + 1) * B:
                    blocks = self.mgr.req_to_blocks.get(request.request_id, [])
                    if s // B < len(blocks):
                        probe = (s, blocks[s // B], blocks[s // B].block_hash)
            except Exception as e:  # noqa: BLE001
                self._off(e)
        out = self._cache(request, num_tokens, *args, **kwargs)
        if probe is not None:
            try:
                s, block, before = probe
                h = block.block_hash
                keyed = block.block_hash_num_tokens
                if h is not None and h != before and not block.is_null and keyed != s:
                    self.tainted[h] = (block.block_id, s, keyed)
                    self.published += 1
                    print(f"{TAINT_TAG} rid={request.request_id} block_id={block.block_id} hash={bytes(h)[:4].hex()} "
                          f"holds_state_at={s} keyed_at={keyed}", flush=True)
            except Exception as e:  # noqa: BLE001
                self._off(e)
        return out

    def get_computed_blocks(self, request, *args, **kwargs):
        out = self._lookup(request, *args, **kwargs)
        if self.on:
            try:
                blocks, hit = out[0], out[1]
                self.lookups += 1
                if hit:
                    self.hits += 1
                    group = blocks.blocks[self.mgr.kv_cache_group_id]
                    block = group[-1] if group else None
                    t = self.tainted.get(block.block_hash) if block is not None else None
                    if t and t[0] == block.block_id:
                        self.tainted_hits += 1
                        print(f"{TAINT_TAG}-HIT rid={request.request_id} hash={bytes(block.block_hash)[:4].hex()} "
                              f"hit_tokens={hit} keyed_at={t[2]} holds_state_at={t[1]}", flush=True)
                if self.lookups % self.EVERY == 0:
                    print(f"{TAINT_TAG}-SUM lookups={self.lookups} hits={self.hits} "
                          f"tainted_published={self.published} tainted_hits={self.tainted_hits}", flush=True)
            except Exception as e:  # noqa: BLE001
                self._off(e)
        return out


def wrap(base):
    """`base` with schedule / update_from_output recording for the logs whose switch is on."""

    class ReqLogScheduler(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._reqlog_on, self._steplog_on = enabled(ENV), enabled(STEP_ENV)
            self._steplog = dict(pending={}, last=None, decode=0)
            self._stepdec_on = enabled(DEC_ENV)
            self._stepdec = dict(pending={}, last=None, t0=None, recs=[])
            if self._stepdec_on:
                try:
                    self._stepdec["engine"] = _dec_engine(self.vllm_config)
                except Exception as e:  # noqa: BLE001
                    self._stepdec_on = False
                    _disabled(DEC_TAG, e)
            self._taint = None
            if enabled(TAINT_ENV):
                try:
                    self._taint = MambaTaint(self.kv_cache_manager, _num_spec(self))
                except Exception as e:  # noqa: BLE001
                    _disabled(TAINT_TAG, e)

        def schedule(self, *args, **kwargs):
            out = super().schedule(*args, **kwargs)
            if self._steplog_on:
                try:
                    _step_scheduled(self._steplog, out, self.requests, _num_spec(self), time.time())
                except Exception as e:  # noqa: BLE001
                    self._steplog_on = False
                    _disabled(STEP_TAG, e)
            if self._stepdec_on:
                try:
                    _dec_scheduled(self._stepdec, out, self.requests, _num_spec(self))
                except Exception as e:  # noqa: BLE001
                    self._stepdec_on = False
                    _disabled(DEC_TAG, e)
            return out

        def update_from_output(self, scheduler_output, model_runner_output):
            now = time.time()
            if self._steplog_on:
                try:
                    line = _step_line(self._steplog, scheduler_output, now)
                    if line:
                        print(line, flush=True)
                except Exception as e:  # noqa: BLE001
                    self._steplog_on = False
                    _disabled(STEP_TAG, e)
            if self._stepdec_on:
                try:
                    line = _dec_record(self._stepdec, scheduler_output, now)
                    if line:
                        print(line, flush=True)
                except Exception as e:  # noqa: BLE001
                    self._stepdec_on = False
                    _disabled(DEC_TAG, e)
            ids, held = None, {}
            if self._reqlog_on:
                try:
                    if not hasattr(self, "_reqlog"):
                        self._reqlog = {}
                    ids = _counts(self._reqlog, scheduler_output, model_runner_output, _num_spec(self),
                                  getattr(self, "num_sampled_tokens_per_step", 1), now)
                    held = {rid: self.requests.get(rid) for rid in ids}  # the base frees finished requests below
                except Exception as e:  # noqa: BLE001 - a log must never stop the engine
                    self._reqlog_on = False
                    _disabled(TAG, e)
            out = super().update_from_output(scheduler_output, model_runner_output)
            if ids:
                try:
                    for rid in ids:
                        req = self.requests.get(rid)
                        if req is None or req.is_finished():
                            st = self._reqlog.pop(rid, None)
                            if st is not None:
                                print(_line(rid, st, req if req is not None else held.get(rid)), flush=True)
                except Exception as e:  # noqa: BLE001
                    self._reqlog_on = False
                    _disabled(TAG, e)
            return out

    ReqLogScheduler.__name__ = ReqLogScheduler.__qualname__ = f"ReqLog{base.__name__}"
    return ReqLogScheduler


def stepdec_scheduler(**kwargs):
    """`--scheduler-cls perfbot_kernels.reqlog.stepdec_scheduler` (overlay/server_extra_args, E14): scheduler() with
    STEPDEC on in every serve arm, so a regime changes only the client."""
    os.environ[DEC_ENV] = "1"
    return scheduler(**kwargs)


def scheduler(**kwargs):
    """`--scheduler-cls perfbot_kernels.reqlog.scheduler`: the class vLLM would pick, wrapped when any switch is on."""
    cfg = kwargs["vllm_config"].scheduler_config
    if cfg.async_scheduling:
        from vllm.v1.core.sched.async_scheduler import AsyncScheduler as base
    else:
        from vllm.v1.core.sched.scheduler import Scheduler as base
    on = {e: enabled(e) for e in ENVS}
    cls = wrap(base) if any(on.values()) else base
    print(f"{TAG} scheduler={cls.__name__} async_scheduling={cfg.async_scheduling} enabled={on[ENV]} "
          f"steplog={on[STEP_ENV]} mamba_taint={on[TAINT_ENV]} stepdec={on[DEC_ENV]}", flush=True)
    return cls(**kwargs)
