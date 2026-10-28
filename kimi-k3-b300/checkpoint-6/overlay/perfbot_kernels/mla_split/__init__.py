"""Route A: MLA prefill context chunks with the KV split into S batch entries of one trtllm call (expedition).

The image's context-chunk call (pinned trtllm_ragged.py run_prefill_context_chunk) launches a grid of
ceil(n / 256) x 12 heads x requests CTAs; at short n most of the 148 SMs idle. Here each request's chunk KV is cut
into S contiguous slices, passed as S batch entries of the same trtllm_ragged_attention_deepseek call, and the S
partials are merged with merge_attn_states in the ln LSE base the server merges in.

This module is the one generator: the server (overlay mla.py, switch PERFBOT_EXP_MLASPLIT) and the kernel gate
(perfbot_kernels.mla_mixbench, variant image_split) both run SplitKVPrefillBackend from backend_class().

Switch PERFBOT_EXP_MLASPLIT: "0" (default) leaves the server path unchanged (server_backend_class returns the
selected class, install_builder_hook does nothing). "1": a step takes route A only if route() holds for its
prefill rows (every row has n > PERFBOT_MLAQ_T, read by mla_absorb.threshold(), and context <=
PERFBOT_MLASPLIT_MAX_CTX), DCP = 8 with interleave 1, the layer has the gate's head count and dims, the query is
fp8, and line M did not mark the step (prefill.perfbot_mlaq). Outside that, every call goes to the image call.
Torch-only at import (vllm is imported lazily); CPU tests in test_cpu.py.
"""

import os
import types

import torch

ENV = "PERFBOT_EXP_MLASPLIT"
MAX_CTX_ENV = "PERFBOT_MLASPLIT_MAX_CTX"
MAX_CTX_DEFAULT = 512000  # = the largest gate context (a domain is what the gate covers)

# The gate's layer (perfbot_kernels.mla_mixbench: served K3 MLA under DCP 8). The router acts only on this layer.
GATE_DCP, GATE_INTERLEAVE = 8, 1
GATE_HEADS, GATE_NOPE, GATE_ROPE, GATE_V = 12, 128, 64, 128

NUM_SM = 148  # B300 SMs (coordinator's default rule)
STEP_Q = 256  # mStepQ of the SM103 HQk192HV128 context cubins: grid X = ceil(n / 256)
SPLIT_ALIGN = 128  # slice boundaries on the Kv128 tile where the length allows


def enabled() -> bool:
    return os.environ.get(ENV, "0") == "1"


def threshold() -> int:
    """T: the same reader line M uses (mla_absorb.threshold, env PERFBOT_MLAQ_T)."""
    from perfbot_kernels import mla_absorb

    return mla_absorb.threshold()


def max_context() -> int:
    return int(os.environ.get(MAX_CTX_ENV, str(MAX_CTX_DEFAULT)))


def route(q_lens, ctx_lens, t: int | None = None, max_ctx: int | None = None) -> bool:
    """The domain, per step: every prefill row has n > T and cached context <= MAX_CTX (any lower context)."""
    t = threshold() if t is None else t
    max_ctx = max_context() if max_ctx is None else max_ctx
    return bool(q_lens) and min(q_lens) > t and max(ctx_lens) <= max_ctx


def layer_in_domain(num_heads, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, dcp_world_size) -> bool:
    return (num_heads, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, dcp_world_size) == (
        GATE_HEADS, GATE_NOPE, GATE_ROPE, GATE_V, GATE_DCP)


# --------------------------------------------------------------------------------------------------------- plan
def default_splits(max_q: int, num_sm: int = NUM_SM, heads: int = GATE_HEADS, step_q: int = STEP_Q) -> int:
    """S = max(1, round(num_sm / (heads x ceil(max_q / step_q)))): about one wave of CTAs (MB-02 sweep: best S)."""
    return max(1, round(num_sm / (heads * -(-max_q // step_q))))


def ctas(n: int, requests: int, splits: int, heads: int = GATE_HEADS, step_q: int = STEP_Q) -> int:
    """Grid size of one context-chunk call: ceil(n / 256) x 12 heads x (requests x S) (no multi-CTA-KV mode)."""
    return -(-n // step_q) * heads * requests * splits


def slice_bounds(kv_len: int, splits: int, align: int = SPLIT_ALIGN) -> list[int]:
    """S + 1 boundaries of S contiguous non-empty slices of [0, kv_len), on `align` where that keeps all non-empty."""
    b = [min(kv_len, round(kv_len * j / splits / align) * align) for j in range(splits + 1)]
    b[0], b[-1] = 0, kv_len
    if any(b[j + 1] <= b[j] for j in range(splits)):
        b = [kv_len * j // splits for j in range(splits + 1)]
    return b


def split_plan(q_lens: list[int], kv_lens: list[int], splits: int, align: int = SPLIT_ALIGN):
    """Batch entries for one context chunk with each request's KV cut into S slices (CPU, no device).

    Entries are request-major (r, s) so the KV slices tile k in order (ragged layout). Entry (r, s) has request r's
    q rows and KV slice s. `q_index` (None for one request: q.repeat(S) is already in entry order) gathers the
    repeated q; `perm` (None for one request) reorders the kernel's entry-major output rows to s-major [S, total_q]
    so the S partials of every row are merged block by block. q_index / perm are int32 CPU tensors built per entry
    (<= requests x S aranges), not per row."""
    S = max(1, min(splits, min(kv_lens)))
    q_off = [0]
    for n in q_lens:
        q_off.append(q_off[-1] + n)
    cu_q, cu_kv, seq = [0], [0], []
    for n, k in zip(q_lens, kv_lens):
        b = slice_bounds(k, S, align)
        for s in range(S):
            cu_q.append(cu_q[-1] + n)
            seq.append(b[s + 1] - b[s])
            cu_kv.append(cu_kv[-1] + seq[-1])
    one = len(q_lens) == 1
    q_index = perm = None
    if not one:
        i32 = torch.int32
        q_index = torch.cat([torch.arange(q_off[r], q_off[r + 1], dtype=i32).repeat(S) for r in range(len(q_lens))])
        perm = torch.cat([torch.arange(cu_q[r * S + s], cu_q[r * S + s] + n, dtype=i32)
                          for s in range(S) for r, n in enumerate(q_lens)])
    return types.SimpleNamespace(S=S, batch=len(q_lens) * S, cu_q=cu_q, cu_kv=cu_kv, seq_lens=seq, max_kv=max(seq),
                                 q_index=q_index, perm=perm)


def device_plan(plan, device):
    """The plan's int32 fields in ONE host-to-device copy (pinned, non_blocking: no sync; the copy keeps the pinned
    block alive until it lands): cu_q_t, cu_kv_t, seq_lens_t, q_index_t, perm_t are views of one device tensor."""
    parts = [torch.tensor(plan.cu_q, dtype=torch.int32), torch.tensor(plan.cu_kv, dtype=torch.int32),
             torch.tensor(plan.seq_lens, dtype=torch.int32)]
    if plan.q_index is not None:
        parts += [plan.q_index, plan.perm]
    flat = torch.cat(parts)
    if torch.device(device).type == "cuda":
        flat = flat.pin_memory().to(device, non_blocking=True)
    views, o = [], 0
    for p in parts:
        views.append(flat[o : o + p.numel()])
        o += p.numel()
    plan.cu_q_t, plan.cu_kv_t, plan.seq_lens_t = views[:3]
    plan.q_index_t, plan.perm_t = (views[3], views[4]) if plan.q_index is not None else (None, None)
    return plan


# ------------------------------------------------------------------------------------------------- step routing
def attach(prefill, q_lens: list[int], ctx_lens: list[int], active: bool) -> None:
    """Per-step CPU info the backend reads: prefill rows' new-token and context lengths, and the route decision."""
    prefill.perfbot_split = types.SimpleNamespace(q_lens=list(q_lens), ctx_lens=list(ctx_lens), active=bool(active))


_hooked = False


def install_builder_hook() -> None:
    """Switch on: wrap MLACommonMetadataBuilder.build to attach() each step's prefill lengths and route decision.
    CPU values only (query_start_loc_cpu, seq_lens_cpu_upper_bound: exact for prefill rows), as mla_absorb's hook."""
    global _hooked
    if _hooked or not enabled():
        return
    from vllm.model_executor.layers.attention import mla_attention as M

    orig = M.MLACommonMetadataBuilder.build

    def build(self, *args, **kwargs):
        md = orig(self, *args, **kwargs)
        p = md.prefill
        if p is None:
            return md
        cm = kwargs["common_attn_metadata"] if "common_attn_metadata" in kwargs else args[1]
        nd, nr = md.num_decodes, md.num_reqs
        qsl = cm.query_start_loc_cpu
        q_lens = (qsl[nd + 1 : nr + 1] - qsl[nd:nr]).tolist()
        seq = cm.seq_lens_cpu_upper_bound[nd:nr].tolist()
        ctx_lens = [s - n for s, n in zip(seq, q_lens)]
        layout_ok = (getattr(self, "dcp_world_size", 1), getattr(self, "cp_kv_cache_interleave_size", 1)) == (
            GATE_DCP, GATE_INTERLEAVE)
        attach(p, q_lens, ctx_lens, layout_ok and route(q_lens, ctx_lens))
        return md

    M.MLACommonMetadataBuilder.build = build
    _hooked = True
    print(f"PERFBOT_EXP_MLASPLIT: on T={threshold()} MAX_CTX={max_context()}", flush=True)


def server_backend_class(cls, num_heads, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, dcp_world_size):
    """The prefill backend class a layer builds: `cls` unchanged unless the switch is on, `cls` is the image's
    TrtllmRaggedPrefillBackend and the layer is the gate's; then SplitKVPrefillBackend (clone() keeps the class)."""
    if not enabled():
        return cls
    split = backend_class()
    image_cls = split.__mro__[1]  # TrtllmRaggedPrefillBackend
    if cls is image_cls and layer_in_domain(num_heads, qk_nope_head_dim, qk_rope_head_dim, v_head_dim,
                                                   dcp_world_size):
        return split
    print(f"PERFBOT_EXP_MLASPLIT: kept {cls.__name__} (heads {num_heads}, dims {qk_nope_head_dim}/{qk_rope_head_dim}/"
          f"{v_head_dim}, dcp {dcp_world_size})", flush=True)
    return cls


# ------------------------------------------------------------------------------------------------------ backend
_counts = {"layer_calls": 0}
_rank0 = None


def _is_rank0() -> bool:
    global _rank0
    if _rank0 is None:
        try:
            from vllm.distributed.parallel_state import get_dcp_group

            _rank0 = get_dcp_group().rank_in_group == 0
        except Exception:  # noqa: BLE001  (bench: no vllm DCP group)
            import torch.distributed as dist

            _rank0 = not dist.is_initialized() or dist.get_rank() == 0
    return _rank0


_cls = None


def backend_class():
    """SplitKVPrefillBackend (one class object per process, so clone() and the server's swap agree)."""
    global _cls
    if _cls is not None:
        return _cls
    from vllm.v1.attention.backends.mla.prefill.trtllm_ragged import TrtllmRaggedPrefillBackend

    class SplitKVPrefillBackend(TrtllmRaggedPrefillBackend):
        """Route A: one trtllm ragged call with batch = requests x S (KV slices), then S-way LSE merge.

        The call is the one in TrtllmRaggedPrefillBackend.run_prefill_context_chunk (pinned trtllm_ragged.py:164)
        with only seq_lens / cum_seq_lens_kv / cum_seq_lens_q / batch_size / q changed (seq_lens on the device, as
        the image's new-token call passes it). The kernel's log2 LSE goes to ln with the same log2_lse_to_ln; the S
        partials merge in place into block 0 with merge_attn_states, every lse contiguous [H, rows]. A chunk with
        S = 1, or a step outside the route, is exactly the image call (super()).

        Plans: built on the first context-chunk call of a step (all chunks at once, CPU lengths only: the step's
        attach() info and chunk.seq_lens, a pinned CPU tensor), reused by every layer of the step, dropped at the
        next prepare_metadata. `splits_override` / `splits_fn` let the bench pick S; the server uses the formula."""

        splits_override: int | None = None
        splits_fn = staticmethod(default_splits)

        def prepare_metadata(self, prefill_metadata) -> None:
            super().prepare_metadata(prefill_metadata)
            self._split_step = None  # (splits_override, {chunk index: plan or None})

        def _step_plans(self, q):
            step = getattr(self, "_split_step", None)
            if step is not None and step[0] == self.splits_override:
                return step[1]
            md = self._prefill_metadata
            info = getattr(md, "perfbot_split", None)
            plans = {}
            if (info is not None and info.active and getattr(md, "perfbot_mlaq", None) is None
                    and q.dtype == torch.float8_e4m3fn):
                for chunk in md.chunked_context.chunks:
                    S = self.splits_override or self.splits_fn(chunk.max_query_len)
                    q_lens = info.q_lens[chunk.request_slice]
                    kv_lens = chunk.seq_lens.tolist()  # pinned CPU tensor (build_mla_chunked_context_metadata)
                    plan = split_plan(q_lens, kv_lens, S) if S > 1 else None
                    if plan is not None and plan.S > 1:
                        assert sum(q_lens) == chunk.token_slice.stop - chunk.token_slice.start, (q_lens, chunk)
                        plan.q_lens, plan.ctx_lens = q_lens, info.ctx_lens[chunk.request_slice]
                        plans[chunk.index] = device_plan(plan, q.device)
            self._split_step = (self.splits_override, plans)
            return plans

        def run_prefill_context_chunk(self, chunk, q, k, v, out=None):
            plans = self._step_plans(q)
            plan = plans.get(chunk.index)
            if plan is None:
                return super().run_prefill_context_chunk(chunk, q, k, v, out)
            from flashinfer.prefill import trtllm_ragged_attention_deepseek
            from vllm.v1.attention.backends.utils import log2_lse_to_ln
            from vllm.v1.attention.ops.merge_attn_states import merge_attn_states

            if chunk.index == min(plans):
                self._log(plan)
            S, nq, heads = plan.S, q.shape[0], q.shape[1]
            if plan.q_index_t is None:
                q_rep = q.repeat(S, 1, 1)
            else:
                q_rep = q.view(torch.uint8).index_select(0, plan.q_index_t).view(q.dtype)
            o = torch.empty(S * nq, heads, v.shape[2], device=q.device, dtype=self._prefill_metadata.output_dtype)
            o, lse = trtllm_ragged_attention_deepseek(
                query=q_rep, key=k, value=v, workspace_buffer=self._workspace_buffer, seq_lens=plan.seq_lens_t,
                max_q_len=chunk.max_query_len, max_kv_len=plan.max_kv, bmm1_scale=self.scale, bmm2_scale=1.0,
                o_sf_scale=1.0, batch_size=plan.batch, window_left=-1, cum_seq_lens_q=plan.cu_q_t,
                cum_seq_lens_kv=plan.cu_kv_t, enable_pdl=False, is_causal=False, return_lse=True, out=o)
            lse = log2_lse_to_ln(lse)  # [S * nq, H], ln
            if plan.perm_t is not None:
                o, lse = o.index_select(0, plan.perm_t), lse.index_select(0, plan.perm_t)
            o = o.view(S, nq, heads, -1)
            lse = lse.view(S, nq, heads).transpose(1, 2).contiguous()  # [S, H, nq]
            for s in range(1, S):
                merge_attn_states(output=o[0], prefix_output=o[0], prefix_lse=lse[0], suffix_output=o[s],
                                  suffix_lse=lse[s], output_lse=lse[0])
            if out is not None:
                out.copy_(o[0])
                return out, lse[0]
            return o[0], lse[0]

        @staticmethod
        def _log(plan):
            """Once per layer call that splits (its first split chunk): the e2e's proof that the path ran."""
            _counts["layer_calls"] += 1
            k = _counts["layer_calls"]
            if not _is_rank0():
                return
            if k == 1:
                print(f"PERFBOT_EXP_MLASPLIT: first split prefill (n {plan.q_lens} ctx {plan.ctx_lens} S {plan.S}) "
                      f"in layer call 1 of the process", flush=True)
            if k >= 64 and k & (k - 1) == 0:
                print(f"PERFBOT_EXP_MLASPLIT: split layer calls {k}", flush=True)

    _cls = SplitKVPrefillBackend
    return _cls
