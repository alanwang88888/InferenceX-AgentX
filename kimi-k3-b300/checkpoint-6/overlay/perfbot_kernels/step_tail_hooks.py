"""Engine hooks of P-11 lever D (units C and E) for the overlay files (item-02, from lane Q b05c50e patches/).

The overlay rule (driver/tests/test_overlay_base.py) allows only added, tagged lines in an overlay/vllm file. A fused
kernel replaces image launches that sit in the middle of image functions, so the image calls are skipped by
one-step tokens instead of edits:

- a hook (draft_prep / the expand_idx_mapping wrapper) runs the fused kernel, then `mark`s one token per image
  launch it replaced, keyed by the image call's own output buffer (kind, data_ptr) or object id;
- the image call site is wrapped (`skip_once`, installed by a tagged line at the module that owns the name) or has
  a tagged early return; it `take`s its token: present = the fused kernel already wrote that buffer this step, so
  it returns the image's return value without a launch; absent = the image function runs.

Every hook path is gated by PERFBOT_EXP_GLUE (perfbot_kernels.glue: C, E). Each `begin` clears the stale tokens
of its unit, so a token never outlives the step that made it. If a fused kernel declines by itself it runs the
image functions in image order (prep.py / prologue.py `_image_path`), so the skips stay correct.
"""

import functools

import torch

from perfbot_kernels.glue import on

_tokens: set = set()


def mark(*keys):
    _tokens.update(keys)


def take(key) -> bool:
    if key in _tokens:
        _tokens.discard(key)
        return True
    return False


def clear(unit):
    for key in [k for k in _tokens if k[0] == unit]:
        _tokens.discard(key)


def skip_once(key_of, result_of):
    """Wrap an image function: when `key_of(args)` holds a token, return `result_of(args)` with no launch."""

    def wrap(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if _tokens and take(key_of(*args, **kwargs)):
                return result_of(*args, **kwargs)
            return fn(*args, **kwargs)

        wrapper.perfbot_p11_image = fn
        return wrapper

    return wrap


def _ptr(t):
    return t.data_ptr()


# ---- unit E: next-step prologue (17 launches -> step_prologue + the #10 image pre-copy) -------------------------

_E = {}  # world and shapes of the current step (set by prologue_begin, used by the expand wrapper)
_HOST_PAD = {}


def host_pad(n):
    """CPU bool scratch for prepare_inputs' two is_padding fills on the hook path (no launch; step_prologue writes the
    GPU buffer, #3)."""
    if _HOST_PAD.get("t") is None or _HOST_PAD["t"].numel() < n:
        _HOST_PAD["t"] = torch.empty(max(n, 1024), dtype=torch.bool)
    return _HOST_PAD["t"][:n]


def _kernel_table_groups(attn_groups, mamba_group_ids):
    """The one non-mamba KV group whose attention groups map kernel block tables (utils.py:380-395), and those
    groups; (None, []) if there is not exactly one."""
    found = [(gid, [g for g in groups if g.kernel_block_size is not None])
             for gid, groups in enumerate(attn_groups) if gid not in mamba_group_ids]
    found = [(gid, groups) for gid, groups in found if groups]
    return found[0] if len(found) == 1 else (None, [])


def _mamba_inputs(state, kv_cache_config, attn_groups, num_reqs_padded, num_tokens):
    """The mamba, KDA and MLA keys of the step_prologue world (lane Q mamba_hybrid.py.diff
    perfbot_prologue_inputs), or None outside the envelope."""
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

    ctx = getattr(state, "_mamba_ctx", None)
    if not getattr(state, "_align_mode", False) or state.recoverssm is not None or ctx is None \
            or not ctx.is_initialized:
        return None
    mamba_group_ids, mamba_spec = state._get_mamba_group_info(kv_cache_config)
    kda = []
    for gid in mamba_group_ids:
        builders = [g.get_metadata_builder(0) for g in attn_groups[gid]]
        builders = [b for b in builders if hasattr(b, "mamba_aligned_state_indices")]
        if len(builders) != 1:
            return None
        b = builders[0]
        # The KDA staging branch (kda_metadata.py:624-631), one full build per group, no RecoverSSM.
        if (b.supports_update_block_table or getattr(b, "use_recoverssm", True) or not b.use_full_cuda_graph
                or not b.use_spec_decode or num_tokens > b.decode_cudagraph_max_bs
                or num_reqs_padded > b.spec_state_indices_tensor.shape[0]):
            return None
        kda.append(b)
    mla_gid, kt = _kernel_table_groups(attn_groups, mamba_group_ids)
    if mla_gid is None or any(
            g.kernel_block_stride != kt[0].kernel_block_stride
            or g.kv_cache_spec.block_size // g.kernel_block_size != kt[0].kernel_block_offsets.numel()
            or g.kernel_block_table.shape != kt[0].kernel_block_table.shape
            or g.kernel_block_table.stride() != kt[0].kernel_block_table.stride() for g in kt):
        return None
    return dict(
        mamba_block_size=mamba_spec.block_size, kernel_block_stride=kt[0].kernel_block_stride,
        mamba_group_ids=list(mamba_group_ids), mla_group_id=mla_gid, conv_state_dim_first=is_conv_state_dim_first(),
        kda_spec_state_indices=[b.spec_state_indices_tensor for b in kda],
        kda_spec_query_start_loc=[b.spec_query_start_loc for b in kda],
        kda_num_accepted=[b.num_accepted_tokens for b in kda],
        kernel_block_tables=[g.kernel_block_table for g in kt], kernel_block_offsets=kt[0].kernel_block_offsets,
        mamba_state_idx=state._mamba_state_idx_gpu, mamba_src_col=state._mamba_src_col_gpu,
        mamba_src_off=state._mamba_src_off_gpu, num_accepted_gpu=state.num_accepted_tokens_gpu,
        aligned_state_indices=ctx.aligned_state_indices, mamba_state_base_addrs=ctx.state_base_addrs,
        mamba_state_block_strides=ctx.state_block_strides, mamba_state_elem_sizes=ctx.state_elem_sizes,
        mamba_state_inner_sizes=ctx.state_inner_sizes, mamba_state_conv_widths=ctx.state_conv_widths,
        mamba_state_group_indices=ctx.state_group_indices, mamba_state_dim_row_count=ctx.state_dim_row_count,
        mamba_state_dim_row_stride=ctx.state_dim_row_stride, mamba_block_table_ptrs=ctx.block_table_ptrs,
        mamba_aligned_block_table_ptrs=ctx.aligned_index_block_table_ptrs,
        mamba_block_table_stride=ctx.block_table_stride_req,
        # carried for the mamba prepare_attn hook (not step_prologue keys)
        _kda_builders=kda, _kt_groups=kt,
    )


def prologue_begin(runner, scheduler_output, batch_req_state, batch_desc) -> bool:
    """Host conditions of unit E (lane Q model_runner.py.diff _step_prologue_world). True: this step takes the hook
    path; the expand_idx_mapping wrapper then runs step_prologue."""
    from vllm import envs
    from vllm.config.compilation import CUDAGraphMode

    clear("E")
    _E.clear()
    runner.model_state.perfbot_prologue_na = None
    if not on("E"):
        return False
    num_tokens = batch_req_state.num_tokens
    if not (not batch_req_state.has_prefill and batch_desc.cg_mode == CUDAGraphMode.FULL
            and batch_desc.num_reqs is not None and batch_desc.num_ubatches == 1
            and scheduler_output.scheduled_spec_decode_tokens and runner.adaptive_verification is None
            and runner.model_state.num_new_sampled_tokens_per_step == 1 and runner.num_speculative_steps == __import__("perfbot_kernels.k5q", fromlist=["k5q"]).K  # PERFBOT K5 item 4 (land-8: 7)
            and runner.decode_query_len == __import__("perfbot_kernels.k5q", fromlist=["k5q"]).Q and envs.VLLM_MOE_SKIP_PADDING and runner.dcp_size > 1  # PERFBOT K5 item 4 (land-8: 8)
            and runner.pcp_manager is None and not runner.lora_config and runner.fast_prefill is None
            and runner.model_config.rswa_window is None
            and bool((batch_req_state.num_scheduled_tokens <= runner.decode_query_len).all())):
        return False
    world = _mamba_inputs(runner.model_state, runner.kv_cache_config, runner.attn_groups, batch_desc.num_reqs,
                          num_tokens)
    if world is None:
        return False
    bt, rs, ib = runner.block_tables, runner.req_states, runner.input_buffers
    world.update(
        has_prefill=False, dcp_size=runner.dcp_size, dcp_rank=runner.dcp_rank, cp_interleave=runner.cp_interleave,
        num_speculative_steps=runner.num_speculative_steps, decode_query_len=runner.decode_query_len,
        query_start_loc=ib.query_start_loc, idx_mapping_prev=None, draft_tokens_prev=None,
        num_computed=rs.num_computed_tokens.gpu, last_sampled=rs.last_sampled_tokens, prefill_len=rs.prefill_len.gpu,
        req_draft_tokens=rs.draft_tokens, all_token_ids=rs.all_token_ids.gpu,
        next_prefill_tokens=rs.next_prefill_tokens, slot_block_tables=[b.gpu for b in bt.block_tables],
        num_blocks=bt.num_blocks.gpu, input_ids=ib.input_ids, positions=ib.positions, is_padding=ib.is_padding,
        seq_lens=ib.seq_lens, slot_mappings=bt.slot_mappings, input_block_tables=bt.input_block_tables,
        dcp_local_seq_lens=ib.dcp_local_seq_lens, block_table_ptrs=bt.block_table_ptrs,
        block_table_strides=bt.block_table_strides, block_sizes=bt.block_sizes_tensor,
        slot_mapping_enabled=bt.slot_mapping_enabled, dcp_sharded=bt.dcp_sharded,
        input_block_table_ptrs=bt.input_block_table_ptrs,
    )
    num_reqs = len(batch_req_state.req_ids)
    _E.update(world=world, runner=runner, R=num_reqs, Rp=batch_desc.num_reqs or num_reqs, T=num_tokens,
              Tp=max(num_tokens, batch_desc.num_tokens))
    return True


def prologue_active() -> bool:
    return "world" in _E


def _expand_run(idx_mapping, total_num_logits, cu_num_logits, decode_query_len):
    """In image order this is the first launch after H1-H3: run step_prologue here (#3-#9, #11-#20, then #10)."""
    from perfbot_kernels.step_tail import step_prologue

    world, runner = _E.pop("world"), _E.pop("runner")
    kda, kt = world.pop("_kda_builders"), world.pop("_kt_groups")
    eim, elp, li, na = step_prologue(**world, R=_E["R"], Rp=_E["Rp"], T=_E["T"], Tp=_E["Tp"], L=total_num_logits,
                                     idx_mapping=idx_mapping, cu_num_logits=cu_num_logits)
    runner.model_state.perfbot_prologue_na = na
    _E.update(logits_indices=li, kda=kda, kt=kt, mla_group_id=world["mla_group_id"])
    ib = runner.input_buffers
    mark(("E", "pos"), ("E", "combine"), ("E", "attn"), ("E", "dcp", _ptr(ib.dcp_local_seq_lens)))
    return eim, elp


def wrap_expand(fn):
    """expand_idx_mapping: on the hook path, the call that runs step_prologue."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if _E.get("world") is not None:
            return _expand_run(*args, **kwargs)
        return fn(*args, **kwargs)

    wrapper.perfbot_p11_image = fn
    return wrapper


wrap_pos = skip_once(lambda *a, **k: ("E", "pos"), lambda *a, **k: None)
wrap_combine = skip_once(lambda *a, **k: ("E", "combine"), lambda *a, **k: _E["logits_indices"])


def attn_views(runner, input_batch):
    """model_runner.prepare_attn after step_prologue (#7, #8): the views block_table.py:172 and :218 return."""
    if not take(("E", "attn")):
        return None
    bt = runner.block_tables
    return (tuple(t[: input_batch.num_reqs_after_padding] for t in bt.input_block_tables),
            bt.slot_mappings[:, : input_batch.num_tokens_after_padding])


def wrap_dcp(fn):
    """cp_utils.prepare_dcp_local_seq_lens: one wrapper for both callers (model runner #11, drafter launch 4)."""

    @functools.wraps(fn)
    def wrapper(dcp_local_seq_lens, *args, **kwargs):
        if _tokens:
            p = _ptr(dcp_local_seq_lens)
            if take(("E", "dcp", p)) or take(("C", "dcp", p)):
                return dcp_local_seq_lens[: kwargs.get("num_reqs_padded")]
        return fn(dcp_local_seq_lens, *args, **kwargs)

    wrapper.perfbot_p11_image = fn
    return wrapper


def mamba_hook(state, input_batch, attn_groups, kv_cache_config, num_reqs):
    """Called from the mamba prepare_attn hook copy: #16-#18 tokens (one per KDA staged buffer) and the #19-#20
    kernel-table token. Returns the int32[num_reqs] num_accepted that step_prologue wrote, or None."""
    na = getattr(state, "perfbot_prologue_na", None)
    state.perfbot_prologue_na = None
    if na is None:
        return None
    for b in _E.get("kda", ()):
        mark(("E", "stage", _ptr(b.spec_state_indices_tensor), num_reqs, input_batch.num_reqs))
    return na


def kernel_table_src(block_tables):
    mla = _E.get("mla_group_id")
    for group in _E.get("kt", ()):
        group.perfbot_kernel_table_src = block_tables[mla]


def wrap_stage(fn):
    """kda_metadata.stage_spec_decode_metadata (#16-#18): skipped for the staged buffer step_prologue wrote."""

    @functools.wraps(fn)
    def wrapper(*args, **kw):
        if _tokens:
            staged = kw["staged_state_indices"]
            if take(("E", "stage", _ptr(staged), staged.shape[0], kw["num_spec_decodes"])):
                return None
        return fn(*args, **kw)

    wrapper.perfbot_p11_image = fn
    return wrapper


# ---- unit C: draft prep (4 launches -> draft_prep) ---------------------------------------------------------------

def draft_prep(spec, input_batch, aux_hidden_states, num_sampled, num_rejected, last_sampled, next_prefill_tokens,
               temperature, seeds, dp_sync_state, dummy_run, is_profile):
    """DFlash/DSpark propose, before the per-group prepare_dflash_inputs loop (lane Q dflash_speculator.py.diff).
    With dp_size 1 and no DP sync, dispatch_cg_and_sync_dp reads host values only (a pure lookup,
    cudagraph_utils.py:523-553), so it can run here as well as at its image site."""
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp

    from perfbot_kernels.step_tail import draft_prep as fused

    clear("C")
    if not on("C") or spec.dp_size != 1 or dp_sync_state is not None:
        return False
    num_reqs = input_batch.num_reqs
    batch_desc, _ = dispatch_cg_and_sync_dp(
        spec.query_cudagraph_manager, num_reqs, num_reqs * spec.num_query_per_req,
        uniform_token_count=spec.num_query_per_req, dp_size=spec.dp_size, dp_rank=spec.dp_rank,
        need_eager=is_profile, dp_sync_state=None)
    if not (batch_desc.cg_mode == CUDAGraphMode.FULL and batch_desc.num_reqs is not None and not dummy_run
            and spec.sample_from_anchor and len(spec.draft_kv_cache_group_ids) == 1
            and spec._aux_staging is not None and bool(aux_hidden_states) and spec._num_rejected is None
            and not spec.supports_decode_metadata_update and spec.dcp_size > 1 and spec.pcp_manager is None
            and spec.eplb_state is None
            and input_batch.num_tokens <= spec._num_graph_context_tokens(batch_desc.num_reqs)):
        return False
    gid = spec.draft_kv_cache_group_id
    bt, ib = spec.block_tables, spec.input_buffers
    fused(
        target_positions=input_batch.positions, target_qsl=input_batch.query_start_loc,
        idx_mapping=input_batch.idx_mapping[:num_reqs], num_sampled=num_sampled, num_rejected=num_rejected,
        last_sampled=last_sampled, next_prefill_tokens=next_prefill_tokens, temperature_in=temperature,
        seeds_in=seeds, block_table=bt.input_block_tables[gid], aux_hidden=aux_hidden_states,
        input_ids=ib.input_ids, positions=ib.positions, is_padding=ib.is_padding,
        query_start_loc=ib.query_start_loc, seq_lens=ib.seq_lens, dcp_local_seq_lens=ib.dcp_local_seq_lens,
        query_slot=bt.slot_mappings[gid], context_positions=spec.context_positions,
        context_slot_mappings=spec._context_slot_mappings, sample_indices=spec.sample_indices,
        sample_pos=spec.sample_pos, sample_idx_mapping=spec.sample_idx_mapping, temperature=spec.temperature,
        seeds=spec.seeds, aux_staging=spec._aux_staging, num_tokens=input_batch.num_tokens,
        num_context=spec._num_graph_context_tokens(batch_desc.num_reqs),
        num_scheduled_tokens=input_batch.num_scheduled_tokens, num_graph_reqs=batch_desc.num_reqs,
        mask_token_id=spec.parallel_drafting_token_id, block_size=bt.block_sizes[gid], cp_rank=bt.cp_rank,
        cp_size=spec.dcp_size, cp_interleave=bt.cp_interleave, num_query_per_req=spec.num_query_per_req,
        num_speculative_steps=spec.num_speculative_steps, max_num_reqs=spec.max_num_reqs,
        max_num_tokens=spec.max_num_tokens, max_model_len=spec.max_model_len,
        sample_from_anchor=spec.sample_from_anchor)
    mark(("C", "dflash", _ptr(bt.slot_mappings[gid])), ("C", "replay", id(spec)),
         ("C", "dcp", _ptr(ib.dcp_local_seq_lens)))
    return True


# prepare_dflash_inputs(input_buffers, query_slot_mapping, ...): launch 1, keyed by the group's query slot buffer.
wrap_dflash = skip_once(lambda input_buffers, query_slot, *a, **k: ("C", "dflash", _ptr(query_slot)),
                        lambda *a, **k: None)


def replay_done(spec) -> bool:
    """_prepare_graph_replay after draft_prep: launches 2 (aux cat) and 3 (PAD fill) are done; its other branches
    are off under the host conditions (_num_rejected None, not dummy, T <= NC)."""
    return bool(_tokens) and take(("C", "replay", id(spec)))
