# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.layers.mamba import checkpoint as perfbot_p07_checkpoint  # PERFBOT P-07
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateCopyFuncsByType
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadataBuilder
from vllm.v1.attention.backends.short_conv_attn import (
    PleShortConvAttentionMetadataBuilder,
    ShortConvAttentionMetadataBuilder,
)
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec
from vllm.v1.kv_cache_interface import PERFBOT_P07_CKPT  # PERFBOT P-07
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.gpu.model_states.recoverssm import RecoverSSMState
from vllm.v1.worker.mamba_utils import (
    MambaSpecDecodeGPUContext,
    get_mamba_group_ids,
    get_mamba_groups,
    preprocess_mamba_align_fused_kernel,
    validate_mamba_state_copy_funcs,
)
from vllm.v1.worker.utils import AttentionGroup
from perfbot_kernels.glue import on as perfbot_on  # PERFBOT P-11
_perfbot_hooks = None  # PERFBOT P-11
if perfbot_on("E"):  # PERFBOT P-11: fused next-step prologue (lane Q unit E)
    from perfbot_kernels import step_tail_hooks as _perfbot_hooks  # PERFBOT P-11


@dataclass
class MambaHybridAttnMetadata(ModelSpecificAttnMetadata):
    is_prefilling: torch.Tensor
    num_accepted_tokens: torch.Tensor | None = None
    num_decode_draft_tokens_cpu: torch.Tensor | None = None

    def get_extra_common_attn_kwargs(
        self, kv_cache_group_id: int, num_reqs: int
    ) -> dict[str, Any]:
        return {"is_prefilling": self.is_prefilling[:num_reqs]}

    def get_extra_attn_kwargs(
        self, attn_metadata_builder: Any, num_reqs: int
    ) -> dict[str, Any]:
        if not isinstance(
            attn_metadata_builder,
            (
                Mamba2AttentionMetadataBuilder,
                GDNAttentionMetadataBuilder,
                ShortConvAttentionMetadataBuilder,
                PleShortConvAttentionMetadataBuilder,
            ),
        ):
            return {}
        return {
            "num_accepted_tokens": None
            if self.num_accepted_tokens is None
            else self.num_accepted_tokens[:num_reqs],
            "num_decode_draft_tokens_cpu": None
            if self.num_decode_draft_tokens_cpu is None
            else self.num_decode_draft_tokens_cpu[:num_reqs],
        }


class MambaHybridModelState(DefaultModelState):
    """Model state for hybrid attention + Mamba / linear-attention models."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        self.cache_config = vllm_config.cache_config
        self.num_accepted_tokens_gpu = torch.ones(
            self.max_num_reqs, dtype=torch.int32, device=self.device
        )
        # Pre-copy "align" prefix-cache state (V2). The migration of each
        # request's mamba state across block boundaries runs as a fused GPU
        # kernel reusing the postprocess copy machinery, so the per-step src
        # columns and the running state_idx are kept GPU-resident.
        self._align_mode = self.cache_config.mamba_cache_mode == "align"
        self.recoverssm = (
            RecoverSSMState() if self.cache_config.use_kda_recoverssm else None
        )
        if self._align_mode:
            self._mamba_state_idx_gpu = torch.zeros(
                self.max_num_reqs, dtype=torch.int32, device=self.device
            )
            self._mamba_src_col_gpu = torch.full(
                (self.max_num_reqs,), -1, dtype=torch.int32, device=self.device
            )
            self._mamba_src_off_gpu = torch.zeros(
                self.max_num_reqs, dtype=torch.int32, device=self.device
            )
            self._mamba_ctx: MambaSpecDecodeGPUContext | None = None
            self._mamba_group_ids: list[int] = []
            self._mamba_spec: MambaSpec | None = None
            self._mamba_state_copy_funcs: MambaStateCopyFuncsByType | None = None
            self._p07_add_start = np.full(self.max_num_reqs, -1, dtype=np.int64)  # PERFBOT P-07: start of the row's first chunk

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        super().add_request(req_index, new_req_data)
        # Must reset the speculative acceptance count in this idx which could be stale.
        self.num_accepted_tokens_gpu[req_index].fill_(1)
        if self._align_mode:
            # Seed the running state block from the resumed/prefilled position.
            # The column is in mamba blocks. cache_config.block_size is the
            # min over prefix-cacheable groups, so a drafter group with its own
            # smaller block (e.g. DFlash) pulls it below mamba_block_size.
            mamba_block_size = self.cache_config.mamba_block_size
            assert mamba_block_size is not None
            self._mamba_state_idx_gpu[req_index].fill_(
                (new_req_data.num_computed_tokens - 1) // mamba_block_size
            )
            if PERFBOT_P07_CKPT:  # PERFBOT P-07
                self._p07_add_start[req_index] = new_req_data.num_computed_tokens  # PERFBOT P-07

    def _get_mamba_group_info(
        self, kv_cache_config: KVCacheConfig
    ) -> tuple[list[int], MambaSpec]:
        if self._mamba_spec is None:
            mamba_groups = get_mamba_groups(kv_cache_config)
            mamba_spec = next(iter(mamba_groups))
            assert all(
                spec.block_size == mamba_spec.block_size
                and spec.num_speculative_blocks == mamba_spec.num_speculative_blocks
                and spec.mamba_cache_mode == mamba_spec.mamba_cache_mode
                for spec in mamba_groups
            ), "all mamba groups must share cache scheduling parameters"
            self._mamba_group_ids = get_mamba_group_ids(mamba_groups)
            self._mamba_spec = mamba_spec
        return self._mamba_group_ids, self._mamba_spec

    def initialize_kv_cache(
        self, kv_cache_config: KVCacheConfig, block_tables: BlockTables
    ) -> None:
        if not self._align_mode:
            return
        mamba_group_ids, _ = self._get_mamba_group_info(kv_cache_config)
        if self._mamba_state_copy_funcs is None:
            mamba_groups = get_mamba_groups(kv_cache_config)
            mamba_types = {spec.mamba_type for spec in mamba_groups}
            copy_funcs = self.model.get_mamba_state_copy_funcs(mamba_types)
            validate_mamba_state_copy_funcs(mamba_groups, copy_funcs)
            self._mamba_state_copy_funcs = copy_funcs
        # Both SD and DS conv layouts support a >0 spec-decode shift: the
        # fused pre-copy kernel (``_copy_mamba_state_block``) applies the
        # ``token_bias = num_accepted - 1`` window shift per conv layout
        # (SD: contiguous slice; DS: per-dim-row strided slice), matching
        # the V1 ``get_conv_copy_spec`` semantics.
        self._mamba_ctx = MambaSpecDecodeGPUContext.create(
            max_num_reqs=self.max_num_reqs,
            kv_cache_config=kv_cache_config,
            copy_funcs=self._mamba_state_copy_funcs,
            device=self.device,
            make_buffer=lambda n, dtype: CpuGpuBuffer(
                n, dtype=dtype, device=self.device
            ),
        )
        # Under PP a step's postprocess runs after later steps re-gathered the
        # batch-ordered input tables, so state copies read the per-request-slot
        # tables. Aligned state indices are for the current batch only.
        self._mamba_ctx.initialize_from_forward_context(
            kv_cache_config,
            self.vllm_config.compilation_config.static_forward_context,
            self._mamba_state_copy_funcs,
            [block_tables.block_tables[gid].gpu for gid in mamba_group_ids],
            [block_tables.input_block_tables[gid] for gid in mamba_group_ids],
        )

    def preprocess_state(
        self,
        input_batch: InputBatch,
        block_tables: tuple[torch.Tensor, ...],
        kv_cache_config: KVCacheConfig,
        num_computed_tokens: torch.Tensor,
    ) -> None:
        """Migrate each request's mamba state across block boundaries before the
        forward (V1 align semantics, done on GPU). Runs on real batches only
        (dummy DP/profiling runs skip preprocess_state), and before
        ``prepare_attn`` gathers ``num_accepted_tokens``, so the boundary reset
        is visible to the forward kernels.
        """
        if perfbot_p07_checkpoint.PERFBOT_P07_CKPTLOG is not None:  # PERFBOT P-07 ckptlog: read back the last export before this step's pre-copy
            perfbot_p07_checkpoint.PERFBOT_P07_CKPTLOG.note_step(input_batch, block_tables)  # PERFBOT P-07 ckptlog
        if not self._align_mode:
            return
        num_reqs = input_batch.num_reqs
        if num_reqs == 0:
            return
        if getattr(self, "perfbot_prologue_na", None) is not None:  # PERFBOT P-11
            return  # #9 ran in step_prologue, #10 right after it (prologue.py)  # PERFBOT P-11
        _, mamba_spec = self._get_mamba_group_info(kv_cache_config)
        ctx = self._mamba_ctx
        assert ctx is not None

        # The state-advance + pre-copy kernels run every step; they fast-exit per
        # request when src_col < 0 or src_col == dst_col, so no copy happens on
        # steps that don't cross a block boundary. (Skipping the launch entirely
        # would need a V1-style async-D2H of the actual num_computed, since
        # num_computed_tokens_np is an optimistic mirror under async scheduling;
        # the launch cost is ~0.3% of TPOT, so the GPU fast-exit suffices.)
        block = 256
        grid = (triton.cdiv(num_reqs, block),)
        preprocess_mamba_align_fused_kernel[grid](
            input_batch.idx_mapping,
            self._mamba_state_idx_gpu,
            num_computed_tokens,
            input_batch.query_start_loc,
            self.num_accepted_tokens_gpu,
            self._mamba_src_col_gpu,
            self._mamba_src_off_gpu,
            num_reqs,
            BLOCK_SIZE=block,
            MAMBA_BLOCK_SIZE=mamba_spec.block_size,
        )
        ctx.run_fused_precopy(
            num_reqs,
            self._mamba_state_idx_gpu,
            self._mamba_src_col_gpu,
            self._mamba_src_off_gpu,
            input_batch.idx_mapping,
        )

    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
        ubatch_idx: int = 0,
        model_specific_attn_metadata: ModelSpecificAttnMetadata | None = None,
    ) -> dict[str, Any]:
        assert ubatch_idx == 0, "DBO is not supported"
        assert model_specific_attn_metadata is None
        if getattr(self, "perfbot_prologue_na", None) is not None and not for_capture:  # PERFBOT P-11
            return self._perfbot_p11_prepare_attn(  # PERFBOT P-11
                input_batch, cudagraph_mode, block_tables, slot_mappings, attn_groups,  # PERFBOT P-11
                kv_cache_config, for_capture, ubatch_idx, model_specific_attn_metadata,  # PERFBOT P-11
            )  # PERFBOT P-11
        if cudagraph_mode == CUDAGraphMode.FULL:
            num_reqs = input_batch.num_reqs_after_padding
            num_tokens = input_batch.num_tokens_after_padding
        else:
            num_reqs = input_batch.num_reqs
            num_tokens = input_batch.num_tokens
        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)
        # Prefer the promised bound: a capture dummy's measured max is its even
        # split, not the length the graph must replay.
        max_query_len = input_batch.max_query_len
        if max_query_len is None:
            max_query_len = input_batch.num_scheduled_tokens.max().item()
        seq_lens_cpu_upper_bound = input_batch.seq_lens_cpu_upper_bound
        if for_capture:
            # Capture with worst-case max_seq_len so the graph is valid at any replay.
            max_seq_len = self.max_model_len
        else:
            max_seq_len = seq_lens_cpu_upper_bound[:num_reqs].max().item()

        is_prefilling_np = input_batch.is_prefilling_np
        if input_batch.prefill_runs_as_decode_np is not None:
            # A prompt tail the scheduler padded with placeholder drafts must run
            # as a spec-decode row: the prefill kernels can't roll them back.
            is_prefilling_np = is_prefilling_np & ~input_batch.prefill_runs_as_decode_np
        is_prefilling = torch.zeros(num_reqs, dtype=torch.bool, device="cpu")
        is_prefilling[: input_batch.num_reqs] = torch.from_numpy(is_prefilling_np)
        # During CUDAGraph capture, num_decode_draft_tokens_cpu and num_accepted_tokens
        # are created by attn_metadata_builder.build_for_cudagraph_capture, so we only
        # compute them during actual (non-capture) forward execution.
        num_accepted_tokens = None
        num_decode_draft_tokens_cpu = None
        if not for_capture and self.vllm_config.num_speculative_tokens > 0:
            num_accepted_tokens = self.num_accepted_tokens_gpu.new_ones(num_reqs)
            num_accepted_tokens[: input_batch.num_reqs] = self.num_accepted_tokens_gpu[
                input_batch.idx_mapping
            ]

            # GDN uses >= 0 to select spec-decode rows, so non-decode rows
            # need the -1 sentinel. Decode rows without drafts stay spec rows:
            # only that path applies the previous step's accepted offset.
            num_decode_draft_tokens_np = np.full(num_reqs, -1, dtype=np.int32)
            num_draft_tokens_per_req = input_batch.num_draft_tokens_per_req
            if num_draft_tokens_per_req is None:
                num_draft_tokens_per_req = np.zeros(input_batch.num_reqs, np.int32)
            # Test request state, not num_scheduled_tokens == draft_count+1:
            # adaptive rewrites num_scheduled_tokens to an even split, so that
            # equality rarely holds and would demote every verify row to decode.
            is_decode = (
                ~is_prefilling_np
                & (input_batch.num_scheduled_tokens > 0)
                & (
                    input_batch.num_scheduled_tokens
                    <= self.vllm_config.num_speculative_tokens + 1
                )
            )
            num_decode_draft_tokens_np[: input_batch.num_reqs] = np.where(
                is_decode, num_draft_tokens_per_req, -1
            )
            num_decode_draft_tokens_cpu = torch.from_numpy(num_decode_draft_tokens_np)

        if self._align_mode:
            mamba_group_ids, _ = self._get_mamba_group_info(kv_cache_config)
            aligned_index_builders = []
            for group_idx, group_id in enumerate(mamba_group_ids):
                for group in attn_groups[group_id]:
                    builder = group.get_metadata_builder(0)
                    if hasattr(builder, "mamba_aligned_state_indices"):
                        aligned_index_builders.append((group_idx, builder))
            if aligned_index_builders:
                assert self._mamba_ctx is not None
                all_group_indices = self._mamba_ctx.compute_aligned_state_indices(
                    input_batch.seq_lens, num_reqs
                )
                for group_idx, builder in aligned_index_builders:
                    builder.mamba_aligned_state_indices = all_group_indices[group_idx]

        if PERFBOT_P07_CKPT and self._align_mode:  # PERFBOT P-07: fresh row = first chunk since the row was sent whole
            fresh = self._p07_add_start[input_batch.idx_mapping_np] == input_batch.num_computed_tokens_np  # PERFBOT P-07
            perfbot_p07_checkpoint.PERFBOT_P07_FRESH_ROWS = None if for_capture else np.pad(fresh, (0, num_reqs - fresh.size))  # PERFBOT P-07
        mamba_attn_metadata = MambaHybridAttnMetadata(
            is_prefilling=is_prefilling,
            num_accepted_tokens=num_accepted_tokens,
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
        )
        attn_metadata = build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=max_query_len,
            seq_lens=input_batch.seq_lens,
            max_seq_len=max_seq_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            positions=input_batch.positions,
            model_specific_attn_metadata=mamba_attn_metadata,
            for_cudagraph_capture=for_capture,
            rswa_prefix_lens=input_batch.prompt_lens,
        )
        if self.recoverssm is not None:
            self.recoverssm.record_step(
                attn_metadata, attn_groups, for_capture=for_capture
            )
        return attn_metadata
# PERFBOT P-11
    def _perfbot_p11_prepare_attn(  # PERFBOT P-11
        self,  # PERFBOT P-11
        input_batch: InputBatch,  # PERFBOT P-11
        cudagraph_mode: CUDAGraphMode,  # PERFBOT P-11
        block_tables: tuple[torch.Tensor, ...],  # PERFBOT P-11
        slot_mappings: torch.Tensor,  # PERFBOT P-11
        attn_groups: list[list[AttentionGroup]],  # PERFBOT P-11
        kv_cache_config: KVCacheConfig,  # PERFBOT P-11
        for_capture: bool = False,  # PERFBOT P-11
        ubatch_idx: int = 0,  # PERFBOT P-11
        model_specific_attn_metadata: ModelSpecificAttnMetadata | None = None,  # PERFBOT P-11
    ) -> dict[str, Any]:  # PERFBOT P-11
        assert ubatch_idx == 0, "DBO is not supported"  # PERFBOT P-11
        assert model_specific_attn_metadata is None  # PERFBOT P-11
        if cudagraph_mode == CUDAGraphMode.FULL:  # PERFBOT P-11
            num_reqs = input_batch.num_reqs_after_padding  # PERFBOT P-11
            num_tokens = input_batch.num_tokens_after_padding  # PERFBOT P-11
        else:  # PERFBOT P-11
            num_reqs = input_batch.num_reqs  # PERFBOT P-11
            num_tokens = input_batch.num_tokens  # PERFBOT P-11
        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)  # PERFBOT P-11
        # Prefer the promised bound: a capture dummy's measured max is its even  # PERFBOT P-11
        # split, not the length the graph must replay.  # PERFBOT P-11
        max_query_len = input_batch.max_query_len  # PERFBOT P-11
        if max_query_len is None:  # PERFBOT P-11
            max_query_len = input_batch.num_scheduled_tokens.max().item()  # PERFBOT P-11
        seq_lens_cpu_upper_bound = input_batch.seq_lens_cpu_upper_bound  # PERFBOT P-11
        if for_capture:  # PERFBOT P-11
            # Capture with worst-case max_seq_len so the graph is valid at any replay.  # PERFBOT P-11
            max_seq_len = self.max_model_len  # PERFBOT P-11
        else:  # PERFBOT P-11
            max_seq_len = seq_lens_cpu_upper_bound[:num_reqs].max().item()  # PERFBOT P-11
# PERFBOT P-11
        is_prefilling_np = input_batch.is_prefilling_np  # PERFBOT P-11
        if input_batch.prefill_runs_as_decode_np is not None:  # PERFBOT P-11
            # A prompt tail the scheduler padded with placeholder drafts must run  # PERFBOT P-11
            # as a spec-decode row: the prefill kernels can't roll them back.  # PERFBOT P-11
            is_prefilling_np = is_prefilling_np & ~input_batch.prefill_runs_as_decode_np  # PERFBOT P-11
        is_prefilling = torch.zeros(num_reqs, dtype=torch.bool, device="cpu")  # PERFBOT P-11
        is_prefilling[: input_batch.num_reqs] = torch.from_numpy(is_prefilling_np)  # PERFBOT P-11
        # During CUDAGraph capture, num_decode_draft_tokens_cpu and num_accepted_tokens  # PERFBOT P-11
        # are created by attn_metadata_builder.build_for_cudagraph_capture, so we only  # PERFBOT P-11
        # compute them during actual (non-capture) forward execution.  # PERFBOT P-11
        num_accepted_tokens = None  # PERFBOT P-11
        num_decode_draft_tokens_cpu = None  # PERFBOT P-11
        if not for_capture and self.vllm_config.num_speculative_tokens > 0:  # PERFBOT P-11
            # P-11 #12-#14 ran in step_prologue (the same int32[num_reqs] values); marks #16-#18  # PERFBOT P-11
            num_accepted_tokens = _perfbot_hooks.mamba_hook(  # PERFBOT P-11
                self, input_batch, attn_groups, kv_cache_config, num_reqs  # PERFBOT P-11
            )  # PERFBOT P-11
# PERFBOT P-11
            # GDN uses >= 0 to select spec-decode rows, so non-decode rows  # PERFBOT P-11
            # need the -1 sentinel. Decode rows without drafts stay spec rows:  # PERFBOT P-11
            # only that path applies the previous step's accepted offset.  # PERFBOT P-11
            num_decode_draft_tokens_np = np.full(num_reqs, -1, dtype=np.int32)  # PERFBOT P-11
            num_draft_tokens_per_req = input_batch.num_draft_tokens_per_req  # PERFBOT P-11
            if num_draft_tokens_per_req is None:  # PERFBOT P-11
                num_draft_tokens_per_req = np.zeros(input_batch.num_reqs, np.int32)  # PERFBOT P-11
            # Test request state, not num_scheduled_tokens == draft_count+1:  # PERFBOT P-11
            # adaptive rewrites num_scheduled_tokens to an even split, so that  # PERFBOT P-11
            # equality rarely holds and would demote every verify row to decode.  # PERFBOT P-11
            is_decode = (  # PERFBOT P-11
                ~is_prefilling_np  # PERFBOT P-11
                & (input_batch.num_scheduled_tokens > 0)  # PERFBOT P-11
                & (  # PERFBOT P-11
                    input_batch.num_scheduled_tokens  # PERFBOT P-11
                    <= self.vllm_config.num_speculative_tokens + 1  # PERFBOT P-11
                )  # PERFBOT P-11
            )  # PERFBOT P-11
            num_decode_draft_tokens_np[: input_batch.num_reqs] = np.where(  # PERFBOT P-11
                is_decode, num_draft_tokens_per_req, -1  # PERFBOT P-11
            )  # PERFBOT P-11
            num_decode_draft_tokens_cpu = torch.from_numpy(num_decode_draft_tokens_np)  # PERFBOT P-11
# PERFBOT P-11
        if self._align_mode:  # PERFBOT P-11
            mamba_group_ids, _ = self._get_mamba_group_info(kv_cache_config)  # PERFBOT P-11
            aligned_index_builders = []  # PERFBOT P-11
            for group_idx, group_id in enumerate(mamba_group_ids):  # PERFBOT P-11
                for group in attn_groups[group_id]:  # PERFBOT P-11
                    builder = group.get_metadata_builder(0)  # PERFBOT P-11
                    if hasattr(builder, "mamba_aligned_state_indices"):  # PERFBOT P-11
                        aligned_index_builders.append((group_idx, builder))  # PERFBOT P-11
            if aligned_index_builders:  # PERFBOT P-11
                assert self._mamba_ctx is not None  # PERFBOT P-11
                # P-11 #15 ran in step_prologue: the view mamba_utils.py:1141 returns  # PERFBOT P-11
                all_group_indices = self._mamba_ctx.aligned_state_indices[:, :num_reqs]  # PERFBOT P-11
                for group_idx, builder in aligned_index_builders:  # PERFBOT P-11
                    builder.mamba_aligned_state_indices = all_group_indices[group_idx]  # PERFBOT P-11
            _perfbot_hooks.kernel_table_src(block_tables)  # P-11 #19-#20 ran in step_prologue  # PERFBOT P-11
# PERFBOT P-11
        if PERFBOT_P07_CKPT and self._align_mode:  # PERFBOT P-07: fresh row = first chunk since the row was sent whole  # PERFBOT P-11
            fresh = self._p07_add_start[input_batch.idx_mapping_np] == input_batch.num_computed_tokens_np  # PERFBOT P-07  # PERFBOT P-11
            perfbot_p07_checkpoint.PERFBOT_P07_FRESH_ROWS = None if for_capture else np.pad(fresh, (0, num_reqs - fresh.size))  # PERFBOT P-07  # PERFBOT P-11
        mamba_attn_metadata = MambaHybridAttnMetadata(  # PERFBOT P-11
            is_prefilling=is_prefilling,  # PERFBOT P-11
            num_accepted_tokens=num_accepted_tokens,  # PERFBOT P-11
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,  # PERFBOT P-11
        )  # PERFBOT P-11
        attn_metadata = build_attn_metadata(  # PERFBOT P-11
            attn_groups=attn_groups,  # PERFBOT P-11
            num_reqs=num_reqs,  # PERFBOT P-11
            num_tokens=num_tokens,  # PERFBOT P-11
            query_start_loc_gpu=input_batch.query_start_loc,  # PERFBOT P-11
            query_start_loc_cpu=query_start_loc_cpu,  # PERFBOT P-11
            max_query_len=max_query_len,  # PERFBOT P-11
            seq_lens=input_batch.seq_lens,  # PERFBOT P-11
            max_seq_len=max_seq_len,  # PERFBOT P-11
            block_tables=block_tables,  # PERFBOT P-11
            slot_mappings=slot_mappings,  # PERFBOT P-11
            kv_cache_config=kv_cache_config,  # PERFBOT P-11
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,  # PERFBOT P-11
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,  # PERFBOT P-11
            positions=input_batch.positions,  # PERFBOT P-11
            model_specific_attn_metadata=mamba_attn_metadata,  # PERFBOT P-11
            for_cudagraph_capture=for_capture,  # PERFBOT P-11
            rswa_prefix_lens=input_batch.prompt_lens,  # PERFBOT P-11
        )  # PERFBOT P-11
        if self.recoverssm is not None:  # PERFBOT P-11
            self.recoverssm.record_step(  # PERFBOT P-11
                attn_metadata, attn_groups, for_capture=for_capture  # PERFBOT P-11
            )  # PERFBOT P-11
        return attn_metadata  # PERFBOT P-11

    def postprocess_state(
        self,
        idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor | int,
        num_computed_tokens: torch.Tensor | None = None,
    ) -> None:
        # Chunked prefill does not sample a token, so num_sampled can be 0.
        # Mamba treats num_accepted_tokens=1 as the neutral non-spec value.
        num_reqs = idx_mapping.shape[0]
        if num_reqs:
            if not isinstance(num_sampled, int):
                # idx_mapping may contain -1 sentinels (filtered rows) under PP; the
                # kernel skips them rather than scattering with a host-side gather.
                _scatter_num_accepted_kernel[(num_reqs,)](
                    idx_mapping,
                    num_sampled,
                    self.num_accepted_tokens_gpu,
                )
            else:
                # Fill with single value.
                _fill_num_accepted_kernel[(num_reqs,)](
                    idx_mapping,
                    self.num_accepted_tokens_gpu,
                    max(num_sampled, 1),
                )

        if self.recoverssm is not None:
            self.recoverssm.commit_step(
                num_sampled,
                idx_mapping,
                state_indices=(self._mamba_state_idx_gpu if self._align_mode else None),
                num_accepted_tokens=self.num_accepted_tokens_gpu,
            )

        if not num_reqs:
            return

        # Align: save the running state to the block-aligned position when
        # spec-decode acceptance leaves the sequence non-block-aligned (mirrors
        # the V1 align postprocess). num_computed_tokens already holds the
        # post-step advanced count.
        if (
            self._align_mode
            and num_computed_tokens is not None
            and self._mamba_ctx is not None
        ):
            self._mamba_ctx.run_fused_postprocess_align(
                num_reqs,
                self.num_accepted_tokens_gpu,
                self._mamba_state_idx_gpu,
                num_computed_tokens,
                idx_mapping,
            )


@triton.jit
def _scatter_num_accepted_kernel(
    idx_mapping_ptr,  # [num_reqs] batch_idx -> req_state_idx (-1 to skip)
    num_sampled_ptr,  # [num_reqs]
    num_accepted_ptr,  # [max_num_reqs]
):
    row = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + row)
    if req_state_idx < 0:
        return
    num_sampled = tl.load(num_sampled_ptr + row)
    tl.store(num_accepted_ptr + req_state_idx, tl.maximum(num_sampled, 1))


@triton.jit
def _fill_num_accepted_kernel(
    idx_mapping_ptr,  # [num_reqs] batch_idx -> req_state_idx (-1 to skip)
    num_accepted_ptr,  # [max_num_reqs]
    num_sampled,
):
    row = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + row)
    if req_state_idx < 0:
        return
    tl.store(num_accepted_ptr + req_state_idx, num_sampled)
