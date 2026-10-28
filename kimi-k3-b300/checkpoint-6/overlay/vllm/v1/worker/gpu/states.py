# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor, UvaBackedTensor
# A new request stages all_token_ids[num_computed:] only when no reader of its prompt prefix exists; the full write  # PERFBOT P-07F
# stays the default (a row is suffix-only only if the sampler calls perfbot_p07f_free before apply_staged_writes).  # PERFBOT P-07F
PERFBOT_P07F = __import__("os").environ.get("PERFBOT_EXP_PREFILL_GRAPH", "0") == "1"  # PERFBOT P-07F: read once, at import


class RequestState:
    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        num_speculative_steps: int,
        vocab_size: int,
        device: torch.device,
        num_prefill_lookahead: int = 1,
        use_dense_all_token_ids: bool = False,
    ):
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.max_num_batched_tokens = max_num_batched_tokens
        self.num_speculative_steps = num_speculative_steps
        self.vocab_size = vocab_size
        self.device = device

        self.req_id_to_index: dict[str, int] = {}
        self.index_to_req_id: dict[int, str] = {}
        self.free_indices = list(range(max_num_reqs))

        # NOTE(woosuk): This tensor can be extremely large (e.g., several GBs)
        # depending on the configured max_num_reqs and max_model_len.
        # To save GPU memory, we use UVA instead of GPU by default, but
        # ngram_gpu benefits from dense device residency because it scans
        # active rows repeatedly during proposal.
        self.all_token_ids = StagedWriteTensor(
            (self.max_num_reqs, self.max_model_len),
            dtype=torch.int32,
            device=device,
            uva_instead_of_gpu=not use_dense_all_token_ids,
        )
        # Dense rows serve ngram_gpu, which scans the history: never suffix-only there.  # PERFBOT P-07F
        self.perfbot_p07f = PERFBOT_P07F and not use_dense_all_token_ids  # PERFBOT P-07F
        self._p07f_pending: dict[int, tuple[list[int], int]] = {}  # PERFBOT P-07F: req_idx -> (all_token_ids, num_computed)
        self._p07f_free: set[int] = set()  # PERFBOT P-07F: rows the sampler declared prefix-free this pass
        self._p07f_count = [0, 0, 0]  # PERFBOT P-07F evidence: rows suffix-only, rows full, tokens not staged
        # NOTE(woosuk): Distinguish clearly between prompt_len and prefill_len:
        # - prompt_len: Number of tokens in the user-provided prompt.
        # - prefill_len: Number of tokens passed into the model runner.
        #   This can include the prompt and additional partial output tokens,
        #   so prefill_len >= prompt_len.
        # Usually, prefill_len equals prompt_len, but in cases such as resumption after
        # preemption, prefill_len may be greater. Differentiating between these values
        # is crucial, as certain features such as prompt logprobs or frequency penalties
        # must treat prompt and output tokens separately.
        self.prompt_len = UvaBackedTensor(self.max_num_reqs, dtype=torch.int32)
        self.prefill_len = UvaBackedTensor(self.max_num_reqs, dtype=torch.int32)
        # total_len = prompt_len + output_len. It grows as the request progresses.
        self.total_len = StagedWriteTensor(
            self.max_num_reqs, dtype=torch.int32, device=device
        )

        # Number of computed tokens.
        self.num_computed_prefill_tokens = np.zeros(self.max_num_reqs, dtype=np.int32)
        self.num_computed_tokens = StagedWriteTensor(
            self.max_num_reqs, dtype=torch.int32, device=device
        )
        # Optimistic CPU mirror of num_computed_tokens (upper bound on GPU value).
        self.num_computed_tokens_np = np.zeros(self.max_num_reqs, dtype=np.int32)

        # Last sampled tokens.
        self.last_sampled_tokens = torch.zeros(
            self.max_num_reqs, 1, dtype=torch.int64, device=device
        )

        # Max total seq length (prompt_len + max_tokens).
        self.max_seq_len = np.zeros(self.max_num_reqs, dtype=np.int32)

        # Draft tokens.
        self.draft_tokens = torch.zeros(
            self.max_num_reqs,
            self.num_speculative_steps,
            dtype=torch.int64,
            device=device,
        )

        self.next_prefill_tokens = torch.zeros(
            num_prefill_lookahead, self.max_num_reqs, dtype=torch.int32, device=device
        )

    @property
    def num_reqs(self) -> int:
        return len(self.req_id_to_index)

    def add_request(
        self,
        req_id: str,
        prompt_len: int,
        all_token_ids: list[int],
        num_computed_tokens: int,
        max_tokens: int,
    ) -> None:
        assert len(self.free_indices) > 0, "No free indices"
        req_idx = self.free_indices.pop()
        self.req_id_to_index[req_id] = req_idx
        self.index_to_req_id[req_idx] = req_id

        self.max_seq_len[req_idx] = prompt_len + max_tokens
        self.prompt_len.np[req_idx] = prompt_len
        prefill_len = len(all_token_ids)
        assert prefill_len >= prompt_len, (
            f"prefill_len {prefill_len} < prompt_len {prompt_len}"
        )
        self.prefill_len.np[req_idx] = prefill_len
        self.total_len.stage_write_elem(req_idx, prefill_len)
        if self.perfbot_p07f:  # PERFBOT P-07F: defer the row; the start column is decided in apply_staged_writes
            self._p07f_pending[req_idx] = (all_token_ids, num_computed_tokens)  # PERFBOT P-07F
            self._p07f_free.discard(req_idx)  # PERFBOT P-07F
            all_token_ids = ()  # PERFBOT P-07F: the stock stage_write below returns on an empty write
        self.all_token_ids.stage_write(req_idx, 0, all_token_ids)
        self.num_computed_prefill_tokens[req_idx] = num_computed_tokens
        self.num_computed_tokens_np[req_idx] = num_computed_tokens
        self.num_computed_tokens.stage_write_elem(req_idx, num_computed_tokens)

        if self.num_speculative_steps > 0:
            self.draft_tokens[req_idx].zero_()

    def perfbot_p07f_free(self, req_idx: int) -> None:  # PERFBOT P-07F
        """The sampler found no reader of this row's prompt prefix: stage its suffix only."""  # PERFBOT P-07F
        if req_idx in self._p07f_pending:  # PERFBOT P-07F
            self._p07f_free.add(req_idx)  # PERFBOT P-07F
  # PERFBOT P-07F
    def _p07f_stage_pending(self) -> None:  # PERFBOT P-07F
        # Prefill readers start at num_computed (prepare_prefill_inputs; its lookahead is after it).  # PERFBOT P-07F
        for req_idx, (ids, num_computed) in self._p07f_pending.items():  # PERFBOT P-07F
            start = num_computed if req_idx in self._p07f_free else 0  # PERFBOT P-07F
            self.all_token_ids.stage_write(req_idx, start, ids[start:] if start else ids)  # PERFBOT P-07F
            self._p07f_count[0 if start else 1] += 1  # PERFBOT P-07F evidence: rows suffix-only / full
            self._p07f_count[2] += start  # PERFBOT P-07F evidence: tokens not staged
            if sum(self._p07f_count[:2]) <= 256 or sum(self._p07f_count[:2]) % 64 == 0:  # PERFBOT P-07F evidence: rows 1-256, then every 64
                print("PERFBOT-P07F rows_suffix=%d rows_full=%d tokens_skipped=%d" % tuple(self._p07f_count),  # PERFBOT P-07F
                      file=__import__("sys").stderr, flush=True)  # PERFBOT P-07F
        self._p07f_pending.clear()  # PERFBOT P-07F
        self._p07f_free.clear()  # PERFBOT P-07F
  # PERFBOT P-07F
    def apply_staged_writes(self) -> None:
        self.prompt_len.copy_to_uva()
        self.prefill_len.copy_to_uva()
        self.total_len.apply_write()
        self._p07f_stage_pending()  # PERFBOT P-07F
        self.all_token_ids.apply_write()
        self.num_computed_tokens.apply_write()

    def remove_request(self, req_id: str) -> int | None:
        """Return the freed slot index, or None if the request was not found."""
        req_idx = self.req_id_to_index.pop(req_id, None)
        if req_idx is None:
            return None
        self.index_to_req_id.pop(req_idx, None)
        self._p07f_pending.pop(req_idx, None)  # PERFBOT P-07F
        self._p07f_free.discard(req_idx)  # PERFBOT P-07F
        self.free_indices.append(req_idx)
        return req_idx
