"""PERFBOT-ATKEY: the narrow FlashInfer autotune cache key (W5; coordinator ruling 2026-10-21 03:28Z).

The overlay flashinfer_autotune_cache.py (tag "# PERFBOT BUILD: ATKEY") returns key(vllm_config) in place of upstream's
sha256(VllmConfig.compute_hash()). Upstream's key binds every config field (max_num_seqs, cudagraph_capture_sizes,
additional_config, ...), so each conc and each capture-size set paid a cold tune of ~930 s and drew new tactics
(17-21 of 35 differ between two cold tunes of the same shapes). All 6 key dirs of the warm seed hold the same 35 entry
keys [m 2026-10-21]: the key split tactics, not shapes.

The narrow key keeps only what sets the tuned shapes: model architecture, dtype, quantization, the parallel sizes, the
largest tuning bucket (max_num_batched_tokens), the drafter (method, num_speculative_tokens) and the KV cache dtype.
FlashInfer keys every entry by op, runner, shapes, dtypes and the MoE tp/ep rank, so a record served under other
shapes misses (and tunes) instead of returning a wrong tactic. The FlashInfer version and the arch stay path parts.

PERFBOT_ATKEY_SALT (default empty): a bundle that changes the launch environment of a tuned op (a side stream or SM
carve-out beside the MoE, PDL edges, a fused neighbour) sets it in its arm_env.json and gets its own key and its own
cold tune. A salted arm's tuning is evidence, never the record.

The tuning of record is autotune-record/ (MANIFEST.json + one file per rank). serve.sh lays it under this key before
the server starts, and probes/atkey_check.py proves at server ready that the arm used it unchanged.
"""
import hashlib
import json
import os

SALT_ENV = "PERFBOT_ATKEY_SALT"


def fields(vllm_config) -> dict:
    """The config values the narrow key binds; str() so a torch dtype and a parsed string compare equal."""
    m, p, s = vllm_config.model_config, vllm_config.parallel_config, vllm_config.speculative_config
    return {
        "architecture": str(m.architecture),
        "dtype": str(m.dtype),
        "quantization": str(m.quantization),
        "tensor_parallel_size": p.tensor_parallel_size,
        "pipeline_parallel_size": p.pipeline_parallel_size,
        "data_parallel_size": p.data_parallel_size,
        "decode_context_parallel_size": p.decode_context_parallel_size,
        "enable_expert_parallel": bool(p.enable_expert_parallel),
        "max_num_batched_tokens": vllm_config.scheduler_config.max_num_batched_tokens,
        "spec_method": None if s is None else str(s.method),
        "num_speculative_tokens": None if s is None else s.num_speculative_tokens,
        "kv_cache_dtype": str(vllm_config.cache_config.cache_dtype),
        "salt": os.environ.get(SALT_ENV, ""),
    }


def key_of(values: dict) -> str:
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def key(vllm_config) -> str:
    return key_of(fields(vllm_config))
