"""GPU variants of the bench (README.md). Each is one function `run(case) -> Tensor [n, HL, V]` in the registry.

image       the image's prefill attention for one MLA layer as the server calls it: the overlay method
            MultiHeadLatentAttention._perfbot_prefill_image (the pinned _forward_prefill_fused body), run on a minimal
            layer object, with the pinned impl's _context_parallel_compute_prefill_context, the real
            TrtllmRaggedPrefillBackend, the real build_mla_chunked_context_metadata (the server's chunking) and the
            overlay's DirectDCPKVGatherWorkspace behind an MLADCPManager.
image_split the image path; only the context-chunk attention differs: each request's chunk KV is cut into S slices
            passed as S batch entries of ONE trtllm_ragged_attention_deepseek call (grid ceil(n/256) x 12 x S
            instead of x 1), the S partials merged with merge_attn_states in the ln base the server merges in.
fa4_split   the image path; the context-chunk attention is the pinned FA4 CuTe varlen kernel (fp8 q/k/v, unit
            descales, num_splits=0 heuristic). Optional; guarded by an availability check.
absorbed    the mla_absorb gate chain (perfbot_kernels.mla_absorb.selftest.Rank.chain) on the same inputs.

Everything vllm is imported lazily: this module imports on a CPU host but runs only in the server image.
"""

import types

import torch
import torch.nn.functional as F

import perfbot_kernels.mla_mixbench as mb
from perfbot_kernels.mla_mixbench import HL, L, P, R, V, WORKSPACE_ROWS, W, expected_chunks, register

FP8 = torch.float8_e4m3fn


class KvBProj(torch.nn.Module):
    """Stand-in for the layer's ColumnParallelLinear kv_b_proj (unquantized bf16): the same call the server makes,
    UnquantizedLinearMethod.apply -> default_unquantized_gemm -> torch.nn.functional.linear (pinned
    model_executor/layers/utils.py:86), returning (output, bias) like the real layer."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.params_dtype = weight.dtype

    def forward(self, x):
        return F.linear(x, self.weight), None


# ------------------------------------------------------------------------------------------- prefill backends
def backend_classes():
    """(TrtllmRaggedPrefillBackend, SplitKVPrefillBackend, FA4ContextPrefillBackend). The two subclasses keep the
    image's new-token call (trtllm, causal) and replace only run_prefill_context_chunk. SplitKVPrefillBackend is
    the server's class (perfbot_kernels.mla_split.backend_class): the gate runs exactly the server's code."""
    from vllm.v1.attention.backends.mla.prefill.trtllm_ragged import TrtllmRaggedPrefillBackend

    from perfbot_kernels import mla_split

    SplitKVPrefillBackend = mla_split.backend_class()

    class FA4ContextPrefillBackend(TrtllmRaggedPrefillBackend):
        """Context chunk through vllm_flash_attn.flash_attn_varlen_func(fa_version=4) -> cute.interface._flash_attn_fwd
        (pinned flash_attn_interface.py:176, :406-435), as the pinned FlashAttnPrefillBackend.run_prefill_context_chunk
        calls it, plus fp8 unit descales of shape (batch, kv_heads) (interface.py:606-608) and num_splits=0 (the
        min(SMs // mblocks, 128, nblocks) heuristic, interface.py:329, :713-724). FA4 returns ln LSE [H, total_q]
        (cute/softmax.py:229 multiplies by LN2), the base the trtllm path reaches after log2_lse_to_ln: no
        conversion."""

        def run_prefill_context_chunk(self, chunk, q, k, v, out=None):
            from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

            ones = self.__dict__.setdefault("_descale", {})
            d = ones.get(chunk.num_requests)
            if d is None:
                d = ones[chunk.num_requests] = torch.ones(chunk.num_requests, k.shape[1], dtype=torch.float32,
                                                          device=q.device)
            o, lse = flash_attn_varlen_func(
                q, k, v, max_seqlen_q=chunk.max_query_len, cu_seqlens_q=chunk.query_start_loc,
                max_seqlen_k=chunk.max_seq_len, cu_seqlens_k=chunk.cu_seq_lens, softmax_scale=self.scale,
                causal=False, return_softmax_lse=True, out=out, q_descale=d, k_descale=d, v_descale=d,
                num_splits=0, fa_version=4)
            return o, lse

    return TrtllmRaggedPrefillBackend, SplitKVPrefillBackend, FA4ContextPrefillBackend


def fa4_unavailable(env) -> str | None:
    """CPU-side checks before the first FA4 call: imports, arch // 10 == 10 (fp8, interface.py:633), head dims."""
    try:
        from vllm.vllm_flash_attn.cute import interface as fi
        from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return f"import: {type(e).__name__}: {e}"[:300]
    try:
        arch = fi._get_device_arch()
        if arch // 10 != 10:
            return f"arch {arch}: FA4 fp8 needs arch // 10 == 10"
        fi._validate_head_dims(P + R, V, arch // 10, 16 // torch.empty((), dtype=FP8).element_size())
    except Exception as e:  # noqa: BLE001
        return f"check: {type(e).__name__}: {e}"[:300]
    return None


# ---------------------------------------------------------------------------------------------- image family
def image_layer(env, case):
    """A minimal layer object carrying exactly what _perfbot_prefill_image and the impl read, with the real methods
    bound: MultiHeadLatentAttention._perfbot_prefill_image / _attn_read_kv_cache (overlay mla.py) and
    MLACommonBaseImpl._context_parallel_compute_prefill_context / _concat_k_nope_k_pe (pinned mla_attention.py)."""
    from vllm.model_executor.layers.attention.mla_attention import MLACommonBaseImpl
    from vllm.models.kimi_k3.nvidia.mla import MultiHeadLatentAttention as MLA

    prefill_fn = getattr(MLA, "_perfbot_prefill_image", None) or MLA._forward_prefill_fused  # pinned name
    impl_cls = type("ImageImpl", (), {
        "_context_parallel_compute_prefill_context": MLACommonBaseImpl._context_parallel_compute_prefill_context,
        "_concat_k_nope_k_pe": MLACommonBaseImpl._concat_k_nope_k_pe,
    })
    impl = impl_cls()
    impl.__dict__.update(
        kv_b_proj=env.kv_b_proj, kv_lora_rank=L, qk_rope_head_dim=R, qk_nope_head_dim=P, v_head_dim=V,
        num_heads=HL, kv_cache_dtype="fp8",
        _use_flashinfer_concat_mla_k=False)  # MLACommonImpl: has_flashinfer() and num_heads == 128 -> False at 12
    layer_cls = type("ImageLayer", (), {
        "_perfbot_prefill_image": prefill_fn,
        "_attn_read_kv_cache": MLA._attn_read_kv_cache,
    })
    layer = layer_cls()
    layer.__dict__.update(
        kv_b_proj=env.kv_b_proj, num_local_heads=HL, qk_nope_head_dim=P, qk_rope_head_dim=R, v_head_dim=V,
        kv_lora_rank=L, kv_cache_dtype="fp8", kv_cache=case.cache_u8, impl=impl, dcp_world_size=W,
        _one_scale=env.one, _k_scale=env.one, _k_scale_inv=env.one)
    return layer, prefill_fn


def image_metadata(env, case, backend):
    """attn_metadata for the case's prefill requests (request r: n_r new tokens after ctx_r cached): what
    MLACommonMetadataBuilder.build makes for the prefill rows (pinned mla_attention.py:2347-2388), through the same
    build_mla_chunked_context_metadata call and MLACommonPrefillMetadata dataclass, plus the step info the server's
    builder hook attaches (mla_split.attach, route forced on: the gate tests the split on every case). Only
    `.prefill` is read on this path, so the outer metadata is a SimpleNamespace(prefill=...)."""
    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonPrefillMetadata,
        build_mla_chunked_context_metadata,
    )

    from perfbot_kernels import mla_split

    dev = env.dev
    q_lens, ctxs = [r.n for r in case.reqs], [r.ctx for r in case.reqs]
    qsl = [0]
    for m in q_lens:
        qsl.append(qsl[-1] + m)
    chunked = build_mla_chunked_context_metadata(
        context_lens_cpu=torch.tensor(ctxs, dtype=torch.int32),
        prefill_query_start_loc_cpu=torch.tensor(qsl, dtype=torch.int32),
        chunked_prefill_workspace=env.chunk_ws,
        chunked_prefill_workspace_size=WORKSPACE_ROWS,
        block_size=case.page,
        align_chunk_to_block=True,
        device=dev,
        dcp_world_size=W,
        dcp_local_block_size=1,
        dcp_virtual_block_size=W,
        dcp_manager=env.dcp_manager,
    )
    assert chunked is not None, ctxs
    if len(case.reqs) == 1:
        assert len(chunked.chunks) == expected_chunks(ctxs[0]), (ctxs, len(chunked.chunks))
    prefill = MLACommonPrefillMetadata(
        block_table=case.block_table,
        query_start_loc=torch.tensor(qsl, dtype=torch.int32, device=dev),
        max_query_len=max(q_lens),
        chunked_context=chunked,
        output_dtype=torch.bfloat16,
        q_data_type=FP8,  # --attention-config use_prefill_query_quantization: true, fp8 cache
        prefill_backend=backend,
    )
    backend.prepare_metadata(prefill)  # the builder's call (sets the backend's query lengths)
    mla_split.attach(prefill, q_lens, ctxs, active=True)  # the server's builder hook, after build
    torch.cuda.synchronize()
    return types.SimpleNamespace(prefill=prefill)


def run_image_family(case, name, backend):
    st = case.memo.get(name)
    if st is None:
        layer, fn = image_layer(case.env, case)
        md = image_metadata(case.env, case, backend)
        out = torch.empty(case.n, HL * V, dtype=torch.bfloat16, device=case.env.dev)
        st = case.memo[name] = types.SimpleNamespace(layer=layer, fn=fn, md=md, out=out,
                                                     chunks=len(md.prefill.chunked_context.chunks))
    # positions / cos_sin_cache None: the K3 target's MLA layers are NoPE (overlay mla.py: rotary_emb None).
    st.fn(st.layer, case.q, case.kv_c, case.k_pe, None, None, case.slots, st.md, st.out)
    return st.out.view(case.n, HL, V)


@register("image", m03_key="m03_image_ms")
def image(case):
    return run_image_family(case, "image", case.env.backends["image"])


@register("image_split", rel_bound=1e-2)
def image_split(case):
    return run_image_family(case, "image_split", case.env.backends["image_split"])


@register("fa4_split", rel_bound=2e-2, available=fa4_unavailable)
def fa4_split(case):
    return run_image_family(case, "fa4_split", case.env.backends["fa4_split"])


@register("absorbed", m03_key="m03_absorbed_ms")
def absorbed(case):
    c = case.memo.get("absorbed")
    if c is None:
        from perfbot_kernels import mla_absorb

        dev, n = case.env.dev, case.n
        pad = mla_absorb.bucket(n)
        assert pad <= case.G, (n, case.ctx)
        c = case.memo["absorbed"] = {
            "n": n, "ctx": case.ctx, "G": case.G, "q_nope": case.q_nope, "q_pe": case.q_pe,
            "new_c": case.kv_c, "new_pe": case.k_pe.view(n, R), "cache": case.cache_u8.view(FP8),
            "block_table": case.block_table, "slots": case.slots, "local": case.local,
            "seq_local": torch.tensor([case.local], dtype=torch.int32, device=dev),
            "seq_global": torch.tensor([case.G], dtype=torch.int32, device=dev),
            "pad": pad, "qsl": torch.tensor([0, pad], dtype=torch.int32, device=dev),
            "out": torch.empty(n, HL, V, dtype=torch.bfloat16, device=dev)}
    return case.env.absorb.chain(c)
