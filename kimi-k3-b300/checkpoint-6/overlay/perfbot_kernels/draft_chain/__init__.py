# SPDX-License-Identifier: Apache-2.0
"""Item 15 form F-B in the served draft (DSpark MLA), behind PERFBOT_EXP_DRAFT_CHAIN=1 (read once, at import).

Call sites (insert-only `# PERFBOT P-15` lines; switch off = no call changes, the image path runs unchanged):
  dspark_mla.py  module global `fused_allreduce_rms_norm = draft_chain.routed_ar(fused_allreduce_rms_norm)`:
                 off -> the same function object; on -> ar_norm below.
  dspark_mla.py  K3DSparkForCausalLM.process_weights_after_loading: `draft_chain.prepare(self.model)`.
  mla.py         MultiHeadLatentAttention._apply_q_lora_attention, first lines: `out = draft_chain.attention(...)`;
                 None -> the image path.

Kernel 1 (draft_ar, two-shot Lamport AR + residual + RMSNorm), routed by the padded draft M (DESIGN.md §8.1): served
one-shot at M <= 16 (served M 7, 14), two-shot at 17..64 (served M 21..56), served above 64. The bits are equal on both
paths. Parity = the call index % 2 (10 calls per draft step: L0 post, L1-L4 in + post, final; an even count).
Kernels 2 + 3 (draft_query + lse_combine, Lamport, one DcpBox for the draft): kernel 2 replaces fused_q_kv_rmsnorm,
q_b, W_UK bmm, the q-concat + FP8 cache insert and the DCP q gather; tokenspeed (impl.forward_mqa) reads the q
mailbox; kernel 3 replaces the DCP combine and the W_UV bmm. Parity = layer index % 2 (5 calls per step, odd).
Why one parity may repeat across steps (layer 4 -> next step's layer 0) [e]: between two pair calls there is always a
draft AR collective (served flashinfer or two-shot). A rank leaves it only after every rank published its AR input,
which every rank does after its own kernel 3 finished (stream order; the AR publish follows griddepcontrol.wait). So
all ranks finished reading a mailbox before any rank writes the same parity again. Checked by selftest/b300_chain.py
(graph replays vs eager, fresh inputs per replay).
Declines (image path, one log line per reason): switch off, TP != 8 or DCP != 8, heads per rank != 8, not q-LoRA,
not the tokenspeed backend (FP8 query, base-2 LSE, no padded heads), KV cache not plain fp8, q_b weight not bf16, rope table not [*, 64] fp32, a batch with prefill rows, M > 64,
no metadata (dummy run), a non-contiguous qkv / tokenspeed output, unexpected index dtypes.
"""

import os
from types import SimpleNamespace

import torch

SWITCH = os.environ.get("PERFBOT_EXP_DRAFT_CHAIN") == "1"
SERVED_MAX_M = __import__("perfbot_kernels.k5q", fromlist=["k5q"]).SERVED_MAX_M  # PERFBOT K5 item 10 (land-8: 16) # kernel 1: served one-shot at padded M <= 16, two-shot above (i15-ar-v2 crossover 14 -> 21)
CAP = 64  # mailbox rows (DcpBox, TwoShot)
WORLD, HEADS, H = 8, 8, 7168
_said = set()
_state = None


def say(msg):
    if msg not in _said:
        _said.add(msg)
        print(f"PERFBOT-DRAFT-CHAIN {msg}", flush=True)


def routed_ar(served):
    """dspark_mla's module-global AR: the same function object when the switch is off."""
    if not SWITCH:
        return served

    def ar(hidden_states, residual, norm):
        out = ar_norm(hidden_states, residual, norm)
        return served(hidden_states, residual, norm) if out is None else out

    return ar


def ar_norm(x, residual, norm, state=None):
    """Kernel 1 two-shot at 16 < M <= 64 on a prepared site; None -> the served op."""
    state = state or _state
    idx = getattr(norm, "_perfbot_ar_idx", None)
    m = x.shape[0]
    if state is None or idx is None or not SERVED_MAX_M < m <= CAP:
        return None
    if x.dtype != torch.bfloat16 or not x.is_contiguous() or not residual.is_contiguous():
        say("decline ar: dtype / layout")
        return None
    normed = torch.empty_like(x)
    rout = torch.empty_like(x)
    state.ar.launch(idx % 2, x, residual, norm.weight, norm.variance_epsilon, normed.view(1, m, H), rout.view(1, m, H))
    return normed, rout


class State:
    """Per process: the TP two-shot box (kernel 1) and the draft's DcpBox with kernels 2 + 3 on it."""

    def __init__(self, tp_group, dcp_group, device):
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.mla.dcp import DcpBox

        from perfbot_kernels.draft_ar import TwoShotBox
        from perfbot_kernels.draft_query import LamportBox as QueryBox
        from perfbot_kernels.lse_combine import LamportBox as ValueBox

        self.ar = TwoShotBox(tp_group.device_group, device, tp_group.rank_in_group)
        self.box = DcpBox(dcp_group.device_group, device, dcp_group.world_size, dcp_group.rank_in_group, HEADS,
                          cap=CAP)
        self.k2, self.k3 = QueryBox(self.box), ValueBox(self.box)


def layer_weights(attn):
    """Kernel 2 / 3 weights of one draft MLA layer, in the layouts the gates ran on (contiguous; W_UV latent axis
    contiguous); None + a reason when the layer does not fit."""
    if attn.q_lora_rank is None or attn.num_local_heads != HEADS:
        return None, "not q-LoRA / heads per rank != 8"
    impl = attn.impl  # kernel 2 publishes an FP8 query and kernel 3 merges a base-2 LSE: tokenspeed's contract
    if ("Tokenspeed" not in type(impl).__name__ or getattr(impl, "supports_quant_query_input", None) is not True
            or getattr(impl, "lse_base_on_e", None) is not False or getattr(impl, "q_pad_num_heads", None) is not None):
        return None, f"attention backend {type(impl).__name__}"
    if attn.kv_cache_dtype not in ("fp8", "fp8_e4m3"):
        return None, f"kv cache {attn.kv_cache_dtype}"
    qb = attn.q_b_proj.weight
    if qb.dtype != torch.bfloat16 or tuple(qb.shape) != (HEADS * 192, 1536):
        return None, f"q_b weight {qb.dtype} {tuple(qb.shape)}"
    rope = getattr(attn.rotary_emb, "cos_sin_cache", None)
    if rope is None or rope.dtype != torch.float32 or rope.shape[-1] != 64 or not rope.is_contiguous():
        return None, "rope table"
    k2 = (attn.q_a_layernorm.weight.data.contiguous(), attn.kv_a_layernorm.weight.data.contiguous(),
          qb.data.contiguous(), attn.W_UK_T.data.contiguous())
    wuv = attn.W_UV.data.transpose(1, 2).contiguous().transpose(1, 2)  # [8, 512, 128], latent stride 1
    return SimpleNamespace(k2=k2, wuv=wuv), None


def prepare(model, tp_group=None, dcp_group=None):
    """Draft model, after its weights are final: build the boxes (collective: every rank calls this) and tag the
    sites. Off or any decline: no site is tagged and the image path runs."""
    global _state
    if not SWITCH:
        return
    from vllm.distributed import get_dcp_group, get_tp_group

    tp_group, dcp_group = tp_group or get_tp_group(), dcp_group or get_dcp_group()
    if tp_group.world_size != WORLD or dcp_group.world_size != WORLD:
        say(f"off: TP {tp_group.world_size} DCP {dcp_group.world_size}")
        return
    per_layer = []
    for layer in model.layers:
        w, why = layer_weights(layer.self_attn)
        if w is None:
            say(f"off: {why}")
            return
        per_layer.append(w)
    calls = []  # AR sites in call order: L0 post, then (in, post) per layer, then the final norm
    for i, layer in enumerate(model.layers):
        calls += ([layer.input_layernorm] if i else []) + [layer.post_attention_layernorm]
    calls.append(model.final_norm)
    if len(calls) % 2:
        say(f"off: {len(calls)} AR calls per step (odd: the kernel 1 parity would repeat)")
        return
    device = model.layers[0].self_attn.q_b_proj.weight.device
    _state = State(tp_group, dcp_group, device)
    for i, layer in enumerate(model.layers):
        layer.self_attn._perfbot_chain = (_state, i, per_layer[i])
    for idx, norm in enumerate(calls):
        norm._perfbot_ar_idx = idx
    say(f"on: {len(model.layers)} layers, {len(calls)} AR sites, two-shot at {SERVED_MAX_M} < M <= {CAP}")


def attention(attn, positions, qkv):
    """mla.py hook: kernels 2 + 3 around tokenspeed for an all-decode draft batch; None -> the image path."""
    chain = getattr(attn, "_perfbot_chain", None)
    if chain is None:
        return None
    from vllm.forward_context import get_forward_context

    ctx = get_forward_context()
    if ctx.attn_metadata is None:
        return None
    md = ctx.attn_metadata[attn.layer_name]
    out = torch.empty(qkv.shape[0], HEADS * 128, dtype=qkv.dtype, device=qkv.device)
    return out if core(attn, chain, positions, qkv, md, ctx.slot_mapping[attn.layer_name], out) else None


def core(attn, chain, positions, qkv, md, slot_mapping, out):
    """Kernels 2 + 3 for one layer; writes out[:M]. False (nothing launched) when the batch does not fit."""
    state, idx, w = chain
    m = md.num_actual_tokens
    if md.num_decode_tokens != m or not 1 <= m <= CAP:
        say("decline attention: prefill rows or M > 64")
        return False
    seq, qsl = md.decode.seq_lens, md.query_start_loc[: md.num_decodes + 1]
    if (not qkv.is_contiguous() or positions.dtype != torch.int64 or slot_mapping.dtype != torch.int64
            or seq.dtype != torch.int32 or qsl.dtype != torch.int32):
        say("decline attention: qkv layout / index dtype")
        return False
    cache = attn.kv_cache.view(torch.uint8)
    if cache.dim() != 3 or cache.shape[-1] != 576:
        say(f"decline attention: kv cache shape {tuple(cache.shape)}")
        return False
    parity = idx % 2
    state.k2.run(qkv[:m], w.k2, attn.rotary_emb.cos_sin_cache, positions[:m], cache[None], slot_mapping[:m].view(1, m),
                 attn._q_scale_inv, attn._k_scale_inv, attn.rms_norm_eps, parity=parity)
    mqa_q = state.box.q[parity, :m].view(torch.float8_e4m3fn)  # [M, 64, 576]: the gathered query
    rows, lse = attn.impl.forward_mqa(mqa_q, attn._attn_read_kv_cache(), md, attn)
    if not rows.is_contiguous() or not lse.is_contiguous():
        # tokenspeed already read the mailbox and kernel 3 must re-arm it: a hard error, not a decline
        raise RuntimeError(f"PERFBOT P-15: tokenspeed output strides {rows.stride()} / {lse.stride()}")
    state.k3.run(rows.view(1, m, WORLD * HEADS, -1), lse.view(1, m, -1), seq[None], qsl, w.wuv,
                 out[:m].view(1, m, HEADS * 128), parity=parity)
    return True
