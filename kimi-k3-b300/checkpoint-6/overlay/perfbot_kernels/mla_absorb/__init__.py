"""Line M: short MLA prefill chunks over a long cached context through the absorbed (latent MQA) decode path.

Fact (AG-02n, docs/expedition/LOG.md): at the agentic point a mixed step that prefills 120-218 new tokens over a
~100k cached context takes 106-305 ms (decode step 14.5 ms); ~45 % is MLA prefill attention over the whole
DCP-gathered context (kv gather, kv_b_proj up-projection, fp8 casts, dense fmha, merge), paid per chunk whatever
the number of new tokens. The absorbed path reads the fp8 latent cache in place on each DCP rank:
  q_nope @ W_UK_T -> fused q-concat + latent cache insert (the decode op) -> DCP q gather -> tokenspeed_mla_decode
  (q_len = n, causal in DCP global coordinates) -> DCP lse combine -> W_UV.
Same ops as the decode branch of MultiHeadLatentAttention._attention; only the query length differs.

Switch PERFBOT_EXP_MLAQ: "0" (default; landed 2026-10-18 by R-M1r + R-M2s, suspended the same day by ruling (c) after
M-04q: the absorbed path carries an extra ~0.03 rel-L2 that the fp8 q does not explain) runs the image path; "1" a prefill whose requests all
have q_len <= PERFBOT_MLAQ_T (default 512, the landed arm's) and a cached context in [PERFBOT_MLAQ_MIN_CTX, PERFBOT_MLAQ_MAX_CTX] (default 1024..300000, the
kernel gate's declared domain) runs absorbed; "shadow" runs both, keeps the image output and the image cache contents, and logs one
`PERFBOT-MLAQ` line per layer call on DCP rank 0 (n, context, image us, absorbed us, difference): the real-kernel
wall of both paths at served shapes (the T crossover) and an in-server numerics check. Shadow syncs per call.
"""

import contextlib
import os
import sys
import types
from pathlib import Path

import torch

ENV = "PERFBOT_EXP_MLAQ"
_hooked = False
_calls = {"absorbed": 0, "shadow": 0}


def mode() -> str:
    return os.environ.get(ENV, "0")  # suspended 2026-10-18 (M-04q, ruling (c)); "1" = absorbed, behind the flag


def threshold() -> int:
    return int(os.environ.get("PERFBOT_MLAQ_T", "512"))  # the landed arms' T


def min_context() -> int:
    """Cached tokens a prefill row needs for the absorbed path (PERFBOT_MLAQ_MIN_CTX, default 1024)."""
    return int(os.environ.get("PERFBOT_MLAQ_MIN_CTX", "1024"))


def max_context() -> int:
    """Most cached tokens a prefill row may have on the absorbed path (PERFBOT_MLAQ_MAX_CTX, default 300000): the
    gate's largest correctness case. M-03 saw the largest image-vs-absorbed differences beyond it (440-516k)."""
    return int(os.environ.get("PERFBOT_MLAQ_MAX_CTX", "300000"))


def bucket(n: int) -> int:
    """Query rows the kernel runs for n new tokens: the next power of two >= 8. tokenspeed_mla compiles one
    kernel per q_len (seq_len_q is in its compile key), so served lengths map onto <= 11 shapes. The block is
    padded at the FRONT: the kernel bounds row q by K - (q_len - 1) + q, so the real rows keep their causal
    bounds; a dummy row with no visible key gets the kernel's -1e6 sentinel (finite, ~0 weight) and is dropped."""
    return max(8, 1 << (n - 1).bit_length())


def install_builder_hook() -> None:
    """Attach `prefill.perfbot_mlaq` (per-request lengths the absorbed path needs) when the switch is on.

    MLACommonMetadataBuilder.build keeps only decode rows' DCP-local lengths; the prefill rows' global and
    DCP-local sequence lengths come from the same CommonAttentionMetadata. CPU values come from the CPU copies
    the builder already holds (no device sync)."""
    global _hooked
    if _hooked or mode() == "0":
        return
    from vllm.model_executor.layers.attention import mla_attention as M

    orig = M.MLACommonMetadataBuilder.build

    def build(self, *args, **kwargs):  # callers pass keywords (gpu/attn_utils.py) plus model-specific extras
        md = orig(self, *args, **kwargs)
        cm = kwargs["common_attn_metadata"] if "common_attn_metadata" in kwargs else args[1]
        p = md.prefill
        if p is None or getattr(self, "dcp_world_size", 1) <= 1 or cm.dcp_local_seq_lens is None:
            return md
        nd, nr = md.num_decodes, md.num_reqs
        qsl = cm.query_start_loc_cpu
        q_lens = (qsl[nd + 1 : nr + 1] - qsl[nd:nr]).tolist()
        limit = threshold() if mode() == "1" else 4096  # shadow samples every size the DCP workspace takes
        pads = [bucket(n) for n in q_lens]
        if not q_lens or max(q_lens) > limit or sum(pads) > 8192:  # 8192: the direct DCP workspace
            return md
        seq_cpu = cm.seq_lens_cpu_upper_bound[nd:nr].tolist()  # exact for prefill rows
        n_part = self.dcp_world_size * self.cp_kv_cache_interleave_size
        if min(seq_cpu) < n_part or any(b > s for b, s in zip(pads, seq_cpu)):
            return md  # every DCP rank holds a key; a padded row stays inside the sequence
        ctxs = [s - n for s, n in zip(seq_cpu, q_lens)]
        if min(ctxs) < min_context() or (mode() == "1" and max(ctxs) > max_context()):
            return md  # the gated domain (M-01g: near-empty contexts tie the image error; shadow samples beyond max)
        local_max = [((s + n_part - 1) // n_part) * self.cp_kv_cache_interleave_size for s in seq_cpu]
        cu = [0]
        for b in pads:
            cu.append(cu[-1] + b)
        p.perfbot_mlaq = types.SimpleNamespace(
            q_lens=q_lens, pads=pads, seq_lens_cpu=seq_cpu, local_max=local_max,
            interleave=self.cp_kv_cache_interleave_size,
            seq_lens=cm.seq_lens[nd:nr], local_seq_lens=cm.dcp_local_seq_lens[nd:nr],
            padded_query_start_loc=torch.tensor(cu, dtype=torch.int32).to(cm.seq_lens.device, non_blocking=True),
            block_table=p.block_table)
        return md

    M.MLACommonMetadataBuilder.build = build
    _hooked = True


def attend(layer, q, kv_c_normed, k_pe, positions, cos_sin_cache, slot_mapping, info, out) -> torch.Tensor:
    """Absorbed attention for the prefill rows; writes out [n, H_local * v_head_dim]. Inserts the new tokens'
    latent into the paged cache first (the decode op), so each query sees itself and its predecessors.
    Returns the query the kernel was given before padding and the DCP gather: [n, H_local, L + R], fp8 (times
    layer._q_scale) on an fp8 cache (reference_report's q8 truth)."""
    q_nope, q_pe = q.split([layer.qk_nope_head_dim, layer.qk_rope_head_dim], dim=-1)
    ql_nope = torch.bmm(q_nope.transpose(0, 1), layer.W_UK_T).transpose(0, 1)
    mqa_q = layer._decode_concat_cache(ql_nope, q_pe, kv_c_normed, k_pe, positions, cos_sin_cache, slot_mapping)
    padded = pad_front(mqa_q, info.q_lens, info.pads)
    padded = layer.dcp_manager.query_gather(padded)
    cache = layer._attn_read_kv_cache()
    lat, lse, start = [], [], 0
    for r, b in enumerate(info.pads):
        md = types.SimpleNamespace(
            num_decodes=1, num_decode_tokens=b, max_seq_len=info.local_max[r], causal=True,
            decode=types.SimpleNamespace(block_table=info.block_table[r : r + 1], seq_lens=info.local_seq_lens[r : r + 1],
                                         dcp_tot_seq_lens=info.seq_lens[r : r + 1]))
        o, l = layer.impl.forward_mqa(padded[start : start + b], cache, md, layer)
        lat.append(o)
        lse.append(l)
        start += b
    latent = lat[0] if len(lat) == 1 else torch.cat(lat)
    lse_all = lse[0] if len(lse) == 1 else torch.cat(lse)
    latent = layer.dcp_manager.combine(latent, lse_all, seq_lens=info.local_seq_lens,
                                       query_start_loc=info.padded_query_start_loc)
    layer._v_up_proj(unpad_front(latent, info.q_lens, info.pads), out=out)
    _calls["absorbed"] += 1
    k = _calls["absorbed"]
    if k & (k - 1) == 0 and k >= 64 and layer.impl.dcp_rank == 0:  # 64, 128, ...: how often the path ran
        print(f"PERFBOT_EXP_MLAQ: absorbed layer calls {k}", flush=True)
    if k == 1:
        print(f"PERFBOT_EXP_MLAQ: first absorbed prefill (rows {info.q_lens} -> {info.pads}) in {layer.layer_name}",
              flush=True)
    return mqa_q


_warmed: set = set()


def served_pages(backend=None) -> list:
    """Every page the served kv cache can have: the backend's int kernel block sizes. The page is part of the kernel's
    compile key. The served page is fixed after warm() (weight load) by KV-cache init: image v1/worker/utils.py
    select_common_block_size(manager block, [backend]) returns the manager block when the backend lists it, else the
    largest listed int that divides it; either way a listed int. K3 serves 64: the hybrid alignment sets the manager
    block to 1536 (P-01c server.log interface.py:911), not listed, so 64. warm() at the load-time cache_config.block_size
    (16) compiled a page never served, and every bucket compiled inside served requests (P-01c, 2026-10-18)."""
    if backend is None:
        from vllm.v1.attention.backends.mla.tokenspeed_mla import TokenspeedMLABackend as backend
    return sorted(s for s in backend.get_supported_kernel_block_sizes() if isinstance(s, int))


def warm(layer) -> None:
    """Mode "1": run tokenspeed_mla_decode once per bucket <= T on dummy data (this rank only, no collective) so
    its per-q_len CuTe DSL compile happens at weight load. Same static arguments as forward_mqa: fp8 q/cache,
    every page in served_pages(), gathered head count, causal, LSE, cp_world = the DCP size."""
    heads = layer.q_pad_num_heads or layer.num_local_heads * layer.dcp_world_size
    if heads in _warmed or mode() != "1":  # once per gathered head count (target, draft)
        return
    _warmed.add(heads)
    import time

    from tokenspeed_mla import get_num_sm, tokenspeed_mla_decode

    dev = layer.W_UV.device
    entry = layer.kv_lora_rank + layer.qk_rope_head_dim
    work = torch.empty(get_num_sm(dev) * heads * 8 * (layer.kv_lora_rank + 1) * 4, dtype=torch.int8, device=dev)
    t0, done, pages = time.time(), [], served_pages()
    for page in pages:
        b = 8
        while b <= threshold():
            n = (b + page - 1) // page + 1
            tokenspeed_mla_decode(
                query=torch.zeros(1, b, heads, entry, dtype=torch.float8_e4m3fn, device=dev),
                kv_cache=torch.zeros(n, page, entry, dtype=torch.float8_e4m3fn, device=dev),
                workspace_buffer=work, kv_lora_rank=layer.kv_lora_rank, qk_rope_head_dim=layer.qk_rope_head_dim,
                block_tables=torch.arange(n, dtype=torch.int32, device=dev).view(1, n),
                seq_lens=torch.tensor([b], dtype=torch.int32, device=dev), max_seq_len=b, softmax_scale=1.0,
                output_scale=1.0, enable_pdl=False, return_lse=True, causal_mask=True,
                causal_seqs=torch.tensor([b * layer.dcp_world_size], dtype=torch.int32, device=dev),
                cp_world=layer.dcp_world_size, cp_rank=layer.impl.dcp_rank)
            if page == pages[0]:
                done.append(b)
            b *= 2
    torch.cuda.synchronize(dev)
    print(f"PERFBOT_EXP_MLAQ: warmed buckets {done} (heads {heads}, pages {pages}) in {time.time() - t0:.1f} s", flush=True)


def pad_front(x, q_lens, pads):
    """[sum n, ...] -> [sum pad, ...]: each request's rows at the END of its padded block, zeros before."""
    if q_lens == pads:
        return x
    out = x.new_zeros((sum(pads),) + tuple(x.shape[1:]))
    s = d = 0
    for n, b in zip(q_lens, pads):
        out[d + b - n : d + b] = x[s : s + n]
        s, d = s + n, d + b
    return out


def unpad_front(x, q_lens, pads):
    if q_lens == pads:
        return x
    parts, d = [], 0
    for n, b in zip(q_lens, pads):
        parts.append(x[d + b - n : d + b])
        d += b
    return parts[0] if len(parts) == 1 else torch.cat(parts)


def shadow_report(layer, info, image_out, absorbed_out, t_img, t_abs) -> None:
    """One line per call on DCP rank 0: shapes, both walls (CUDA events), image-vs-absorbed difference."""
    torch.cuda.synchronize()
    _calls["shadow"] += 1
    if getattr(layer, "dcp_manager", None) is None or layer.impl.dcp_rank != 0:
        return
    a, b = image_out.float(), absorbed_out.float()
    rel = ((a - b).norm() / a.norm().clamp_min(1e-30)).item()
    n = sum(info.q_lens)
    ctx = max(s - q for s, q in zip(info.seq_lens_cpu, info.q_lens))
    print(f"PERFBOT-MLAQ layer={layer.layer_name} reqs={len(info.q_lens)} n={n} ctx_max={ctx} "
          f"image_us={t_img[0].elapsed_time(t_img[1]) * 1e3:.1f} absorbed_us={t_abs[0].elapsed_time(t_abs[1]) * 1e3:.1f} "
          f"rel_l2={rel:.4g} max_abs={(a - b).abs().max().item():.4g} calls={_calls['shadow']}", flush=True)


_ref_taken: dict = {}


def global_positions(W: int, lm: int, I: int, dev=None) -> torch.Tensor:
    """Global position of row (rank k, local j) in a [W * lm] gather of padded shards: DCP deals blocks of I tokens
    round-robin over the W ranks (vllm get_dcp_local_seq_lens), so p = (j // I) * I * W + k * I + j % I."""
    j = torch.arange(lm, device=dev).repeat(W)
    k = torch.arange(W, device=dev).repeat_interleave(lm)
    return (j // I) * I * W + k * I + j % I


def shard_rows(p: torch.Tensor, W: int, lm: int, I: int) -> torch.Tensor:
    """Row of global position p in a [W * lm] gather of padded shards: the inverse of global_positions."""
    return ((p // I) % W) * lm + (p // (I * W)) * I + p % I


def ref_attention(lat, qn, qp, w_uk_t, w_uv, limit, scale, L, chunk=65536, tails=None):
    """fp64 un-absorbed MLA attention per head: k = [c W_UK | k_pe], v = c W_UV; row r sees keys <= limit[r].
    lat: fp64 [K, L + R], or (K, fn) with fn(a, b) -> fp64 rows [a, b) (converted per chunk, never all at once);
    qn [r, HL, P], qp [r, HL, R]; w_uk_t [HL, P, L]; w_uv [HL, L, V]. w_uk_t None: qn is already the absorbed
    (latent) query [r, HL, L] and the scores are qn . c + qp . k_pe (the absorbed path's math, q8 truth). Keys in
    chunks with an online softmax (the whole [G, P] per-head K at 500k context would not fit beside the server).
    tails {name: fp64 [n, L + R]}: alternative keys at positions [K, K + n). The first K keys are read once; each
    tail continues a copy of their softmax state. Returns ref [r, HL, V] without tails, else
    {name: (ref, p_tail [r, HL])} with p_tail = the softmax mass on the tail keys."""
    K, get = (lat.shape[0], lambda a, b: lat[a:b]) if torch.is_tensor(lat) else lat
    dev, f64 = qn.device, torch.float64
    rows, HL, V = qn.shape[0], qn.shape[1], w_uv.shape[-1]
    names = list(tails) if tails else [None]
    ref = {k: torch.empty((rows, HL, V), dtype=f64, device=dev) for k in names}
    mass = {k: torch.empty((rows, HL), dtype=f64, device=dev) for k in names}
    for h in range(HL):
        wk, wv = None if w_uk_t is None else w_uk_t[h].double(), w_uv[h].double()  # [P, L], [L, V]
        q1, q2 = qn[:, h].double(), qp[:, h].double()

        def step(st, keys, a, tail):
            m, s_sum, acc, s_tail = st
            cl, pe = keys[:, :L], keys[:, L:]
            s = ((q1 @ cl.T if wk is None else q1 @ (cl @ wk.T).T) + q2 @ pe.T) * scale
            s = s.masked_fill(torch.arange(a, a + keys.shape[0], device=dev)[None, :] > limit[:, None], float("-inf"))
            m_new = torch.maximum(m, s.max(-1).values)
            c_old = torch.exp(m - m_new)  # 0 on the first chunk (m = -inf, m_new finite: limit >= 0)
            p = torch.exp(s - m_new[:, None])
            return (m_new, s_sum * c_old + p.sum(-1), acc * c_old[:, None] + p @ (cl @ wv),
                    s_tail * c_old + (p.sum(-1) if tail else 0))

        z = torch.zeros((rows,), dtype=f64, device=dev)
        st = (torch.full((rows,), float("-inf"), dtype=f64, device=dev), z, torch.zeros((rows, V), dtype=f64, device=dev), z)
        for a in range(0, K, chunk):
            st = step(st, get(a, min(K, a + chunk)), a, False)
        for k in names:
            m, s_sum, acc, s_tail = st if k is None else step(st, tails[k], K, True)
            ref[k][:, h] = acc / s_sum[:, None]
            mass[k][:, h] = s_tail / s_sum
    return ref[None] if not tails else {k: (ref[k], mass[k]) for k in names}


def ref_samples() -> int:
    """M-04: fp64 reference calls per (layer, region) in shadow mode (PERFBOT_MLAQ_REF, default 0 = off)."""
    return int(os.environ.get("PERFBOT_MLAQ_REF", "0"))


def pshift() -> int:
    """REF-locator (QUEUE.md, brief 11): in shadow REF samples, also run the absorbed path through the P-shift kernel
    with this shift and report it against every truth (PERFBOT_MLAQ_PSHIFT, default 0 = off)."""
    return int(os.environ.get("PERFBOT_MLAQ_PSHIFT", "0"))


P_FMA = "fma_c = (0.0 - row_max_new) * softmax_params.softmax_scale_log2\n"
P_LSE = "cute.math.log2(row_sum, fastmath=True)\n"


def pshift_source(src, shift):
    """H-P8 (M-H02): mla_decode_fp8.py with P scaled by 2^shift before its fp8 store (the softmax exp2 offset, :3327;
    row_sum follows, so O / row_sum is unchanged) and lse lowered by shift at both epilogue sites (:3883, :3914)."""
    if src.count(P_FMA) != 1 or src.count(P_LSE) != 2:
        raise ValueError(f"pshift: unexpected source ({src.count(P_FMA)} fma sites, {src.count(P_LSE)} lse sites)")
    return src.replace(P_FMA, P_FMA[:-1] + f" + {float(shift)}\n").replace(P_LSE, P_LSE[:-1] + f" - {float(shift)}\n")


_pshift: dict = {}


def pshift_parts(shift, out_dir=None):
    """(kernel class, compile getter, source sha[:12]) of mla_decode_fp8 with P shifted, built once per shift and
    process. The patched source is written to a per-process file (the DSL reads kernel source through inspect, so an
    exec'd string would compile the stock file) and loaded as a private module next to the stock one; the getter is a
    fresh cache over the stock getter's function, so stock and shifted compiles never share an entry."""
    if shift not in _pshift:
        import functools
        import hashlib
        import importlib.util
        import tempfile

        import tokenspeed_mla.mla_decode as tmd
        import tokenspeed_mla.mla_decode_fp8 as f8

        new = pshift_source(Path(f8.__file__).read_text(), shift)
        path = Path(out_dir or tempfile.mkdtemp(prefix="perfbot-pshift-")) / f"mla_decode_fp8_pshift{shift}_{os.getpid()}.py"
        path.write_text(new)
        name = f"{f8.__name__}_pshift{shift}"
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        _pshift[shift] = (mod.BlackwellMultiHeadLatentAttentionForwardFP8,
                          functools.cache(tmd._get_compiled_mla_kernel.__wrapped__),
                          hashlib.sha256(new.encode()).hexdigest()[:12])
    return _pshift[shift]


@contextlib.contextmanager
def pshift_kernel(shift, out_dir=None):
    """tokenspeed_mla_decode runs the P-shift kernel inside the block (mla_decode looks up the fp8 class and the
    compile getter as module globals at call time); the stock class and getter are restored on exit."""
    import tokenspeed_mla.mla_decode as tmd

    cls, getter, _ = pshift_parts(shift, out_dir)
    keep = tmd.BlackwellMultiHeadLatentAttentionForwardFP8, tmd._get_compiled_mla_kernel
    tmd.BlackwellMultiHeadLatentAttentionForwardFP8, tmd._get_compiled_mla_kernel = cls, getter
    try:
        yield
    finally:
        tmd.BlackwellMultiHeadLatentAttentionForwardFP8, tmd._get_compiled_mla_kernel = keep


MAP_REL_MAX = 0.035  # map_rel above this flags the sample bad_map=1: fp8 e4m3 rounding gives ~0.027, a wrong map ~1.4
TRUTHS = ("t8", "mix")
FP8_MAX = 448.0  # e4m3fn largest finite
Q_SUB, Q_TINY = 2.0 ** -6, 2.0 ** -9  # e4m3 smallest normal, smallest subnormal


def fp8_quant(x, s: float):
    """fp64 x -> the fused decode concat's fp8 query (vllm csrc fused_kimi_k3_mla_key_concat_kv_cache_kernel.cu
    copyChunk8: bf16 input, x * fp32(1 / s), clamp +-448, E4M3 round to nearest) -> fp64 times s."""
    inv = torch.tensor(1.0 / s, dtype=torch.float32)
    return (x.to(torch.bfloat16).float() * inv).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).double() * s


def q_sums(qa, q8d, s: float, s_layer: float, L: int):
    """Sums for the q statistics (all-reducible): per part (latent [..., :L], pe [..., L:]) sum (q8d - qa)^2,
    sum qa^2, #(|qa| / s < 2^-6), #(|qa| / s < 2^-9), numel; then over both parts sum (fp8_quant(qa, x) - qa)^2 for
    x = s (emulation of the served scale) and x = s_layer. qa: fp64 pre-quantization absorbed q; q8d: dequantized."""
    out = []
    for a, b in ((qa[..., :L], q8d[..., :L]), (qa[..., L:], q8d[..., L:])):
        out += [(b - a).pow(2).sum(), a.pow(2).sum(), (a.abs() < Q_SUB * s).sum(), (a.abs() < Q_TINY * s).sum(),
                a.numel()]
    out += [(fp8_quant(qa, x) - qa).pow(2).sum() for x in (s, s_layer)]
    return torch.stack([torch.as_tensor(v, dtype=torch.float64, device=qa.device) for v in out])


def q_fields(v, amax, s: float, s_layer: float) -> dict:
    """REF-line q statistics from reduced q_sums v and the reduced absmax (latent, pe)."""
    v, (al, ap) = v.tolist(), amax.tolist()
    lat, pe = v[:5], v[5:10]
    norm = lat[1] + pe[1]
    f = {"q_scale": s, "q_amax_lat": al, "q_amax_pe": ap, "q_fill": max(al, ap) / (FP8_MAX * s),
         "q_rel": ((lat[0] + pe[0]) / norm) ** 0.5}
    for name, p in (("lat", lat), ("pe", pe)):
        f.update({f"q_rel_{name}": (p[0] / max(p[1], 1e-300)) ** 0.5, f"q_f6_{name}": p[2] / p[4],
                  f"q_f9_{name}": p[3] / p[4]})
    f.update(q_rel_emu=(v[10] / norm) ** 0.5, q_scale_layer=s_layer, q_rel_layer=(v[11] / norm) ** 0.5)
    return f


def ref_due(layer, info):
    """The region ("in" / "out") reference_report samples this call in, else None; changes no counter. The same
    on every rank (layer, metadata and per-layer counters only), so a caller may run collectives on it."""
    if (torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()) or len(info.q_lens) != 1:
        return None
    region = "in" if info.seq_lens_cpu[0] - info.q_lens[0] <= max_context() else "out"
    return region if _ref_taken.get((layer.layer_name, region), 0) < ref_samples() else None


@torch.no_grad()
def reference_report(layer, info, q, kv_c_normed, k_pe, image_out, absorbed_out, rows: int = 16,
                     chunk: int = 65536, group=None, force: bool = False, q8=None, q_scale=None, extra=None):
    """M-04 (shadow only, single-request calls): both paths' error against two fp64 truths on real activations.

    Truths = un-absorbed attention in fp64 (k_nope = c W_UK, v = c W_UV, per head), causal, for up to `rows` query
    rows (evenly spaced, last row included), over keys that differ only in the n new tokens:
      t8  every key as stored in the fp8 cache (the new tokens' rows too, as the insert left them), times the scale;
      mix the stored context and the new tokens' bf16 latent (what each path was given).
    The cache shards are all-gathered as uint8 (G x 576 bytes) and converted to fp64 per 64k-key chunk; the context
    is read once and both tails continue its softmax state (ref_attention tails). The target is NoPE (no rotary):
    q_pe and k_pe enter as they are. Sampling is the same on every rank (it depends only on the layer, the metadata
    and per-layer counters), so the collectives match; nothing runs while a stream captures. Region "in" = cached
    context within max_context(), "out" = beyond it; up to ref_samples() calls of each per layer (force=True: always;
    group: the DCP process group, default vllm's). Reads only: cache, block table and metadata are not written.
    One key=value line per sample on DCP rank 0: rel-L2 over all heads (sums all-reduced over the DCP group) of image
    and absorbed vs each truth and their ratio; pnew_* = the softmax mass (truth mix, mean over all heads) on the new
    keys, min/median/max over the sampled rows; map_rel = the gathered new-token rows vs their bf16 source (fp8
    rounding ~0.027 if the position map is right, ~1.4 if not), bad_map=1 above MAP_REL_MAX.
    q8 (attend's return, [n, HL, L + R]) with q_scale (layer._q_scale): a third truth q8 = the t8 keys with the
    exact query the absorbed kernel used, dequantized (q8 x q_scale), in latent space (scores q8_lat . c + q8_pe . k_pe,
    out = softmax . c then W_UV), and q statistics over all n rows: the pre-quantization absorbed q
    qa = [q_nope W_UK | q_pe] in fp64; per part absmax (all-reduced max), fraction of |qa| / q_scale below 2^-6 / 2^-9,
    quantization rel-L2 ||q8 x q_scale - qa|| / ||qa||; q_fill = absmax / (448 q_scale); q_rel_emu = fp8_quant at
    q_scale (checks the emulation against q_rel); q_scale_layer = absmax / 224 and q_rel_layer = fp8_quant there.
    extra {name: [n, HL * V] output}: more paths on the same call (absorbed_ps = the absorbed path through the
    P-shift kernel, pshift()), each reported as <name>_rel_<truth> and ratio_<name>_<truth> (vs image).
    q8 None: no q8 fields. Returns the line's values (every rank) plus "ref" {truth: [r, HL, V] this rank's heads},
    "sel", "pnew_rows"."""
    if (torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()) or len(info.q_lens) != 1:
        return None
    n, G = info.q_lens[0], info.seq_lens_cpu[0]
    ctx = G - n
    region = "in" if ctx <= max_context() else "out"
    if not force:
        if ref_due(layer, info) is None:
            return None
        key = (layer.layer_name, region)
        _ref_taken[key] = _ref_taken.get(key, 0) + 1
    import time

    import torch.distributed as dist

    if group is None:
        from vllm.distributed.parallel_state import get_dcp_group

        group = get_dcp_group().device_group
    t0 = time.time()
    W, I, rank = layer.dcp_world_size, info.interleave, layer.impl.dcp_rank
    dev = q.device
    # This rank's shard of the request: local index j -> paged slot; padded to the common local_max for the gather.
    cache = layer._attn_read_kv_cache()
    page = cache.shape[1]
    flat = cache.view(-1, cache.shape[-1])
    lm = info.local_max[0]
    j = torch.arange(lm, device=dev)
    bt = info.block_table[0].long()
    slots = bt[torch.clamp(j // page, max=bt.numel() - 1)] * page + j % page
    shard = flat[slots].view(torch.uint8).contiguous()
    gathered = torch.empty((W * lm, shard.shape[1]), dtype=torch.uint8, device=dev)
    dist.all_gather_into_tensor(gathered, shard, group=group)
    k_scale = float(layer._k_scale)

    def stored(a, b):  # fp64 stored latent of global positions [a, b)
        rows_ = shard_rows(torch.arange(a, b, device=dev), W, lm, I)
        return gathered[rows_].view(cache.dtype).double() * k_scale

    new = torch.cat([kv_c_normed[:n].reshape(n, -1), k_pe[:n].reshape(n, -1)], -1).double()
    tail8 = stored(ctx, G)
    map_rel = ((tail8 - new).norm() / new.norm().clamp_min(1e-30)).item()
    L = layer.kv_lora_rank
    sel = torch.unique(torch.linspace(0, n - 1, min(rows, n), device=dev).round().long())
    qd = q[:n][sel].double()  # [r, HL, P + R]
    qn, qp = qd[..., : layer.qk_nope_head_dim], qd[..., layer.qk_nope_head_dim :]
    HL, V = qd.shape[1], layer.v_head_dim
    truth = ref_attention((ctx, stored), qn, qp, layer.W_UK_T, layer.W_UV, ctx + sel, layer.scale, L, chunk,
                          tails={"t8": tail8, "mix": new})
    truths, qv = TRUTHS, torch.zeros(0, dtype=torch.float64, device=dev)
    if q8 is not None:
        qs = float(q_scale)
        q8d = q8[:n].double() * (qs if q8.element_size() == 1 else 1.0)  # a bf16 q8 (no quant query input) is dequantized
        truth.update(ref_attention((ctx, stored), q8d[sel][..., :L], q8d[sel][..., L:], None, layer.W_UV, ctx + sel,
                                   layer.scale, L, chunk, tails={"q8": tail8}))
        truths = TRUTHS + ("q8",)
        qall, P = q[:n].double(), layer.qk_nope_head_dim
        qa = torch.cat([torch.einsum("nhp,hpl->nhl", qall[..., :P], layer.W_UK_T.double()), qall[..., P:]], -1)
        amax = torch.stack([qa[..., :L].abs().max(), qa[..., L:].abs().max()])
        dist.all_reduce(amax, op=dist.ReduceOp.MAX, group=group)
        s_layer = amax.max().item() / (FP8_MAX / 2)  # headroom 2x
        qv = q_sums(qa, q8d, qs, s_layer, L)
    outs = {"image": image_out, "absorbed": absorbed_out, **(extra or {})}
    outs = {k: o[:n][sel].view(sel.numel(), HL, V).double() for k, o in outs.items()}
    sums, m = [], len(outs) + 1  # per truth: one error per path, then the truth's norm
    for t in truths:
        ref = truth[t][0]
        sums += [(o - ref).pow(2).sum() for o in outs.values()] + [ref.pow(2).sum()]
    k0, k1 = m * len(truths), m * len(truths) + sel.numel()
    vec = torch.cat([torch.stack(sums), truth["mix"][1].sum(1), qv])
    dist.all_reduce(vec, group=group)
    res = {"layer": layer.layer_name, "region": region, "n": n, "ctx": ctx, "rows": sel.numel()}
    for i, t in enumerate(truths):
        e = {name: (vec[m * i + j] / vec[m * i + m - 1]).sqrt().item() for j, name in enumerate(outs)}
        res.update({f"{name}_rel_{t}": v for name, v in e.items()})
        res[f"ratio_{t}"] = e["absorbed"] / max(e["image"], 1e-30)
        res.update({f"ratio_{name}_{t}": e[name] / max(e["image"], 1e-30) for name in extra or {}})
    pnew = (vec[k0:k1] / (HL * W)).tolist()
    s = sorted(pnew)
    res.update(pnew_min=s[0], pnew_med=s[len(s) // 2], pnew_max=s[-1], map_rel=map_rel,
               bad_map=int(map_rel > MAP_REL_MAX))
    if q8 is not None:
        res.update(q_fields(vec[k1:], amax, qs, s_layer))
    res["s"] = time.time() - t0
    if rank == 0:
        print("PERFBOT-MLAQ-REF " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                             for k, v in res.items()), flush=True)
    res.update(ref={t: truth[t][0] for t in truths}, sel=sel, pnew_rows=pnew)
    return res
