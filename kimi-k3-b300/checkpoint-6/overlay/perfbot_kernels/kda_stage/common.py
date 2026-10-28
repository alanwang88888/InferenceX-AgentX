"""Shared harness of the KDA stage: weights, served state, image control chain, fp64 reference, timing.

Served point (round-3 nsys, conc 6, K=7 spec): FULL graph at T = 8 N tokens, the spec-verify path, TP 8 -> H 12 local
heads, K = V = 128, hidden 7168, in_proj rows [q 1536 | k 1536 | v 1536 | og 1536 | f_a 128 | b 12 | pad 4] = 6288.

State layout (image mamba_utils, SD conv layout):
  conv pool  [pool, 3 + 7, 4608] bf16 per block; the kernels see it as (pool, dim, state_len) through .transpose(-1, -2)
  ssm pool   [pool, 12, 128, 128] fp32 (V rows, K contiguous); slots at any stride (the server's hybrid pool)
  spec_idx   [N, 8] int32: slot of the state after token t; spec_idx[:, 0] is also the conv slot; <= 0 = null
  accepted   [N] int32 in 1..8: the start state is at spec_idx[n, accepted - 1], the conv window starts at column
             accepted - 1
"""

import json
import os
import statistics
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

H, D, HID, TP, NSPEC, WIDTH = 12, 128, 7168, 8, 7, 4
P = H * D  # 1536 local projection
QKV = 3 * P  # 4608 conv channels
SLOTS = NSPEC + 1  # tokens per request on the verify path
STATE_LEN = WIDTH - 1 + NSPEC  # 10
COLS_RAW = 4 * P + D + H  # 6284
COLS = (COLS_RAW + 15) // 16 * 16  # 6288, the image's 16-alignment
EPS = 1e-5
SCALE = D ** -0.5
MODEL = os.environ.get("PERFBOT_MODEL_DIR", "/model")


def results_path(name: str) -> Path:
    root = Path(os.environ.get("PERFBOT_WORKSPACE", "/workspace"))
    out = root / os.environ.get("PERFBOT_RESULTS", "results/measure")
    out.mkdir(parents=True, exist_ok=True)
    return out / name


# ---------------------------------------------------------------- weights


def _find(d, key):
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for v in d.values():
            r = _find(v, key)
            if r is not None:
                return r
    return None


def model_config(model: str = MODEL) -> dict:
    cfg = json.loads((Path(model) / "config.json").read_text())
    lac = _find(cfg, "linear_attn_config") or {}
    return {"lower_bound": lac.get("gate_lower_bound"), "num_heads": lac.get("num_heads"),
            "head_dim": lac.get("head_dim"), "conv": lac.get("short_conv_kernel_size"), "raw": lac}


def kda_layers(model: str = MODEL) -> list[str]:
    """Prefixes of the checkpoint's KDA layers, in layer order."""
    wm = json.loads((Path(model) / "model.safetensors.index.json").read_text())["weight_map"]
    pre = [k[: -len(".q_conv1d.weight")] for k in wm if k.endswith("self_attn.q_conv1d.weight")]
    return sorted(pre, key=lambda p: int(p.split(".layers.")[1].split(".")[0]))


@dataclass
class Layer:
    w_in: torch.Tensor  # [COLS, HID] bf16, image row order q|k|v|og|f_a|b|pad
    w_fb: torch.Tensor  # [P, D] bf16
    w_conv: torch.Tensor  # [QKV, WIDTH] fp32, oldest tap first
    a_log: torch.Tensor  # [H] fp32
    dt_bias: torch.Tensor  # [P] fp32
    w_norm: torch.Tensor  # [D] fp32 (the image's decode copy is fp32; o_norm.weight dtype kept in `norm_dtype`)
    w_o: torch.Tensor  # [HID, P] bf16
    lower_bound: float
    name: str


def load_layer(prefix: str, rank: int = 0, model: str = MODEL, device="cuda", lower_bound=None) -> Layer:
    from safetensors import safe_open

    wm = json.loads((Path(model) / "model.safetensors.index.json").read_text())["weight_map"]
    handles = {}

    def get(name, rows=None, cols=None):
        f = wm[f"{prefix}.{name}"]
        if f not in handles:
            handles[f] = safe_open(str(Path(model) / f), framework="pt", device="cpu")
        sl = handles[f].get_slice(f"{prefix}.{name}")
        shape = sl.get_shape()
        if rows is not None and cols is not None:
            t = sl[rows[0]:rows[1], cols[0]:cols[1]]
        elif rows is not None:
            t = sl[rows[0]:rows[1]]
        elif cols is not None:
            t = sl[:, cols[0]:cols[1]]
        else:
            t = sl[:]
        return t, shape

    r0, r1 = rank * P, (rank + 1) * P
    h0, h1 = rank * H, (rank + 1) * H
    parts = [get(f"{n}_proj.weight", rows=(r0, r1))[0] for n in ("q", "k", "v", "g")]
    parts.append(get("f_a_proj.weight")[0])  # replicated
    parts.append(get("b_proj.weight", rows=(h0, h1))[0])
    w_in = torch.cat([p.to(torch.bfloat16) for p in parts], 0)
    w_in = torch.cat([w_in, torch.zeros(COLS - COLS_RAW, HID, dtype=torch.bfloat16)], 0)
    w_fb = get("f_b_proj.weight", rows=(r0, r1))[0].to(torch.bfloat16)
    convs = []
    for n in ("q", "k", "v"):
        t, _ = get(f"{n}_conv1d.weight", rows=(r0, r1))
        convs.append(t.reshape(P, WIDTH).float())
    w_conv = torch.cat(convs, 0)
    a, a_shape = get("A_log")
    a_log = a.reshape(-1)[h0:h1].float() if len(a_shape) != 1 else get("A_log", rows=(h0, h1))[0].float()
    dt, _ = get("dt_bias")
    dt_bias = dt.reshape(-1)[r0:r1].float()
    w_norm = get("o_norm.weight")[0].float()
    w_o = get("o_proj.weight", cols=(r0, r1))[0].to(torch.bfloat16)
    if lower_bound is None:
        lower_bound = model_config(model)["lower_bound"]
    to = lambda t: t.contiguous().to(device)
    return Layer(to(w_in), to(w_fb), to(w_conv), to(a_log), to(dt_bias), to(w_norm), to(w_o), float(lower_bound),
                 prefix)


# ---------------------------------------------------------------- served state


@dataclass
class State:
    x: torch.Tensor  # [T, HID] bf16 (the layer input after its norm)
    conv: torch.Tensor  # [pool, STATE_LEN, QKV] bf16 (SD storage)
    ssm: torch.Tensor  # [pool, H, D, D] fp32 view, slot stride may exceed H*D*D
    spec_idx: torch.Tensor  # [N, SLOTS] int32
    accepted: torch.Tensor  # [N] int32
    qsl: torch.Tensor  # [N + 1] int32
    valid: torch.Tensor  # [N] bool: request has a real (non-null) slot

    @property
    def n(self):
        return self.spec_idx.shape[0]

    @property
    def t(self):
        return self.x.shape[0]

    def conv_view(self):  # (pool, dim, state_len), as the image kernels take it
        return self.conv.transpose(-1, -2)

    def clone(self) -> "State":
        ssm_store = self.ssm.as_strided((self.ssm.shape[0], self.ssm.stride(0)), (self.ssm.stride(0), 1))
        store = ssm_store.clone()
        ssm = store.as_strided(self.ssm.shape, self.ssm.stride())
        rs = self.spec_idx.stride(0)  # keep the block-table row stride (and the columns past SLOTS)
        idx = self.spec_idx.as_strided((self.n, rs), (rs, 1)).clone()[:, :SLOTS]
        return State(self.x.clone(), self.conv.clone(), ssm, idx, self.accepted.clone(), self.qsl.clone(),
                     self.valid.clone())


def make_state(n: int, seed: int = 0, null_requests: int = 0, slot_pad: int = 1024, device="cuda",
               x_std: float = 1.0, idx_row_stride: int = SLOTS) -> State:
    """N requests x SLOTS tokens. Slots are a random permutation of 1..pool-1 (slot 0 = null); the last
    `null_requests` requests carry null slots (padded requests). The ssm pool slot stride is H*D*D + slot_pad floats
    (the server's slots are not dense). idx_row_stride > SLOTS: spec_idx is the [:, :SLOTS] slice of an
    [N, idx_row_stride] int32 table, as the server's block_table_tensor[:N, :8]; the columns past SLOTS hold the
    pool's unused slots, so a kernel that reads them writes slots the check expects untouched."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    pool = 1 + n * SLOTS + 3
    perm = torch.randperm(pool - 1, generator=g)[: n * SLOTS] + 1
    spec_idx = perm.view(n, SLOTS).to(torch.int32)
    if idx_row_stride > SLOTS:
        spare = torch.tensor(sorted(set(range(1, pool)) - set(perm.tolist())), dtype=torch.int32)
        wide = spare[torch.arange(n * idx_row_stride) % len(spare)].view(n, idx_row_stride)
        wide[:, :SLOTS] = spec_idx
        spec_idx = wide[:, :SLOTS]
    valid = torch.ones(n, dtype=torch.bool)
    if null_requests:
        spec_idx[n - null_requests:] = 0
        valid[n - null_requests:] = False
    accepted = torch.randint(1, SLOTS + 1, (n,), generator=g, dtype=torch.int32)
    qsl = torch.arange(0, n + 1, dtype=torch.int32) * SLOTS
    x = (torch.randn(n * SLOTS, HID, generator=g) * x_std).to(torch.bfloat16)
    conv = (torch.randn(pool, STATE_LEN, QKV, generator=g) * 0.5).to(torch.bfloat16)
    stride = H * D * D + slot_pad
    store = torch.randn(pool, stride, generator=g) * 0.02
    ssm = store.to(device).as_strided((pool, H, D, D), (stride, D * D, D, 1))
    dev = lambda t: t.to(device)
    rs = spec_idx.stride(0)
    idx = spec_idx.as_strided((n, rs), (rs, 1)).to(device)[:, :SLOTS]  # .to() of a slice would drop the stride
    return State(dev(x), dev(conv), ssm, idx, dev(accepted), dev(qsl), dev(valid))


def next_round(st: State, seed: int) -> None:
    """Advance to the next verify round in place: new x, new acceptance counts; pools carry over."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    st.x.copy_(torch.randn(st.x.shape, generator=g).to(torch.bfloat16))
    st.accepted.copy_(torch.randint(1, SLOTS + 1, (st.n,), generator=g, dtype=torch.int32))


# ---------------------------------------------------------------- image control chain


class Control:
    """The image's six launches on the spec-verify path, exactly as kda.py runs them (ops called directly)."""

    def __init__(self, lw: Layer):
        from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
        from vllm.models.kimi_k3.nvidia.ops.third_party.kda import fused_recurrent_kda
        from vllm.third_party.flash_linear_attention.ops.kda import FusedRMSNormGated

        self.lw = lw
        self.conv_update = causal_conv1d_update
        self.recurrent = fused_recurrent_kda
        from vllm.config import VllmConfig, set_current_vllm_config

        # A CustomOp needs a current vLLM config to construct; the server dispatches it to forward_cuda (its trace
        # shows the Triton layer_norm_gated_fwd_kernel), so __call__ runs forward_cuda explicitly.
        with set_current_vllm_config(VllmConfig()):
            self.o_norm = FusedRMSNormGated(D, activation="sigmoid").to(lw.w_norm.device)
        with torch.no_grad():
            self.o_norm.weight.copy_(lw.w_norm.to(self.o_norm.weight.dtype))

    def in_proj(self, st: State):
        return F.linear(st.x, self.lw.w_in)

    def core(self, st: State, proj: torch.Tensor, core: torch.Tensor):
        """Launches 2-5: f_b_proj, conv, recurrent, o_norm. Writes core [1, T, H, D] bf16; updates the pools."""
        lw, t = self.lw, st.t
        mixed_qkv, og, f_a, beta = proj.split([QKV, P, D, H] + ([COLS - COLS_RAW] if COLS > COLS_RAW else []),
                                              dim=-1)[:4]
        g1 = F.linear(f_a, lw.w_fb).view(1, t, H, D)
        conv_out = torch.empty_like(mixed_qkv)
        qkv = self.conv_update(mixed_qkv, st.conv_view(), lw.w_conv, None, activation="silu",
                               conv_state_indices=st.spec_idx[:, 0], num_accepted_tokens=st.accepted,
                               query_start_loc=st.qsl, max_query_len=SLOTS, validate_data=False, out=conv_out)
        q, k, v = (x.reshape(1, t, H, D) for x in qkv.split(P, dim=-1))
        self.recurrent(q=q, k=k, v=v, raw_g=g1, raw_beta=beta.unsqueeze(0), A_log=lw.a_log, dt_bias=lw.dt_bias,
                       lower_bound=lw.lower_bound, initial_state=st.ssm, cu_seqlens=st.qsl,
                       ssm_state_indices=st.spec_idx, num_accepted_tokens=st.accepted, out=core)
        core.copy_(self.o_norm.forward_cuda(core, og.view(t, H, D)))
        return core

    def o_proj(self, core: torch.Tensor):
        return F.linear(core.view(-1, P), self.lw.w_o)

    def __call__(self, st: State, core: torch.Tensor):
        return self.o_proj(self.core(st, self.in_proj(st), core))


# ---------------------------------------------------------------- fp64 reference


def reference(lw: Layer, st: State, proj: torch.Tensor | None = None) -> dict:
    """float64 over the same bf16 inputs and weights, no intermediate rounding. Does not modify `st`.
    proj: the bf16 [T, COLS] projection a stage after in_proj received (its true input); None = x @ w_in in fp64.
    Returns out [T, HID], core [T, P], proj [T, COLS], and for valid requests: ssm_slots {(n, t): [H, D, D]},
    conv_new {n: [STATE_LEN, QKV]} (the conv slot after the call)."""
    f = torch.float64
    x = st.x.to(f)
    proj = x @ lw.w_in.to(f).T if proj is None else proj.to(f)
    qkv_raw, og, f_a, b = proj[:, :QKV], proj[:, QKV:QKV + P], proj[:, QKV + P:QKV + P + D], proj[:, 4 * P + D:COLS_RAW]
    g1 = f_a @ lw.w_fb.to(f).T
    w = lw.w_conv.to(f)
    a = torch.exp(lw.a_log.to(f))
    core = torch.zeros(st.t, P, dtype=f, device=x.device)
    slots, conv_new = {}, {}
    for n in range(st.n):
        if not bool(st.valid[n]):
            continue
        r0, acc = int(st.qsl[n]), int(st.accepted[n])
        cslot = int(st.spec_idx[n, 0])
        old = st.conv[cslot].to(f)  # [STATE_LEN, QKV]
        o = acc - 1
        ext = torch.cat([old[o:o + WIDTH - 1], qkv_raw[r0:r0 + SLOTS]], 0)  # [3 + 8, QKV]
        conv_new[n] = torch.cat([old[o + 1:o + WIDTH - 1], qkv_raw[r0:r0 + SLOTS]], 0)
        conv = torch.stack([(ext[t:t + WIDTH] * w.T).sum(0) for t in range(SLOTS)], 0)
        conv = conv * torch.sigmoid(conv)
        S = st.ssm[int(st.spec_idx[n, acc - 1])].to(f).clone()  # [H, V, K]
        for t in range(SLOTS):
            row = r0 + t
            q = conv[t, :P].view(H, D)
            k = conv[t, P:2 * P].view(H, D)
            v = conv[t, 2 * P:].view(H, D)
            q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * SCALE
            k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
            gate = lw.lower_bound * torch.sigmoid(a[:, None] * (g1[row].view(H, D) + lw.dt_bias.to(f).view(H, D)))
            beta = torch.sigmoid(b[row])
            S = S * torch.exp(gate)[:, None, :]
            vn = (v - torch.einsum("hvk,hk->hv", S, k)) * beta[:, None]
            S = S + vn[:, :, None] * k[:, None, :]
            ov = torch.einsum("hvk,hk->hv", S, q)
            ov = ov * torch.rsqrt((ov * ov).mean(-1, keepdim=True) + EPS) * lw.w_norm.to(f)
            ov = ov * torch.sigmoid(og[row].view(H, D))
            core[row] = ov.reshape(P)
            slots[(n, t)] = S.clone()
    out = core @ lw.w_o.to(f).T
    return {"out": out, "core": core, "proj": proj, "ssm_slots": slots, "conv_new": conv_new}


def errors(lw: Layer, before: State, after: State, out: torch.Tensor, core: torch.Tensor, ref: dict) -> dict:
    """Max-abs and max-rel (over the reference's max magnitude) errors of one implementation's results against the
    fp64 reference, valid requests only."""
    rows = torch.cat([torch.arange(int(before.qsl[n]), int(before.qsl[n]) + SLOTS) for n in range(before.n)
                      if bool(before.valid[n])]).to(out.device)

    def e(got, want):
        got, want = got.double(), want.double()
        d = (got - want).abs().max().item()
        return {"max_abs": d, "rel": d / max(want.abs().max().item(), 1e-30)}

    res = {"out": e(out.view(-1, HID)[rows], ref["out"][rows]), "core": e(core.reshape(-1, P)[rows], ref["core"][rows])}
    if ref["ssm_slots"]:
        got = torch.stack([after.ssm[int(before.spec_idx[n, t])] for (n, t) in ref["ssm_slots"]])
        res["ssm"] = e(got, torch.stack(list(ref["ssm_slots"].values())))
    if ref["conv_new"]:
        got = torch.stack([after.conv[int(before.spec_idx[n, 0])] for n in ref["conv_new"]])
        res["conv"] = e(got, torch.stack(list(ref["conv_new"].values())))
    return res


# ---------------------------------------------------------------- timing


def graph_timed(fn, reps: int = 50, repeats: int = 5) -> float:
    """Median µs per fn() over CUDA-graph replays (fn may launch many kernels)."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    for _ in range(5):
        graph.replay()
    times = []
    for _ in range(repeats):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(reps):
            graph.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1000.0 / reps)
    del graph
    return statistics.median(times)
