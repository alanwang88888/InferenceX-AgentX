# SPDX-License-Identifier: Apache-2.0
"""8-rank selftest of the draft_chain integration (item 15), TP8 = DCP8, no server, no weights files.

It drives the integration code itself, not the kernels alone. It calls draft_chain.prepare on a stand-in draft model
(5 layers of real-shaped weights), routed_ar, and draft_chain.core with stand-in metadata. tokenspeed is a stub: it
records the q it was given and returns seeded rows / LSE. The served reference is the served op chain (b300_pair's
Served + MLADCPManager + bmm, the same as i15-pair).
Checks, one JSON line each ("probe": "i15-chain"):
  prepare     the switch is on, 10 AR sites tagged 0..9 in call order, 5 layers chained with parity idx % 2;
              the overlay's dspark_mla module global is the routed AR (not the served function).
  route/M     kernel 1 through routed_ar: the served op at M <= 16, two-shot at 17..64; normed + residual bits equal
              to the served op on every rank (M 8, 16, 24, 32, 40, 48, 56, 64).
  attn/M/salt draft_chain.attention (the mla.py hook's entry; forward context stubbed): cache bytes equal served (kernel 2 G1); the q the stub saw inside kernel 2's G2 envelope, and
              the served q too; out inside kernel 3's fp64 envelope, and the served out too.
  graph/M     the parity probe: one draft step (5 x [AR in, core, AR post], final AR = 10 AR + 5 pair calls) captured
              in a CUDA graph, REPLAYS replays back to back with fresh inputs per replay, no host sync. Every replay's
              outputs must be bit-equal to a fenced eager run of the same inputs (fence = sync + barrier between
              calls, so no race is possible there).
Exit: 0 every check passed; 1 a check failed; 2 a crash, a rank failure or the wall budget ran out. The summary line
leads with "rc" and "rc_meaning".
Run: PERFBOT_EXP_DRAFT_CHAIN=1 python3 -B -m perfbot_kernels.draft_chain.selftest.b300_chain   (PYTHONPATH = overlay)
"""

import json
import os
import socket
import sys
import time
from types import SimpleNamespace

import torch
import torch.multiprocessing as mp

os.environ["PERFBOT_EXP_DRAFT_CHAIN"] = "1"  # before draft_chain is imported (the switch is read at import)

TP, H, EPS, LAYERS = 8, 7168, 1e-5, 5
ROUTE_MS = (8, 16, 24, 32, 40, 48, 56, 64)
ATTN_MS = (7, 16, 21, 35, 56)
GRAPH_MS = (16, 24, 40, 56)
REPLAYS = 12
WALL_BUDGET_S = 600
RC_MEANING = {0: "ok", 1: "a check failed", 2: "crash / rank failed / budget"}


def emit(row):
    print(json.dumps({"probe": "i15-chain", **row}), flush=True)


def check(name, ok, **info):
    return {"name": name, "ok": bool(ok), **info}


def bits(a, b):
    return bool(torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16)))


def q_envelope(qjudge, m, q_c, pos, weights, table, one, mine, served):
    """Kernel 2's G2 over the 8 source head blocks (as b300_pair.q_g2): (mine ok, served ok, bytes diff)."""
    mine, served = mine.reshape(m, 64, 576), served.reshape(m, 64, 576)
    ok_m = ok_s = True
    nd = 0
    for s in range(8):
        ref, err = qjudge.query_reference(q_c, weights[2][s * 1536 : (s + 1) * 1536], weights[3][s * 8 : (s + 1) * 8],
                                          table[pos], one)
        h = slice(s * 8, (s + 1) * 8)
        ok_m &= qjudge.envelope("q", mine[:, h], ref, err)["ok"]
        ok_s &= qjudge.envelope("served_q", served[:, h], ref, err)["ok"]
        nd += qjudge.code_distance(mine[:, h], served[:, h])[0]
    return ok_m, ok_s, nd


class TokenspeedStub:
    """tokenspeed stand-in (its contract: FP8 query, base-2 LSE, no padded heads): records the q it was given;
    returns this rank's seeded rows / LSE."""

    supports_quant_query_input, lse_base_on_e, q_pad_num_heads = True, False, None

    def __init__(self, dev):
        self.rows = torch.empty(64, 64, 512, dtype=torch.bfloat16, device=dev)
        self.lse = torch.empty(64, 64, dtype=torch.float32, device=dev)
        self.seen = torch.empty(64, 64, 576, dtype=torch.uint8, device=dev)

    def forward_mqa(self, q, cache, md, layer):
        m = q.shape[0]
        self.seen[:m].copy_(q.view(torch.uint8))
        return self.rows[:m], self.lse[:m]


def make_model(dev, rank, weights, table, w_uv, blocks, bs):
    gq, gkv, w_qb, w_uk = weights
    from vllm.model_executor.layers.layernorm import RMSNorm

    def norm():
        n = RMSNorm(H, eps=EPS).to(dev, torch.bfloat16)
        with torch.no_grad():
            n.weight.copy_((1 + 0.1 * torch.randn(H, generator=torch.Generator().manual_seed(7))).to(torch.bfloat16))
        return n

    layers = []
    for i in range(LAYERS):
        attn = SimpleNamespace(
            q_lora_rank=1536, num_local_heads=8, kv_cache_dtype="fp8", layer_name=f"draft.{i}",
            q_b_proj=SimpleNamespace(weight=w_qb[rank * 1536 : (rank + 1) * 1536]),
            rotary_emb=SimpleNamespace(cos_sin_cache=table),
            q_a_layernorm=SimpleNamespace(weight=gq), kv_a_layernorm=SimpleNamespace(weight=gkv),
            W_UK_T=w_uk[rank * 8 : (rank + 1) * 8], W_UV=w_uv,
            kv_cache=torch.zeros(blocks, bs, 576, dtype=torch.uint8, device=dev),
            _q_scale_inv=torch.ones(1, device=dev), _k_scale_inv=torch.ones(1, device=dev), rms_norm_eps=EPS,
            impl=TokenspeedStub(dev))
        attn._attn_read_kv_cache = (lambda a: (lambda: a.kv_cache))(attn)
        layers.append(SimpleNamespace(self_attn=attn, input_layernorm=norm(), post_attention_layernorm=norm()))
    return SimpleNamespace(layers=layers, final_norm=norm())


def metadata(x, m):
    s = x.qsl.numel() - 1
    return SimpleNamespace(num_actual_tokens=m, num_decode_tokens=m, num_decodes=s,
                           decode=SimpleNamespace(seq_lens=x.seq), query_start_loc=x.qsl)


def run_rank(rank, port, t0):
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import get_tp_group, init_distributed_environment, initialize_model_parallel
    from vllm.distributed.parallel_state import get_dcp_group

    vllm_config = VllmConfig()
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(world_size=TP, rank=rank, distributed_init_method=f"tcp://127.0.0.1:{port}",
                                     local_rank=rank, backend="nccl")
        initialize_model_parallel(tensor_model_parallel_size=TP, decode_context_model_parallel_size=TP)
        import vllm._custom_ops  # noqa: F401  (registers the direct DCP ops)
        from vllm.models.common.ops.fused_allreduce_rms_norm import fused_allreduce_rms_norm

        from perfbot_kernels import draft_chain
        from perfbot_kernels.draft_query.selftest import judge as qjudge
        from perfbot_kernels.draft_query.selftest.gpu import BLOCKS, BS, Served, make_weights, rope_table
        from perfbot_kernels.lse_combine.b300_pair import Inputs, fence, served_manager
        from perfbot_kernels.lse_combine.selftest import judge
        from perfbot_kernels.lse_combine.selftest.gpu import make_wuv, served_wuv

        tp, dcp = get_tp_group(), get_dcp_group()
        group = tp.device_group
        manager, info = served_manager(vllm_config, dev)
        weights, table = make_weights(dev), rope_table(dev)
        served_q = Served(weights, table)
        w_uv = make_wuv(dev, ranks=8)[rank * 8 : (rank + 1) * 8]
        one = torch.ones(1, dtype=torch.float32, device=dev)
        out = {"checks": []}
        add = out["checks"].append

        # ---- prepare
        model = make_model(dev, rank, weights, table, w_uv, BLOCKS, BS)
        draft_chain.prepare(model, tp, dcp)
        st = draft_chain._state
        norms = [n for i, l in enumerate(model.layers)
                 for n in ([l.input_layernorm] if i else []) + [l.post_attention_layernorm]] + [model.final_norm]
        add(check("prepare", st is not None and all(hasattr(l.self_attn, "_perfbot_chain") for l in model.layers) and draft_chain.SWITCH
                  and [getattr(n, "_perfbot_ar_idx", None) for n in norms] == list(range(10))
                  and [l.self_attn._perfbot_chain[1] for l in model.layers] == list(range(LAYERS)),
                  served_path=info))
        import vllm.forward_context as fctx
        import vllm.models.kimi_k3.nvidia.dspark_mla as dspark

        ar = dspark.fused_allreduce_rms_norm  # the overlay's module global: routed_ar(served)
        add(check("dspark_global", ar is not fused_allreduce_rms_norm and draft_chain.SWITCH,
                  module=getattr(dspark, "__file__", "")))

        # ---- kernel 1 routing, bits vs the served op
        for m in ROUTE_MS:
            g = torch.Generator().manual_seed(100 * m + rank)
            x = torch.randn(m, H, generator=g).to(dev, torch.bfloat16)
            res = torch.randn(m, H, generator=torch.Generator().manual_seed(100 * m)).to(dev, torch.bfloat16)
            ok, path = True, []
            for k in (1, 2):  # both parities
                n = norms[k]
                took = draft_chain.ar_norm(x.clone(), res, n) is not None
                fence(group)
                path.append("twoshot" if took else "served")
                n_s, r_s = fused_allreduce_rms_norm(x.clone(), res, n)
                n_c, r_c = ar(x.clone(), res, n)
                fence(group)
                ok &= bits(n_s, n_c) and bits(r_s, r_c)
            want = "served" if m <= draft_chain.SERVED_MAX_M else "twoshot"
            add(check(f"route/M{m}", ok and path == [want, want], path=path[0]))

        # ---- core: kernels 2 + 3 through the hook, vs the served chain
        for m in ATTN_MS:
            for kind, layer in (("randn", 0), ("empty", 1)):
                attn = model.layers[layer].self_attn
                x = Inputs(m, rank, TP, dev, kind, seed=1000 * m + layer)
                attn.impl.rows[:m].copy_(x.rows)
                attn.impl.lse[:m].copy_(x.lse)
                attn.kv_cache.zero_()
                ctx = SimpleNamespace(attn_metadata={attn.layer_name: metadata(x, m)},
                                      slot_mapping={attn.layer_name: x.slots[0]})
                fctx.get_forward_context = lambda: ctx  # attention() imports it from the module at call time
                fence(group)
                out_k = draft_chain.attention(attn, x.pos, x.qkv)
                fence(group)
                took = out_k is not None
                out_k = out_k if took else torch.full((m, 1024), float("nan"), dtype=torch.bfloat16, device=dev)
                cache_s = torch.zeros(1, BLOCKS, BS, 576, dtype=torch.uint8, device=dev)
                mqa, q_c = served_q.rank(rank, x.qkv, x.pos, cache_s[0], x.slots[0], one, one)
                gathered = manager.query_gather(mqa.view(torch.float8_e4m3fn))
                lat = manager.combine(x.rows, x.lse, seq_lens=x.seq, query_start_loc=x.qsl)
                out_s = served_wuv(lat, w_uv)
                fence(group)
                q_ok, sq_ok, nd = q_envelope(qjudge, m, q_c, x.pos[:m], weights, table, one,
                                             attn.impl.seen[:m], gathered.view(torch.uint8))
                xs, vs = judge.owner_inputs(x.rows_all, x.lse_all, x.empty_all, rank)
                _, _, ref, err = judge.reference(xs, vs, w_uv)
                o_ok = judge.envelope("out", out_k, ref, err)["ok"]
                so_ok = judge.envelope("served_out", out_s, ref, err)["ok"]
                c_ok = bool(torch.equal(attn.kv_cache, cache_s[0]))
                add(check(f"attn/M{m}/{kind}", took and c_ok and q_ok and sq_ok and o_ok and so_ok, took=took,
                          cache_equal=c_ok, q_in_envelope=q_ok, served_q_in_envelope=sq_ok, out_in_envelope=o_ok,
                          served_out_in_envelope=so_ok, q_bytes_diff_vs_served=nd))

        # ---- the parity probe: graph replays vs fenced eager
        for m in GRAPH_MS:
            late = torch.tensor([time.time() - t0 > WALL_BUDGET_S - 90], device=dev)
            torch.distributed.broadcast(late, tp.ranks[0], group=group)  # one decision for all ranks
            if late.item():
                add(check(f"graph/M{m}", False, skipped="budget"))
                continue
            xin = torch.empty(11, m, H, dtype=torch.bfloat16, device=dev)  # AR partial inputs (in place: served)
            res0 = torch.empty(m, H, dtype=torch.bfloat16, device=dev)
            ins = [Inputs(m, rank, TP, dev, "randn", seed=50_000 + 100 * r + m) for r in range(REPLAYS)]
            # every replay's inputs staged on the GPU first, so load() is device copies only (no host stall)
            xs_all = [torch.randn(11, m, H, generator=torch.Generator().manual_seed(9000 + 31 * r + rank))
                      .to(dev, torch.bfloat16) for r in range(REPLAYS)]
            rs_all = [torch.randn(m, H, generator=torch.Generator().manual_seed(9000 + 31 * r))
                      .to(dev, torch.bfloat16) for r in range(REPLAYS)]
            took = []
            qkv = torch.empty_like(ins[0].qkv)
            pos, slots = ins[0].pos.clone(), ins[0].slots[0].clone()
            md = metadata(ins[0], m)
            attn_out = torch.empty(LAYERS, m, 1024, dtype=torch.bfloat16, device=dev)
            outs = torch.empty(10, m, H, dtype=torch.bfloat16, device=dev)

            def load(r):
                xin.copy_(xs_all[r])
                res0.copy_(rs_all[r])
                attn_out.fill_(float("nan"))  # a skipped kernel 3 leaves NaN: never equal, never in envelope
                outs.fill_(float("nan"))
                qkv.copy_(ins[r].qkv)
                for l in model.layers:
                    l.self_attn.impl.rows[:m].copy_(ins[r].rows)
                    l.self_attn.impl.lse[:m].copy_(ins[r].lse)

            def step(between=None):
                res, k = res0, 0
                for i, l in enumerate(model.layers):
                    if i:
                        h, res = ar(xin[k], res, l.input_layernorm)
                        outs[k].copy_(h)
                        k += 1
                        between and between()
                    took.append(draft_chain.core(l.self_attn, l.self_attn._perfbot_chain, pos, qkv, md, slots,
                                                 attn_out[i]))
                    between and between()
                    h, res = ar(xin[k], res, l.post_attention_layernorm)
                    outs[k].copy_(h)
                    k += 1
                    between and between()
                h, _ = ar(xin[10], res, model.final_norm)
                outs[9].copy_(h)

            want = []
            for r in range(REPLAYS):  # fenced eager reference
                load(r)
                fence(group)
                step(between=lambda: fence(group))
                fence(group)
                want.append((outs.clone(), attn_out.clone()))
            load(0)
            fence(group)
            step()  # warm-up (compiles are done; the allocator settles before capture)
            fence(group)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                step()
            fence(group)
            got = []
            for r in range(REPLAYS):  # back to back: host copies and replays, no sync, no barrier
                load(r)
                graph.replay()
                got.append((outs.clone(), attn_out.clone()))
            torch.cuda.synchronize()
            bad = [r for r in range(REPLAYS) if not (bits(got[r][0], want[r][0]) and bits(got[r][1], want[r][1]))]
            fence(group)
            nan = [r for r in range(REPLAYS) if bool(torch.isnan(got[r][1]).any())]
            add(check(f"graph/M{m}", not bad and not nan and all(took) and len(took) == LAYERS * (REPLAYS + 2),
                      replays=REPLAYS, bad_replays=bad, nan_replays=nan, core_calls=len(took)))
        fence(group)
    return out


def worker(rank, port, t0, queue):
    try:
        queue.put((rank, run_rank(rank, port, t0)))
    except BaseException as e:
        queue.put((rank, {"error": f"{type(e).__name__}: {e}"}))
        raise


def summary_line(rc, **row):
    emit({"summary": True, "rc": rc, "rc_meaning": RC_MEANING[rc], **row})
    return rc


def main():
    t0 = time.time()
    assert torch.cuda.device_count() >= TP, f"needs {TP} GPUs, sees {torch.cuda.device_count()}"
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, port, t0, queue)) for r in range(TP)]
    for p in procs:
        p.start()
    res = {}
    deadline = t0 + WALL_BUDGET_S
    while len(res) < TP and time.time() < deadline:
        try:
            r, v = queue.get(timeout=max(1.0, deadline - time.time()))
            res[r] = v
            if "error" in v:
                break
        except Exception:
            break
    for p in procs:
        p.join(timeout=5)
        if p.is_alive():
            p.kill()
    errors = {r: v["error"] for r, v in res.items() if "error" in v}
    if errors or len(res) < TP:
        return summary_line(2, ok=False, errors=errors, ranks_reported=sorted(res), seconds=round(time.time() - t0, 1))
    failed, skipped = [], []
    for i, c in enumerate(res[0]["checks"]):
        per = [res[r]["checks"][i] for r in range(TP)]
        ok = all(p["ok"] for p in per)
        emit({"check": c["name"], "ok": ok, "ranks_failed": [r for r in range(TP) if not per[r]["ok"]],
              **{k: v for k, v in c.items() if k not in ("name", "ok")}})
        (skipped if c.get("skipped") else failed if not ok else []).append(c["name"])
    rc = 2 if skipped else 1 if failed else 0
    return summary_line(rc, ok=rc == 0, checks=len(res[0]["checks"]), failed=failed, skipped=skipped,
                        device=torch.cuda.get_device_name(0), seconds=round(time.time() - t0, 1))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # an uncaught error must not exit 1 (1 = a check failed)
        sys.exit(summary_line(2, ok=False, errors={"main": f"{type(e).__name__}: {e}"}))
