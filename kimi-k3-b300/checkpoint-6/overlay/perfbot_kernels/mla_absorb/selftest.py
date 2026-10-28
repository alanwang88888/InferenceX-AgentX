"""Kernel-only gate of line M (absorbed short-prefill MLA under DCP 8). Run in the server env (PERFBOT_PRE_EXEC):

  python3 -B -m perfbot_kernels.mla_absorb.selftest [--check] [--time]

8 processes (one per GPU, NCCL + symmetric memory). Per case (n new tokens after a context of ctx tokens, global
length G = ctx + n, strided DCP shards: global position p lives on rank p % 8 at local index p // 8; target MLA
shape: 12 heads/rank, latent 512 + rope 64, qk_nope 128, v 128, fp8 latent cache, page 64):
  candidate = the production ops the server's absorbed path calls: ql = q_nope @ W_UK_T (bf16 bmm) ->
    fused_mla_decode_q_concat_kv_cache_insert (fp8 q + fp8 latent insert of the new tokens; slot -1 off-rank) ->
    DirectDCPQGatherWorkspace.gather -> tokenspeed_mla_decode(q_len n, causal, causal_seqs = G, cp_world 8) ->
    DirectDCPA2AWorkspace.lse_reduce -> bmm W_UV.
  reference = fp64 attention in the un-absorbed form (k_nope = W_UK c, v = c W_UV) over the stored context latent
    (fp8 values) and the new tokens' bf16 latent, causal (token i sees global positions <= ctx + i).
  dense = numerics emulation of the image dense prefill: fp8 (unscaled) q, k = [fp8(bf16(W_UK c)), fp8(k_pe)],
    v = fp8(bf16(c W_UV)), fp32 attention (the error yardstick only; its wall is measured in-server, shadow mode).
  --check  cases (n, ctx): (8,0) (8,5) (37,1000) (120,30000) (256,5) (256,100) (256,30000) (512,30000)
           (120,100000) (512,100000) (120,300000) (512,300000):
           rel-L2 vs fp64 of candidate <= ROUTED_BOUND (1.02) x dense on every case and rank where
           min_context() <= ctx <= max_context()
           (the builder routes only those; shorter contexts are causal checks, <= 1.1x dense); order control (ctx <= n, where
           the new tokens are a large share of the keys): the same chain with the attention BEFORE the insert must
           be far off (>= 10x the candidate error, or non-finite);
           CUDA-graph capture of the chain at (120, 30000) and (8, 1000), 2 replays on new q: bit-equal to eager.
  --time   chain wall (graph replay, median of 5 x 20) for n in 8..2048 x ctx in 30k/100k/300k; rank 0 recorded.
GATE: check passes on all 8 ranks and the chain at (120, 100000) <= 1000 us (the image dense prefill measured
1.8-5.8 ms per layer at these shapes in AG-02n). Record: $PERFBOT_WORKSPACE/$PERFBOT_RESULTS/mlaq-gate.json.
"""

import argparse
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch

import perfbot_kernels.mla_absorb as mla_absorb

W, HL, L, R, P, V, PAGE = 8, 12, 512, 64, 128, 128, 64
H = W * HL
MAXT = 2048
SCALE = (P + R) ** -0.5
PORT = 29561
CHECK = [(8, 0), (8, 5), (37, 1000), (120, 30000), (256, 5), (256, 100), (256, 30000), (512, 30000),
         (120, 100000), (512, 100000), (120, 300000), (512, 300000)]  # up to the router's max_context()
TIME_N = [8, 64, 120, 256, 512, 1024, 2048]
TIME_CTX = [30000, 100000, 300000]
BAR_US = 1000.0
ROUTED_BOUND = 1.02  # candidate rel-L2 <= this x dense, every routed case and rank (was 1.0 until M-05g)
FP8 = torch.float8_e4m3fn


def results_path(name="mlaq-gate.json") -> Path:
    root = Path(os.environ.get("PERFBOT_WORKSPACE", "/workspace"))
    out = root / os.environ.get("PERFBOT_RESULTS", "results/measure")
    out.mkdir(parents=True, exist_ok=True)
    return out / name


def log(tag, obj):
    print(f"{tag} {json.dumps(obj, default=str)}", flush=True)


class Rank:
    def __init__(self, rank, dev):
        import torch.distributed as dist

        self.rank, self.dev, self.dist = rank, dev, dist
        group = dist.group.WORLD
        try:
            import torch.distributed._symmetric_memory as symm_mem

            symm_mem.enable_symm_mem_for_group(group.group_name)
        except Exception:  # noqa: BLE001
            pass
        import vllm._C_stable_libtorch  # noqa: F401  (registers torch.ops._C.direct_dcp_*)
        from tokenspeed_mla import get_num_sm, tokenspeed_mla_decode
        from vllm.models.kimi_k3.nvidia.ops.fused_mla_key_concat_kv_cache import (
            fused_mla_decode_q_concat_kv_cache_insert,
        )
        from vllm.v1.attention.ops import dcp

        self.decode, self.insert = tokenspeed_mla_decode, fused_mla_decode_q_concat_kv_cache_insert
        self.qws = dcp.DirectDCPQGatherWorkspace(group, dev, MAXT, HL, L + R, FP8, 1, None)
        self.aws = dcp.DirectDCPA2AWorkspace(group, dev, MAXT, HL, L, torch.bfloat16, 1)
        # vLLM tokenspeed_mla.py _get_workspace: num_sms * H * 8 * (L + 1) * 4 bytes
        self.work = torch.empty(get_num_sm(dev) * H * 8 * (L + 1) * 4, dtype=torch.int8, device=dev)
        self.one = torch.ones(1, dtype=torch.float32, device=dev)
        g = torch.Generator(device=dev).manual_seed(7)
        self.w_uk_t = (torch.randn(H, P, L, device=dev, generator=g) * L ** -0.5).to(torch.bfloat16)
        self.w_uv = (torch.randn(H, L, V, device=dev, generator=g) * L ** -0.5).to(torch.bfloat16)
        self.mine = slice(rank * HL, (rank + 1) * HL)

    def barrier(self):
        torch.cuda.synchronize()
        self.dist.barrier(device_ids=[self.dev.index])

    # ---------------------------------------------------------------- one case
    def case(self, n, ctx, seed):
        """Global inputs (same on every rank) and this rank's paged shard of the context."""
        dev, G = self.dev, ctx + n
        g = torch.Generator(device=dev).manual_seed(1000 + seed * 31 + n * 7 + ctx)
        c = {"n": n, "ctx": ctx, "G": G}
        c["ctx_lat"] = torch.randn(ctx, L + R, device=dev, generator=g).to(FP8)  # stored context latent
        c["new_c"] = torch.randn(n, L, device=dev, generator=g).to(torch.bfloat16)
        c["new_pe"] = torch.randn(n, R, device=dev, generator=g).to(torch.bfloat16)
        c["q_nope"] = torch.randn(n, H, P, device=dev, generator=g).to(torch.bfloat16)
        c["q_pe"] = torch.randn(n, H, R, device=dev, generator=g).to(torch.bfloat16)
        local = max(0, (G - self.rank + W - 1) // W)
        pages = max(1, (local + PAGE - 1) // PAGE)
        cache = torch.zeros(pages * PAGE, L + R, dtype=FP8, device=dev)
        pos = torch.arange(self.rank, max(ctx, self.rank), W, device=dev)  # empty when ctx <= rank
        cache[: pos.numel()] = c["ctx_lat"][pos]
        c["cache"] = cache.view(pages, PAGE, L + R)
        c["block_table"] = torch.arange(pages, dtype=torch.int32, device=dev).view(1, pages)
        p_new = torch.arange(ctx, G, device=dev)
        c["slots"] = torch.where(p_new % W == self.rank, p_new // W, torch.full_like(p_new, -1)).to(torch.int64)
        c["local"] = local
        c["seq_local"] = torch.tensor([local], dtype=torch.int32, device=dev)
        c["seq_global"] = torch.tensor([G], dtype=torch.int32, device=dev)
        c["pad"] = mla_absorb.bucket(n)
        assert c["pad"] <= G, (n, ctx)
        c["qsl"] = torch.tensor([0, c["pad"]], dtype=torch.int32, device=dev)
        c["out"] = torch.empty(n, HL, V, dtype=torch.bfloat16, device=dev)
        return c

    def chain(self, c, insert_first=True):
        """The candidate on this rank's local heads; writes c['out']."""
        ql = torch.bmm(c["q_nope"][:, self.mine].transpose(0, 1), self.w_uk_t[self.mine]).transpose(0, 1)
        q_pe = c["q_pe"][:, self.mine]

        def ins():
            return self.insert(ql, q_pe, c["new_c"], c["new_pe"], c["cache"].view(-1, c["cache"].shape[-2], L + R), c["slots"],
                               q_scale_inv=self.one, cache_scale_inv=self.one)

        if insert_first:
            mqa_q = ins()
        else:  # order control: the query without the insert, attention, then the insert
            mqa_q = self.insert(ql, q_pe, c["new_c"], c["new_pe"], torch.zeros_like(c["cache"]), c["slots"],
                                q_scale_inv=self.one, cache_scale_inv=self.one)
        n, b = c["n"], c["pad"]
        q = self.qws.gather(mla_absorb.pad_front(mqa_q, [n], [b]))  # the server's front padding to a bucket
        o, lse = self.decode(query=q.view(1, b, H, L + R), kv_cache=c["cache"], workspace_buffer=self.work,
                             kv_lora_rank=L, qk_rope_head_dim=R, block_tables=c["block_table"], seq_lens=c["seq_local"],
                             max_seq_len=max(1, c["local"]), softmax_scale=SCALE, output_scale=1.0, enable_pdl=False,
                             return_lse=True, causal_mask=True, causal_seqs=c["seq_global"], cp_world=W, cp_rank=self.rank)
        if not insert_first:
            ins()
        lat = self.aws.lse_reduce(o.view(b, H, L), lse.view(b, H), False, c["seq_local"], c["qsl"])
        lat = mla_absorb.unpad_front(lat, [n], [b])
        torch.bmm(lat.transpose(0, 1), self.w_uv[self.mine], out=c["out"].transpose(0, 1))
        return c["out"]

    def keys(self, c):
        """Global latent of every position as the reference sees it: stored context (fp8 values), new tokens bf16."""
        new = torch.cat([c["new_c"], c["new_pe"]], -1)
        return torch.cat([c["ctx_lat"].double(), new.double()], 0)

    def reference(self, c, dense=False):
        n, ctx, G = c["n"], c["ctx"], c["G"]
        lat = self.keys(c)  # [G, 576]
        cl, pe = lat[:, :L], lat[:, L:]
        wk, wv = self.w_uk_t[self.mine].double(), self.w_uv[self.mine].double()  # [HL,P,L], [HL,L,V]
        qn, qp = c["q_nope"][:, self.mine].double(), c["q_pe"][:, self.mine].double()  # [n,HL,*]
        mask = torch.arange(G, device=self.dev)[None, :] <= (ctx + torch.arange(n, device=self.dev))[:, None]
        outs = []
        for h in range(HL):
            k_nope = cl @ wk[h].T  # [G, P]
            v = cl @ wv[h]  # [G, V]
            kp, q1, q2 = pe, qn[:, h], qp[:, h]
            if dense:  # image dense prefill numerics: bf16 up-projection, unscaled fp8 q/k/v, fp32 attention
                f8 = lambda x: x.to(torch.bfloat16).to(FP8).double()
                k_nope, v, kp, q1, q2 = f8(k_nope), f8(v), f8(kp), f8(q1), f8(q2)
            s = (q1 @ k_nope.T + q2 @ kp.T) * SCALE
            s = s.masked_fill(~mask, float("-inf"))
            outs.append(torch.softmax(s, -1) @ v)
        return torch.stack(outs, 1)  # [n, HL, V]

    @staticmethod
    def err(x, ref):
        d = x.double() - ref
        return {"rel_l2": (d.norm() / ref.norm()).item(), "max_abs": d.abs().max().item()}

    # ---------------------------------------------------------------- modes
    def check(self):
        res, ok = [], True
        for i, (n, ctx) in enumerate(CHECK):
            c = self.case(n, ctx, i)
            cand = self.chain(c).clone()
            self.barrier()
            ref = self.reference(c)
            e_c, e_d = self.err(cand, ref), self.err(self.reference(c, dense=True), ref)
            c2 = self.case(n, ctx, i)
            e_o = self.err(self.chain(c2, insert_first=False).clone(), ref)
            self.barrier()
            row = {"n": n, "ctx": ctx, "cand": e_c, "dense_emul": e_d, "attention_before_insert": e_o,
                   "finite": bool(torch.isfinite(cand).all())}
            # The order control can only show where the new tokens are a large share of the keys (ctx <= n);
            # a non-finite result there counts as far off.
            far = not (e_o["rel_l2"] < 10 * e_c["rel_l2"])
            row["order_control_applies"] = ctx <= n
            # Routed domain (the builder's [min_context, max_context]): non-inferior, error <= ROUTED_BOUND x the
            # image's (coordinator ruling 2026-10-18 after M-05g: at >= 100k both paths sit on the fp8 floor ~0.027
            # and a strict <= flips on the draw; a mechanism error shows as ratio >> 1). Shorter contexts are causal
            # checks only (all keys are new tokens, fp8-quantized at a different point than the image's): <= 1.1x.
            row["routed"] = mla_absorb.min_context() <= ctx <= mla_absorb.max_context()
            bound = ROUTED_BOUND if row["routed"] else 1.1
            row["ok"] = (row["finite"] and e_c["rel_l2"] <= bound * e_d["rel_l2"]
                         and (far or ctx > n))
            ok = ok and row["ok"]
            res.append(row)
            log("CASE", {"rank": self.rank, **row})
        graph = self.graph()
        return {"ok": ok and graph["ok"], "cases": res, "graph": graph}

    def graph(self):
        out, ok = [], True
        for n, ctx in ((120, 30000), (8, 1000)):
            c = self.case(n, ctx, 99)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                self.chain(c)
            torch.cuda.current_stream().wait_stream(s)
            self.barrier()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                self.chain(c)
            self.barrier()
            same = []
            for k in range(2):
                g = torch.Generator(device=self.dev).manual_seed(500 + k)
                c["q_nope"].copy_(torch.randn(c["q_nope"].shape, device=self.dev, generator=g).to(torch.bfloat16))
                gr.replay()
                self.barrier()
                got = c["out"].clone()
                self.chain(c)
                self.barrier()
                same.append(bool(torch.equal(got, c["out"])))
            ok = ok and all(same)
            out.append({"n": n, "ctx": ctx, "replays_bit_equal": same})
        return {"ok": ok, "shapes": out}

    def time(self):
        rows = []
        for ctx in TIME_CTX:
            for n in TIME_N:
                c = self.case(n, ctx, 7)
                self.chain(c)
                self.barrier()
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    for _ in range(4):
                        self.chain(c)
                self.barrier()
                ts = []
                for _ in range(5):
                    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    self.barrier()
                    a.record()
                    for _ in range(5):
                        gr.replay()
                    b.record()
                    torch.cuda.synchronize()
                    ts.append(a.elapsed_time(b) * 1e3 / 20)
                row = {"n": n, "ctx": ctx, "chain_us": round(statistics.median(ts), 1)}
                rows.append(row)
                log("TIME", {"rank": self.rank, **row})
                del gr, c
        at = [r["chain_us"] for r in rows if r["n"] == 120 and r["ctx"] == 100000]
        return {"ok": bool(at) and at[0] <= BAR_US, "rows": rows, "bar_us": BAR_US}


def _rank_main(rank, port, out_dir, modes):
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=W, device_id=dev)
    rec = {"rank": rank}
    path = Path(out_dir, f"mlaq-rank{rank}.json")
    save = lambda: path.write_text(json.dumps(rec, indent=1, default=str))
    try:
        rk = Rank(rank, dev)
        for name in modes:
            t0 = time.time()
            try:
                rec[name] = getattr(rk, name)()
            except Exception:  # noqa: BLE001
                rec[name] = {"ok": False, "error": traceback.format_exc()}
                traceback.print_exc()
            rec[name]["seconds"] = round(time.time() - t0, 1)
            save()
    except Exception:  # noqa: BLE001
        rec["error"] = traceback.format_exc()
        traceback.print_exc()
    save()
    print(f"MLAQ rank={rank} done {[(n, (rec.get(n) or {}).get('ok')) for n in modes]}", flush=True)
    try:
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    for f in ("check", "time"):
        ap.add_argument(f"--{f}", action="store_true")
    ap.add_argument("--deadline-s", type=float, default=1380.0)
    args = ap.parse_args()
    modes = [n for n in ("check", "time") if getattr(args, n)]
    path = results_path()
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_rank_main, args=(r, PORT, str(path.parent), modes)) for r in range(W)]
    t0 = time.time()
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=max(1.0, args.deadline_s - (time.time() - t0)))
    for p in procs:
        if p.is_alive():
            p.terminate()
    ranks = []
    for r in range(W):
        f = path.parent / f"mlaq-rank{r}.json"
        ranks.append(json.loads(f.read_text()) if f.is_file() else {"rank": r, "error": "no record"})
    record = {"item": "expedition-M-mlaq", "modes": modes, "exitcodes": [p.exitcode for p in procs],
              "seconds": round(time.time() - t0, 1),
              "errors": {r["rank"]: r["error"][-1500:] for r in ranks if "error" in r}}
    parts = {}
    for n in modes:
        recs = [r.get(n) or {} for r in ranks]
        parts[n] = recs[0].get("ok") is True if n == "time" else all(x.get("ok") is True for x in recs)
        record[n] = {"ok": parts[n], "rank0": recs[0],
                     "failed_ranks": [r["rank"] for r, x in zip(ranks, recs) if x.get("ok") is not True]}
    record["parts_ok"] = parts
    record["gate_pass"] = bool(set(parts) == {"check", "time"} and all(parts.values()))
    path.write_text(json.dumps(record, indent=1, default=str))
    print(f"PERFBOT-MLAQ-GATE parts={parts} gate_pass={record['gate_pass']} record={path}", flush=True)
    return 0 if record["gate_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
