"""CPU tests of the pure parts of the fp64 reference and of reftest.py (torch only, no GPU, no vllm):

  python3 -I overlay/perfbot_kernels/mla_absorb/test_reftest_cpu.py

Chunked vs dense reference (both truths, the new-key mass), the latent-space (absorbed) truth vs the un-absorbed one,
shard_rows vs global_positions, reference_report's two-truth bookkeeping, its q8 truth and q statistics, map_rel flag
and key=value line (one fake 2-rank DCP group), the tokenspeed planner mirror and the decode workspace bound, the image
emulation's merge, and the R7 case bookkeeping.
"""

import contextlib
import io
import sys
import types
import unittest
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # overlay/ (the package root on the server's PYTHONPATH)

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from perfbot_kernels import mla_absorb  # noqa: E402
from perfbot_kernels.mla_absorb import reftest as rt  # noqa: E402

F64, FP8 = torch.float64, torch.float8_e4m3fn


def problem(seed=0, ctx=37, n=5, hl=2, l_=16, r_=4, p_=8, v_=8):
    g = torch.Generator().manual_seed(seed)
    rnd = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    return types.SimpleNamespace(
        ctx=ctx, n=n, hl=hl, L=l_, R=r_, P=p_, V=v_, ctx_lat=rnd(ctx, l_ + r_).to(FP8),
        new_c=rnd(n, l_).to(torch.bfloat16), new_pe=rnd(n, 1, r_).to(torch.bfloat16),
        q=rnd(n, hl, p_ + r_).to(torch.bfloat16), w_uk_t=rnd(hl, p_, l_) * l_ ** -0.5, w_uv=rnd(hl, l_, v_) * l_ ** -0.5,
        scale=(p_ + r_) ** -0.5)


def latents(pb):
    new = torch.cat([pb.new_c, pb.new_pe.view(pb.n, -1)], -1)
    lat8 = torch.cat([pb.ctx_lat.double(), new.to(FP8).double()])  # the insert stores fp8(new) (cache scale 1)
    latm = torch.cat([pb.ctx_lat.double(), new.double()])
    return lat8, latm


class Reference(unittest.TestCase):
    def test_chunked_equals_dense_both_truths(self):
        pb = problem()
        lat8, latm = latents(pb)
        sel = torch.tensor([0, 2, 4])
        qd = pb.q[sel].double()
        qn, qp = qd[..., : pb.P], qd[..., pb.P :]
        got = mla_absorb.ref_attention((pb.ctx, lambda a, b: lat8[a:b]), qn, qp, pb.w_uk_t, pb.w_uv, pb.ctx + sel,
                                       pb.scale, pb.L, chunk=7, tails={"t8": lat8[pb.ctx :], "mix": latm[pb.ctx :]})
        for name, lat in (("t8", lat8), ("mix", latm)):
            ref, mass = rt.dense_reference(lat, qn, qp, pb.w_uk_t, pb.w_uv, pb.ctx + sel, pb.scale, pb.L, pb.ctx)
            self.assertLess(rt.rel(got[name][0], ref), 1e-12, name)
            self.assertLess((got[name][1] - mass).abs().max().item(), 1e-12, name)
        self.assertGreater(rt.rel(got["t8"][0], got["mix"][0]), 1e-4)  # the truths differ (fp8 vs bf16 tail)
        plain = mla_absorb.ref_attention(latm, qn, qp, pb.w_uk_t, pb.w_uv, pb.ctx + sel, pb.scale, pb.L, chunk=11)
        self.assertLess(rt.rel(plain, got["mix"][0]), 1e-12)  # tensor form (mixbench gate) = tails form

    def test_latent_truth_equals_unabsorbed(self):
        pb = problem(5)
        lat8, latm = latents(pb)
        sel = torch.tensor([0, 1, 4])
        qd = pb.q[sel].double()
        qn, qp = qd[..., : pb.P], qd[..., pb.P :]
        ql = torch.einsum("rhp,hpl->rhl", qn, pb.w_uk_t.double())  # exact (fp64) absorbed query
        tails = {"t8": lat8[pb.ctx :], "mix": latm[pb.ctx :]}
        un = mla_absorb.ref_attention((pb.ctx, lambda a, b: lat8[a:b]), qn, qp, pb.w_uk_t, pb.w_uv, pb.ctx + sel,
                                      pb.scale, pb.L, chunk=7, tails=tails)
        ab = mla_absorb.ref_attention((pb.ctx, lambda a, b: lat8[a:b]), ql, qp, None, pb.w_uv, pb.ctx + sel,
                                      pb.scale, pb.L, chunk=7, tails=tails)
        for t in tails:
            self.assertLess(rt.rel(ab[t][0], un[t][0]), 1e-12, t)
            self.assertLess((ab[t][1] - un[t][1]).abs().max().item(), 1e-12, t)

    def test_shard_rows_inverts_global_positions(self):
        for w, lm, i in ((8, 13, 1), (8, 16, 4), (3, 7, 2)):
            pos = mla_absorb.global_positions(w, lm, i)
            p = torch.arange(int(pos.max()) + 1)
            p = p[torch.isin(p, pos)]
            self.assertTrue(torch.equal(pos[mla_absorb.shard_rows(p, w, lm, i)], p), (w, lm, i))


class FakeGroup:
    """Two-rank DCP group seen from rank 0: all_gather returns both padded shards (and checks rank 0's), all_reduce
    doubles (rank 1 is given rank 0's values)."""

    def __init__(self, shards):
        self.shards = shards

    def __enter__(self):
        self.saved = dist.all_gather_into_tensor, dist.all_reduce

        def gather(out, shard, group=None):
            assert torch.equal(shard, self.shards[0]), "rank 0's shard (slots / pages) differs"
            out.copy_(torch.cat(self.shards))

        def reduce(t, op=None, group=None):
            if op != dist.ReduceOp.MAX:  # rank 1 = rank 0: max is the identity
                t.mul_(2)

        dist.all_gather_into_tensor, dist.all_reduce = gather, reduce
        return self

    def __exit__(self, *a):
        dist.all_gather_into_tensor, dist.all_reduce = self.saved


def report_setup(pb, lat_stored, page=4, w=2):
    """rank 0's paged cache (pages in reverse order: the block table matters) and the fake group's shards."""
    G = pb.ctx + pb.n
    lm = -(-G // w)
    shards = []
    for k in range(w):
        rows = torch.zeros(lm, pb.L + pb.R, dtype=FP8)
        pos = torch.arange(k, G, w)
        rows[: pos.numel()] = lat_stored[pos].to(FP8)
        shards.append(rows.view(torch.uint8))
    pages = -(-lm // page)
    cache = torch.zeros(pages * page, pb.L + pb.R, dtype=FP8)
    order = torch.arange(pages).flip(0)
    for j in range(pages):
        cache[order[j] * page : (order[j] + 1) * page][: max(0, min(page, lm - j * page))] = (
            shards[0].view(FP8)[j * page : (j + 1) * page])
    layer = types.SimpleNamespace(
        layer_name="layers.3", dcp_world_size=w, impl=types.SimpleNamespace(dcp_rank=0), _k_scale=1.0,
        kv_lora_rank=pb.L, qk_nope_head_dim=pb.P, v_head_dim=pb.V, scale=pb.scale, W_UK_T=pb.w_uk_t, W_UV=pb.w_uv,
        _attn_read_kv_cache=lambda: cache.view(pages, page, -1))
    info = types.SimpleNamespace(q_lens=[pb.n], seq_lens_cpu=[G], interleave=1, local_max=[lm],
                                 block_table=order.view(1, -1).to(torch.int32))
    return layer, info, [s.contiguous() for s in shards]


def call(pb, layer, info, shards, image, absorbed, **kw):
    out = io.StringIO()
    with FakeGroup(shards), contextlib.redirect_stdout(out):
        res = mla_absorb.reference_report(layer, info, pb.q, pb.new_c, pb.new_pe, image, absorbed, group=object(),
                                          force=True, **kw)
    return res, out.getvalue().splitlines()


def absorbed_q(pb):
    """fp64 pre-quantization absorbed query [n, HL, L + R] = [q_nope W_UK | q_pe]."""
    qd = pb.q.double()
    return torch.cat([torch.einsum("nhp,hpl->nhl", qd[..., : pb.P], pb.w_uk_t.double()), qd[..., pb.P :]], -1)


class Report(unittest.TestCase):
    def test_two_truths_bookkeeping_and_line(self):
        pb = problem(1)
        lat8, latm = latents(pb)
        layer, info, shards = report_setup(pb, lat8)
        sel = torch.arange(pb.n)
        qd = pb.q.double()
        refs = {t: rt.dense_reference(lat, qd[..., : pb.P], qd[..., pb.P :], pb.w_uk_t, pb.w_uv, pb.ctx + sel,
                                      pb.scale, pb.L, pb.ctx) for t, lat in (("t8", lat8), ("mix", latm))}
        image, absorbed = refs["mix"][0].reshape(pb.n, -1), refs["t8"][0].reshape(pb.n, -1)
        res, lines = call(pb, layer, info, shards, image, absorbed)
        self.assertLess(res["image_rel_mix"], 1e-12)  # chunked truth vs dense: rounding only
        self.assertLess(res["absorbed_rel_t8"], 1e-12)
        want = rt.rel(refs["mix"][0], refs["t8"][0])
        self.assertAlmostEqual(res["image_rel_t8"], want, delta=1e-9 * want)
        self.assertAlmostEqual(res["absorbed_rel_mix"], rt.rel(refs["t8"][0], refs["mix"][0]), delta=1e-10)
        pnew = refs["mix"][1].mean(1)  # mean over heads (the fake rank 1 repeats rank 0's heads)
        self.assertLess(max(abs(a - b) for a, b in zip(res["pnew_rows"], pnew.tolist())), 1e-12)
        self.assertTrue(rt.MAP_REL_LO <= res["map_rel"] <= rt.MAP_REL_MAX, res["map_rel"])
        self.assertEqual(res["bad_map"], 0)
        self.assertEqual(len(lines), 1)
        kv = dict(w.split("=", 1) for w in lines[0].split()[1:])
        self.assertEqual(lines[0].split()[0], "PERFBOT-MLAQ-REF")
        for k in ("layer", "n", "ctx", "image_rel_t8", "absorbed_rel_t8", "ratio_t8", "image_rel_mix",
                  "absorbed_rel_mix", "ratio_mix", "pnew_min", "pnew_med", "pnew_max", "map_rel", "bad_map", "s"):
            self.assertIn(k, kv)
        self.assertEqual(int(kv["bad_map"]), 0)
        self.assertFalse(any(k.startswith("q_") or k.endswith("_q8") for k in kv))  # no q8: no q8 fields

    def test_q8_truth_and_stats(self):
        pb = problem(6)
        lat8, latm = latents(pb)
        layer, info, shards = report_setup(pb, lat8)
        s = torch.tensor([0.0371])  # a static fp32 q scale like layer._q_scale (not 1: the dequantization is tested)
        qa = absorbed_q(pb)
        q8 = (qa.to(torch.bfloat16).float() * s.reciprocal()).clamp(-448, 448).to(FP8)  # the fused concat's fp8 query
        q8d = q8.double() * s.item()
        sel = torch.arange(pb.n)
        eye = torch.eye(pb.L, dtype=F64).expand(pb.hl, pb.L, pb.L)  # dense_reference with W_UK = I: latent scores
        fake = rt.dense_reference(lat8, q8d[..., : pb.L], q8d[..., pb.L :], eye, pb.w_uv, pb.ctx + sel, pb.scale, pb.L,
                                  pb.ctx)[0]  # an absorbed output made with the same quantized q, independent code
        mix = rt.dense_reference(latm, pb.q.double()[..., : pb.P], pb.q.double()[..., pb.P :], pb.w_uk_t, pb.w_uv,
                                 pb.ctx + sel, pb.scale, pb.L, pb.ctx)[0]
        res, lines = call(pb, layer, info, shards, mix.reshape(pb.n, -1), fake.reshape(pb.n, -1), q8=q8,
                          q_scale=s)
        self.assertLess(res["absorbed_rel_q8"], 1e-12)  # the q8 truth reproduces the fake absorbed output
        self.assertLess(res["ratio_q8"], 1e-9)
        self.assertGreater(res["image_rel_q8"], 1e-4)
        self.assertGreater(res["absorbed_rel_t8"], 1e-3)  # vs the bf16-q truth the q quantization shows
        self.assertLess(res["image_rel_mix"], 1e-12)
        self.assertAlmostEqual(res["q_rel_emu"], res["q_rel"], delta=1e-12)  # emulation = the given q8
        self.assertTrue(1e-3 < res["q_rel"] < 0.1, res["q_rel"])
        al, ap = qa[..., : pb.L].abs().max().item(), qa[..., pb.L :].abs().max().item()
        self.assertAlmostEqual(res["q_amax_lat"], al, delta=1e-12 * al)
        self.assertAlmostEqual(res["q_amax_pe"], ap, delta=1e-12 * ap)
        self.assertAlmostEqual(res["q_fill"], max(al, ap) / (448 * s.item()), delta=1e-12)
        self.assertAlmostEqual(res["q_scale_layer"], max(al, ap) / 224, delta=1e-12)
        self.assertEqual(len(lines), 1)
        kv = dict(w.split("=", 1) for w in lines[0].split()[1:])
        for k in ("image_rel_q8", "absorbed_rel_q8", "ratio_q8", "q_scale", "q_amax_lat", "q_amax_pe", "q_fill",
                  "q_f6_lat", "q_f9_lat", "q_f6_pe", "q_f9_pe", "q_rel", "q_rel_lat", "q_rel_pe", "q_rel_emu",
                  "q_scale_layer", "q_rel_layer", "ratio_t8", "s"):
            self.assertIn(k, kv)

    def test_q_stats_known_values(self):
        L_, r_ = 8, 4
        lat = torch.tensor([1.0, 2.0 ** -8, 2.0 ** -8, 2.0 ** -11], dtype=F64).repeat(6 * L_ // 4).view(3, 2, L_)
        qa = torch.cat([lat, torch.full((3, 2, r_), 2.0, dtype=F64)], -1)
        q8d = mla_absorb.fp8_quant(qa, 1.0)
        self.assertEqual(q8d[..., :L_].unique().tolist(), [0.0, 2.0 ** -8, 1.0])  # 2^-11 flushes, 2^-8 is subnormal
        amax = torch.stack([qa[..., :L_].abs().max(), qa[..., L_:].abs().max()])
        s_layer = 2.0 / 224
        f = mla_absorb.q_fields(mla_absorb.q_sums(qa, q8d, 1.0, s_layer, L_), amax, 1.0, s_layer)
        self.assertEqual((f["q_f6_lat"], f["q_f9_lat"], f["q_f6_pe"], f["q_f9_pe"]), (0.75, 0.25, 0.0, 0.0))
        want = (2.0 ** -22 / (1 + 2 * 2.0 ** -16 + 2.0 ** -22)) ** 0.5
        self.assertAlmostEqual(f["q_rel_lat"], want, delta=1e-15)
        self.assertEqual(f["q_rel_pe"], 0.0)
        self.assertAlmostEqual(f["q_rel_emu"], f["q_rel"], delta=1e-15)
        self.assertEqual((f["q_amax_lat"], f["q_amax_pe"], f["q_fill"]), (1.0, 2.0, 2.0 / 448))
        self.assertLess(f["q_rel_layer"], 1e-15)  # x / s_layer = 112 x: 112, 7/16, 7/128, 224 are all exact in e4m3
        self.assertEqual(mla_absorb.fp8_quant(torch.tensor([1000.0, -1000.0], dtype=F64), 1.0).tolist(), [448.0, -448.0])

    def test_bad_map_flag(self):
        pb = problem(2)
        lat8, _ = latents(pb)
        wrong = lat8.clone()
        wrong[pb.ctx :] = lat8[pb.ctx :].roll(1, 0)  # new tokens' rows at the wrong positions
        layer, info, shards = report_setup(pb, wrong)
        z = torch.zeros(pb.n, pb.hl * pb.V)
        res, lines = call(pb, layer, info, shards, z, z)
        self.assertGreater(res["map_rel"], rt.MAP_REL_MAX)
        self.assertEqual(res["bad_map"], 1)
        self.assertIn("bad_map=1", lines[0])

    def test_skips(self):
        pb = problem(3)
        layer, info, shards = report_setup(pb, latents(pb)[0])
        z = torch.zeros(pb.n, pb.hl * pb.V)
        two = types.SimpleNamespace(**{**vars(info), "q_lens": [2, 3]})
        self.assertIsNone(mla_absorb.reference_report(layer, two, pb.q, pb.new_c, pb.new_pe, z, z, force=True))
        self.assertEqual(mla_absorb.ref_samples(), 0)  # PERFBOT_MLAQ_REF unset: off without force
        self.assertIsNone(mla_absorb.reference_report(layer, info, pb.q, pb.new_c, pb.new_pe, z, z))

    def test_extra_path(self):  # REF-locator: absorbed_ps reported per truth, the two base paths unchanged
        pb = problem(4)
        lat8, latm = latents(pb)
        layer, info, shards = report_setup(pb, lat8)
        qd, sel = pb.q.double(), torch.arange(pb.n)
        refs = {t: rt.dense_reference(lat, qd[..., : pb.P], qd[..., pb.P :], pb.w_uk_t, pb.w_uv, pb.ctx + sel,
                                      pb.scale, pb.L, pb.ctx)[0].reshape(pb.n, -1) for t, lat in (("t8", lat8), ("mix", latm))}
        base, _ = call(pb, layer, info, shards, refs["mix"], refs["t8"])
        res, lines = call(pb, layer, info, shards, refs["mix"], refs["t8"], extra={"absorbed_ps": refs["t8"]})
        for k in ("image_rel_t8", "absorbed_rel_t8", "ratio_t8", "image_rel_mix", "absorbed_rel_mix", "ratio_mix"):
            self.assertAlmostEqual(res[k], base[k], delta=1e-12)
        self.assertLess(res["absorbed_ps_rel_t8"], 1e-12)
        self.assertAlmostEqual(res["ratio_absorbed_ps_mix"], res["ratio_mix"], delta=1e-12)
        kv = dict(w.split("=", 1) for w in lines[0].split()[1:])
        for k in ("absorbed_ps_rel_t8", "absorbed_ps_rel_mix", "ratio_absorbed_ps_t8", "ratio_absorbed_ps_mix"):
            self.assertIn(k, kv)

    def test_ref_due(self):
        pb = problem(5)
        layer, info, shards = report_setup(pb, latents(pb)[0])
        z = torch.zeros(pb.n, pb.hl * pb.V)
        with unittest.mock.patch.dict("os.environ", {"PERFBOT_MLAQ_REF": "1"}):
            mla_absorb._ref_taken.clear()
            self.assertEqual(mla_absorb.ref_due(layer, info), "in")
            self.assertEqual(mla_absorb.ref_due(layer, info), "in")  # no counter change
            with FakeGroup(shards), contextlib.redirect_stdout(io.StringIO()):
                self.assertIsNotNone(mla_absorb.reference_report(layer, info, pb.q, pb.new_c, pb.new_pe, z, z,
                                                                 group=object()))
            self.assertIsNone(mla_absorb.ref_due(layer, info))  # the one sample of this (layer, region) taken
            two = types.SimpleNamespace(**{**vars(info), "q_lens": [2, 3]})
            mla_absorb._ref_taken.clear()
            self.assertIsNone(mla_absorb.ref_due(layer, two))
        mla_absorb._ref_taken.clear()


class PShift(unittest.TestCase):
    def test_swap_and_restore(self):  # fake tokenspeed_mla: the swap, the separate compile cache, the restore
        import functools
        import tempfile

        d = Path(tempfile.mkdtemp())
        src = ("class BlackwellMultiHeadLatentAttentionForwardFP8:\n    def k(self, row_max_new, softmax_params, row_sum, cute):\n"
               "        " + mla_absorb.P_FMA + "        a = " + mla_absorb.P_LSE + "        b = " + mla_absorb.P_LSE)
        (d / "mla_decode_fp8.py").write_text(src)
        pkg = types.ModuleType("tokenspeed_mla")
        f8 = types.ModuleType("tokenspeed_mla.mla_decode_fp8")
        f8.__file__ = str(d / "mla_decode_fp8.py")
        f8.BlackwellMultiHeadLatentAttentionForwardFP8 = stock = type("Stock", (), {})
        tmd = types.ModuleType("tokenspeed_mla.mla_decode")
        tmd.BlackwellMultiHeadLatentAttentionForwardFP8 = stock

        @functools.cache
        def getter(x):
            return (tmd.BlackwellMultiHeadLatentAttentionForwardFP8, x)

        tmd._get_compiled_mla_kernel = getter
        pkg.mla_decode, pkg.mla_decode_fp8 = tmd, f8
        mods = {"tokenspeed_mla": pkg, "tokenspeed_mla.mla_decode": tmd, "tokenspeed_mla.mla_decode_fp8": f8}
        with unittest.mock.patch.dict(sys.modules, mods):
            mla_absorb._pshift.clear()
            self.assertIs(getter(1)[0], stock)
            with mla_absorb.pshift_kernel(8, d):
                cls = tmd.BlackwellMultiHeadLatentAttentionForwardFP8
                self.assertIsNot(cls, stock)
                self.assertIs(tmd._get_compiled_mla_kernel(1)[0], cls)  # own cache: not the stock entry for 1
            self.assertIs(tmd.BlackwellMultiHeadLatentAttentionForwardFP8, stock)
            self.assertIs(tmd._get_compiled_mla_kernel, getter)
            self.assertIs(getter(1)[0], stock)
            with self.assertRaises(RuntimeError), mla_absorb.pshift_kernel(8, d):
                raise RuntimeError("inside")
            self.assertIs(tmd._get_compiled_mla_kernel, getter)  # restored on an exception too
            written = list(d.glob("mla_decode_fp8_pshift8_*.py"))
            self.assertEqual(len(written), 1)  # built once per shift and process
            self.assertIn("softmax_scale_log2 + 8.0", written[0].read_text())
        mla_absorb._pshift.clear()


class Planner(unittest.TestCase):
    def test_plan_and_workspace_bound(self):
        cc, sm = (10, 3), 148
        # the M-04k crash shape: M128, fold 1, split 5, 96 * 13 * 5 * 513 * 4 bytes
        self.assertEqual(rt.ts_plan(1, 13, 96, 512, sm, cc, 9602), (128, 1, 5, 96 * 13 * 5 * 513 * 4))
        self.assertEqual(rt.ts_plan(1, 38, 96, 512, sm, cc, 9602)[2:], (1, 0))  # split 1 from q_len 38 at B = 1
        self.assertEqual(rt.ts_plan(1, 37, 96, 512, sm, cc, 9602)[2], 2)
        self.assertEqual(rt.ts_plan(1, 4, 16, 512, sm, (10, 0), 1000)[:2], (64, 4))  # the SM100 M64 family
        image = sm * 96 * 8 * 513 * 4  # vllm _get_workspace with _TOKENSPEED_MAX_Q_LEN = 8
        bound = rt.ws_bound(96, 512, sm, cc, max_b=16, max_q=15)
        self.assertLessEqual(bound, 74 * 96 * 513 * 4)  # B * q_len * split <= num_sm // 2 whenever split > 1
        self.assertLess(bound, image)
        for q in range(1, 16):  # the M128 split does not depend on the context length
            self.assertEqual(len({rt.ts_plan(1, q, 96, 512, sm, cc, s) for s in (1, 9600, 64000)}), 1)
        print(f"decode workspace bound (B<=16, q_len<=15, 148 SMs) {bound} B; image buffer {image} B")


class Emulation(unittest.TestCase):
    def test_merge_and_quantization_level(self):
        pb = problem(4, ctx=40, n=6)
        lat8, latm = latents(pb)
        sel = torch.tensor([0, 3, 5])
        # kv_b_proj rows per head: [k_nope (W_UK_T[h], P x L) | v (W_UV[h]^T, V x L)]
        w_kv = torch.cat([pb.w_uk_t, pb.w_uv.transpose(1, 2)], 1).reshape(-1, pb.L).to(torch.bfloat16)
        args = (lambda a, b: lat8[a:b], pb.ctx)
        tail = (pb.new_c, pb.new_pe.view(pb.n, -1), pb.q, sel, w_kv, pb.scale)
        one = rt.image_emulation(*args, [(0, 40)], *tail)[0]
        three = rt.image_emulation(*args, [(0, 13), (13, 29), (29, 40)], *tail)[0]
        self.assertLess(rt.rel(three, one), 2e-2)  # chunking changes only bf16 rounding of the partials
        qd = pb.q[sel].double()
        wk = w_kv.view(pb.hl, -1, pb.L)[:, : pb.P].double()
        wv = w_kv.view(pb.hl, -1, pb.L)[:, pb.P :].transpose(1, 2).double()
        ref = rt.dense_reference(latm, qd[..., : pb.P], qd[..., pb.P :], wk, wv, pb.ctx + sel, pb.scale, pb.L, pb.ctx)[0]
        e = rt.rel(one, ref)
        self.assertTrue(1e-3 < e < 0.2, e)  # fp8 level, not exact, not broken


class R7Bookkeeping(unittest.TestCase):
    def test_cases_ids_and_open(self):
        a, b = rt.r7_cases()
        self.assertEqual((len(a), len(b)), (60, 20))
        self.assertEqual(rt.parse_id(rt.case_id("ABS", 128, 76800, 1, "spawn2")),
                         {"phase": "ABS", "q_len": 128, "ctx": 76800, "causal": 1})
        recs = [{"id": a[0], "start": True}, {"id": a[0], "rc": "ok", **rt.parse_id(a[0])}, {"id": a[1], "start": True},
                {"id": "A:13:76800:1", "rc": "CUDADialectError: cudaErrorInvalidValue", **rt.parse_id("A:13:76800:1")}]
        self.assertEqual(rt.r7_open(recs), [a[1]])
        c = rt.r7_c_cases(recs)
        self.assertEqual(len(c), 6)  # 13 @ 76800 causal is already in the fixed C set
        recs.append({"id": "B:74:76800:0", "rc": "x", **rt.parse_id("B:74:76800:0")})
        self.assertEqual(rt.r7_c_cases(recs)[-1], "C:74:76800:0")
        self.assertEqual(rt.short_error(RuntimeError("\x1b[91merror:\x1b[0m bad\n  value")), "RuntimeError: error: bad value")


if __name__ == "__main__":
    unittest.main(verbosity=1)
