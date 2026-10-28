"""Kernel-only GPU gate for EXPERTLOG (one GPU). Run in the server env (serve.sh PERFBOT_PRE_EXEC):

  python3 -B -m perfbot_kernels.expertlog.selftest

On moe_stage.selftest's synthetic K3 MoE layer (896 experts, top-16, per-rank shapes) driven through the overlay's
real TrtLlmMxfp4ExpertsMonolithic.apply, with the module flag _PERFBOT_EXPERTLOG flipped in place:
  untouched  apply() with the log on leaves x and the carrier (packed top-K + logits) bit-identical.
  output     log on vs off. Image path (M 128 > stage MAX_M): finalized rows and expert weights bit for bit.
             Stage path (M 8, 48): expert weights and permute map bit for bit; the stored rows bit for bit, or, when
             two off runs already differ (fc2's cross-CTA fp32 red.add order, moe_stage selftest graph_probe),
             within that probe's tolerance (rtol 2^-6, atol 1e-3 x max).
  counts     eager: rec[slot, 0] == distinct ids over the first `live` rows.
  graph      apply() with the log on captured in a CUDA graph (M 8, 48, 128): three replays, each with a new carrier
             and a new (slot, live); rec equals each replay's own count (the graph reads ctl at replay; a sync
             inside capture would raise).
Writes $W/$PERFBOT_RESULTS/expertlog-gate.json; exits 1 on any failed check.
"""

import json
import sys
import traceback

import torch

SHAPES = [8, 48, 128]


def _rows(mst, tail, out, m):
    if getattr(out, "perfbot_prefilled", False):
        return tail.rows(m).clone()
    return mst.tail_finalize(out, m)


def _run(b, tail, mst, inp, on: bool):
    b.mod._PERFBOT_EXPERTLOG = on
    tail.arm(1)
    out = b.candidate(inp)
    torch.cuda.synchronize()
    return out, _rows(mst, tail, out, inp.m)


def _eq(a, b):
    return bool(torch.equal(a.view(torch.int16), b.view(torch.int16)))


def check_shape(b, tail, mst, el, m: int) -> dict:
    inp = b.inputs(m)
    x0, c0 = inp.x.clone(), inp.carrier.clone()
    off1, r_off1 = _run(b, tail, mst, inp, False)
    off2, r_off2 = _run(b, tail, mst, inp, False)
    on, r_on = _run(b, tail, mst, inp, True)
    rec = {"stage_path": bool(getattr(on, "perfbot_prefilled", False)),
           "untouched": bool(torch.equal(inp.x.view(torch.int16), x0.view(torch.int16))
                             and torch.equal(inp.carrier.view(torch.int32), c0.view(torch.int32))),
           "weights_equal": _eq(on.expert_weights, off1.expert_weights),
           "rows_bitwise": _eq(r_on, r_off1), "off_off_bitwise": _eq(r_off2, r_off1),
           "rows_max_abs_diff": (r_on.float() - r_off1.float()).abs().max().item()}
    if rec["stage_path"]:
        rec["perm_equal"] = bool(torch.equal(on.expanded_idx_to_permuted_idx, off1.expanded_idx_to_permuted_idx))
    ref = r_off1.float()
    close = bool(torch.allclose(r_on.float(), ref, rtol=2 ** -6, atol=1e-3 * ref.abs().max().item()))
    rec["rows_ok"] = rec["rows_bitwise"] or (rec["stage_path"] and not rec["off_off_bitwise"] and close)
    st = el.state(b.dev)
    live = max(1, m - 3)
    st.set_step(0, live)
    _run(b, tail, mst, inp, True)
    want = len(set(inp.ids[:live].reshape(-1).tolist()))
    got = st.read(1, 1)[0][0]
    rec["count_eager"], rec["count_want"] = got, want
    rec["counts_ok"] = got == want
    rec["graph"] = graph_probe(b, tail, mst, el, m)
    rec["ok"] = all(rec[k] for k in ("untouched", "weights_equal", "rows_ok", "counts_ok")) \
        and rec.get("perm_equal", True) and rec["graph"]["ok"]
    return rec


def graph_probe(b, tail, mst, el, m: int) -> dict:
    inp = b.inputs(m)
    b.mod._PERFBOT_EXPERTLOG = True
    tail.arm(1)
    b.candidate(inp)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        b.candidate(inp)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        b.candidate(inp)
    st = el.state(b.dev)
    want = []
    for slot in range(3):
        new = b.inputs(m)
        inp.x.copy_(new.x)
        inp.carrier.copy_(new.carrier)
        live = max(1, m - 2 * slot)
        st.set_step(slot, live)
        tail.arm(1)
        graph.replay()
        want.append(len(set(new.ids[:live].reshape(-1).tolist())))
    torch.cuda.synchronize()
    got = [r[0] for r in st.read(3, 1)]
    del graph
    return {"ok": got == want, "got": got, "want": want}


def main() -> int:
    record, ok = {"item": "expertlog", "device": torch.cuda.get_device_name()}, False
    path = None
    try:
        import perfbot_kernels.expertlog as el
        import perfbot_kernels.moe_stage as ms
        import perfbot_kernels.moe_stage.selftest as mst

        path = mst.results_path("expertlog-gate.json")
        b = mst.Bench()
        tail = mst.FakeTail(b.dev)
        ms.set_target(b.dev, tail.buf.data_ptr(), tail.flags, mst.SELF_RANK, mst.MAXM_TAIL)
        record["module_file"] = b.mod.__file__
        record["flag_at_import"] = bool(b.mod._PERFBOT_EXPERTLOG)
        shapes = {}
        for m in SHAPES:
            shapes[f"M{m}"] = check_shape(b, tail, mst, el, m)
            print(f"EXPERTLOG-CHECK M={m} {json.dumps(shapes[f'M{m}'])}", flush=True)
        record["shapes"] = shapes
        ok = all(r["ok"] for r in shapes.values())
    except Exception:
        record["error"] = traceback.format_exc()
        traceback.print_exc()
    record["ok"] = ok
    if path is not None:
        path.write_text(json.dumps(record, indent=1, default=str))
    print(f"PERFBOT-EXPERTLOG-GATE ok={ok} record={path}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
