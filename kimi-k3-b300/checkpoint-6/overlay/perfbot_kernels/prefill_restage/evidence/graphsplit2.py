# PERFBOT P-07F: item 05 ride-along reader (CPU, reads an nsys sqlite export); see RIDEALONG.md.
"""Per prefill step, rank 0 (device 0): graph vs eager kernels, device gaps charged to the kernel before the gap
(by family), host API time on the launching thread. Step bounds = families.csv (mark, mark + wall). CPU only."""
import sqlite3, re, sys, json, csv, collections
db, fam_csv = sys.argv[1], sys.argv[2]
c = sqlite3.connect(db)
S = {r[0]: r[1] for r in c.execute("select id, value from StringIds")}
walls = {}
for r in csv.DictReader(open(fam_csv)):
    walls[int(r['step'])] = (float(r['wall_ms']), int(r['n']), r['ctx'], int(r['batch']))
marks = {}
for st, txt, tid in c.execute("select start, text, textId from NVTX_EVENTS where text like 'perfbot_step_%' or textId in (select id from StringIds where value like 'perfbot_step_%')"):
    t = txt if txt else S.get(tid, "")
    m = re.match(r"perfbot_step_(\d+) ", t)
    if m and int(m[1]) in walls: marks.setdefault(int(m[1]), []).append(st)
def fam(n):
    n = n.lower()
    for k, keys in [("mla_ctx", ["fmha", "dcp_kv_gather", "materialize_kv", "gather_and_maybe_dequant", "catarraybatched", "increment_epoch", "kvconcatpack", "qkvquantkvcache", "merge_attn", "computelse"]),
                    ("kda", ["flash_kda", "causal_conv1d", "gather_initial", "layer_norm_gated", "store_cache_checkpoint", "kda", "chunk_"]),
                    ("moe", ["bmm_", "routing", "finalize", "quantizemxfp8", "situ_and_mul", "moe"]),
                    ("gemm", ["nvjet", "cublas", "gemm", "sm100_xmma", "cutlass"]),
                    ("comm", ["allreduce", "nccl", "direct_dcp", "multimem", "lamport"]),
                    ("norm_elt", ["rms_norm", "elementwise", "index", "cat", "triton", "copy", "fill", "reduce"])]:
        if any(x in n for x in keys): return k
    return "other"
out = []
for step in sorted(int(x) for x in sys.argv[3].split(",")):
    wall, n, ctx, b = walls[step]
    st = min(marks[step]); en = st + int(wall * 1e6)
    ks = c.execute("select k.start, k.end, k.graphId, k.correlationId, s.value, k.globalPid from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName where k.deviceId=0 and k.start>=? and k.start<? order by k.start", (st, en)).fetchall()
    # launching host thread(s): runtime rows correlated to these kernels
    pid = ks[0][5]
    corr = [k[3] for k in ks[:200]]
    tids = {r[0] for r in c.execute(f"select globalTid from CUPTI_ACTIVITY_KIND_RUNTIME where correlationId in ({','.join(map(str,corr))}) and (globalTid >> 24) = (? >> 24)", (pid,))}
    api = collections.Counter(); api_ms = collections.Counter()
    for tid in tids:
        for s0, e0, nid in c.execute("select start, end, nameId from CUPTI_ACTIVITY_KIND_RUNTIME where globalTid=? and start>=? and start<?", (tid, st, en)):
            nm = S.get(nid, "?"); nm = re.sub(r"_v\d+$", "", nm); api[nm] += 1; api_ms[nm] += (e0 - s0) * 1e-6
    gaps_by = collections.Counter(); kcount = collections.Counter(); gcount = collections.Counter(); kms = collections.Counter()
    cur_end = None; prev = None
    for s0, e0, gid, corr_, name, _p in ks:
        f = ("G:" if gid else "E:") + fam(name)
        kcount[f] += 1; kms[f] += (e0 - s0) * 1e-6
        if cur_end is not None and s0 > cur_end: gaps_by[prev] += (s0 - cur_end) * 1e-6
        if cur_end is None or e0 > cur_end: cur_end = e0; prev = f
    graphs = len({k[2] for k in ks if k[2]})
    row = dict(step=step, batch=b, n=n, ctx=ctx, wall=wall, kernels=len(ks), graph_kernels=sum(v for k, v in kcount.items() if k.startswith("G:")),
               distinct_graph_ids=graphs, inner_gap_ms=round(sum(gaps_by.values()), 2), host_tail_ms=round((en - cur_end) * 1e-6, 2),
               gap_after=dict((k, round(v, 2)) for k, v in gaps_by.most_common()), kernels_by=dict(kcount.most_common()),
               kernel_ms_by=dict((k, round(v, 1)) for k, v in kms.most_common()),
               api_top=dict((k, [api[k], round(api_ms[k], 1)]) for k, _ in api_ms.most_common(8)), host_threads=len(tids))
    print(json.dumps(row)); sys.stdout.flush()
