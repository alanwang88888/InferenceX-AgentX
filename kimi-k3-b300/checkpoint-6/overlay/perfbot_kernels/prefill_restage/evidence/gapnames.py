# PERFBOT P-07F: item 05 ride-along reader (CPU, reads an nsys sqlite export); see RIDEALONG.md.
"""For given steps: device gaps > 50 us on device 0 — kernel before / after, and host runtime calls overlapping the gap. CPU only."""
import sqlite3, re, sys, csv, collections
db, fam_csv = sys.argv[1], sys.argv[2]
c = sqlite3.connect(db)
S = {r[0]: r[1] for r in c.execute("select id, value from StringIds")}
walls = {int(r['step']): float(r['wall_ms']) for r in csv.DictReader(open(fam_csv))}
for step in [int(x) for x in sys.argv[3].split(",")]:
    st = c.execute("select min(start) from NVTX_EVENTS where text like ? ", (f"perfbot_step_{step} %",)).fetchone()[0]
    en = st + int(walls[step] * 1e6)
    ks = c.execute("select k.start, k.end, k.graphId, s.value, k.globalPid, k.correlationId from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName where k.deviceId=0 and k.start>=? and k.start<? order by k.start", (st, en)).fetchall()
    pid = ks[0][4]
    tid = c.execute("select globalTid from CUPTI_ACTIVITY_KIND_RUNTIME where correlationId=? and (globalTid>>24)=(?>>24)", (ks[0][5], pid)).fetchone()[0]
    rts = c.execute("select start, end, nameId from CUPTI_ACTIVITY_KIND_RUNTIME where globalTid=? and start>=? and start<? order by start", (tid, st - 10**8, en)).fetchall()
    pair = collections.Counter(); pair_ms = collections.Counter(); hostcall = collections.Counter()
    cur_end = None; prev = None
    gaps = []
    for s0, e0, gid, name, _, _ in ks:
        if cur_end is not None and s0 - cur_end > 50_000:
            gaps.append((cur_end, s0, prev, ("G:" if gid else "E:") + name[:40]))
        if cur_end is None or e0 > cur_end: cur_end = e0; prev = ("G:" if gid else "E:") + name[:40]
    for g0, g1, a, b in gaps:
        pair[(a, b)] += 1; pair_ms[(a, b)] += (g1 - g0) * 1e-6
        # host calls active during the gap
        for r0, r1, nid in rts:
            if r1 > g0 and r0 < g1:
                nm = re.sub(r"_v\d+$", "", S.get(nid, "?")); hostcall[nm] += (min(r1, g1) - max(r0, g0)) * 1e-6
    tot = sum(pair_ms.values())
    print(f"== step {step}: {len(gaps)} gaps > 50us, {tot:.1f} ms")
    for k, v in pair_ms.most_common(8): print(f"   {v:6.2f} ms x{pair[k]:3d}  {k[0]}  ->  {k[1]}")
    print("   host API inside gaps (ms):", {k: round(v, 2) for k, v in hostcall.most_common(6)})
