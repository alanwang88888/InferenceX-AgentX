# PERFBOT P-07F: item 05 ride-along reader (CPU, reads an nsys sqlite export); see RIDEALONG.md.
"""Timeline of the first ~60 device events of a step + host runtime calls + NVTX ranges on the launching thread. CPU only."""
import sqlite3, re, sys, csv
db, fam_csv, step = sys.argv[1], sys.argv[2], int(sys.argv[3])
c = sqlite3.connect(db)
S = {r[0]: r[1] for r in c.execute("select id, value from StringIds")}
walls = {int(r['step']): float(r['wall_ms']) for r in csv.DictReader(open(fam_csv))}
st, tid0 = c.execute("select start, globalTid from NVTX_EVENTS where text like ? order by start limit 1", (f"perfbot_step_{step} %",)).fetchone()
en = st + int(walls[step] * 1e6)
ks = c.execute("select k.start, k.end, k.graphId, s.value, k.correlationId, k.globalPid from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName where k.deviceId=0 and k.start>=? and k.start<? order by k.start limit 80", (st - 5_000_000, en)).fetchall()
pidhi = ks[-1][5] >> 24
rt = {r[0]: (r[1], r[2], r[3]) for r in c.execute("select correlationId, start, nameId, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME where start>=? and start<? and (globalTid>>24)=?", (st - 30_000_000, en, pidhi))}
tid = tid0
print("mark thread", tid0)
ev = []
for s0, e0, gid, name, corr, _ in ks:
    h = rt.get(corr)
    ev.append((s0, f"DEV {'G' if gid else 'E'} {name[:45]:45s} dur {(e0-s0)/1e3:7.1f}us launched_at {(h[0]-st)/1e6 if h else float('nan'):8.3f}ms by {h[2] if h else '?'}"))
for s0, e0, txt, tidn in c.execute("select n.start, n.end, coalesce(n.text, s.value), n.globalTid from NVTX_EVENTS n left join StringIds s on s.id=n.textId where n.start>=? and n.start<? and (n.globalTid>>24)=? ", (st - 30_000_000, st + 40_000_000, pidhi)):
    ev.append((s0, f"NVTX {str(txt)[:60]} dur {((e0 or s0)-s0)/1e6:.3f}ms tid {tidn}"))
for t, s in sorted(ev)[:140]:
    print(f"{(t-st)/1e6:9.3f} {s}")
