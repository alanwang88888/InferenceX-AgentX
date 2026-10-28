# PERFBOT P-07F: item 05 ride-along reader (CPU, reads an nsys sqlite export); see RIDEALONG.md.
"""Per step: host API span on the launching thread vs device busy; how long the device waits for the host. CPU only."""
import sqlite3, re, sys, csv
db, fam_csv = sys.argv[1], sys.argv[2]
c = sqlite3.connect(db)
S = {r[0]: r[1] for r in c.execute("select id, value from StringIds")}
F = {int(r['step']): r for r in csv.DictReader(open(fam_csv))}
print("step n ctx wall | first_apply_write->first_prepare_prefill (host prologue hole) | fwd: host launch span, dev span, dev busy, launch lag at end")
for step in [int(x) for x in sys.argv[3].split(",")]:
    r = F[step]; wall = float(r['wall_ms'])
    st, tid = c.execute("select start, globalTid from NVTX_EVENTS where text like ? order by start limit 1", (f"perfbot_step_{step} %",)).fetchone()
    en = st + int(wall * 1e6)
    pidhi = tid >> 24
    ks = c.execute("select k.start, k.end, s.value, k.correlationId from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName where k.deviceId=0 and k.start>=? and k.start<? order by k.start", (st, en)).fetchall()
    rt = {x[0]: x[1] for x in c.execute("select correlationId, start from CUPTI_ACTIVITY_KIND_RUNTIME where globalTid=? and start>=? and start<?", (tid, st - 5_000_000, en))}
    aw = next((k for k in ks if '_apply_write' in k[2]), None)
    pp = next((k for k in ks if '_prepare_prefill_inputs' in k[2]), None)
    # forward = from the kernel after get_aligned_state_indices (end of prologue) to the last kernel
    i0 = next((i for i, k in enumerate(ks) if 'get_aligned_state_indices' in k[2]), 0)
    fw = ks[i0 + 1:]
    hl = [rt[k[3]] for k in fw if k[3] in rt]
    busy = 0; ce = None
    for s0, e0, *_ in fw:
        if ce is None or s0 >= ce: busy += e0 - s0; ce = e0
        elif e0 > ce: busy += e0 - ce; ce = e0
    lag_end = (fw[-1][0] - (rt.get(fw[-1][3]) or fw[-1][0])) * 1e-6
    print(step, r['n'], r['ctx'], wall, '|', round(((pp[0] if pp else 0) - (aw[0] if aw else 0)) * 1e-6, 2), '|',
          round((max(hl) - min(hl)) * 1e-6, 1), round((fw[-1][1] - fw[0][0]) * 1e-6, 1), round(busy * 1e-6, 1), round(lag_end, 1))
