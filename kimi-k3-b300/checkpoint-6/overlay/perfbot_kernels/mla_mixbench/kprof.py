"""Per-kernel breakdown of one MLA prefill layer call at the split-sweep shapes (coordinator request 2026-10-18:
measure, do not infer, what the ~1.8-2.0 ms per layer left after route A is, before a glue kernel is written).

  python3 -B -m perfbot_kernels.mla_mixbench.kprof [--variants image,image_split,absorbed] [--calls 5] [--deadline-s S]

8 processes, the bench's inputs (bench.make_case) and variants. Per shape and variant: warm-up, then `--calls`
calls under torch.profiler (CUDA activity, rank 0 only; every rank runs the same calls, the collectives need them).
Per kernel name on rank 0: device us per layer call and launches per call. Also per call: the sum of kernel time,
the synced wall (bench.time_eager), and wall - busy = time the GPU waited (host launch gaps, collective skew).
NCCL / all-gather kernel time includes waiting for the slowest rank.
Lines: `MIXBENCH-PROF n= ctx= variant= kernel=<name> us_per_call= launches_per_call=` and
`MIXBENCH-PROF-SUM n= ctx= variant= busy_us= wall_us= idle_us= kernels=`; exit 0 iff every shape and variant
produced a record. Record: $PERFBOT_WORKSPACE/$PERFBOT_RESULTS/mixbench-profile.json.
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import torch

import perfbot_kernels.mla_mixbench as mb
from perfbot_kernels.mla_mixbench import W
from perfbot_kernels.mla_mixbench import bench

PORT = bench.PORT + 7


def kernel_table(prof, calls):
    """{kernel name: (us per call, launches per call)} from the profiler's device events."""
    acc = {}
    for e in prof.events():
        if str(getattr(e, "device_type", "")).split(".")[-1] != "CUDA":
            continue
        us = e.time_range.elapsed_us()
        t, c = acc.get(e.name, (0.0, 0))
        acc[e.name] = (t + us, c + 1)
    return {k: (round(t / calls, 2), round(c / calls, 2)) for k, (t, c) in sorted(acc.items(), key=lambda x: -x[1][0])}


def run_profile(env, args, rec, save, t0):
    shapes = mb.load_shapes()
    for i in bench.sweep_indices(shapes):
        if env.any_flag(time.time() - t0 > args.deadline_s - 120):
            rec["stopped_at"] = {"shape": i}
            break
        sh = shapes[i]
        case = bench.make_case(env, sh, i)
        row = rec["shapes"][str(i)] = {"idx": i, "n": sh["n"], "ctx": sh["ctx"], "variants": {}}
        for name in args.variants:
            fn = mb.VARIANTS[name].run
            _, first, err = bench.warm(env, fn, case)
            v = row["variants"][name] = {"first_call_s": first}
            if err:
                v["error"] = err
                continue
            v["synced_us"], _ = bench.time_eager(env, lambda: fn(case), True)
            env.barrier()
            torch.cuda.synchronize()
            if env.rank == 0:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                    for _ in range(args.calls):
                        fn(case)
                    torch.cuda.synchronize()
                v["kernels"] = kernel_table(prof, args.calls)
                v["busy_us"] = round(sum(t for t, _ in v["kernels"].values()), 1)
            else:
                for _ in range(args.calls):
                    fn(case)
                torch.cuda.synchronize()
        save()
        del case
        torch.cuda.empty_cache()


def report_profile(args, ranks, path, t0, exitcodes) -> int:
    r0 = ranks[0]
    ok = "error" not in r0 and all(e == 0 for e in exitcodes)
    rows = []
    for i in bench.sweep_indices(mb.load_shapes()):
        row = r0.get("shapes", {}).get(str(i))
        if row is None:
            ok = False
            continue
        for name in args.variants:
            v = row["variants"].get(name, {})
            if "kernels" not in v:
                ok = False
                print(f"MIXBENCH-PROF n={row['n']} ctx={row['ctx']} variant={name} ERROR {v.get('error', 'no record')[-300:]!r}",
                      flush=True)
                continue
            for k, (us, cnt) in v["kernels"].items():
                print(f"MIXBENCH-PROF n={row['n']} ctx={row['ctx']} variant={name} kernel={k[:160]} us_per_call={us} "
                      f"launches_per_call={cnt}", flush=True)
            print(f"MIXBENCH-PROF-SUM n={row['n']} ctx={row['ctx']} variant={name} busy_us={v['busy_us']} "
                  f"wall_us={v['synced_us']} idle_us={round(v['synced_us'] - v['busy_us'], 1)} "
                  f"kernels={len(v['kernels'])} launches={round(sum(c for _, c in v['kernels'].values()), 1)}", flush=True)
        rows.append(row)
    record = {"item": "expedition-route-A-profile", "args": vars(args), "exitcodes": exitcodes,
              "seconds": round(time.time() - t0, 1), "rows": rows,
              "errors": {r["rank"]: r["error"][-1500:] for r in ranks if "error" in r}}
    path.write_text(json.dumps(record, indent=1, default=str))
    ok = ok and not record["errors"]
    print(f"PERFBOT-MIXBENCH-PROFILE done={ok} record={path}", flush=True)
    return 0 if ok else 1


def _rank_main(rank, port, out_dir, args):
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=W, device_id=dev)
    import perfbot_kernels.mla_mixbench.variants  # noqa: F401  (registers the variants in this process)

    t0 = time.time()
    rec = {"rank": rank, "shapes": {}}
    path = Path(out_dir, f"mixbench-profile-rank{rank}.json")
    save = lambda: path.write_text(json.dumps(rec, indent=1, default=str))
    try:
        with torch.inference_mode():
            env = bench.Env(rank, dev)
            run_profile(env, args, rec, save, t0)
    except Exception:  # noqa: BLE001
        rec["error"] = traceback.format_exc()
        traceback.print_exc()
    rec["seconds"] = round(time.time() - t0, 1)
    save()
    print(f"MIXBENCH-PROF rank={rank} done error={'error' in rec}", flush=True)
    try:
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="image,image_split,absorbed")
    ap.add_argument("--calls", type=int, default=5)
    ap.add_argument("--deadline-s", type=float, default=600.0)
    args = ap.parse_args()
    args.variants = [v for v in args.variants.split(",") if v]
    import perfbot_kernels.mla_mixbench.variants  # noqa: F401

    unknown = [v for v in args.variants if v not in mb.VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants {unknown}; registered {sorted(mb.VARIANTS)}")
    path = bench.results_path("mixbench-profile.json")
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_rank_main, args=(r, PORT, str(path.parent), args)) for r in range(W)]
    t0 = time.time()
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=max(1.0, args.deadline_s + 60 - (time.time() - t0)))
    for p in procs:
        if p.is_alive():
            p.terminate()
    ranks = []
    for r in range(W):
        f = path.parent / f"mixbench-profile-rank{r}.json"
        ranks.append(json.loads(f.read_text()) if f.is_file() else {"rank": r, "error": "no record"})
    return report_profile(args, ranks, path, t0, [p.exitcode for p in procs])


if __name__ == "__main__":
    sys.exit(main())
