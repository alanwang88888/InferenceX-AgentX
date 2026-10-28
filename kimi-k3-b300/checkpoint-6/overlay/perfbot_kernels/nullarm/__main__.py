"""python3 -B -m perfbot_kernels.nullarm <log> [<record.json>]: the shared run paths' reader of a gate's output.
Prints the last PERFBOT-NULLARM record of <log> (and writes it to <record.json>), exit 0; with no complete record it
prints `PERFBOT-VERDICT INCOMPLETE: no null arm` and exits 78."""
import json
import sys
from pathlib import Path

from perfbot_kernels.nullarm import complete, read

INCOMPLETE_RC = 78

rec = read(Path(sys.argv[1]).read_text(errors="replace"))
if len(sys.argv) > 2:
    Path(sys.argv[2]).write_text(json.dumps(rec) + "\n")
if not complete(rec):
    print("PERFBOT-VERDICT INCOMPLETE: no null arm", flush=True)
    sys.exit(INCOMPLETE_RC)
print(f"PERFBOT-VERDICT null_spread_us={rec['null_spread_us']:.3f} "
      + " ".join(f"{r}={v['verdict']}" for r, v in sorted(rec["rows"].items())), flush=True)
