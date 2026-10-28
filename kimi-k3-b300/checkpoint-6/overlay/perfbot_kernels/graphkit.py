"""graphkit: the program kit for kernel-only harnesses that compare captured CUDA graphs (graph-pair fillers).

Why (item 05, B300, 2026-10-21) [m]: a re-captured copy of the same tiny-kernel graph can time 24-36 % apart
(i05inst); the cause is the address of its small tensors, a 4 KiB-granular hash with two classes, 1.003 / 1.392
us per dependent node, ~50 % slow (i05addr). GEMM-sized graphs keep a ~0.2 % instance floor. Ruling (perf-bot-8e): a
graph-pair verdict holds only through its own re-captured null inside bound.

One generator for every harness:
  kit = Kit(TorchBackend(torch, device))       # or any backend with the same 8 methods (FakeBackend in the test)
  y = kit.small("y", 1, torch.float32)         # rule 1: every small tensor is a view of ONE allocation per arm set
  kit.arm("tiny84", lambda: [y.add_(1) for _ in range(84)])
  kit.null("tiny84")                           # rule 2: the re-captured null is an arm (same body, captured again)
  res = kit.run(windows=30, k=10)              # rule 3: order rotated by one per window: every arm, null included,
                                               #         visits every position
  ok, why, stats = kit.gate(res)               # rules 2-4: VOID on null, A/A, position or coverage, bounds from BOUNDS
  print(kit.addr_line())                       # KIT-ADDR {json}: base, span and every small view's address
  print(line("TAG", enum if ok else "VOID", {...}))
The harness keeps its own question, enum and bits check; the kit owns capture, timing order and the VOID gate.

v2 (2026-10-22, after i08s3): Kit(..., shared_pool=True) captures every arm of the set into ONE graph memory pool
(torch.cuda.graph_pool_handle), so intermediates the graphs allocate and free during capture reuse the same addresses;
the re-captured null then differs from its arm only in capture order. kit.pool_line() prints KIT-POOL {json}: the mode
and, per arm, the allocated / reserved bytes the capture added. i08s3 [m] saw a same-sign capture-order offset of
0.4-0.6 us (1.0-1.5 %) on 40-66 us graphs with v1 (private pool per capture); i05kit2 calibrates v2 against v1.
Default shared_pool=False keeps v1 behaviour; arm(..., shared=bool) overrides it per arm (calibration only)."""
import json

VOID_FRAC = 0.05   # null, A/A and position spread bound, as a fraction of the arm's median
ALIGN = 512        # small views start on 512 B boundaries (one 4 KiB hash block holds 8 of them)
# Bounds and floors, each with its source; a harness prints the rows it relies on.
BOUNDS = {
    "void_frac": {"value": VOID_FRAC, "src": "i05lnode/i05hide/i05inst pre-registered rule"},
    "tiny_node_fast_us": {"value": 1.003, "src": "i05addr tree c42bb313, line 585e9b7e"},
    "tiny_node_slow_us": {"value": 1.392, "src": "i05addr tree c42bb313, line 585e9b7e"},
    "tiny_far_share": {"value": 0.496, "src": "i05addr tree c42bb313, line 585e9b7e"},
    "hash_block_bytes": {"value": 4096, "src": "i05addr: class constant inside 8/8 4 KiB blocks"},
    "tiny_chain_instance_spread": {"value": 0.357, "src": "i05inst tree d870b705 (fresh addresses)"},
    "gemm_instance_spread": {"value": 0.0016, "src": "i05inst tree d870b705 (F.linear [8x35840]x[35840x7168])"},
}


def median(xs):
    s = sorted(xs)
    n = len(s)
    return (s[n // 2] + s[(n - 1) // 2]) / 2


def line(tag, enum, rec):
    """The harness verdict line: `<TAG> <enum> {json}`, sorted str keys."""
    return f"{tag} {enum} " + json.dumps(rec, sort_keys=True, default=str)


class TorchBackend:
    """CUDA graphs on one device. Methods: alloc, view, capture, replay, event, elapsed_us, sleep, sync."""

    def __init__(self, torch, device, sleep_cycles=2_000_000):
        self.t, self.dev, self.cycles = torch, device, sleep_cycles
        if torch.device(device).type == "cuda":  # a CPU device serves the CPU test of alloc / view only
            torch.cuda.set_device(device)

    def alloc(self, nbytes):
        self.buf = self.t.zeros(nbytes, dtype=self.t.uint8, device=self.dev)
        return self.buf.data_ptr()

    def view(self, offset, numel, dtype):
        n = numel * self.t.empty((), dtype=dtype).element_size()
        return self.buf[offset:offset + n].view(dtype)

    def capture(self, body, pool=None):
        for _ in range(2):  # warm-up outside capture (handles, workspaces)
            body()
        self.sync()
        g = self.t.cuda.CUDAGraph()
        with self.t.cuda.graph(g, pool=pool):
            body()
        return g

    def pool(self):
        return self.t.cuda.graph_pool_handle()

    def mem(self):
        return self.t.cuda.memory_allocated(self.dev), self.t.cuda.memory_reserved(self.dev)

    def replay(self, g):
        g.replay()

    def event(self):
        e = self.t.cuda.Event(enable_timing=True)
        e.record()
        return e

    def elapsed_us(self, a, b):
        return a.elapsed_time(b) * 1e3

    def sleep(self):
        self.t.cuda._sleep(self.cycles)  # the host enqueues the next block while the GPU sleeps

    def sync(self):
        self.t.cuda.synchronize()


class Kit:
    def __init__(self, backend, span=1 << 20, shared_pool=False):
        self.b, self.span, self.shared_pool = backend, span, shared_pool
        self.base = backend.alloc(span)
        self.views, self.next, self.arms, self.nulls, self.pools = {}, 0, {}, {}, {}
        self.handle = None

    def _capture(self, name, body, shared):
        """Capture into the kit's one pool (v2) or a private pool (v1); record what the capture added."""
        shared = self.shared_pool if shared is None else shared
        if shared and self.handle is None:
            self.handle = self.b.pool()
        mem = getattr(self.b, "mem", lambda: (0, 0))  # a v1 backend (no mem / pool) still works in v1 mode
        a0, r0 = mem()
        g = self.b.capture(body, self.handle) if shared else self.b.capture(body)
        a1, r1 = mem()
        self.pools[name] = {"shared": bool(shared), "alloc_added": a1 - a0, "reserved_added": r1 - r0}
        return g

    def small(self, name, numel, dtype, itemsize=4):
        """Rule 1: a small tensor carved from the one shared allocation; its address is recorded. itemsize: bytes per
        element of dtype (4 for fp32 / int32, 2 for bf16)."""
        size = numel * itemsize
        if self.next + size > self.span:
            raise ValueError(f"small {name}: {size} B does not fit in the {self.span} B kit span")
        off = self.next
        self.next = (off + size + ALIGN - 1) // ALIGN * ALIGN
        self.views[name] = {"offset": off, "addr": self.base + off, "bytes": size}
        return self.b.view(off, numel, dtype)

    def arm(self, name, body, shared=None):
        if name in self.arms:
            raise ValueError(f"arm {name} twice")
        self.arms[name] = (body, self._capture(name, body, shared))

    def null(self, of):
        """Rule 2: the same body captured again (same pool mode as its arm), timed as its own arm `null:<of>`."""
        self.nulls[f"null:{of}"] = of
        self.arms[f"null:{of}"] = (self.arms[of][0],
                                   self._capture(f"null:{of}", self.arms[of][0], self.pools[of]["shared"]))

    def run(self, windows, k=10, sync_every=5):
        """Rule 3: windows x k replays per arm; arm j sits at position (j - w) mod n in window w."""
        keys = list(self.arms)
        n = len(keys)
        t = {a: [] for a in keys}
        pos = {a: [] for a in keys}
        pend = []
        for w in range(windows):
            for p in range(n):
                a = keys[(p + w) % n]
                self.b.sleep()
                s = self.b.event()
                for _ in range(k):
                    self.b.replay(self.arms[a][1])
                pend.append((a, p, s, self.b.event()))
            if w % sync_every == sync_every - 1 or w == windows - 1:
                self.b.sync()
                for a, p, s, e in pend:
                    t[a].append(self.b.elapsed_us(s, e) / k)
                    pos[a].append(p)
                pend = []
        return {"t": t, "pos": pos, "windows": windows, "narms": n}

    def gate(self, res):
        """Rules 2-4 -> (ok, why, stats). stats: median, A/A, null diff and position spread per arm."""
        n, W = res["narms"], res["windows"]
        stats = {}
        for a, ts in res["t"].items():
            m = median(ts)
            byp = {}
            for v, p in zip(ts, res["pos"][a]):
                byp.setdefault(p, []).append(v)
            pm = [median(v) for v in byp.values()]
            stats[a] = {"median_us": m, "aa_us": abs(median(ts[0::2]) - median(ts[1::2])),
                        "pos_spread": (max(pm) - min(pm)) / m, "positions": len(byp)}
        if W < n:
            return False, f"windows {W} < arms {n}: not every position covered", stats
        for a, s in stats.items():
            if s["aa_us"] > VOID_FRAC * s["median_us"]:
                return False, f"{a}: A/A {s['aa_us']:.3f} > {VOID_FRAC:.0%} of {s['median_us']:.3f} us", stats
        for nl, of in self.nulls.items():
            d = stats[nl]["median_us"] - stats[of]["median_us"]
            stats[nl]["null_us"] = d
            if abs(d) > VOID_FRAC * stats[of]["median_us"]:
                return False, f"{nl}: re-captured null {d:+.3f} us > {VOID_FRAC:.0%} of {of}", stats
            if stats[nl]["pos_spread"] > VOID_FRAC:
                return False, f"{nl}: position spread {stats[nl]['pos_spread']:.1%} > {VOID_FRAC:.0%}", stats
        if not self.nulls:
            return False, "no re-captured null arm (ruling: a graph-pair verdict needs its own)", stats
        return True, "", stats

    def pool_line(self):
        return "KIT-POOL " + json.dumps({"shared_pool": self.shared_pool, "arms": self.pools}, sort_keys=True)

    def addr_line(self):
        return "KIT-ADDR " + json.dumps({"base": self.base, "span": self.span, "views": self.views}, sort_keys=True)
