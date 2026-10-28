"""G-row ladder (expedition item G, LOG G-split): the image's decode GEMM router (vllm/models/kimi_k3/nvidia/
low_latency_gemm.py) sends a local (N, K) to dsv3_fused_a_gemm or the CuTe skinny GEMM only for M <= 16 (both
kernels reject M > 16); at k >= 2-3 decode rows (M = 8 x rows, q_len 8) six call sites fall back to cuBLAS nvjet +
splitKreduce. No new kernel: the candidate is the image's own router applied per row chunk of at most `chunk` rows
(rows are independent, so each chunk is the served M <= 16 call on a slice). selftest.py times it against the served
route at the served shapes.

SITES: the six G-split call sites (TP8 local shapes from the K3 config: hidden 7168, q_lora 1536, 96 heads x 192 /
8 = 2304, v 96 x 128 / 8 = 1536, 2 shared experts x 2 x 3072 / 8 = 1536, latent 3584), calls per decode step
(93 layers: 24 MLA, 69 KDA, 92 MoE), and their G-split slope share (ms per row). Pure Python; no torch here.
"""

M_PER_ROW = 8  # q_len 8 (num_speculative_tokens 7)
ROWS = (1, 2, 3, 4, 5, 6)
CHUNKS = (8, 16)
SITES = (  # (name, N, K, calls per step, aux stream, G-split slope ms/row)
    ("E1 latent down", 3584, 7168, 92, True, 0.108),
    ("E1 shared gate_up", 1536, 7168, 92, True, 0.086),
    ("M1 q_b", 2304, 1536, 24, False, 0.018),
    ("E3 shared down", 7168, 768, 92, True, 0.016),
    ("M1 g_proj", 1536, 7168, 24, True, 0.014),
    ("M1 qkv_a", 2112, 7168, 24, False, 0.011),
)


def shapes():
    """Distinct (N, K) over SITES, in SITES order."""
    out = []
    for _, n, k, *_ in SITES:
        if (n, k) not in out:
            out.append((n, k))
    return out


def chunks(m, size):
    """Row slices [(start, stop)] of at most `size` rows covering m rows."""
    if m < 1 or size < 1:
        raise ValueError(f"m {m}, size {size}")
    return [(a, min(a + size, m)) for a in range(0, m, size)]


def step_saving_ms(us, rows):
    """Per-step saving (ms) at `rows` decode rows from us[(N, K, M)] = {"served": t, "chunk<c>": t, ...}: per site,
    calls x (served - best candidate), clamped at 0 (the router would keep the served call). Kernel time, not exposed
    time: four of the six sites run on the aux stream (SITES[4]); the serve pair rules on exposure."""
    m = rows * M_PER_ROW
    tot = 0.0
    for _, n, k, calls, _, _ in SITES:
        t = us.get((n, k, m))
        if t:
            best = min((v for b, v in t.items() if b != "served"), default=t["served"])
            tot += calls * max(0.0, t["served"] - best) * 1e-3
    return tot
