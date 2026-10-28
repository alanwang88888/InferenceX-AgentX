"""E1 overlap harness (expedition G / ENDSTATE row E1, LOG G-split + KO-01): one K3 MoE layer's front, kernel only,
with the served two-stream structure, to measure how much aux-stream GEMM time the main stream hides at M 8 and 48.

Served structure (image vLLM 3696c772 + overlay; one aux stream for both forks, vllm.utils.torch_utils.aux_stream):
  fork "front"   KimiMoE._maybe_overlap_router_and_down_proj (overlay model.py:1466-1476, maybe_execute_in_parallel):
                 main  RouterTopK (torch.ops.perfbot_router_topk.route, model.py:1435)
                 aux   latent down 3584x7168 (ReplicatedLinear -> KimiK3LowLatencyLinearMethod: dsv3 M 2..8, else
                       F.linear = cuBLAS)
                 join  main waits the aux event before the experts.
  fork "experts" MoERunner._forward_impl -> SharedExperts (moe_runner.py:942 sync, :638 / :680 orders,
                 shared_experts.py:139 aux.wait_stream(main), :148-150 run on aux then main.wait_stream(aux)):
                 main  routed experts = TrtLlmMxfp4ExpertsMonolithic.apply -> perfbot_kernels.moe_stage
                       (MoePermute, MoeGemm fc1, MoeGemm fc2 into the tail's buffer)
                 aux   KimiMLP: gate_up 1536x7168, situ_and_mul, shared down 7168x768 (dsv3 M 1..16, else cuBLAS)
                 join  main waits aux; the latent tail (8-rank collective) follows and is not in this harness.
Serial fallback (served code, aux stream off): front runs route then latent down (maybe_execute_in_parallel with
aux_stream None); experts runs the shared MLP first (SharedExpertsOrder.NO_OVERLAP, moe_runner.py:638), then routed.

Pure Python; no torch here. selftest.py runs it on one GPU.
"""

M_PER_ROW = 8  # q_len 8 (num_speculative_tokens 7)
ROWS = (1, 6)
MS = tuple(r * M_PER_ROW for r in ROWS)  # (8, 48)
HIDDEN, LATENT, E, TOPK = 7168, 3584, 896, 16
SHARED_I = 2 * 3072 // 8  # 2 shared experts x 3072 / TP 8 = 768 per rank
GEMMS = {  # name: local (N, K) of the bf16 weight (TP8), as skinny_ladder.SITES
    "latent_down": (LATENT, HIDDEN),
    "gate_up": (2 * SHARED_I, HIDDEN),
    "shared_down": (HIDDEN, SHARED_I),
}
ROUTER = (E, HIDDEN)  # gate weight [896, 7168] bf16, bias [896] fp32
SITU_BETA, SITU_LINEAR_BETA = 4.0, 25.0  # K3 SiTU constants (moe_stage.selftest ALPHA, BETA)
ROUTE_SCALE = 1.0

# (phase, main-stream kernels, aux-stream kernels): descriptive (the record carries it); the selftest runs whole sides
# through the served functions, which fix the order.
PHASES = (
    ("front", ("RouterTopK",), ("latent_down",)),
    ("experts", ("MoePermute", "MoeGemm fc1", "MoeGemm fc2"), ("gate_up", "situ_and_mul", "shared_down")),
)
SCOPES = {"front": ("front",), "experts": ("experts",), "layer": ("front", "experts")}
# mode: (main side on, aux side on, placement). A side that is off returns a precomputed result and launches nothing.
# placement: "aux" = the served fork/join; "serial" = the served code with the aux stream off; "auxfirst" = the
# image's maybe_execute_in_parallel(launch_aux_first=True) on both forks (a candidate: not served today).
MODES = {
    "main": (True, False, "aux"),
    "aux": (False, True, "aux"),
    "both": (True, True, "aux"),
    "serial": (True, True, "serial"),
    "auxfirst": (True, True, "auxfirst"),
}
PLACEMENTS = ("aux", "serial", "auxfirst")


def gemm_backend(plan, m):
    """The backend the image's low-latency method runs for M rows: the plan entry's name, else F.linear (cuBLAS)."""
    entry = plan.get(m) if plan else None
    return entry[0] if entry else "cublas"


def summarize(us):
    """Derived numbers (us) from {mode: median us} of one (M, scope); None where an input is missing.
    exposed_aux = both - main (aux time the main stream did not hide), hidden_aux = aux - exposed_aux,
    overlap_gain = serial - both (> 0: the fork pays), auxfirst_delta = auxfirst - both (< 0: launch order helps)."""
    def d(a, b):
        return None if us.get(a) is None or us.get(b) is None else round(us[a] - us[b], 2)

    exposed = d("both", "main")
    return {
        "exposed_aux_us": exposed,
        "hidden_aux_us": None if exposed is None or us.get("aux") is None else round(us["aux"] - exposed, 2),
        "overlap_gain_us": d("serial", "both"),
        "auxfirst_delta_us": d("auxfirst", "both"),
    }
