"""Server adapter: one KimiK3DeltaAttention layer + its KDA metadata -> core()'s (lw, st) API.

The overlay kda.py builds one ServedCore per layer when PERFBOT_EXP_KDA is on and calls run(proj, out) from
forward(), after in_proj and before f_b_proj. run() returns None after it launched core() (the layer then skips
f_b_proj, the conv update, the recurrence and o_norm), or the reason it declined (the layer runs the image path).
Each decline class is logged once.

Served only: a pure spec-decode batch (no prefill, no plain decode) without RecoverSSM. In the FULL CUDA graph the
metadata is the padded batch (padded rows: spec_idx 0 = the null block, accepted 1), so n = spec_idx rows and
T = n * 8 = the captured token count; core() writes zeros for a null request.

Compile and the weight checks run on the first eager forward (the profile run, after weight loading); a first call
inside CUDA-graph capture declines, so the server never compiles under capture.

PERFBOT_EXP_KDA_CORE selects the served core variant: "full" (default) or "landed" (the previous served kernel,
for a same-tree e2e control).
"""

import os
from types import SimpleNamespace

import torch

from . import core as k

_SHAPES = {"w_fb": ((k.P, k.D), torch.bfloat16), "w_conv": ((3 * k.P, k.WIDTH), torch.float32),
           "a_log": ((k.H,), torch.float32), "dt_bias": ((k.P,), torch.float32), "w_norm": ((k.D,), torch.float32)}


class ServedCore:
    def __init__(self, layer, logger):
        self.layer = layer
        self.log = logger
        self.lw = None
        self.fatal = None
        self.logged = set()
        self.variant = os.environ.get("PERFBOT_EXP_KDA_CORE", "full")
        if self.variant not in ("full", "landed"):
            raise ValueError(f"PERFBOT_EXP_KDA_CORE={self.variant!r}: expected full or landed")

    def _decline(self, cls: str, detail: str = "") -> str:
        if cls not in self.logged:
            self.logged.add(cls)
            self.log.info("PERFBOT_EXP_KDA %s: image path (%s) %s", self.layer.prefix, cls, detail)
        return cls

    def _warm(self) -> None:
        L = self.layer
        w = L.conv1d.weight
        lw = SimpleNamespace(w_fb=L.f_b_proj.weight, w_conv=w.view(w.size(0), w.size(-1)), a_log=L.A_log,
                             dt_bias=L.dt_bias, w_norm=L.perfbot_norm_weight, lower_bound=L.gate_lower_bound)
        bad = [f"{n} {tuple(getattr(lw, n).shape)} {getattr(lw, n).dtype}" for n, (shape, dt) in _SHAPES.items()
               if tuple(getattr(lw, n).shape) != shape or getattr(lw, n).dtype != dt
               or not getattr(lw, n).is_contiguous()]
        if lw.lower_bound is None:
            bad.append("gate_lower_bound None")
        if not bad and not torch.equal(lw.w_norm, L.o_norm.weight.float()):
            bad.append("w_norm copy != o_norm.weight")
        if bad:
            self.fatal = "; ".join(bad)
            self.log.warning("PERFBOT_EXP_KDA %s: disabled: %s", L.prefix, self.fatal)
            return
        k.compiled(lw.w_fb.device, lw.lower_bound, variant=self.variant)
        self.lw = lw

    def run(self, proj: torch.Tensor, out: torch.Tensor, attn_metadata) -> str | None:
        L = self.layer
        capturing = torch.cuda.is_current_stream_capturing()
        if self.lw is None and self.fatal is None and not capturing:
            self._warm()
        if self.fatal is not None:
            return self._decline("disabled", self.fatal)
        if self.lw is None:
            return self._decline("not warmed before capture")
        m = attn_metadata.get(L.prefix) if isinstance(attn_metadata, dict) else None
        if m is None:
            return "no metadata"
        if L.use_recoverssm:
            return self._decline("recoverssm")
        if m.num_prefills or m.num_decodes or not m.num_spec_decodes:
            return self._decline("not pure spec decode")
        si = m.spec_state_indices_tensor
        st = SimpleNamespace(n=si.shape[0], t=proj.shape[0], conv=L.kv_cache[0], ssm=L.kv_cache[1], spec_idx=si,
                             accepted=m.num_accepted_tokens)
        reason = k.check(self.lw, st, proj, out)
        if reason is not None:
            return self._decline("check " + reason.split()[0], reason)
        k.core(self.lw, st, proj, out, variant=self.variant)
        return None
