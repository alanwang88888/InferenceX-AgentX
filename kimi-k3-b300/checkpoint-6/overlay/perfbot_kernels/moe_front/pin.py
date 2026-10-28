"""Item 17 R5 rung "pin": the served WsGemm with its register weight loads pinned above griddepcontrol.wait.

Served-signature entry: ``WsGemmPin`` takes the served ``WsGemm`` constructor arguments and ``compile_publish``
the served ``moe/front.py`` arguments, so the served path switches by one assignment:
``front.compile_publish = pin.compile_publish``. The kernel is ws_gemm.WsGemm(ready=True): it drains cp.async,
XORs the weight words and adds them (never: key = slot != 0x7FFFFFFF) to a global word before the wait, so ptxas
keeps the 14 weight LDGs above ACQBULK (scratch/r5/SASS.txt). Same math: bit-equal outputs (scratch/r5)."""
import functools

import cutlass as c
import cutlass.cute as cute
import torch
from cutlass import Int64, Uint32
from cutlass.cute.runtime import make_fake_stream

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common import primitives as P
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import fake
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe import front as F

from .ws_gemm import WsGemm


class WsGemmPin(WsGemm):
    def __init__(self, n, k, m, rt, kq, staged, split, valid, ks=1, tg=0, ch=0):
        super().__init__(n, k, m, rt, kq, staged, split, valid, ks, tg, ch, ready=True)


@functools.cache
def compile_publish(n, split, valid, batch, device, ks, geometry=None):
    """moe/front.py compile_publish with WsGemmPin (same arguments, same compiled call shape)."""
    rt, kq, staged, tg, ch = geometry or F.front_geometry(batch, n, ks)
    kernel = WsGemmPin(n, 7168, batch, rt, kq, staged, split, valid, ks, tg, ch)
    args: tuple = (fake((batch, 7168)), fake((n, 7168)), fake((1, 64, F.WIDTH)))
    args += (c.Int64(0), c.Int32(0), c.Int32(0), fake((batch, 896), None, c.Float32))
    with torch.accelerator.device_index(device):
        return cute.compile(kernel, *args, make_fake_stream())


@P.asm()
def unicast_words(address, words, *, kind="f32"):
    """Single-GPU tests only (item 08 corun.unicast_words): multicast_words' operands as a plain relaxed store;
    multimem needs >= 2 GPUs. Patch ``gemm.ws_gemm.multicast_words`` before compiling."""
    vector, registers = P._vector(len(words), 1)
    f32 = kind == "f32"
    values = (P.bits_float(v) if f32 else Uint32(v) for v in words)
    constraints = "l" + (",f" if f32 else ",r") * len(words) + ",~{memory}"
    code = f"st.relaxed.gpu.global{vector}.{'f32' if f32 else 'b32'} [$0],{registers};"
    return code, constraints, (Int64(address), *values)
