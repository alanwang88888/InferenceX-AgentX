"""Item 17 R0 fork of the served RouteSitu (vLLM 49cb2bbf moe/front.py) with optional %globaltimer stamps.

``kernel``, ``route`` and ``copy`` are copies of the served code; ``select``, ``publish`` and ``expert`` are
inherited. ``stamps`` (const_expr): off = the served kernel; on = one extra Int64 argument (the stamp buffer
address, last) and the stamps of ``RS_SLOTS``: %clock64 at ``buf[cta * RS_NS + slot]`` plus one (globaltimer,
clock64) edge at ``buf[cta * RS_NS + RS_EDGE]`` (stamps.py). Route CTAs are
[0, batch), copy CTAs [batch, 2 batch)."""
import cutlass as c
import cutlass.cute as cute
from cutlass import Int32, Uint32, Uint64

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.mailbox import poll, rearm
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.primitives import (
    alloc,
    bits_float,
    div_rn,
    redux,
    sigmoid,
    store_words,
    view,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe.experts import plan_chunk
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe.front import RouteSitu as ServedRouteSitu
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe.front import ordered
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe.shared import situ

from .stamps import edge, stamp

# entry; poll = the CTA's polled cells all seen (last warp; route: logits, copy: latent); keys = sigmoid +
# bias keys in smem (route, tid 0 after the barrier); cutoff = 16th group maximum known (route); top16 =
# candidates ranked (route); out = route: ids + weights written (warp 0), copy: latent + plan written (last
# warp); release = before launch_dependents (last warp); exit = after it (last warp).
RS_SLOTS = ("entry", "poll", "keys", "cutoff", "top16", "out", "release", "exit")
RS_EDGE = len(RS_SLOTS)  # (globaltimer, clock64) of one edge at the CTA's entry
RS_NS = RS_EDGE + 2
_S = {n: i for i, n in enumerate(RS_SLOTS)}


class RouteSitu(ServedRouteSitu):
    def __init__(self, *a, stamps=False, **kw):
        super().__init__(*a, **kw)
        self.stamps = stamps
        if stamps and not (self.lean and self.polled):
            raise ValueError("RouteSitu stamps: lean polled path only (batch > 8)")

    @cute.jit
    def mark(self, sb, cta, name: c.Constexpr, when):
        if c.const_expr(self.stamps):
            if when:
                stamp(sb, cta * RS_NS + _S[name])

    @cute.kernel
    def kernel(self, args):
        logits, bias, ids, weights, mailbox, latent, gate_up, hidden = args[0:8]
        scale, beta, linear_beta = args[8:11]
        plan = args[11] if self.plan else None
        sb = args[11 + int(self.plan)] if self.stamps else None
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        self.mark(sb, cta, "entry", tid == 0)
        if c.const_expr(self.stamps):
            if tid == 0:
                edge(sb, cta * RS_NS + RS_EDGE)
        token, role = cta % self.batch, cta // self.batch
        if c.const_expr(self.early):
            cute.arch.griddepcontrol_launch_dependents()
        for r in c.range_constexpr(len(self.roles)):
            if role == r:
                if c.const_expr(self.roles[r] == "route"):
                    self.route(logits, bias, ids, weights, scale, token, tid, mailbox, sb, cta)
                elif c.const_expr(self.roles[r] == "copy"):
                    # Lamport: no dependency wait; the sentinel poll is the barrier.
                    base = mailbox.iterator + token * mailbox.shape[2] + tid * 8
                    if tid < 192:
                        self.copy(base, latent, token, tid, 2, plan, sb, cta)
                    else:
                        self.copy(base, latent, token, tid, 1, plan, sb, cta)
                else:
                    cute.arch.griddepcontrol_wait()
                    rounds = (self.shared + 255) // 256  # issue every load first
                    pair = cute.make_rmem_tensor((rounds, 2), c.Float32)
                    for i in c.range_constexpr(rounds):
                        if tid + 256 * i < self.shared:
                            pair[i, 0] = gate_up[token, tid + 256 * i]
                            pair[i, 1] = gate_up[token, self.shared + tid + 256 * i]
                    for i in c.range_constexpr(rounds):
                        if tid + 256 * i < self.shared:
                            gate = c.Float32(c.BFloat16(pair[i, 0]))
                            up = c.Float32(c.BFloat16(pair[i, 1]))
                            value = situ(gate, up, beta, linear_beta)
                            hidden[token, tid + 256 * i] = c.BFloat16(value)
        # Late release: the persistent experts must not take SMs from the down GEMM.
        self.mark(sb, cta, "release", tid % 32 == 0)
        if c.const_expr(not self.early):
            cute.arch.griddepcontrol_launch_dependents()
        self.mark(sb, cta, "exit", tid % 32 == 0)

    @cute.jit
    def copy(self, base, latent, token, tid, count: c.Constexpr[int], plan=None, sb=None, cta=None):
        got = poll(base, 2048, count)
        if c.const_expr(self.stamps):
            cute.arch.sync_warp()
            self.mark(sb, cta, "poll", tid % 32 == 0)
        for i in c.range_constexpr(count):
            dst = latent.iterator + token * latent.stride[0] + (tid + 256 * i) * 8
            words = [got[i, k] for k in range(4)]
            store_words(dst, words)
            rearm(base + i * 2048)
            if c.const_expr(plan is not None):
                plan_chunk(plan, token, tid + 256 * i, words)
        self.mark(sb, cta, "out", tid % 32 == 0)

    @cute.jit
    def route(self, logits, bias, ids, weights, scale, token, tid, mailbox, sb=None, cta=None):
        """Served ``route`` plus stamps (lean polled path)."""
        warp, lane = tid // 32, tid % 32
        smem = c.utils.SmemAllocator()
        km = alloc(smem, Uint32, (8, 32))
        cand, dense = (alloc(smem, Uint64, shape) for shape in ((8, 128), 64))
        sig = alloc(smem, c.Float32, 1024)
        counts = alloc(smem, Int32, 8)
        cutoff = alloc(smem, Uint32, 1)
        chosen = alloc(smem, Int32, 16)
        if c.const_expr(self.lean):  # candidate count; the chosen experts' sigmoids
            found, picked = alloc(smem, Int32, 1), alloc(smem, c.Float32, 16)
        # Thread t owns experts t + 256 i, or 4 t + i when polling 16-byte chunks.
        offsets = cute.make_rmem_tensor(4, c.Float32)
        for i in c.range_constexpr(4):  # bias is a weight: load before the wait
            offsets[i] = c.Float32(0)
            if self.expert(tid, i) < self.experts:
                offsets[i] = bias[self.expert(tid, i)]
        raw = cute.make_rmem_tensor(4, c.Float32)
        key, top = cute.make_rmem_tensor(4, Uint32), Uint32(0)
        for i in c.range_constexpr(4):
            key[i] = Uint32(0)
        if c.const_expr(self.polled):
            if tid < 224:
                cell = mailbox.iterator + token * mailbox.shape[2] + 3584 + tid * 8
                got = poll(cell, 0, 1)
                if c.const_expr(self.stamps):
                    cute.arch.sync_warp()
                    self.mark(sb, cta, "poll", lane == 0)
                rearm(cell)
                for i in c.range_constexpr(4):
                    raw[i] = bits_float(got[0, i])
                if c.const_expr(self.lean):  # all four experts: one block, no branches
                    scores = [sigmoid(raw[i]) for i in range(4)]
                    for i in c.range_constexpr(4):
                        sig[4 * tid + i] = scores[i]
                        key[i] = ordered(scores[i] + offsets[i])
            if c.const_expr(not self.lean):
                cute.arch.griddepcontrol_wait()
        else:
            cute.arch.griddepcontrol_wait()
            for i in c.range_constexpr(4):
                if self.expert(tid, i) < self.experts:
                    raw[i] = logits[token, self.expert(tid, i)]
        for i in c.range_constexpr(4):
            expert = self.expert(tid, i)
            if c.const_expr(not (self.lean and self.polled)):  # noqa: SIM102
                if expert < self.experts:
                    score = sigmoid(raw[i])
                    sig[expert] = score
                    key[i] = ordered(score + offsets[i])
            top = max(top, key[i])
        km[warp, lane] = top
        if c.const_expr(self.lean):  # noqa: SIM102
            if tid == 0:
                found[0] = Int32(0)
        cute.arch.sync_threads()
        self.mark(sb, cta, "keys", tid == 0)
        if warp == 0:
            group = km[0, lane]
            for w in c.range_constexpr(1, 8):
                group = max(group, km[w, lane])
            km[0, lane] = group
            cute.arch.sync_warp()
            rank = Int32(0)
            for j in c.range_constexpr(32):
                rank += Int32(km[0, j] > group)
            low = redux(group if rank < 16 else Uint32(0xFFFFFFFF))
            if lane == 0:
                cutoff[0] = low
        cute.arch.sync_threads()
        self.mark(sb, cta, "cutoff", tid == 0)
        if c.const_expr(self.lean):
            dense = view(cand.iterator, 0, Uint64, (1024,))
            self.select(key, cutoff[0], (dense, sig, chosen, picked, found))
            self.mark(sb, cta, "top16", tid == 0)
            self.publish(chosen, picked, ids, weights, scale, token)
            self.mark(sb, cta, "out", tid == 0)
            return
        threshold, offset = cutoff[0], Int32(0)
        for i in c.range_constexpr(4):
            expert = self.expert(tid, i)
            hit = (expert < self.experts) & (key[i] >= threshold)
            ballot = cute.arch.vote_ballot_sync(hit)
            if hit:
                below = cute.arch.popc(ballot & ((Uint32(1) << lane) - Uint32(1)))
                packed = (Uint64(key[i]) << 32) | Uint64(~Uint32(expert))
                cand[warp, offset + below] = packed
            offset += cute.arch.popc(ballot)
        if lane == 0:
            counts[warp] = offset
        cute.arch.sync_threads()
        if warp == 0:
            n = Int32(0)
            for v in c.range_constexpr(8):
                n += counts[v]
            for half in c.range_constexpr(2):  # compact the regions densely
                index = lane + 32 * half
                rest, region = Int32(index), Int32(0)
                for v in c.range_constexpr(8):
                    if (region == v) & (rest >= counts[v]):
                        rest -= counts[v]
                        region += 1
                value = Uint64(0)
                if index < n:
                    value = cand[region, rest]
                dense[index] = value
            cute.arch.sync_warp()
            if n <= 64:
                for half in c.range_constexpr(2):
                    mine, rank = dense[lane + 32 * half], Int32(0)
                    for j in c.range_constexpr(64):
                        rank += Int32(dense[j] > mine)
                    if (lane + 32 * half < n) & (rank < 16):
                        chosen[rank] = Int32(~Uint32(mine))
            else:  # rare: rank every candidate against all regions
                for q in range(lane, n, 32):
                    rest, region = Int32(q), Int32(0)
                    for v in c.range_constexpr(8):
                        if (region == v) & (rest >= counts[v]):
                            rest -= counts[v]
                            region += 1
                    mine, rank = cand[region, rest], Int32(0)
                    for v in c.range_constexpr(8):
                        for j in range(counts[v]):
                            rank += Int32(cand[v, j] > mine)
                    if rank < 16:
                        chosen[rank] = Int32(~Uint32(mine))
            cute.arch.sync_warp()
            expert, value = chosen[lane % 16], c.Float32(0)
            if lane < 16:
                ids[token, lane] = expert
                value = sig[expert]
            total = value
            for shift in c.range_constexpr(4):
                total += cute.arch.shuffle_sync_bfly(total, 1 << shift)
            if lane < 16:
                value = value * scale
                if c.const_expr(self.renorm):
                    value = div_rn(value, total + c.Float32(1e-20))
                weights[token, lane] = c.BFloat16(value)
