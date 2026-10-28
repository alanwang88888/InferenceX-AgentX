"""PERFBOT-KDA-CKPT: evidence log for P-07 risk c (GPU stream order around the KDA prefill checkpoint export).

Switch: PERFBOT_EXP_KDA_CKPT_LOG=1. The overlay checkpoint.py reads it once, at import, and imports this module only
then. Only the value "1" is on. With the switch off, nothing of this module runs or loads. The log works with
PERFBOT_EXP_KDA_CKPT on or off: the base path exports too, so the CKPT-off arm is the control for the log itself.

Risk c. In a step with a prefill checkpoint, the GPU runs in stream order: the copy-on-write copy of the initial block
(model_runner.update_requests), the Triton state pre-copy (MambaHybridModelState.preprocess_state), the FlashKDA
forward that writes the state at offset T into a workspace row, and the export (MambaPrefillCheckpointExporter.export
-> store_cache_checkpoints_kernel) that copies that row into the checkpoint block. A wrong order puts a wrong state
into the block, or a later write overwrites it.

The check, per exported row (state index != NULL_BLOCK_ID and offset > 0, the kernel's own condition):
1. on_export (checkpoint.py, right after the export launch, on the current stream): the digest of the SOURCE that the
   kernel reads: the workspace row (recurrent state), and the raw qkv window (conv state, only when offset >= the
   state length, as in the kernel; else "-").
2. note_step (mamba_hybrid.py, first statement of preprocess_state of the next real step): the digest of the
   DESTINATION block, addressed as the kernel addresses it.
3. Equal = no fault (eq=1). Not equal = a risk c fault (eq=0).
Read point. note_step runs before the next step's pre-copy, KV-connector load and forward. It runs after that step's
update_requests, which zeroes newly allocated blocks and applies copy-on-write copies into them. Those writes go only
to blocks of rows in that batch. So a row is "unchecked" (eq=u, counted apart, never a mismatch) when its request is
not in the next batch (freed), or its batch row no longer holds the block at the checkpoint column, or any other row
of the batch holds the block id (reassigned or shared), or a second export of the watched layer came first.

Digest: the bytes as uint8, widened to int64, times the weight 1 + (byte position mod 65521), summed. Equal bytes give
equal digests. The sum is below 255 * 65521 * row bytes, so it does not overflow int64 for rows below 5e11 bytes.
One changed byte changes the sum by d * w, with 0 < |d| < 256 and 0 < w < 65522, so it never gives the same digest.

Scope: only the first KDA layer that exports (the first recurrent_state tensor seen). One layer is enough for order:
every KDA layer runs the same launch sequence on the same stream. Capture: KDA _forward is
@eager_break_during_capture (models/kimi_k3/nvidia/kda.py:904); in a piecewise graph it runs eagerly at replay, so
the hooks run there. During a capture (a BreakableCUDAGraphCapture is active, or the stream is capturing) the hooks do
nothing. preprocess_state does not run in dummy or capture runs (model_runner.py:1833 is on the real-batch branch).

Cost per export of the watched layer: one device-wide torch.cuda.synchronize before each digest (coordinator ruling:
a FlashKDA on a side stream cannot race the source digest), then one host read (.tolist), at the export and at the next
step's read-back,
plus GPU work of about 8 x (row bytes) per row (the int64 widening). Other layers: one data_ptr compare.

Lines (engine stdout, flushed):
  PERFBOT-KDA-CKPT export=<n> req=<request id|-> row=<batch row> block=<id> src=<hex16> dst=<hex16>
    csrc=<hex16|-> cdst=<hex16|-> eq=<0|1|u>
  PERFBOT-KDA-CKPT-SUM exports=<n> checked=<n> mismatch=<n> unchecked=<n>   (each time exports passes 100 more)
Any error prints "PERFBOT-KDA-CKPT disabled: <error>" once and turns the log off. It never reaches the engine.
"""

import torch

ENV = "PERFBOT_EXP_KDA_CKPT_LOG"
TAG = "PERFBOT-KDA-CKPT"
EVERY = 100
_MOD = 65521


def digests(rows: torch.Tensor) -> torch.Tensor:
    """(n, ...) tensor -> (n,) int64 digests of each row's bytes, on the row's device (see the module docstring)."""
    n = rows.shape[0]
    b = rows.contiguous().reshape(n, -1).view(torch.uint8).to(torch.int64)
    w = torch.arange(b.shape[1], dtype=torch.int64, device=b.device) % _MOD + 1
    return (b * w).sum(dim=1)


def _hex(v) -> str:
    return "-" if v is None else f"{v & 0xFFFFFFFFFFFFFFFF:016x}"


def _capturing() -> bool:
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

    if BreakableCUDAGraphCapture.is_active():
        return True
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


class Log:
    def __init__(self):
        self.on = True
        self.layer = None  # data_ptr of the watched layer's recurrent_state
        self.dst = None  # (recurrent_state, conv_state, width, state_len, recurrent row size) of that layer
        self.req_ids = None  # request ids of the step whose preprocess_state ran last, in batch order
        self.pending = []  # exported rows that wait for the next step's read-back
        self.exports = self.checked = self.mismatch = self.unchecked = self.summed = 0

    def off(self, e):
        self.on, self.pending = False, []
        print(f"{TAG} disabled: {type(e).__name__}: {e}", flush=True)

    def _print(self, p, dst, cdst, eq):
        print(f"{TAG} export={p['no']} req={p['req']} row={p['row']} block={p['block']} src={_hex(p['src'])} "
              f"dst={_hex(dst)} csrc={_hex(p['csrc'])} cdst={_hex(cdst)} eq={eq}", flush=True)

    def _sum(self):
        while self.exports >= self.summed + EVERY:
            self.summed += EVERY
            print(f"{TAG}-SUM exports={self.exports} checked={self.checked} mismatch={self.mismatch} "
                  f"unchecked={self.unchecked}", flush=True)

    def on_export(self, ck, raw_qkv, conv_state, rec_ckpt, rec_state, cu_seqlens, rec_rows, state_len, null_id):
        """Source digests of one export launch (checkpoint.py, after store_cache_checkpoints_kernel)."""
        if not self.on:
            return
        try:
            if _capturing():
                return
            if self.layer is None:
                self.layer = rec_state.data_ptr()
            if rec_state.data_ptr() != self.layer:
                return
            n = ck.checkpoint_offsets.numel()
            if n == 0:
                return
            dev = rec_state.device
            width, size = raw_qkv.shape[-1], rec_ckpt[0].numel()
            off = ck.checkpoint_offsets.to(torch.int64)
            sidx = ck.state_indices.to(torch.int64)
            rows = rec_rows.to(torch.int64) if rec_rows is not None else torch.arange(n, device=dev)
            if rec_state.is_cuda:  # device-wide: a side-stream FlashKDA has finished before the source digest reads
                torch.cuda.synchronize(dev)
            # The kernel reads rec_ckpt + row * stride(0) + [0, size) and the window x[cu + off - L : cu + off].
            src = digests(rec_ckpt.as_strided((rec_ckpt.shape[0], size), (rec_ckpt.stride(0), 1))[rows])
            tok = cu_seqlens[:n].to(torch.int64)[:, None] + off[:, None] - state_len
            tok = (tok + torch.arange(state_len, device=dev)).clamp(0, raw_qkv.shape[0] - 1)
            csrc = digests(raw_qkv[tok].transpose(1, 2).to(conv_state.dtype))
            req_rows = ck.request_rows if ck.request_rows is not None else torch.arange(n, device=dev)
            cols = ck.block_cols if ck.block_cols is not None else torch.full((n,), -1, device=dev)
            meta = torch.stack([sidx, off, src, csrc, req_rows.to(torch.int64), cols.to(torch.int64)]).tolist()
            for p in self.pending:  # no read-back since the last export of this layer
                self.unchecked += 1
                self._print(p, None, None, "u")
            self.pending = []
            self.dst = (rec_state, conv_state, width, state_len, size)
            for block, o, s, c, row, col in zip(*meta):
                if block == null_id or o <= 0:
                    continue
                self.exports += 1
                req = self.req_ids[row] if self.req_ids is not None and row < len(self.req_ids) else "-"
                self.pending.append(dict(no=self.exports, req=req, row=row, block=block, col=col, src=s,
                                         csrc=c if o >= state_len else None))
        except Exception as e:  # noqa: BLE001 - a log must never stop the engine
            self.off(e)

    def note_step(self, input_batch, block_tables):
        """Read back the last export's blocks, then keep this step's request ids (mamba_hybrid.preprocess_state)."""
        if not self.on:
            return
        try:
            if self.pending:
                self._readback(input_batch, block_tables)
            self.req_ids = list(input_batch.req_ids[: input_batch.num_reqs])
        except Exception as e:  # noqa: BLE001
            self.off(e)

    def _readback(self, input_batch, block_tables):
        pending, self.pending = self.pending, []
        rec_state, conv_state, width, state_len, size = self.dst
        dev, k, nr = rec_state.device, len(pending), input_batch.num_reqs
        blocks = torch.tensor([p["block"] for p in pending], dtype=torch.int64, device=dev)
        if rec_state.is_cuda:  # device-wide: every stream's writes since the export have finished
            torch.cuda.synchronize(dev)
        dst = digests(rec_state.as_strided((rec_state.shape[0], size), (rec_state.stride(0), 1))[blocks])
        cdst = digests(conv_state[blocks][:, :width, :state_len].contiguous())
        where = {rid: i for i, rid in enumerate(input_batch.req_ids[:nr])}
        own = [where.get(p["req"], -1) for p in pending]
        rr = torch.tensor([max(r, 0) for r in own], dtype=torch.int64, device=dev)
        holds = torch.zeros((k, nr), dtype=torch.int64, device=dev)  # per pending row: block id hits per batch row
        cells = torch.full((k,), -1, dtype=torch.int64, device=dev)  # the own row's cell at the checkpoint column
        for bt in block_tables:
            bt = bt[:nr].to(torch.int64)
            holds += (bt.unsqueeze(0) == blocks.view(k, 1, 1)).sum(dim=2)
            cc = torch.tensor([p["col"] if 0 <= p["col"] < bt.shape[1] else 0 for p in pending], device=dev)
            hit = (bt[rr, cc] == blocks) & torch.tensor([0 <= p["col"] < bt.shape[1] for p in pending], device=dev)
            cells = torch.where(hit, blocks, cells)
        flat = torch.cat([dst, cdst, cells, holds.flatten()]).tolist()
        for i, p in enumerate(pending):
            d, cd, cell = flat[i], flat[k + i], flat[2 * k + i]
            hold = flat[3 * k + i * nr: 3 * k + (i + 1) * nr]
            others = sum(h for r, h in enumerate(hold) if r != own[i])
            cd = cd if p["csrc"] is not None else None
            if own[i] < 0 or cell != p["block"] or others:
                self.unchecked += 1
                self._print(p, d, cd, "u")
                continue
            eq = d == p["src"] and cd == p["csrc"]
            self.checked += 1
            self.mismatch += not eq
            self._print(p, d, cd, int(eq))
        self._sum()


LOG = Log()
on_export = LOG.on_export
note_step = LOG.note_step
