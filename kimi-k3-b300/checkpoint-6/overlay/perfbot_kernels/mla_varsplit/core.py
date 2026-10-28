"""01d: the tokenspeed FP8 split kernel with the unit table and the empty-unit exit (P-14 L6, letter O).

core_fp8.py is GENERATED: render() applied to the image's tokenspeed_mla/mla_decode_fp8.py (BASE_SHA256, vLLM
49cb2bbf image; byte-equal to lane-n-p08 kernel/incumbent). Every change is an added line tagged TAG; no base line
is edited or removed (test_cpu.py regenerates and checks both). Regenerate:

  python -I overlay/perfbot_kernels/mla_varsplit/core.py <base mla_decode_fp8.py>

Unit table (kernel.py writes it after the vector): block_split_kvs[B + u] for the grid unit u = z * B + b (one CTA
pair x 6 M tiles) names the (request, split) unit it runs, encoded the same way, plus EMPTY (bit 16) when z >= the
request's split. The table lists the live units of the requests in descending len / split (the longest CTAs dispatch
first), then the empty units. tokenspeed compiles block_split_kvs with the batch's shape, so the kernel gets the [:B]
view and reads the table past it.

Empty-unit exit: the CTA of an EMPTY unit executes PTX exit after its local mbarrier init and before the cluster
arrive, TMEM alloc and griddepcontrol_wait. Its table load is issued at the kernel top, so the check costs live CTAs
no load latency (one load, no dependent cache_seqs read: an exit check at the top with two dependent loads cost the
live CTAs 2-4 % at one-wave batches [m, B200, order_ab]). Bit-equal: z >= split_b gives k_index >= k_tile_total, so in
the base kernel all four warp loops of such a CTA skip work (k_tile_count > 0 guards; write_empty_output needs mAccO
None, and the exit needs mAccO), it writes no global memory, and the reducer reads only splits below split_b
(mla_decode_fp8.py:1859-1876). PDL rule: z = 0 is below every split, so z = 0 units never exit and every grid keeps
live CTAs that pass griddepcontrol_wait; the grid completes only after its predecessor completed, as before. The
table is VarSplit's store, read before the wait like the base kernel's MMA and compute warps read block_split_kvs.
Both CTAs of a cluster share (s, b, z) (x is not remapped), so they exit together; neither has arrived on a
cross-CTA barrier.
"""

import hashlib
import sys
from pathlib import Path

TAG = "# PERFBOT P-14 01d"
BASE_SHA256 = "4a8e10557f8c5aa49a6ef1f7ac560406ffcf642fa01c88509b4094f194ad10ff"
HERE = Path(__file__).resolve().parent
OUT = HERE / "core_fp8.py"

EMPTY = 1 << 16  # __init__.EMPTY
ENTRY = """\
        if cutlass.const_expr(self.is_var_split_kv and mAccO is not None):
            perfbot_entry = self.perfbot_entry(block_split_kvs, cache_seqs)
"""
EXIT = f"""\
        if cutlass.const_expr(self.is_var_split_kv and mAccO is not None):
            if perfbot_entry >= {EMPTY}:
                nvvm.exit()
"""
UNIT = """\
                blk_coord = self.perfbot_unit(blk_coord, block_split_kvs, cache_seqs)
"""
METHODS = f"""\
    @cute.jit
    def perfbot_unit(self, blk_coord, block_split_kvs, cache_seqs):
        # The unit this CTA runs: table entry B + z * B + b (perfbot_kernels/mla_varsplit/core.py).
        if cutlass.const_expr(self.is_var_split_kv):
            batch = cache_seqs.shape[0]
            unit = block_split_kvs[batch + blk_coord[3] * batch + blk_coord[2]] % {EMPTY}
            blk_coord = (blk_coord[0], blk_coord[1], unit % batch, unit // batch)
        return blk_coord

    @cute.jit
    def perfbot_entry(self, block_split_kvs, cache_seqs):
        # This CTA's table entry (grid unit z * B + b); >= EMPTY: the unit has no k tile (core.py: the exit).
        _, by, bz = cute.arch.block_idx()
        batch = cache_seqs.shape[0]
        return block_split_kvs[batch + bz * batch + by % batch]

"""
# (anchor line, its count in the base, lines inserted, after (True) or before the anchor)
PATCHES = (
    ("        is_leader_cta = mma_tile_coord_v == 0\n", 1, ENTRY, True),
    ("        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mnk, is_relaxed=True)\n", 1, EXIT, False),
    ("                blk_coord = work_tile.tile_idx\n", 5, UNIT, True),
    ("    @cute.jit\n    def get_k_tile_count(\n", 1, METHODS, False),
)


def tagged(block):
    return "".join(f"{line}  {TAG}\n" if line.strip() else "\n" for line in block.splitlines())


def render(base: str) -> str:
    assert hashlib.sha256(base.encode()).hexdigest() == BASE_SHA256, "not the image's mla_decode_fp8.py"
    for anchor, count, block, after in PATCHES:
        assert base.count(anchor) == count, (anchor, base.count(anchor))
        base = base.replace(anchor, anchor + tagged(block) if after else tagged(block) + anchor)
    return base


def kernel_class():
    from . import core_fp8

    return core_fp8.BlackwellMultiHeadLatentAttentionForwardFP8


if __name__ == "__main__":
    OUT.write_text(render(Path(sys.argv[1]).read_text()))
