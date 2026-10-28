"""K 7 -> 5 (roadmap v2 row 1): the one source of every landed constant that was keyed on K = 7 / q = 8.

Every K-keyed site of land-8 (item 01 precards/K5-Q8-CONSTANTS.md, 11 items) imports its value from here, so no site
keeps a q = 8 literal and falls back silently. Rows are verify rows per step = Q x requests (specialized decode `batch`
is rows, runtime/providers.py:25); draft M = K x requests. Tags: [c] read off code, [e] derived here.

consts(K) is the generator; this module serves consts(5). consts(7) returns land-8's literals exactly (test_cpu), so
the K = 7 reference arm of the proof restores them from the same formulas.
"""


def consts(K):
    Q = K + 1  # verify rows per request = decode_query_len = 1 + num_speculative_blocks

    def rows(reqs):
        return Q * reqs

    # q x 96 heads packs into ceil(96 Q / 128) M128 tiles: 5 at Q 6, 6 at Q 8; one L6 unit = one CTA pair per tile;
    # UNITS = units in one wave of 148 SMs: 14 at Q 6, 12 at Q 8. tokenspeed's planner serves splits UNITS // reqs at
    # B1-B5: 14 / 7 / 4 / 3 / 2 at Q 6 (CTAs 140 / 140 / 120 / 120 / 100), 12 / 6 / 4 / 3 / 2 at Q 8 [m, tokenspeed_mla
    # 2.0.2 _get_split_kv_and_workspace_size run in the DSL 4.8.0 env, item 01 planner probe]. Its input floor
    # 148 // (2 B Q) is 12 / 6 / 4 / 3 / 2 at Q 6: a floor, not the split (item 02 review s4 read the floor).
    tiles = -(-96 * Q // 128)
    units = 148 // (2 * tiles)
    return {
        "K": K, "Q": Q,
        # 1 mla_core fork (item 03 E2b + MLA-LOOP): requests 1-5 (MAX_REQS counts requests). K7: 8 <= rows <= 40.
        "MLACORE_ROWS": (rows(1), rows(5)),
        # 2-3 L6 var split: B5 at K5 two waves (2 units, units + 1 - 5) [m i01k5r5: 28 beats 14 on eq241 and skew341
        # over 4 ranks, -0.110 ms/step; coordinator 10:56Z]; at K7 one wave (units, ...) [c land-8, B5 unmeasured
        # there]. B4 / B3 two-wave units scaled from Q 8's 15 / 16 of 12 [e; HOLD as served, i01k5r5]; cap 2 x served.
        # K7: UNITS 12, {5: (12, 8), 4: (15, 6), 3: (16, 8)}.
        "TILES": tiles, "UNITS": units,
        "RULES": {5: ((2 if K == 5 else 1) * units, units + 1 - 5), 4: (round(15 * units / 12), 2 * (units // 4)),
                  3: (16 if K == 5 else round(16 * units / 12), 2 * (units // 3))},  # K5 B3 16 [m i01k5r5c, 4/4 ranks]
        # 4-5 unit E prologue: num_speculative_steps == K, decode_query_len == Q, spec state blocks per request == Q.
        "SPEC_BLOCKS": Q,
        # 6 08L lean experts at 3 / 4 / 5 requests; drain at 5 requests. K7: {24, 32, 40}, {40: 8}. K5 adds 2 requests
        # [m i01k5b2: lean fork -2.18 us at row 12 once S = 8, bit-equal to the image; coordinator 12:57Z].
        "LEAN_ROWS": frozenset({rows(3), rows(4), rows(5)} | ({rows(2)} if K == 5 else set())), "DRAIN": {rows(5): 8},
        # 7 pin (item 17) at 2-5 requests. K7: (16, 24, 32, 40).
        "PIN_ROWS": tuple(rows(r) for r in range(2, 6)),
        # 8 S SharedDown SMs: 3 requests 5 SMs, 5 requests 4 SMs. K7: ",17-24:5,33-40:4". K5: measured per row, the
        # smallest S within 0.3 us of the best [m i01k5s2b, i01k5b2: SharedDown bit-equal to the served S, REF64 v4]:
        # 2 / 3 / 4 / 5 requests 8 / 6 / 5 / 5 SMs; 9 = SPLIT's first split batch (front.py "9-12:6"). Coordinator 12:57Z.
        "S_TAIL": (f",9-{rows(2)}:8,{rows(2) + 1}-{rows(3)}:6,{rows(3) + 1}-{rows(4)}:5,{rows(4) + 1}-{rows(5)}:5" if K == 5
                   else f",{rows(2) + 1}-{rows(3)}:5,{rows(4) + 1}-{rows(5)}:4"),
        # 9 X2 join off at 2-5 requests. K7: ",9-40:0".
        "X2_TAIL": f",{rows(1) + 1}-{rows(5)}:0",
        # 10 draft chain kernel 1: one-shot at padded M <= 16. The crossover is in M (AR payload rows), measured
        # 14 -> 21 at K7; K5 draft M 15 and 20 sit inside that unmeasured band: kept at 16 [e, M-keyed].
        "SERVED_MAX_M": 16,
        # 11 CUDA graph capture sizes: draft M (K r) and verify rows (Q r) for r 1..8 requests, then land-8's list from
        # 56 up (desk K5-Q8-CONSTANTS item 11; coordinator 05:48Z). K5 adds draft 35 / 40 (7 / 8 requests). land-8 took
        # draft r 1..5 only: consts(7) is land-8 + {42, 49} (land-8 padded those to 48 / 56).
        "CAPTURE": sorted({K * r for r in range(1, 9)} | {Q * r for r in range(1, 9)}
                          | {56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 256, 512, 1024, 2048, 4096, 8192}),
    }


globals().update(consts(5))  # K = 5: the recipe override_c1..c8 value
