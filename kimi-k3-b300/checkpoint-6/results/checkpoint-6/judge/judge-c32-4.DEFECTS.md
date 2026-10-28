# judge-c32-4 known defects (sidecar; judge-c32-4 itself is frozen: SHA256SUMS 3095b5e5 covers its README)

## D1: the band depends on the order of the runs in the spec (item 05 07:38Z, coord RULE 07:38Z)
- c32.py takes the 95 % band from a null bootstrap with a fixed seed (0) over the control readings in list order.
  The same arms in a different run order give a different band.
- Seen at n_cand 3 on the same 8 stock arms [m, c32.py 53bf1a26]:
  - aa1,aa22,aa3,p0a (pre-registered order, read 1 of record): official 1.847, consistent 2.460
  - aa3,p0a,aa1,aa22 (item 05): 1.762 / 2.327
  - p0a,aa3,aa1,aa22 (item 01): 1.776 / 2.324
  - Range seen: official 1.762-1.847, consistent 2.324-2.460.
- Rule for checkpoint 6 (coord 07:38Z): judge-c32-4 is not touched. The band of record = item 03's call with the
  pre-registered spec order. Every other order = INFO.
- Fix (reads after checkpoint 6): judge-c32-5, a new directory. Sort arms by name before the seeded bootstrap, or seed
  per sorted arm set. Proof: two run orders give byte-equal BAND rows. A coord ruling comes before any data is read.
