# Step 5 — Global binary BB join count

`05_global_bb_join_count.ipynb` tests whether annual HP–LA LSOAs share more boundaries with other HP–LA LSOAs than expected under random label assignment.

## Frozen choices

- PySAL fuzzy contiguity with `predicate="intersects"` and no buffering;
- binary weights;
- 999 permutations;
- fixed seed `20260823 + year`;
- one spatial island reported and excluded from the formal statistic.

The verified observed BB joins are 1,803, 1,740 and 1,626 for 2001, 2011 and 2021. The test supports global clustering only; it does not identify local significant clusters.

The optional plotting scripts describe BB-neighbour participation for visualisation. Their GIS outputs are not distributed.
