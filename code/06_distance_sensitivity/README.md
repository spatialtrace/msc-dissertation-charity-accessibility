# Step 6 — 20 km road-distance sensitivity

This branch repeats E2SFCA using road distance rather than modelled travel time while keeping the common geography, demand, capacity, boundary-support and A* standardisation rules fixed.

1. Run `e2sfca/06a_distance_e2sfca_20km.ipynb` or its `run_workflow.py`.
2. Run `downstream/run_workflow.py` for the core trajectory outputs, then use
   `downstream/06b_distance_hpla_trajectories.ipynb` for the appended ICB and
   urban–rural decomposition.

The distance branch is robustness evidence and is not the dissertation's main accessibility specification.
