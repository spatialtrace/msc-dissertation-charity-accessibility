# Reproducibility guide

## 1. Environment

The final verification used Python 3.11.15 and the package versions recorded in [`environment.yml`](environment.yml) and [`requirements-lock.txt`](requirements-lock.txt). GDAL/`ogr2ogr` and a PROJ installation compatible with GeoPandas are also required for road preparation.

Set the private project root before running any stage:

```bash
export DISSERTATION_DATA_ROOT="/path/to/private/dissertation-data"
```

All JSON configurations use `${DISSERTATION_DATA_ROOT}`. The workflow expands this environment variable at runtime, so no username or machine-specific path is committed.

## 2. Expected private inputs

The private root retains the original internal structure used by the dissertation, including:

- `final_data_and_analysis/Data_Spine/charity_rebuild_v2/`
- `final_data_and_analysis/unified-lsoa/`
- `final_data_and_analysis/halo-20km/`
- `final_data_and_analysis/Travel_Time/`
- `final_data_and_analysis/Road_network(Distance)/`
- the licensed and historical source directories referenced by Step 1

See [`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md) for sources and licence restrictions. The repository does not provide these private inputs.

## 3. Run order and hand-offs

### Step 1 — data preparation

Run the charity rebuild and common-geography notebook before preparing boundary and road-network support. `01_common2021_harmonisation.ipynb` stops after harmonisation and provider assignment; it does not contain the superseded internal 20 km OD-cache branch.

### Step 2 — descriptive summaries

Run `01_descriptive_mapping_summary.ipynb` from its own directory. It verifies the common geography and source totals before generating descriptive outputs.

### Step 3 — main E2SFCA

Run `python run_workflow.py` from `code/03_main_travel_time_e2sfca`. The workflow must pass availability, snapping, OD impedance, provider-ratio, accessibility, standardisation, conservation and completeness QA.

### Step 4 — HP–LA and trajectories

Run `icb_urban_rural_decomposition.ipynb` from its own directory. It reads the executed E2SFCA A/A* values unchanged, applies strict annual median rules and then builds three-year trajectories.

### Step 5 — global BB join count

Run `05_global_bb_join_count.ipynb`. The formal test uses binary fuzzy contiguity, 999 permutations and fixed year-specific seeds. Its expected observed BB values are 1,803, 1,740 and 1,626.

### Step 6 — distance sensitivity

Run `06a_distance_e2sfca_20km.ipynb`, followed by `06b_distance_hpla_trajectories.ipynb`. This branch is a robustness specification and does not replace the travel-time main model.

## 4. Verification gates

A run is acceptable only when all of the following hold:

- exactly 3,411 unique target LSOAs in every year;
- harmonised additive counts conserve source totals within the recorded tolerance;
- eligible-charity counts are 2,276, 3,313 and 3,996;
- every main annual accessibility file is complete and finite where required;
- fixed 2001 Care50-weighted A* benchmark is retained;
- HP–LA and trajectory counts sum to 3,411;
- Global BB inputs match the SHA-256 values in its input manifest;
- no configured input changes during a downstream run.

## 5. Public-release boundary

Only aggregate tables, QA records and non-geographic charts are committed. LSOA-level results are required for full execution but remain in the licensed/private environment. This is a transparent access limitation, not evidence that the omitted files were unused.
