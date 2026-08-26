# Step 1 — Data preparation

## Critical decisions

- Reconstruct charity presence for 2001, 2011 and 2021 using the documented registration, care-eligibility, financial and address rules.
- Harmonise additive Census counts to the fixed 2021 LSOAs before recalculating rates.
- Assign registered-charity coordinates to the same repaired 2021 geography.
- Represent capacity as the provider-LSOA sum of charity-level `log(1 + income_2021_gbp)`.
- Prepare exact 20 km external-provider and 40 km competing-demand support plus the 45 km road-network extraction used for routing.

## Main entry points

- `rebuild_charity_data.py`
- `01_common2021_harmonisation.ipynb`
- `prepare_exact_20km_40km_halo.py`
- `filter_distance_road_network_45km.py`
- `build_travel_time_screening_graph_45km.py`
- `prepare_provider_centred_travel_time_package.py`

The former supporting charity and income-check branches are not included because they contain historical/base-pipeline material rather than the final reporting path.
