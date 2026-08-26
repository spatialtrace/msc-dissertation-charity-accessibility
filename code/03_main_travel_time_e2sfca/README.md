# Step 3 — Main travel-time E2SFCA

## Fixed specification

- 30-minute modelled free-flow travel-time catchment;
- bands of 0–10, 10–20 and 20–30 minutes;
- weights 1.00, 0.68 and 0.22;
- external providers selected within an exact 20 km support envelope;
- competing external demand selected within an exact 40 km support envelope;
- true network impedance determines pair eligibility;
- internal outputs restricted to the 3,411 South West target LSOAs;
- A* divided by the fixed 2001 South West Care50-weighted mean A.

Run `python run_workflow.py` from this directory. The code executes input, graph, OD, E2SFCA Step 1, E2SFCA Step 2, standardisation and final conservation/completeness gates in sequence.

The undirected free-flow graph does not model one-way restrictions, turn penalties, congestion, individual journeys or service use.
