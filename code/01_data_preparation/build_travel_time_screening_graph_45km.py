#!/usr/bin/env python3
"""Build a DfT-default travel-time graph from the fixed 45 km OSMM network.

This script changes only the edge cost. It reuses the selected, pre-filtered
45 km RoadLink layer and the existing endpoint plus grade-separation topology.
It does not calculate OD matrices, E2SFCA, or downstream results.

The graph is undirected because the selected OSMM extract has no usable
directionality or turn-restriction fields. Edge weights are modelled car travel
time in minutes, not observed door-to-door journey time.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.sparse import csr_matrix, load_npz, save_npz, triu
from scipy.sparse.csgraph import connected_components


SOURCE_URL = (
    "https://www.gov.uk/government/publications/journey-time-statistics-guidance/"
    "journey-time-statistics-notes-and-definitions-2019"
)

# Values supplied by the user and verified against DfT Journey Time Statistics
# 2019, Table 2. The OSMM routeHierarchy value Local Road is the confirmed
# naming counterpart of the DfT table label Local Street.
SPEED_PROFILE = (
    ("Motorway", "Motorway", 83.8),
    ("A Road Primary", "A Road Primary", 47.0),
    ("A Road", "A Road", 38.0),
    ("B Road Primary", "B Road Primary", 31.7),
    ("B Road", "B Road", 40.7),
    ("Minor Road", "Minor Road", 36.9),
    ("Local Road", "Local Street", 25.6),
    ("Local Access Road", "Local Access Road", 22.4),
    ("Restricted Local Access Road", "Restricted Local Access Road", 24.5),
    ("Secondary Access Road", "Secondary Access Road", 42.3),
    (
        "Restricted Secondary Access Road",
        "Restricted Secondary Access Road",
        45.6,
    ),
)

SPEED_KMH = {route: speed for route, _, speed in SPEED_PROFILE}
DFT_LABEL = {route: label for route, label, _ in SPEED_PROFILE}


def refuse_overwrite(paths: list[Path]) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        joined = "\n".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing outputs:\n{joined}")


def read_links(gpkg: Path) -> gpd.GeoDataFrame:
    links = gpd.read_file(
        gpkg,
        layer="RoadLink",
        engine="pyogrio",
        columns=[
            "operationalState",
            "formOfWay",
            "routeHierarchy",
            "length",
            "startGradeSeparation",
            "endGradeSeparation",
            "geometry",
        ],
    )
    if not links["operationalState"].eq("Open").all():
        raise AssertionError("The selected 45 km RoadLink layer contains non-Open links")
    return links


def hierarchy_audit(links: gpd.GeoDataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit = (
        links.assign(
            speed_kmh=links["routeHierarchy"].map(SPEED_KMH),
            dft_table_label=links["routeHierarchy"].map(DFT_LABEL),
        )
        .groupby("routeHierarchy", dropna=False, as_index=False)
        .agg(
            roadlink_count=("routeHierarchy", "size"),
            total_length_m=("length", "sum"),
            speed_kmh=("speed_kmh", "first"),
            dft_table_label=("dft_table_label", "first"),
        )
        .sort_values("routeHierarchy", na_position="first")
        .reset_index(drop=True)
    )
    audit["total_length_km"] = audit["total_length_m"] / 1000.0
    audit["speed_status"] = np.where(
        audit["speed_kmh"].isna(), "UNMATCHED", "matched"
    )
    audit["source"] = "DfT Journey Time Statistics 2019 Table 2"
    audit["source_url"] = SOURCE_URL
    audit["mapping_note"] = np.where(
        audit["routeHierarchy"].eq("Local Road"),
        "OSMM Local Road mapped to DfT Local Street, confirmed by user",
        "exact routeHierarchy label match",
    )

    present = set(links["routeHierarchy"].dropna().unique())
    absent = pd.DataFrame(SPEED_PROFILE, columns=[
        "routeHierarchy", "dft_table_label", "speed_kmh"
    ])
    absent = absent.loc[~absent["routeHierarchy"].isin(present)].copy()
    absent["status"] = "defined in speed profile but absent from current 45 km extract"
    return audit, absent


def graph_keys(links: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    geometries = links.geometry.array
    start = shapely.get_point(geometries, 0)
    end = shapely.get_point(geometries, -1)
    start_key = np.column_stack(
        [
            np.rint(shapely.get_x(start) * 1000).astype(np.int64),
            np.rint(shapely.get_y(start) * 1000).astype(np.int64),
            links["startGradeSeparation"].fillna(0).to_numpy(dtype=np.int64),
        ]
    )
    end_key = np.column_stack(
        [
            np.rint(shapely.get_x(end) * 1000).astype(np.int64),
            np.rint(shapely.get_y(end) * 1000).astype(np.int64),
            links["endGradeSeparation"].fillna(0).to_numpy(dtype=np.int64),
        ]
    )
    nodes, inverse = np.unique(
        np.vstack([start_key, end_key]), axis=0, return_inverse=True
    )
    edge_count = len(links)
    return nodes, inverse[:edge_count], inverse[edge_count:]


def build_graph(
    links: gpd.GeoDataFrame,
) -> tuple[np.ndarray, csr_matrix, dict[str, int]]:
    speed = links["routeHierarchy"].map(SPEED_KMH).to_numpy(dtype=float)
    length_m = links["length"].to_numpy(dtype=float)
    missing_speed = ~np.isfinite(speed)
    if missing_speed.any():
        missing = (
            links.loc[missing_speed, "routeHierarchy"]
            .value_counts(dropna=False)
            .to_dict()
        )
        raise AssertionError(f"Unmatched routeHierarchy values; graph not built: {missing}")

    # Required equation: time_min = length_km / speed_kmh * 60.
    time_min = (length_m / 1000.0) / speed * 60.0
    nodes, u, v = graph_keys(links)
    valid = (
        (u != v)
        & np.isfinite(length_m)
        & (length_m > 0)
        & np.isfinite(time_min)
        & (time_min > 0)
    )
    invalid_or_self_loop = int((~valid).sum())
    u, v, time_min = u[valid], v[valid], time_min[valid]

    # Preserve the existing topology: undirected endpoint pairs, keyed at 1 mm
    # plus grade separation. For parallel source links, retain the least-time
    # edge, exactly analogous to the existing graph's minimum-length rule.
    a, b = np.minimum(u, v), np.maximum(u, v)
    edge_key = a.astype(np.int64) * len(nodes) + b.astype(np.int64)
    order = np.argsort(edge_key)
    edge_key, a, b, time_min = (
        edge_key[order],
        a[order],
        b[order],
        time_min[order],
    )
    first = np.r_[0, np.flatnonzero(np.diff(edge_key)) + 1]
    duplicate_edges_collapsed = int(len(edge_key) - len(first))
    a, b = a[first], b[first]
    time_min = np.minimum.reduceat(time_min, first)
    graph = csr_matrix((time_min, (a, b)), shape=(len(nodes), len(nodes)))
    graph = graph + graph.T

    if graph.nnz == 0 or (graph.data <= 0).any() or not np.isfinite(graph.data).all():
        raise AssertionError("Travel-time graph contains invalid weights")
    if (graph != graph.T).nnz or np.any(graph.diagonal()):
        raise AssertionError("Travel-time graph is not symmetric with a zero diagonal")

    return nodes, graph, {
        "input_roadlinks": len(links),
        "missing_speed_roadlinks": int(missing_speed.sum()),
        "invalid_or_self_loop_edges_removed": invalid_or_self_loop,
        "duplicate_endpoint_pairs_collapsed": duplicate_edges_collapsed,
    }


def assert_same_topology(
    nodes: np.ndarray,
    graph: csr_matrix,
    baseline_nodes_path: Path,
    baseline_graph_path: Path,
) -> tuple[csr_matrix, np.ndarray]:
    baseline_nodes = np.load(baseline_nodes_path, mmap_mode="r")
    baseline_graph = load_npz(baseline_graph_path).tocsr()
    if nodes.shape != baseline_nodes.shape or not np.array_equal(nodes, baseline_nodes):
        raise AssertionError("Travel-time nodes differ from the existing 45 km nodes")

    new_structure = graph.copy()
    old_structure = baseline_graph.copy()
    new_structure.data = np.ones_like(new_structure.data, dtype=np.uint8)
    old_structure.data = np.ones_like(old_structure.data, dtype=np.uint8)
    topology_difference = (new_structure != old_structure).nnz
    if topology_difference:
        raise AssertionError(
            f"Travel-time graph topology differs at {topology_difference} matrix entries"
        )
    return baseline_graph, baseline_nodes


def time_summary(graph: csr_matrix) -> dict[str, float]:
    values = triu(graph, k=1, format="csr").data
    return {
        "time_min_min": float(values.min()),
        "time_min_median": float(np.median(values)),
        "time_min_p95": float(np.quantile(values, 0.95)),
        "time_min_max": float(values.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-gpkg", type=Path, required=True)
    parser.add_argument("--baseline-graph", type=Path, required=True)
    parser.add_argument("--baseline-nodes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    graph_path = args.output_dir / "road_graph_travel_time_min_45km.npz"
    nodes_path = args.output_dir / "road_nodes_xyz_45km.npy"
    qa_dir = args.output_dir / "qa"
    hierarchy_path = qa_dir / "route_hierarchy_speed_audit.csv"
    unmatched_path = qa_dir / "route_hierarchy_unmatched.csv"
    absent_path = qa_dir / "speed_profile_categories_absent_from_extract.csv"
    graph_audit_path = qa_dir / "travel_time_graph_45km_audit.csv"
    output_paths = [
        graph_path,
        nodes_path,
        hierarchy_path,
        unmatched_path,
        absent_path,
        graph_audit_path,
    ]
    refuse_overwrite(output_paths)
    qa_dir.mkdir(parents=True, exist_ok=True)

    links = read_links(args.input_gpkg)
    hierarchy, absent = hierarchy_audit(links)
    unmatched = hierarchy.loc[hierarchy["speed_kmh"].isna()].copy()
    hierarchy.to_csv(hierarchy_path, index=False, encoding="utf-8-sig")
    unmatched.to_csv(unmatched_path, index=False, encoding="utf-8-sig")
    absent.to_csv(absent_path, index=False, encoding="utf-8-sig")

    if not unmatched.empty:
        raise AssertionError(
            "Unmatched routeHierarchy values were written to "
            f"{unmatched_path}; no graph was built"
        )

    nodes, graph, metrics = build_graph(links)
    baseline_graph, baseline_nodes = assert_same_topology(
        nodes,
        graph,
        args.baseline_nodes,
        args.baseline_graph,
    )
    component_count, labels = connected_components(graph, directed=False)
    component_sizes = np.bincount(labels)
    baseline_component_count, _ = connected_components(baseline_graph, directed=False)
    metrics.update(
        {
            "graph_nodes": int(graph.shape[0]),
            "graph_edges_undirected": int(graph.nnz // 2),
            "graph_components": int(component_count),
            "largest_component_nodes": int(component_sizes.max()),
            "baseline_graph_nodes": int(baseline_graph.shape[0]),
            "baseline_graph_edges_undirected": int(baseline_graph.nnz // 2),
            "baseline_graph_components": int(baseline_component_count),
            "node_count_change": int(graph.shape[0] - baseline_graph.shape[0]),
            "edge_count_change": int((graph.nnz - baseline_graph.nnz) // 2),
            "component_count_change": int(component_count - baseline_component_count),
            "nodes_exactly_equal_baseline": bool(np.array_equal(nodes, baseline_nodes)),
            "adjacency_exactly_equal_baseline": True,
            "graph_is_symmetric": bool((graph != graph.T).nnz == 0),
            "graph_has_zero_diagonal": bool(not np.any(graph.diagonal())),
            "ferrylink_edges_used": 0,
            **time_summary(graph),
        }
    )
    graph_audit = pd.DataFrame([metrics])

    # Save only after every topology and weight assertion has passed.
    save_npz(graph_path, graph)
    np.save(nodes_path, nodes)
    graph_audit.to_csv(graph_audit_path, index=False, encoding="utf-8-sig")

    print("ROUTE HIERARCHY AUDIT")
    print(hierarchy.to_string(index=False))
    print("\nSPEED-PROFILE CATEGORIES ABSENT FROM EXTRACT")
    print(absent.to_string(index=False))
    print("\nTRAVEL-TIME GRAPH QA")
    print(graph_audit.to_string(index=False))


if __name__ == "__main__":
    main()
