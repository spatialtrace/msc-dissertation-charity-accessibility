#!/usr/bin/env python3
"""Build the car-access 45 km graph from a pre-filtered OSMM GPKG.

This is data preparation only. It does not calculate OD matrices, E2SFCA, or
any downstream dissertation analysis. The input RoadLink layer must contain
only Open links in the six user-selected ``formOfWay`` categories.
"""

from __future__ import annotations

import argparse
import sqlite3
import subprocess
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import shapely
from scipy.sparse import csr_matrix, save_npz
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


pyproj.network.set_network_enabled(False)

RETAINED_FORM_OF_WAY = (
    "Single Carriageway",
    "Dual Carriageway",
    "Roundabout",
    "Slip Road",
    "Traffic Island Link",
    "Traffic Island Link At Junction",
)


def export_filtered_gpkg(baseline: Path, destination: Path) -> None:
    """Create the final two-layer GPKG from the unfiltered 45 km extract."""
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing filtered GPKG: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    quoted_categories = ",".join(f"'{value}'" for value in RETAINED_FORM_OF_WAY)
    sql = (
        "SELECT * FROM RoadLink WHERE operationalState = 'Open' "
        f"AND formOfWay IN ({quoted_categories})"
    )
    subprocess.run(
        [
            "ogr2ogr",
            "--config",
            "OGR_SQLITE_SYNCHRONOUS",
            "OFF",
            "-f",
            "GPKG",
            "-unsetFieldWidth",
            "-gt",
            "65536",
            str(destination),
            str(baseline),
            "-dialect",
            "SQLITE",
            "-sql",
            sql,
            "-nln",
            "RoadLink",
        ],
        check=True,
    )
    subprocess.run(
        [
            "ogr2ogr",
            "-f",
            "GPKG",
            "-update",
            "-append",
            str(destination),
            str(baseline),
            "study_area_buffer_45km",
            "-nln",
            "study_area_buffer_45km",
        ],
        check=True,
    )


def readonly_connection(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def source_counts(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    with readonly_connection(path) as connection:
        state = pd.read_sql_query(
            'SELECT operationalState, COUNT(*) AS roadlinks FROM "RoadLink" '
            'GROUP BY operationalState ORDER BY operationalState',
            connection,
        )
        form = pd.read_sql_query(
            'SELECT formOfWay, COUNT(*) AS source_open_roadlinks FROM "RoadLink" '
            'WHERE operationalState = \'Open\' GROUP BY formOfWay ORDER BY formOfWay',
            connection,
        )
        roadlinks = int(
            pd.read_sql_query('SELECT COUNT(*) AS n FROM "RoadLink"', connection)["n"].iloc[0]
        )
        ferries = int(
            pd.read_sql_query('SELECT COUNT(*) AS n FROM "FerryLink"', connection)["n"].iloc[0]
        )
    return state, form, roadlinks, ferries


def build_graph(filtered_gpkg: Path, output_dir: Path) -> tuple[np.ndarray, dict[str, object]]:
    links = gpd.read_file(
        filtered_gpkg,
        layer="RoadLink",
        engine="pyogrio",
        columns=[
            "operationalState",
            "formOfWay",
            "length",
            "startGradeSeparation",
            "endGradeSeparation",
            "geometry",
        ],
    )
    observed = set(links["formOfWay"].dropna().unique())
    if observed != set(RETAINED_FORM_OF_WAY):
        raise AssertionError(f"Unexpected retained formOfWay values: {sorted(observed)}")
    if not links["operationalState"].eq("Open").all():
        raise AssertionError("Filtered RoadLink layer contains a non-Open link")

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
    weights = links["length"].to_numpy(dtype=float)
    nodes, inverse = np.unique(
        np.vstack([start_key, end_key]), axis=0, return_inverse=True
    )
    edge_count = len(weights)
    u, v = inverse[:edge_count], inverse[edge_count:]
    valid = (u != v) & np.isfinite(weights) & (weights > 0)
    invalid_or_self_loop = int((~valid).sum())
    u, v, weights = u[valid], v[valid], weights[valid]
    a, b = np.minimum(u, v), np.maximum(u, v)
    edge_key = a.astype(np.int64) * len(nodes) + b.astype(np.int64)
    order = np.argsort(edge_key)
    edge_key, a, b, weights = (
        edge_key[order],
        a[order],
        b[order],
        weights[order],
    )
    first = np.r_[0, np.flatnonzero(np.diff(edge_key)) + 1]
    duplicate_edges_collapsed = int(len(edge_key) - len(first))
    a, b = a[first], b[first]
    weights = np.minimum.reduceat(weights, first)
    graph = csr_matrix((weights, (a, b)), shape=(len(nodes), len(nodes)))
    graph = graph + graph.T
    if graph.nnz == 0 or (graph.data <= 0).any() or not np.isfinite(graph.data).all():
        raise AssertionError("Filtered graph contains invalid weights")
    if (graph != graph.T).nnz or np.any(graph.diagonal()):
        raise AssertionError("Filtered graph is not symmetric with a zero diagonal")

    save_npz(output_dir / "road_graph_open_with_ferries_45km.npz", graph)
    np.save(output_dir / "road_nodes_xyz_45km.npy", nodes)
    component_count, labels = connected_components(graph, directed=False)
    component_sizes = np.bincount(labels)
    largest_component = int(component_sizes.argmax())
    metrics = {
        "filtered_open_roadlinks": len(links),
        "invalid_or_self_loop_edges_removed": invalid_or_self_loop,
        "duplicate_endpoint_pairs_collapsed": duplicate_edges_collapsed,
        "graph_nodes": graph.shape[0],
        "graph_edges_undirected": graph.nnz // 2,
        "graph_components": component_count,
        "largest_component_id": largest_component,
        "largest_component_nodes": int(component_sizes[largest_component]),
        "surface_grade_zero_nodes": int((nodes[:, 2] == 0).sum()),
    }
    return nodes, metrics


def snap_audit(nodes: np.ndarray, package_root: Path) -> pd.DataFrame:
    surface = nodes[:, 2] == 0
    tree = cKDTree(nodes[surface, :2] / 1000.0)
    rows: list[dict[str, object]] = []

    demand = pd.read_csv(
        package_root / "boundaries/competing_demand_halo_40km_2021_lsoa_list.csv"
    )
    demand_xy = demand[["representative_easting", "representative_northing"]].to_numpy(float)
    distance, _ = tree.query(demand_xy, k=1)
    rows.append(
        {
            "spatial_role": "external_competing_demand_2021_lsoa_representative_points",
            "year": 2021,
            "rows": len(demand),
            "missing_coordinates": int(np.isnan(demand_xy).any(axis=1).sum()),
            "minimum_nearest_surface_node_m": float(np.min(distance)),
            "median_nearest_surface_node_m": float(np.median(distance)),
            "p95_nearest_surface_node_m": float(np.quantile(distance, 0.95)),
            "maximum_nearest_surface_node_m": float(np.max(distance)),
            "used_in_od": False,
        }
    )

    for year in (2001, 2011, 2021):
        frame = pd.read_csv(
            package_root / f"charities/external_eligible_charities_{year}.csv",
            low_memory=False,
        )
        points = gpd.GeoDataFrame(
            frame[["charity_number", "lat", "long"]].copy(),
            geometry=gpd.points_from_xy(frame["long"], frame["lat"]),
            crs="EPSG:4326",
        ).to_crs("EPSG:27700")
        xy = np.column_stack([points.geometry.x, points.geometry.y])
        distance, _ = tree.query(xy, k=1)
        rows.append(
            {
                "spatial_role": "external_provider_exact_historical_points",
                "year": year,
                "rows": len(frame),
                "missing_coordinates": int(frame[["lat", "long"]].isna().any(axis=1).sum()),
                "minimum_nearest_surface_node_m": float(np.min(distance)),
                "median_nearest_surface_node_m": float(np.median(distance)),
                "p95_nearest_surface_node_m": float(np.quantile(distance, 0.95)),
                "maximum_nearest_surface_node_m": float(np.max(distance)),
                "used_in_od": False,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-gpkg", type=Path, required=True)
    parser.add_argument("--filtered-gpkg", type=Path, required=True)
    parser.add_argument("--build-filtered-gpkg", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    qa_dir = args.output_dir / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)

    if args.build_filtered_gpkg:
        export_filtered_gpkg(args.baseline_gpkg, args.filtered_gpkg)
    if not args.filtered_gpkg.is_file():
        raise FileNotFoundError(args.filtered_gpkg)

    state, form, source_roadlinks, source_ferries = source_counts(args.baseline_gpkg)
    nodes, metrics = build_graph(args.filtered_gpkg, args.output_dir)

    form["included_in_graph"] = form["formOfWay"].isin(RETAINED_FORM_OF_WAY)
    form["action"] = np.where(form["included_in_graph"], "retained", "excluded")
    form["selection_rule"] = (
        "operationalState=Open and formOfWay in the six user-selected car-access categories"
    )
    form.to_csv(qa_dir / "road_form_of_way_audit.csv", index=False, encoding="utf-8-sig")

    open_source = int(
        state.loc[state["operationalState"].eq("Open"), "roadlinks"].iloc[0]
    )
    nonopen = source_roadlinks - open_source
    excluded_open = open_source - int(metrics["filtered_open_roadlinks"])
    exclusion = pd.concat(
        [
            state.assign(filter_dimension="operationalState").rename(
                columns={"operationalState": "category", "roadlinks": "source_rows"}
            ),
            form[["formOfWay", "source_open_roadlinks"]]
            .rename(columns={"formOfWay": "category", "source_open_roadlinks": "source_rows"})
            .assign(filter_dimension="formOfWay"),
            pd.DataFrame(
                [{"filter_dimension": "network_layer", "category": "FerryLink", "source_rows": source_ferries}]
            ),
        ],
        ignore_index=True,
    )
    retained = set(RETAINED_FORM_OF_WAY)
    exclusion["action"] = np.select(
        [
            (exclusion["filter_dimension"] == "operationalState") & (exclusion["category"] == "Open"),
            (exclusion["filter_dimension"] == "formOfWay") & exclusion["category"].isin(retained),
        ],
        ["eligible_for_formOfWay_filter", "retained"],
        default="excluded",
    )
    exclusion.to_csv(qa_dir / "road_filter_exclusion_audit.csv", index=False, encoding="utf-8-sig")

    study = gpd.read_file(
        args.package_root / "boundaries/halo_20km_spatial.gpkg", layer="study_boundary"
    ).to_crs(27700)
    union = study.geometry.union_all()
    outer_band = gpd.GeoDataFrame(
        geometry=[union.buffer(45_000).difference(union.buffer(40_000))], crs=27700
    )
    outer_links = gpd.read_file(
        args.filtered_gpkg, layer="RoadLink", engine="pyogrio", columns=[], mask=outer_band
    )

    audit = pd.DataFrame(
        [
            {
                "road_source": "OS MasterMap Highways Network December 2021 existing local order 3004697",
                "study_boundary_definition": "dissolved seven April-2023 South West ICB polygons",
                "road_buffer_distance_m": 45_000,
                "source_roadlink_rows_intersecting_buffer": source_roadlinks,
                "source_open_roadlink_rows": open_source,
                "retained_open_roadlink_rows": metrics["filtered_open_roadlinks"],
                "excluded_nonopen_roadlink_rows": nonopen,
                "excluded_open_disallowed_formofway_rows": excluded_open,
                "source_ferrylink_rows": source_ferries,
                "ferrylink_rows_used_in_graph": 0,
                "roadlinks_intersecting_outer_40_45km_band": len(outer_links),
                **metrics,
                "graph_method": "Selected Open RoadLink endpoints keyed to 1 mm plus grade separation; undirected duplicate pairs retain minimum length; no FerryLinks",
                "road_type_rule_status": "applied user-selected six-category car-access formOfWay filter; all ordinary local/unclassified Single Carriageway retained",
                "legacy_graph_filename_note": "road_graph_open_with_ferries_45km.npz retained for path compatibility; graph contains no FerryLinks",
                "used_in_od": False,
                "used_in_e2sfca": False,
            }
        ]
    )
    audit.to_csv(qa_dir / "road_network_45km_audit.csv", index=False, encoding="utf-8-sig")
    snap_audit(nodes, args.package_root).to_csv(
        qa_dir / "road_snap_preparation_audit.csv", index=False, encoding="utf-8-sig"
    )
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
