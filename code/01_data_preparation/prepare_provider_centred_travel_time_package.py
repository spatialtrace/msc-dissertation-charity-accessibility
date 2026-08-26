#!/usr/bin/env python3
"""Prepare an isolated 30-minute travel-time E2SFCA data package.

The package is data preparation only. It does not calculate an OD matrix,
E2SFCA accessibility, mismatch, trajectories, LISA, or spatial models.

All outputs are new files under ``final_data_and_analysis/Travel_Time``. The
existing distance-based halo packages and Step 1-5 workflow are protected by
before/after SHA-256 checks. Existing output files are never overwritten.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import sqlite3
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import shapely
from scipy.sparse import bmat, csr_matrix, load_npz, save_npz, triu
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree


pyproj.network.set_network_enabled(False)

ROOT = Path(os.environ["DISSERTATION_DATA_ROOT"]).expanduser().resolve()
PACKAGE = ROOT / "final_data_and_analysis"
OUTPUT_ROOT = PACKAGE / "Travel_Time"
BASE_RULES_PATH = PACKAGE / "halo-20km/prepare_halo_20km.py"

ROAD_FOLDER = "Road_network(Travel_Time)"
DEMAND_FOLDER = "Demand(Travel_time)"
SUPPLY_FOLDER = "Supply(Travel_time)"
BOUNDARY_FOLDER = "Boundary(Travel_time)"
QA_FOLDER = "QA(Travel_time)"
SCRIPT_FOLDER = "Scripts(Travel_time)"

YEARS = (2001, 2011, 2021)
MAX_CAR_SPEED_KMH = 83.8
MAIN_CATCHMENT_MIN = 30.0
TIME_BANDS_MIN = (10.0, 20.0, 30.0)
PRIMARY_WEIGHTS = (1.00, 0.68, 0.22)
PROVIDER_CANDIDATE_DISTANCE_M = MAX_CAR_SPEED_KMH * MAIN_CATCHMENT_MIN / 60 * 1000
PROVIDER_CENTRED_DEMAND_DISTANCE_M = PROVIDER_CANDIDATE_DISTANCE_M
ROUTING_MARGIN_M = 5_000.0
CONNECTOR_SPEED_KMH = 25.6

EXPECTED_INTERNAL_LSOAS = {2001: 3230, 2011: 3285, 2021: 3411}
EXPECTED_INTERNAL_CHARITIES = {2001: 2276, 2011: 3313, 2021: 3996}

RETAINED_FORM_OF_WAY = (
    "Single Carriageway",
    "Dual Carriageway",
    "Roundabout",
    "Slip Road",
    "Traffic Island Link",
    "Traffic Island Link At Junction",
)

# DfT Journey Time Statistics 2019, Table 2. The Local Road to Local
# Street naming correspondence was explicitly confirmed by the user.
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
    ("Restricted Secondary Access Road", "Restricted Secondary Access Road", 45.6),
)
SPEED_KMH = {route: speed for route, _, speed in SPEED_PROFILE}
DFT_LABEL = {route: label for route, label, _ in SPEED_PROFILE}
DFT_SOURCE_URL = (
    "https://www.gov.uk/government/publications/journey-time-statistics-guidance/"
    "journey-time-statistics-notes-and-definitions-2019"
)

ROAD_SOURCE_DIR = (
    ROOT
    / "分析历史/data/Road/Download_OSMM_Highways_2021_GB_3004697"
    / "MasterMap Highways Network_roads_6440382"
)
ROADLINK_PATTERN = "Highways_Roads_RoadLink_FULL_*.gml.gz"
FERRYLINK_PATTERN = "Highways_Roads_FerryLink_FULL_*.gml.gz"

CHARITY_PACKAGE = PACKAGE / "Data_Spine/charity_rebuild_v2"
CHARITY_STAGE = CHARITY_PACKAGE / "05_geocoding/active_care_geocoded.parquet"
CHARITY_FINAL_PATHS = {
    year: CHARITY_PACKAGE / f"07_final_outputs/{year}charity.csv" for year in YEARS
}
COVARIATE_PATHS = {year: PACKAGE / f"covariates/{year}.csv" for year in YEARS}

SCREENING_ROAD_DIR = PACKAGE / "halo-20km/travel_time_45km"
SCREENING_GRAPH_PATH = SCREENING_ROAD_DIR / "road_graph_travel_time_min_45km.npz"
SCREENING_NODES_PATH = SCREENING_ROAD_DIR / "road_nodes_xyz_45km.npy"
SCREENING_AUDIT_PATH = SCREENING_ROAD_DIR / "qa/travel_time_graph_45km_audit.csv"


def load_base_rules():
    spec = importlib.util.spec_from_file_location("halo_base_rules", BASE_RULES_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load base rules: {BASE_RULES_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_base_rules()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path, *, float_format: str | None = "%.15g") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format=float_format)


def write_json(value: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_command(args: list[str]) -> None:
    subprocess.run(args, check=True)


def stable_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    if not root.exists():
        return []
    excluded_suffixes = ("-shm", "-wal")
    return [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.name != ".DS_Store"
        and not path.name.startswith("~$")
        and not path.name.endswith(excluded_suffixes)
    ]


def snapshot(paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(set(paths), key=lambda item: str(item)):
        stat = path.stat()
        rows.append(
            {
                "path": str(path),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": sha256(path),
            }
        )
    return pd.DataFrame(rows)


def target_relative(path: Path) -> Path:
    return OUTPUT_ROOT / path


def ensure_output_preflight() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    expected_existing = {
        ROAD_FOLDER,
        DEMAND_FOLDER,
        SUPPLY_FOLDER,
        SCRIPT_FOLDER,
    }
    for folder in expected_existing:
        target = OUTPUT_ROOT / folder
        target.mkdir(parents=True, exist_ok=True)
    for folder in (BOUNDARY_FOLDER, QA_FOLDER):
        target = OUTPUT_ROOT / folder
        if target.exists() and any(target.iterdir()):
            raise FileExistsError(f"Refusing to use non-empty output folder: {target}")
    for folder in (ROAD_FOLDER, DEMAND_FOLDER, SUPPLY_FOLDER):
        target = OUTPUT_ROOT / folder
        existing = [path for path in target.rglob("*") if path.is_file()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite existing outputs: {existing}")
    allowed_root_files = {".DS_Store"}
    unexpected_root = [
        path for path in OUTPUT_ROOT.iterdir()
        if path.is_file() and path.name not in allowed_root_files
    ]
    if unexpected_root:
        raise FileExistsError(f"Unexpected existing Travel_Time root files: {unexpected_root}")


PROVIDER_COLUMNS = [
    "analysis_year",
    "provider_scope",
    "uid",
    "charity_number",
    "charity_name",
    "registration_date",
    "removal_date",
    "presence_source",
    "cc_spell_active_at_census",
    "presence_date_disagrees_with_cc_spells",
    "care_strict",
    "care_route",
    "care_route_classification",
    "care_route_text",
    "care_route_both",
    "service_evidence",
    "postcode",
    "native_lsoa_code",
    "provider_lsoa_2021_code",
    "provider_lsoa_2021_name",
    "provider_lsoa_in_demand_support",
    "provider_lsoa_is_internal_2021",
    "provider_assignment_method",
    "nearest_boundary_distance_m",
    "provider_point_distance_from_study_area_m",
    "screening_provider_node_id",
    "screening_provider_snap_distance_m",
    "screening_provider_connector_time_min",
    "screening_min_time_from_internal_demand_min",
    "screening_reachable_within_30min",
    "screening_graph_source",
    "screening_connector_rule",
    "lat",
    "long",
    "address_source",
    "address_evidence_date",
    "address_year_offset",
    "address_method",
    "address_quality",
    "historical_address_flag",
    "data_spine_primary_postcode",
    "supplementary_postcode_count",
    "direct_company_link_count",
    "company_number",
    "income_2021_gbp",
    "log1p_income_2021_gbp",
    "selection_method",
    "finance_source",
    "finance_extract",
    "finance_year_offset",
    "fye_offset_days",
    "fye",
    "cpih_source_year",
    "cpih_index_source",
    "cpih_index_2021",
    "cpih_multiplier_to_2021",
    "cpih_series",
    "cpih_source_url",
    "source_authority",
    "used_in_od",
    "used_in_e2sfca",
]


def canonicalise_provider(frame: pd.DataFrame, year: int, scope: str) -> pd.DataFrame:
    work = frame.copy()
    if scope == "internal_authoritative":
        work["provider_scope"] = scope
        work["native_lsoa_code"] = work[f"lsoa_{year}"].astype("string")
    else:
        work["analysis_year"] = year
        work["provider_scope"] = scope
        work["charity_name"] = work["charity_name"]
        work["registration_date"] = work["registerdate"]
        work["removal_date"] = work["removeddate"]
        work["postcode"] = work["historical_postcode"]
        work["native_lsoa_code"] = work["lsoa_code"].astype("string")
    income = pd.to_numeric(work["income_2021_gbp"], errors="coerce")
    if income.dropna().lt(0).any():
        raise AssertionError(f"{year} {scope} contains negative income_2021_gbp")
    work["income_2021_gbp"] = income
    work["log1p_income_2021_gbp"] = np.where(income.notna(), np.log1p(income), np.nan)
    work["source_authority"] = (
        str(CHARITY_FINAL_PATHS[year]) if scope == "internal_authoritative" else str(CHARITY_STAGE)
    )
    work["used_in_od"] = False
    work["used_in_e2sfca"] = False
    return work


def create_base_context() -> dict[str, object]:
    """Load the frozen geography and define only the 42 km candidate area."""

    boundaries: dict[int, gpd.GeoDataFrame] = {}
    boundary_audit: list[dict[str, object]] = []
    internal_codes: dict[int, set[str]] = {}
    for year in YEARS:
        boundaries[year], audit = BASE.repair_lsoa_layer(year)
        boundary_audit.append(audit)
        covariates = pd.read_csv(COVARIATE_PATHS[year], usecols=["lsoa_code"])
        covariates["lsoa_code"] = covariates["lsoa_code"].astype("string").str.strip()
        if len(covariates) != EXPECTED_INTERNAL_LSOAS[year] or not covariates["lsoa_code"].is_unique:
            raise AssertionError(f"{year} internal LSOA authority is not the frozen workflow")
        internal_codes[year] = set(covariates["lsoa_code"])

    icbs = gpd.read_file(BASE.ICB_PATH).to_crs("EPSG:27700")
    study_union = icbs.geometry.union_all()
    if not study_union.is_valid:
        study_union = study_union.make_valid()
    provider_candidate_area = study_union.buffer(PROVIDER_CANDIDATE_DISTANCE_M)
    provider_candidate_ring = provider_candidate_area.difference(study_union)
    if not provider_candidate_area.is_valid or not provider_candidate_ring.is_valid:
        raise AssertionError("Invalid provider candidate extent")
    return {
        "boundaries": boundaries,
        "boundary_audit": pd.DataFrame(boundary_audit),
        "internal_codes": internal_codes,
        "study_union": study_union,
        "provider_candidate_area": provider_candidate_area,
        "provider_candidate_ring": provider_candidate_ring,
    }


def load_screening_network(context: dict[str, object]) -> dict[str, object]:
    """Build a virtual multi-source screen on the audited 45 km time graph."""

    for path in (SCREENING_GRAPH_PATH, SCREENING_NODES_PATH, SCREENING_AUDIT_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Missing screening-network authority: {path}")
    audit = pd.read_csv(SCREENING_AUDIT_PATH)
    if len(audit) != 1:
        raise AssertionError("45 km travel-time graph audit must contain one row")
    row = audit.iloc[0]
    required_true = (
        int(row["missing_speed_roadlinks"]) == 0
        and str(row["adjacency_exactly_equal_baseline"]).lower() == "true"
        and str(row["graph_is_symmetric"]).lower() == "true"
        and int(row["ferrylink_edges_used"]) == 0
    )
    if not required_true:
        raise AssertionError("45 km travel-time screening graph failed its frozen audit")

    graph = load_npz(SCREENING_GRAPH_PATH).tocsr()
    nodes = np.load(SCREENING_NODES_PATH, mmap_mode="r")
    if graph.shape[0] != graph.shape[1] or graph.shape[0] != len(nodes):
        raise AssertionError("45 km screening graph and node array are inconsistent")
    component_count, component_labels = connected_components(graph, directed=False)
    largest_component = int(np.bincount(component_labels).argmax())

    layer_2021 = context["boundaries"][2021]
    internal = layer_2021.loc[
        layer_2021["lsoa_code"].isin(context["internal_codes"][2021])
    ].copy()
    if len(internal) != EXPECTED_INTERNAL_LSOAS[2021]:
        raise AssertionError("Internal 2021 demand geography is incomplete")
    internal["geometry"] = internal.geometry.representative_point()

    surface_nodes = np.flatnonzero(nodes[:, 2] == 0)
    surface_tree_all = cKDTree(nodes[surface_nodes, :2] / 1000.0)
    scilly = internal.loc[
        internal["lsoa_name"].str.contains("Isles of Scilly", case=False, na=False)
    ]
    if len(scilly) != 1:
        raise AssertionError("Could not identify the Isles of Scilly demand component")
    _, scilly_position = surface_tree_all.query(
        [scilly.geometry.iloc[0].x, scilly.geometry.iloc[0].y], k=1
    )
    scilly_component = int(component_labels[surface_nodes[int(scilly_position)]])
    valid_components = {largest_component, scilly_component}
    valid_nodes = np.flatnonzero(np.isin(component_labels, list(valid_components)))
    surface_valid_nodes = valid_nodes[nodes[valid_nodes, 2] == 0]
    network_tree = cKDTree(nodes[surface_valid_nodes, :2] / 1000.0)

    internal_xy = np.column_stack([internal.geometry.x, internal.geometry.y])
    internal_snap_m, internal_position = network_tree.query(internal_xy, k=1)
    internal_node = surface_valid_nodes[internal_position]
    internal_connector_min = internal_snap_m / 1000.0 / CONNECTOR_SPEED_KMH * 60.0
    source_edges = (
        pd.DataFrame({"node": internal_node, "connector_min": internal_connector_min})
        .groupby("node", as_index=False)["connector_min"]
        .min()
    )
    source_weights = np.maximum(
        source_edges["connector_min"].to_numpy(dtype=float), np.finfo(float).eps
    )
    source_column = csr_matrix(
        (
            source_weights,
            (source_edges["node"].to_numpy(dtype=int), np.zeros(len(source_edges), dtype=int)),
        ),
        shape=(graph.shape[0], 1),
    )
    augmented = bmat(
        [[graph, source_column], [source_column.T, csr_matrix((1, 1))]],
        format="csr",
    )
    virtual_source = graph.shape[0]
    minimum_time = dijkstra(
        augmented,
        directed=False,
        indices=virtual_source,
        limit=MAIN_CATCHMENT_MIN,
    )[:-1]
    if not np.isfinite(minimum_time[internal_node]).all():
        raise AssertionError("The multi-source reachability screen lost internal demand nodes")
    return {
        "network_tree": network_tree,
        "surface_valid_nodes": surface_valid_nodes,
        "minimum_time_to_node": minimum_time,
        "component_count": component_count,
        "largest_component": largest_component,
        "scilly_component": scilly_component,
        "internal_demand_points": internal,
        "internal_snap_m": internal_snap_m,
        "graph_sha256": sha256(SCREENING_GRAPH_PATH),
        "nodes_sha256": sha256(SCREENING_NODES_PATH),
    }


def prepare_network_screened_providers(
    build_root: Path,
    context: dict[str, object],
) -> dict[str, object]:
    """Apply Data Spine rules, then retain only network-reachable external supply."""

    boundaries = context["boundaries"]
    study_union = context["study_union"]
    provider_ring = context["provider_candidate_ring"]
    internal_2021 = context["internal_codes"][2021]
    screening = load_screening_network(context)
    network_tree = screening["network_tree"]
    surface_valid_nodes = screening["surface_valid_nodes"]
    minimum_time_to_node = screening["minimum_time_to_node"]

    stage = pd.read_parquet(CHARITY_STAGE)
    if not stage["care_strict"].fillna(False).all():
        raise AssertionError("Data Spine selected stage includes non-care_strict rows")
    full_2021 = boundaries[2021]
    rep_2021 = full_2021[["lsoa_code", "lsoa_name", "geometry"]].copy()
    rep_2021["geometry"] = rep_2021.geometry.representative_point()
    rep_lookup = rep_2021.set_index("lsoa_code")

    internal_outputs: dict[int, pd.DataFrame] = {}
    candidate_outputs: dict[int, pd.DataFrame] = {}
    reachable_outputs: dict[int, pd.DataFrame] = {}
    combined_outputs: dict[int, pd.DataFrame] = {}
    screen_rows: list[dict[str, object]] = []
    candidate_lists: list[pd.DataFrame] = []
    supply_dir = build_root / SUPPLY_FOLDER

    for year in YEARS:
        internal_raw = pd.read_csv(CHARITY_FINAL_PATHS[year], low_memory=False)
        if len(internal_raw) != EXPECTED_INTERNAL_CHARITIES[year]:
            raise AssertionError(f"{year} internal final charity count changed")
        if internal_raw["charity_number"].duplicated().any():
            raise AssertionError(f"{year} internal charity IDs are duplicated")
        internal = canonicalise_provider(internal_raw, year, "internal_authoritative")
        internal = BASE.assign_charities_to_target(internal, full_2021)

        all_year = stage.loc[stage["target_year"].eq(year)].copy()
        geocoded = all_year.loc[all_year["geocoded"].fillna(False)].copy()
        candidate_points = gpd.GeoDataFrame(
            geocoded,
            geometry=gpd.points_from_xy(geocoded["long"], geocoded["lat"]),
            crs="EPSG:4326",
        ).to_crs("EPSG:27700")
        exact_candidate = candidate_points.geometry.within(provider_ring)
        external_raw = geocoded.loc[exact_candidate.to_numpy()].copy()
        internal_ids = set(pd.to_numeric(internal["charity_number"], errors="raise").astype(int))
        external_raw["charity_number"] = pd.to_numeric(
            external_raw["charity_number"], errors="raise"
        ).astype(int)
        external_raw = external_raw.loc[
            ~external_raw["charity_number"].isin(internal_ids)
        ].copy()
        if external_raw["charity_number"].duplicated().any():
            raise AssertionError(f"{year} external provider candidate IDs are duplicated")
        if len(external_raw) and not external_raw["historical_address_flag"].fillna(False).all():
            raise AssertionError(f"{year} external provider candidate lacks historical-address evidence")
        if external_raw[["historical_postcode", "lat", "long"]].isna().any().any():
            raise AssertionError(f"{year} external provider candidate location is incomplete")
        candidate = canonicalise_provider(external_raw, year, "external_42km_candidate")
        candidate = BASE.assign_charities_to_target(candidate, full_2021)

        for frame in (internal, candidate):
            frame["provider_lsoa_is_internal_2021"] = frame[
                "provider_lsoa_2021_code"
            ].isin(internal_2021)
            frame["provider_lsoa_in_demand_support"] = pd.NA
            exact_points = gpd.GeoDataFrame(
                frame[["charity_number", "lat", "long"]].copy(),
                geometry=gpd.points_from_xy(frame["long"], frame["lat"]),
                crs="EPSG:4326",
            ).to_crs("EPSG:27700")
            frame["provider_point_distance_from_study_area_m"] = (
                exact_points.geometry.distance(study_union).to_numpy()
            )

        internal["screening_provider_node_id"] = pd.NA
        internal["screening_provider_snap_distance_m"] = pd.NA
        internal["screening_provider_connector_time_min"] = pd.NA
        internal["screening_min_time_from_internal_demand_min"] = pd.NA
        internal["screening_reachable_within_30min"] = pd.NA
        internal["screening_graph_source"] = str(SCREENING_GRAPH_PATH)
        internal["screening_connector_rule"] = "not applicable to internal authoritative providers"

        if len(candidate):
            provider_x = candidate["provider_lsoa_2021_code"].map(rep_lookup.geometry.x)
            provider_y = candidate["provider_lsoa_2021_code"].map(rep_lookup.geometry.y)
            if provider_x.isna().any() or provider_y.isna().any():
                raise AssertionError(f"{year} external provider LSOA representative point missing")
            provider_xy = np.column_stack([provider_x, provider_y])
            snap_m, position = network_tree.query(provider_xy, k=1)
            provider_node = surface_valid_nodes[position]
            provider_connector_min = snap_m / 1000.0 / CONNECTOR_SPEED_KMH * 60.0
            total_time_min = minimum_time_to_node[provider_node] + provider_connector_min
            reachable = np.isfinite(total_time_min) & (
                total_time_min <= MAIN_CATCHMENT_MIN + 1e-12
            )
        else:
            snap_m = np.array([], dtype=float)
            provider_node = np.array([], dtype=int)
            provider_connector_min = np.array([], dtype=float)
            total_time_min = np.array([], dtype=float)
            reachable = np.array([], dtype=bool)

        candidate["screening_provider_node_id"] = provider_node
        candidate["screening_provider_snap_distance_m"] = snap_m
        candidate["screening_provider_connector_time_min"] = provider_connector_min
        candidate["screening_min_time_from_internal_demand_min"] = total_time_min
        candidate["screening_reachable_within_30min"] = reachable
        candidate["screening_graph_source"] = str(SCREENING_GRAPH_PATH)
        candidate["screening_connector_rule"] = (
            "provider-LSOA and internal fixed-2021 LSOA representative points snapped to "
            "nearest valid surface node; both connectors costed at 25.6 km/h"
        )
        reachable_external = candidate.loc[reachable].copy()
        reachable_external["provider_scope"] = "external_network_reachable_30min"

        internal["provider_scope"] = "internal_authoritative"
        missing_internal = set(PROVIDER_COLUMNS) - set(internal.columns)
        missing_candidate = set(PROVIDER_COLUMNS) - set(candidate.columns)
        if missing_internal or missing_candidate:
            raise AssertionError(
                f"{year} canonical provider fields missing: internal={sorted(missing_internal)}, "
                f"candidate={sorted(missing_candidate)}"
            )
        internal = internal[PROVIDER_COLUMNS].copy()
        candidate = candidate[PROVIDER_COLUMNS].copy()
        reachable_external = reachable_external[PROVIDER_COLUMNS].copy()
        combined = pd.concat([internal, reachable_external], ignore_index=True)
        combined["charity_number"] = pd.to_numeric(
            combined["charity_number"], errors="raise"
        ).astype(int)
        if combined["charity_number"].duplicated().any():
            raise AssertionError(f"{year} internal/network-reachable provider overlap")

        internal_outputs[year] = internal
        candidate_outputs[year] = candidate
        reachable_outputs[year] = reachable_external
        combined_outputs[year] = combined
        write_csv(candidate, supply_dir / f"external_candidate_charities_42km_{year}.csv")

        candidate_list = candidate[
            [
                "analysis_year",
                "charity_number",
                "provider_lsoa_2021_code",
                "provider_point_distance_from_study_area_m",
                "screening_min_time_from_internal_demand_min",
                "screening_reachable_within_30min",
            ]
        ].copy()
        candidate_lists.append(candidate_list)
        finite_time = total_time_min[np.isfinite(total_time_min)]
        selected_time = total_time_min[reachable]
        screen_rows.append(
            {
                "year": year,
                "national_active_care_strict_rows": len(all_year),
                "national_geocoded_rows": len(geocoded),
                "external_exact_point_candidates_42km": len(candidate),
                "candidate_provider_lsoas": candidate["provider_lsoa_2021_code"].nunique(),
                "network_reachable_external_charities_30min": len(reachable_external),
                "network_reachable_external_provider_lsoas_30min": reachable_external[
                    "provider_lsoa_2021_code"
                ].nunique(),
                "network_excluded_candidate_charities": int((~reachable).sum()),
                "finite_screen_times": len(finite_time),
                "unreachable_or_over_graph_limit": int((~np.isfinite(total_time_min)).sum()),
                "candidate_exact_point_max_distance_from_study_m": (
                    float(candidate["provider_point_distance_from_study_area_m"].max())
                    if len(candidate) else 0.0
                ),
                "provider_lsoa_snap_median_m": float(np.median(snap_m)) if len(snap_m) else np.nan,
                "provider_lsoa_snap_p95_m": float(np.quantile(snap_m, 0.95)) if len(snap_m) else np.nan,
                "provider_lsoa_snap_max_m": float(np.max(snap_m)) if len(snap_m) else np.nan,
                "selected_min_time_min": float(np.min(selected_time)) if len(selected_time) else np.nan,
                "selected_max_time_min": float(np.max(selected_time)) if len(selected_time) else np.nan,
                "screening_catchment_min": MAIN_CATCHMENT_MIN,
                "connector_speed_kmh": CONNECTOR_SPEED_KMH,
                "full_od_matrix_created": False,
            }
        )
        print(
            f"PROVIDER_SCREEN {year} candidates={len(candidate)} "
            f"reachable_30min={len(reachable_external)}",
            flush=True,
        )

    write_csv(
        pd.concat(candidate_lists, ignore_index=True),
        build_root / BOUNDARY_FOLDER / "external_charity_42km_network_screen_list_2001_2011_2021.csv",
    )
    return {
        "internal": internal_outputs,
        "candidates": candidate_outputs,
        "external_reachable": reachable_outputs,
        "combined": combined_outputs,
        "screen_audit": pd.DataFrame(screen_rows),
        "screening": screening,
    }


def create_provider_centred_support_and_demand(
    build_root: Path,
    context: dict[str, object],
    providers: dict[str, object],
) -> dict[str, object]:
    """Create only demand areas needed by network-reachable external supply."""

    boundaries = context["boundaries"]
    internal_codes = context["internal_codes"]
    study_union = context["study_union"]
    full_2021 = boundaries[2021]
    rep_2021 = full_2021[["lsoa_code", "lsoa_name", "geometry"]].copy()
    rep_2021["geometry"] = rep_2021.geometry.representative_point()
    rep_lookup = rep_2021.set_index("lsoa_code")

    centre_rows: list[dict[str, object]] = []
    for year in YEARS:
        reachable = providers["external_reachable"][year]
        for code in sorted(reachable["provider_lsoa_2021_code"].astype(str).unique()):
            centre_rows.append(
                {
                    "year": year,
                    "provider_lsoa_2021_code": code,
                    "provider_lsoa_2021_name": rep_lookup.at[code, "lsoa_name"],
                    "reachable_charities": int(
                        reachable["provider_lsoa_2021_code"].astype(str).eq(code).sum()
                    ),
                    "geometry": rep_lookup.at[code, "geometry"],
                }
            )
    if not centre_rows:
        raise AssertionError("No network-reachable external providers were found")
    provider_centres = gpd.GeoDataFrame(centre_rows, geometry="geometry", crs="EPSG:27700")
    unique_centres = provider_centres.drop_duplicates("provider_lsoa_2021_code").copy()
    provider_support = unique_centres.geometry.buffer(
        PROVIDER_CENTRED_DEMAND_DISTANCE_M
    ).union_all()
    demand_support_geometry = study_union.union(provider_support)
    road_core_geometry = context["provider_candidate_area"].union(demand_support_geometry)
    routing_footprint = road_core_geometry.buffer(ROUTING_MARGIN_M)
    for label, geometry in (
        ("provider_support", provider_support),
        ("demand_support", demand_support_geometry),
        ("road_core", road_core_geometry),
        ("routing_footprint", routing_footprint),
    ):
        if geometry.is_empty or not geometry.is_valid:
            raise AssertionError(f"Invalid {label} geometry")
    print(
        f"PROVIDER_CENTRED_EXTENT unique_external_provider_lsoas={len(unique_centres)} "
        f"routing_margin_m={ROUTING_MARGIN_M:.0f}",
        flush=True,
    )

    demand_extent: dict[int, gpd.GeoDataFrame] = {}
    extent_rows: list[dict[str, object]] = []
    demand_lists: list[pd.DataFrame] = []
    for year in (2021, 2001, 2011):
        layer = boundaries[year].copy()
        representatives = layer.geometry.representative_point()
        if year == 2021:
            spatial_candidate = representatives.within(demand_support_geometry)
            source_selection_rule = "2021 representative point within provider-centred support"
        else:
            # Counts-first areal harmonisation requires every selected 2021 target
            # to have a historical source. Querying the selected target polygons
            # gives the minimal topological source closure. build_crosswalk then
            # retains only positive-area intersections and normalises them.
            target_union = demand_extent[2021].geometry.union_all()
            candidate_index = layer.sindex.query(
                target_union, predicate="intersects"
            )
            spatial_candidate = pd.Series(False, index=layer.index)
            if len(candidate_index):
                candidate_index = np.unique(candidate_index)
                spatial_candidate.iloc[candidate_index] = True
            source_selection_rule = (
                "historical LSOA intersects selected fixed-2021 demand-support polygons; "
                "crosswalk retains positive-area intersections"
            )
        selected = layer["lsoa_code"].isin(internal_codes[year]) | spatial_candidate
        demand = layer.loc[selected].copy()
        demand["representative_easting"] = representatives.loc[selected].x
        demand["representative_northing"] = representatives.loc[selected].y
        demand["representative_distance_from_study_area_m"] = representatives.loc[
            selected
        ].distance(study_union)
        demand["study_scope"] = np.where(
            demand["lsoa_code"].isin(internal_codes[year]),
            "internal_study",
            "external_provider_centred_support",
        )
        demand["country"] = np.where(
            demand["lsoa_code"].str.startswith("E"), "England", "Wales"
        )
        demand = demand.sort_values("lsoa_code").reset_index(drop=True)
        missing_internal = internal_codes[year] - set(demand["lsoa_code"])
        if missing_internal:
            raise AssertionError(f"{year} demand support omits {len(missing_internal)} internal LSOAs")
        demand_extent[year] = demand
        print(
            f"DEMAND_SUPPORT {year} lsoas={len(demand)} "
            f"internal={int(demand['study_scope'].eq('internal_study').sum())}",
            flush=True,
        )
        demand_list = demand.drop(columns="geometry").copy()
        demand_list.insert(0, "year", year)
        demand_list["provider_centred_candidate_radius_m"] = (
            PROVIDER_CENTRED_DEMAND_DISTANCE_M
        )
        demand_lists.append(demand_list)
        extent_rows.append(
            {
                "year": year,
                "extent_role": "internal_plus_provider_centred_demand_support",
                "candidate_radius_m": PROVIDER_CENTRED_DEMAND_DISTANCE_M,
                "lsoas": len(demand),
                "internal_lsoas": int(demand["study_scope"].eq("internal_study").sum()),
                "external_lsoas": int(
                    demand["study_scope"].eq("external_provider_centred_support").sum()
                ),
                "england_lsoas": int(demand["country"].eq("England").sum()),
                "wales_lsoas": int(demand["country"].eq("Wales").sum()),
                "duplicate_lsoa_codes": int(demand["lsoa_code"].duplicated().sum()),
                "source_selection_rule": source_selection_rule,
            }
        )

    demand_target = demand_extent[2021]
    demand_target_codes = set(demand_target["lsoa_code"])
    for year in YEARS:
        provider_codes = set(
            providers["combined"][year]["provider_lsoa_2021_code"].astype(str)
        )
        if provider_codes - demand_target_codes:
            raise AssertionError(
                f"{year} provider-centred support omits {len(provider_codes - demand_target_codes)} provider LSOAs"
            )

    boundary_dir = build_root / BOUNDARY_FOLDER
    write_csv(
        pd.concat(demand_lists, ignore_index=True).sort_values(["year", "lsoa_code"]),
        boundary_dir / "provider_centred_demand_support_native_lsoa_list_2001_2011_2021.csv",
    )
    gpkg = boundary_dir / "travel_time_extents.gpkg"
    layers = [
        (
            "study_boundary",
            gpd.GeoDataFrame(
                {"definition": ["dissolved seven April-2023 South West ICBs"]},
                geometry=[study_union],
                crs="EPSG:27700",
            ),
        ),
        (
            "provider_candidate_area_42km",
            gpd.GeoDataFrame(
                {"distance_m": [PROVIDER_CANDIDATE_DISTANCE_M]},
                geometry=[context["provider_candidate_area"]],
                crs="EPSG:27700",
            ),
        ),
        (
            "provider_candidate_ring_42km",
            gpd.GeoDataFrame(
                {"distance_m": [PROVIDER_CANDIDATE_DISTANCE_M]},
                geometry=[context["provider_candidate_ring"]],
                crs="EPSG:27700",
            ),
        ),
        ("network_reachable_provider_points", provider_centres),
        (
            "provider_centred_demand_support",
            gpd.GeoDataFrame(
                {
                    "candidate_radius_m": [PROVIDER_CENTRED_DEMAND_DISTANCE_M],
                    "unique_external_provider_lsoas": [len(unique_centres)],
                },
                geometry=[demand_support_geometry],
                crs="EPSG:27700",
            ),
        ),
        (
            "road_core_extent",
            gpd.GeoDataFrame(
                {"definition": ["provider candidate area union provider-centred demand support"]},
                geometry=[road_core_geometry],
                crs="EPSG:27700",
            ),
        ),
        (
            "routing_footprint_margin_5km",
            gpd.GeoDataFrame(
                {"routing_margin_m": [ROUTING_MARGIN_M]},
                geometry=[routing_footprint],
                crs="EPSG:27700",
            ),
        ),
        ("provider_centred_demand_lsoa_2021", demand_target),
    ]
    for index, (layer_name, frame) in enumerate(layers):
        frame.to_file(
            gpkg,
            layer=layer_name,
            driver="GPKG",
            mode="w" if index == 0 else "a",
        )
    demand_points = demand_target.copy()
    demand_points["geometry"] = demand_points.geometry.representative_point()
    demand_points.to_file(
        gpkg,
        layer="provider_centred_demand_points_2021",
        driver="GPKG",
        mode="a",
    )

    native_demand: dict[int, pd.DataFrame] = {}
    harmonised_demand: dict[int, pd.DataFrame] = {}
    crosswalks: dict[int, pd.DataFrame] = {}
    demand_audit_rows: list[dict[str, object]] = []
    conservation_rows: list[dict[str, object]] = []
    duplicate_rows: list[dict[str, object]] = []
    demand_dir = build_root / DEMAND_FOLDER
    for year in YEARS:
        keep = set(demand_extent[year]["lsoa_code"])
        native = BASE.extract_native_demand(
            year,
            keep,
            demand_extent[year][["lsoa_code", "lsoa_name"]],
        )
        native["study_scope"] = native["lsoa_code"].map(
            demand_extent[year].set_index("lsoa_code")["study_scope"]
        )
        native["provider_centred_candidate_radius_m"] = (
            PROVIDER_CENTRED_DEMAND_DISTANCE_M
        )
        native["main_catchment_min"] = MAIN_CATCHMENT_MIN
        numeric = native[["care50_num", "population_5plus"]]
        if (
            numeric.isna().any().any()
            or (numeric["care50_num"] < 0).any()
            or (numeric["population_5plus"] <= 0).any()
        ):
            raise AssertionError(f"{year} native Care50 support is incomplete or invalid")
        native_demand[year] = native
        write_csv(native, demand_dir / f"native_provider_centred_care50_support_{year}.csv")

        print(f"CROSSWALK_START {year}", flush=True)
        crosswalk = BASE.build_crosswalk(year, demand_extent[year], demand_target)
        print(f"CROSSWALK_READY {year} rows={len(crosswalk)}", flush=True)
        crosswalks[year] = crosswalk
        write_csv(
            crosswalk,
            demand_dir / f"native_to_2021_provider_centred_support_crosswalk_{year}.csv",
        )
        harmonised = BASE.harmonise_demand(year, native, crosswalk, demand_target)
        target_lookup = demand_target.set_index("lsoa_code")
        harmonised["study_scope"] = harmonised["lsoa_2021_code"].map(
            target_lookup["study_scope"]
        )
        harmonised["representative_easting"] = harmonised["lsoa_2021_code"].map(
            target_lookup["representative_easting"]
        )
        harmonised["representative_northing"] = harmonised["lsoa_2021_code"].map(
            target_lookup["representative_northing"]
        )
        harmonised["provider_centred_candidate_radius_m"] = (
            PROVIDER_CENTRED_DEMAND_DISTANCE_M
        )
        harmonised["main_catchment_min"] = MAIN_CATCHMENT_MIN
        if set(harmonised["lsoa_2021_code"]) != demand_target_codes:
            raise AssertionError(f"{year} harmonised target code mismatch")
        harmonised_demand[year] = harmonised
        write_csv(
            harmonised,
            demand_dir / f"harmonised_provider_centred_care50_to_2021_{year}.csv",
        )

        for variable in ("care50_num", "population_5plus"):
            source_total = float(native[variable].sum())
            target_total = float(harmonised[variable].sum())
            difference = target_total - source_total
            relative_error = abs(difference) / max(abs(source_total), 1.0)
            conservation_rows.append(
                {
                    "year": year,
                    "variable": variable,
                    "native_total": source_total,
                    "harmonised_total": target_total,
                    "signed_difference": difference,
                    "absolute_difference": abs(difference),
                    "relative_error": relative_error,
                    "conservation_pass": relative_error < 1e-12,
                }
            )
        source_weight_error = (
            crosswalk.groupby("source_lsoa_code")["source_normalized_weight"].sum() - 1
        ).abs().max()
        demand_audit_rows.append(
            {
                "year": year,
                "native_rows": len(native),
                "unique_native_lsoas": native["lsoa_code"].nunique(),
                "harmonised_rows": len(harmonised),
                "unique_2021_target_lsoas": harmonised["lsoa_2021_code"].nunique(),
                "expected_2021_target_lsoas": len(demand_target_codes),
                "missing_native_counts": int(
                    native[["care50_num", "population_5plus"]].isna().any(axis=1).sum()
                ),
                "missing_harmonised_counts": int(
                    harmonised[["care50_num", "population_5plus"]].isna().any(axis=1).sum()
                ),
                "crosswalk_rows": len(crosswalk),
                "crosswalk_source_weight_max_error": float(source_weight_error),
                "submetre_boundary_fallback_rows": int(
                    crosswalk["crosswalk_method"].eq(
                        "submetre_boundary_edge_nearest_target"
                    ).sum()
                ),
                "used_in_od": False,
                "used_in_e2sfca": False,
            }
        )
        duplicate_rows.extend(
            [
                {
                    "artifact": f"native_provider_centred_care50_support_{year}",
                    "key": "lsoa_code",
                    "rows": len(native),
                    "duplicate_rows": int(native["lsoa_code"].duplicated().sum()),
                },
                {
                    "artifact": f"harmonised_provider_centred_care50_to_2021_{year}",
                    "key": "lsoa_2021_code",
                    "rows": len(harmonised),
                    "duplicate_rows": int(harmonised["lsoa_2021_code"].duplicated().sum()),
                },
            ]
        )

    conservation = pd.DataFrame(conservation_rows)
    if not conservation["conservation_pass"].all():
        raise AssertionError("Care50 counts-first harmonisation failed conservation")
    write_csv(
        pd.concat([harmonised_demand[year] for year in YEARS], ignore_index=True),
        demand_dir / "harmonised_provider_centred_care50_to_2021_long.csv",
    )

    capacity_outputs: dict[int, pd.DataFrame] = {}
    provider_audit_rows: list[dict[str, object]] = []
    overlap_rows: list[dict[str, object]] = []
    supply_dir = build_root / SUPPLY_FOLDER
    target_rep = demand_target[["lsoa_code", "lsoa_name", "geometry"]].copy()
    target_rep["geometry"] = target_rep.geometry.representative_point()
    target_rep_lookup = target_rep.set_index("lsoa_code")
    for year in YEARS:
        internal = providers["internal"][year].copy()
        candidate = providers["candidates"][year]
        external = providers["external_reachable"][year].copy()
        combined = providers["combined"][year].copy()
        for frame in (internal, external, combined):
            frame["provider_lsoa_in_demand_support"] = frame[
                "provider_lsoa_2021_code"
            ].isin(demand_target_codes)
            if not frame["provider_lsoa_in_demand_support"].all():
                raise AssertionError(f"{year} retained provider lies outside demand support")
        providers["internal"][year] = internal
        providers["external_reachable"][year] = external
        providers["combined"][year] = combined
        write_csv(internal, supply_dir / f"internal_eligible_charities_{year}.csv")
        write_csv(external, supply_dir / f"external_network_reachable_charities_30min_{year}.csv")
        write_csv(combined, supply_dir / f"combined_network_reachable_charities_{year}.csv")

        usable = combined.loc[combined["income_2021_gbp"].notna()].copy()
        capacity = (
            usable.groupby("provider_lsoa_2021_code", as_index=False)
            .agg(
                charity_records_with_usable_income=("charity_number", "size"),
                registered_capacity_log1p_income=("log1p_income_2021_gbp", "sum"),
                internal_usable_charities=(
                    "provider_scope",
                    lambda values: int((values == "internal_authoritative").sum()),
                ),
                external_usable_charities=(
                    "provider_scope",
                    lambda values: int((values == "external_network_reachable_30min").sum()),
                ),
            )
        )
        all_counts = combined.groupby("provider_lsoa_2021_code").size().rename(
            "all_eligible_charity_records"
        )
        capacity = capacity.merge(
            all_counts, on="provider_lsoa_2021_code", validate="one_to_one"
        )
        capacity["provider_lsoa_2021_name"] = capacity[
            "provider_lsoa_2021_code"
        ].map(target_rep_lookup["lsoa_name"])
        capacity["representative_easting"] = capacity[
            "provider_lsoa_2021_code"
        ].map(target_rep_lookup.geometry.x)
        capacity["representative_northing"] = capacity[
            "provider_lsoa_2021_code"
        ].map(target_rep_lookup.geometry.y)
        capacity.insert(0, "year", year)
        capacity["capacity_definition"] = "sum of charity-level log1p(income_2021_gbp)"
        capacity["used_in_od"] = False
        capacity["used_in_e2sfca"] = False
        if capacity[["representative_easting", "representative_northing"]].isna().any().any():
            raise AssertionError(f"{year} provider capacity lacks representative coordinates")
        write_csv(capacity, supply_dir / f"provider_lsoa_log1p_capacity_{year}.csv")
        capacity_outputs[year] = capacity

        internal_ids = set(internal["charity_number"])
        external_ids = set(external["charity_number"])
        overlap = internal_ids & external_ids
        overlap_pass = (
            not overlap
            and combined["provider_lsoa_in_demand_support"].all()
            and capacity["provider_lsoa_2021_code"].isin(demand_target_codes).all()
        )
        overlap_rows.append(
            {
                "year": year,
                "internal_external_charity_id_overlap": len(overlap),
                "combined_provider_lsoa_not_in_demand_support": int(
                    (~combined["provider_lsoa_in_demand_support"]).sum()
                ),
                "provider_capacity_lsoa_not_in_demand_support": int(
                    (~capacity["provider_lsoa_2021_code"].isin(demand_target_codes)).sum()
                ),
                "overlap_pass": overlap_pass,
            }
        )
        missing_income = int(combined["income_2021_gbp"].isna().sum())
        provider_audit_rows.append(
            {
                "year": year,
                "internal_authoritative_charities": len(internal),
                "external_42km_candidate_charities": len(candidate),
                "external_network_reachable_charities_30min": len(external),
                "external_network_excluded_charities": len(candidate) - len(external),
                "combined_eligible_charities": len(combined),
                "combined_unique_charity_numbers": combined["charity_number"].nunique(),
                "with_income_2021_gbp": int(combined["income_2021_gbp"].notna().sum()),
                "missing_income_2021_gbp": missing_income,
                "with_charity_level_log1p_income": int(
                    combined["log1p_income_2021_gbp"].notna().sum()
                ),
                "provider_lsoas_with_usable_income_capacity": len(capacity),
                "providers_outside_demand_support": int(
                    (~combined["provider_lsoa_in_demand_support"]).sum()
                ),
                "used_in_od": False,
                "used_in_e2sfca": False,
            }
        )
        for artifact, frame, key in (
            (f"internal_eligible_charities_{year}", internal, "charity_number"),
            (f"external_candidate_charities_42km_{year}", candidate, "charity_number"),
            (f"external_network_reachable_charities_30min_{year}", external, "charity_number"),
            (f"combined_network_reachable_charities_{year}", combined, "charity_number"),
            (f"provider_lsoa_log1p_capacity_{year}", capacity, "provider_lsoa_2021_code"),
        ):
            duplicate_rows.append(
                {
                    "artifact": artifact,
                    "key": key,
                    "rows": len(frame),
                    "duplicate_rows": int(frame[key].duplicated().sum()),
                }
            )

    write_csv(
        pd.concat([providers["combined"][year] for year in YEARS], ignore_index=True),
        supply_dir / "combined_network_reachable_charities_long.csv",
    )
    write_csv(
        pd.concat([capacity_outputs[year] for year in YEARS], ignore_index=True),
        supply_dir / "provider_lsoa_log1p_capacity_long.csv",
    )
    providers["capacity"] = capacity_outputs
    providers["audit"] = pd.DataFrame(provider_audit_rows)
    providers["overlap_audit"] = pd.DataFrame(overlap_rows)
    providers["duplicate_rows"] = duplicate_rows[len(duplicate_rows) - 15 :]

    return {
        "demand_extent": demand_extent,
        "native_demand": native_demand,
        "harmonised_demand": harmonised_demand,
        "crosswalks": crosswalks,
        "provider_centres": provider_centres,
        "unique_provider_centres": unique_centres,
        "provider_support": provider_support,
        "demand_support_geometry": demand_support_geometry,
        "road_core_geometry": road_core_geometry,
        "routing_footprint": routing_footprint,
        "boundary_audit": context["boundary_audit"],
        "extent_audit": pd.DataFrame(extent_rows),
        "demand_audit": pd.DataFrame(demand_audit_rows),
        "conservation_audit": conservation,
        "duplicate_rows": duplicate_rows,
    }


def create_road_extract(build_root: Path, boundary_gpkg: Path) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    road_dir = build_root / ROAD_FOLDER
    staging = road_dir / ".road_bbox_staging.gpkg"
    output = road_dir / "OSMM_Highways_2021_SW_provider_centred_car_access.gpkg"
    buffer_layer = "routing_footprint_margin_5km"
    buffer_frame = gpd.read_file(boundary_gpkg, layer=buffer_layer)
    minx, miny, maxx, maxy = buffer_frame.total_bounds

    road_sources = sorted(ROAD_SOURCE_DIR.glob(ROADLINK_PATTERN))
    if not road_sources:
        raise FileNotFoundError(f"No OSMM RoadLink sources: {ROAD_SOURCE_DIR}")
    # Seed the staging GeoPackage with the buffer layer so subsequent -update
    # operations never create or overwrite an unrelated file.
    buffer_frame.to_file(staging, layer=buffer_layer, driver="GPKG")
    stage_sql = (
        "SELECT gml_id, operationalState, formOfWay, routeHierarchy, length, "
        "startGradeSeparation, endGradeSeparation, centrelineGeometry FROM RoadLink"
    )
    inventory_rows: list[dict[str, object]] = []
    for index, source in enumerate(road_sources, start=1):
        inventory_rows.append(
            {
                "source_layer": "RoadLink",
                "source_filename": source.name,
                "source_full_path": str(source),
                "size_bytes": source.stat().st_size,
                "sha256": sha256(source),
            }
        )
        if index == 1 or index % 10 == 0 or index == len(road_sources):
            print(f"ROAD_BBOX_STAGE RoadLink {index}/{len(road_sources)} {source.name}", flush=True)
        run_command(
            [
                "ogr2ogr",
                "--config",
                "OGR_SQLITE_SYNCHRONOUS",
                "OFF",
                "-f",
                "GPKG",
                "-unsetFieldWidth",
                "-update",
                "-append",
                "-gt",
                "65536",
                str(staging),
                f"/vsigzip/{source}",
                "-dialect",
                "OGRSQL",
                "-sql",
                stage_sql,
                "-spat",
                str(minx),
                str(miny),
                str(maxx),
                str(maxy),
                "-dim",
                "XY",
                "-nln",
                "RoadLink",
            ]
        )

    retained_sql = ",".join(f"'{value}'" for value in RETAINED_FORM_OF_WAY)
    sql = (
        f'SELECT r.* FROM "RoadLink" r, "{buffer_layer}" b '
        'WHERE ST_Intersects(r."centrelineGeometry", b.geom) '
        "AND r.operationalState = 'Open' "
        f"AND r.formOfWay IN ({retained_sql})"
    )
    print("ROAD_EXACT_FILTER provider-centred footprint Open six-formOfWay no-FerryLink", flush=True)
    run_command(
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
            str(output),
            str(staging),
            "-dialect",
            "SQLITE",
            "-sql",
            sql,
            "-nln",
            "RoadLink",
        ]
    )
    run_command(
        [
            "ogr2ogr",
            "-f",
            "GPKG",
            "-update",
            "-append",
            str(output),
            str(staging),
            buffer_layer,
            "-nln",
            "routing_footprint_provider_centred_margin_5km",
        ]
    )

    links = gpd.read_file(
        output,
        layer="RoadLink",
        engine="pyogrio",
        columns=[
            "gml_id",
            "operationalState",
            "formOfWay",
            "routeHierarchy",
            "length",
            "startGradeSeparation",
            "endGradeSeparation",
            "geometry",
        ],
    )
    if not len(links):
        raise AssertionError("Provider-centred filtered road extract is empty")
    if not links["operationalState"].eq("Open").all():
        raise AssertionError("Provider-centred road extract contains non-Open links")
    observed_forms = set(links["formOfWay"].dropna().unique())
    if observed_forms != set(RETAINED_FORM_OF_WAY):
        raise AssertionError(f"Unexpected retained formOfWay set: {sorted(observed_forms)}")
    if links["gml_id"].isna().any() or links["gml_id"].duplicated().any():
        raise AssertionError("Provider-centred road extract has missing or duplicate gml_id")
    if links.geometry.isna().any() or links.geometry.is_empty.any():
        raise AssertionError("Provider-centred road extract has null or empty geometry")
    if pd.to_numeric(links["length"], errors="coerce").isna().any():
        raise AssertionError("Provider-centred road extract has non-numeric lengths")

    with sqlite3.connect(output) as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    if integrity != "ok":
        raise AssertionError(f"Provider-centred GeoPackage integrity failure: {integrity}")

    for suffix in ("", "-shm", "-wal"):
        Path(str(staging) + suffix).unlink(missing_ok=True)
    return links, pd.DataFrame(inventory_rows)


def build_travel_graph(build_root: Path, links: gpd.GeoDataFrame, context: dict[str, object], providers):
    road_dir = build_root / ROAD_FOLDER
    qa_dir = build_root / QA_FOLDER
    speed = links["routeHierarchy"].map(SPEED_KMH)
    missing_speed = speed.isna()
    hierarchy = (
        links.assign(speed_kmh=speed, dft_table_label=links["routeHierarchy"].map(DFT_LABEL))
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
    hierarchy["total_length_km"] = hierarchy["total_length_m"] / 1000
    hierarchy["speed_status"] = np.where(hierarchy["speed_kmh"].isna(), "UNMATCHED", "matched")
    hierarchy["speed_source"] = "DfT Journey Time Statistics 2019 Table 2"
    hierarchy["speed_source_url"] = DFT_SOURCE_URL
    hierarchy["mapping_note"] = np.where(
        hierarchy["routeHierarchy"].eq("Local Road"),
        "OSMM Local Road mapped to DfT Local Street; user-confirmed naming correspondence",
        "exact routeHierarchy label match",
    )
    present = set(links["routeHierarchy"].dropna().unique())
    absent = pd.DataFrame(SPEED_PROFILE, columns=["routeHierarchy", "dft_table_label", "speed_kmh"])
    absent = absent.loc[~absent["routeHierarchy"].isin(present)].copy()
    absent["status"] = "defined in speed profile but absent from current provider-centred extract"
    write_csv(hierarchy, qa_dir / "route_hierarchy_speed_audit.csv")
    write_csv(
        hierarchy.loc[hierarchy["speed_kmh"].isna()].copy(),
        qa_dir / "route_hierarchy_unmatched.csv",
    )
    write_csv(absent, qa_dir / "speed_profile_categories_absent_from_extract.csv")
    if missing_speed.any():
        raise AssertionError(
            f"Unmatched routeHierarchy values: {links.loc[missing_speed, 'routeHierarchy'].value_counts(dropna=False).to_dict()}"
        )

    form_audit = (
        links.groupby("formOfWay", dropna=False)
        .agg(roadlink_count=("formOfWay", "size"), total_length_m=("length", "sum"))
        .reset_index()
        .sort_values("formOfWay")
    )
    form_audit["retained"] = True
    form_audit["selection_rule"] = "operationalState=Open and six retained car-access formOfWay categories"
    write_csv(form_audit, qa_dir / "road_form_of_way_audit.csv")

    length_m = pd.to_numeric(links["length"], errors="raise").to_numpy(dtype=float)
    speed_values = speed.to_numpy(dtype=float)
    time_min = (length_m / 1000.0) / speed_values * 60.0
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
    nodes, inverse = np.unique(np.vstack([start_key, end_key]), axis=0, return_inverse=True)
    edge_count = len(links)
    u, v = inverse[:edge_count], inverse[edge_count:]
    valid = (
        (u != v)
        & np.isfinite(length_m)
        & (length_m > 0)
        & np.isfinite(time_min)
        & (time_min > 0)
    )
    invalid_or_self_loop = int((~valid).sum())
    a, b = np.minimum(u[valid], v[valid]), np.maximum(u[valid], v[valid])
    valid_time = time_min[valid]
    edge_key = a.astype(np.int64) * len(nodes) + b.astype(np.int64)
    order = np.argsort(edge_key)
    edge_key, a, b, valid_time = edge_key[order], a[order], b[order], valid_time[order]
    first = np.r_[0, np.flatnonzero(np.diff(edge_key)) + 1]
    duplicate_collapsed = int(len(edge_key) - len(first))
    a, b = a[first], b[first]
    valid_time = np.minimum.reduceat(valid_time, first)
    graph = csr_matrix((valid_time, (a, b)), shape=(len(nodes), len(nodes)))
    graph = graph + graph.T
    if graph.nnz == 0 or not np.isfinite(graph.data).all() or (graph.data <= 0).any():
        raise AssertionError("Travel-time graph contains invalid weights")
    if (graph != graph.T).nnz or np.any(graph.diagonal()):
        raise AssertionError("Travel-time graph is not symmetric with zero diagonal")

    save_npz(road_dir / "road_graph_travel_time_min_provider_centred.npz", graph)
    np.save(road_dir / "road_nodes_xyz_provider_centred.npy", nodes)

    link_map = pd.DataFrame(
        {
            "gml_id": links["gml_id"].astype("string"),
            "routeHierarchy": links["routeHierarchy"].astype("string"),
            "formOfWay": links["formOfWay"].astype("string"),
            "length_m": length_m,
            "speed_kmh": speed_values,
            "time_min": time_min,
            "start_node_id": u,
            "end_node_id": v,
            "valid_graph_edge": valid,
        }
    )
    link_map_path = road_dir / "roadlink_travel_time_endpoint_map_provider_centred.parquet"
    if link_map_path.exists():
        raise FileExistsError(link_map_path)
    link_map.to_parquet(link_map_path, index=False)

    components, labels = connected_components(graph, directed=False)
    component_sizes = np.bincount(labels)
    values = triu(graph, k=1, format="csr").data
    outer_band = context["routing_footprint"].difference(context["road_core_geometry"])
    outer_index = links.sindex.query(outer_band, predicate="intersects")
    outer_count = int(len(np.unique(outer_index)))
    if outer_count <= 0:
        raise AssertionError("Provider-centred extract has no RoadLinks in the 5 km routing margin")

    graph_audit = pd.DataFrame(
        [
            {
                "input_filtered_open_roadlinks": len(links),
                "missing_speed_roadlinks": int(missing_speed.sum()),
                "invalid_or_self_loop_edges_removed": invalid_or_self_loop,
                "duplicate_endpoint_pairs_collapsed": duplicate_collapsed,
                "graph_nodes": graph.shape[0],
                "graph_edges_undirected": graph.nnz // 2,
                "graph_components": components,
                "largest_component_nodes": int(component_sizes.max()),
                "surface_grade_zero_nodes": int((nodes[:, 2] == 0).sum()),
                "graph_is_symmetric": (graph != graph.T).nnz == 0,
                "graph_has_zero_diagonal": not np.any(graph.diagonal()),
                "ferrylink_edges_used": 0,
                "time_min_min": float(values.min()),
                "time_min_median": float(np.median(values)),
                "time_min_p95": float(np.quantile(values, 0.95)),
                "time_min_max": float(values.max()),
                "roadlinks_intersecting_routing_margin_5km": outer_count,
                "edge_cost_definition": "length_km / DfT routeHierarchy speed_kmh * 60",
                "topology_definition": "endpoints keyed to 1 mm plus grade separation; undirected duplicate endpoint pairs retain minimum travel time",
                "used_in_od": False,
                "used_in_e2sfca": False,
            }
        ]
    )
    write_csv(graph_audit, qa_dir / "road_graph_provider_centred_audit.csv")

    surface_nodes = nodes[nodes[:, 2] == 0, :2] / 1000.0
    tree = cKDTree(surface_nodes)
    coverage_rows: list[dict[str, object]] = []

    demand_target = context["demand_extent"][2021]
    demand_points = demand_target.geometry.representative_point()
    demand_xy = np.column_stack([demand_points.x, demand_points.y])
    distance, _ = tree.query(demand_xy, k=1)
    coverage_rows.append(coverage_row("combined_2021_demand_support_representative_points", 2021, distance))

    for year in YEARS:
        capacity = providers["capacity"][year]
        capacity_xy = capacity[["representative_easting", "representative_northing"]].to_numpy(float)
        distance, _ = tree.query(capacity_xy, k=1)
        coverage_rows.append(coverage_row("provider_lsoa_capacity_representative_points", year, distance))
        combined = providers["combined"][year]
        exact_points = gpd.GeoDataFrame(
            combined[["lat", "long"]].copy(),
            geometry=gpd.points_from_xy(combined["long"], combined["lat"]),
            crs="EPSG:4326",
        ).to_crs("EPSG:27700")
        exact_xy = np.column_stack([exact_points.geometry.x, exact_points.geometry.y])
        distance, _ = tree.query(exact_xy, k=1)
        coverage_rows.append(coverage_row("combined_eligible_charity_exact_points", year, distance))
    coverage = pd.DataFrame(coverage_rows)
    write_csv(coverage, qa_dir / "road_coverage_audit.csv")
    return {
        "graph_audit": graph_audit,
        "coverage_audit": coverage,
        "hierarchy_audit": hierarchy,
        "form_audit": form_audit,
    }


def coverage_row(role: str, year: int, distance: np.ndarray) -> dict[str, object]:
    return {
        "spatial_role": role,
        "year": year,
        "rows": len(distance),
        "minimum_nearest_surface_node_m": float(np.min(distance)),
        "median_nearest_surface_node_m": float(np.median(distance)),
        "p95_nearest_surface_node_m": float(np.quantile(distance, 0.95)),
        "p99_nearest_surface_node_m": float(np.quantile(distance, 0.99)),
        "maximum_nearest_surface_node_m": float(np.max(distance)),
        "over_2000m": int((distance > 2000).sum()),
        "coverage_complete": bool(np.isfinite(distance).all()),
        "snap_method_status": "QA only; final OD not run and nearest-link access still required",
    }


def merge_build(build_root: Path) -> None:
    all_files = [path for path in build_root.rglob("*") if path.is_file()]
    conflicts = [OUTPUT_ROOT / path.relative_to(build_root) for path in all_files if (OUTPUT_ROOT / path.relative_to(build_root)).exists()]
    if conflicts:
        raise FileExistsError(f"Refusing to overwrite Travel_Time files: {conflicts}")
    for path in sorted(all_files, key=lambda value: len(value.parts)):
        destination = OUTPUT_ROOT / path.relative_to(build_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        path.rename(destination)


def main() -> None:
    ensure_output_preflight()
    required = [
        BASE_RULES_PATH,
        BASE.ICB_PATH,
        *BASE.LSOA_PATHS.values(),
        *BASE.CENSUS_PATHS.values(),
        CHARITY_STAGE,
        *CHARITY_FINAL_PATHS.values(),
        *COVARIATE_PATHS.values(),
        SCREENING_GRAPH_PATH,
        SCREENING_NODES_PATH,
        SCREENING_AUDIT_PATH,
        *sorted(ROAD_SOURCE_DIR.glob(ROADLINK_PATTERN)),
        *sorted(ROAD_SOURCE_DIR.glob(FERRYLINK_PATTERN)),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen inputs: {missing}")

    protected_roots = [
        PACKAGE / "halo-20km",
        PACKAGE / "5-Step-Analysis",
        PACKAGE / "Road_network(Distance)",
        CHARITY_PACKAGE / "07_final_outputs",
    ]
    protected_files: list[Path] = []
    for root in protected_roots:
        protected_files.extend(stable_files(root))
    protected_before = snapshot(protected_files)

    build_root = Path(tempfile.mkdtemp(prefix=".Travel_Time-build-", dir=str(PACKAGE)))
    for folder in (ROAD_FOLDER, DEMAND_FOLDER, SUPPLY_FOLDER, BOUNDARY_FOLDER, QA_FOLDER):
        (build_root / folder).mkdir(parents=True, exist_ok=True)
    print(f"TRAVEL_TIME_BUILD_ROOT {build_root}", flush=True)

    try:
        context = create_base_context()
        providers = prepare_network_screened_providers(build_root, context)
        support = create_provider_centred_support_and_demand(build_root, context, providers)
        boundary_gpkg = build_root / BOUNDARY_FOLDER / "travel_time_extents.gpkg"
        links, road_inventory = create_road_extract(build_root, boundary_gpkg)
        road = build_travel_graph(build_root, links, support, providers)

        duplicate_audit = pd.DataFrame(support["duplicate_rows"])
        if (duplicate_audit["duplicate_rows"] != 0).any():
            raise AssertionError("Duplicate QA failed")
        overlap_audit = providers["overlap_audit"]
        if not overlap_audit["overlap_pass"].all():
            raise AssertionError("Provider-demand overlap QA failed")

        qa_dir = build_root / QA_FOLDER
        write_csv(support["boundary_audit"], qa_dir / "boundary_repair_audit.csv")
        write_csv(support["extent_audit"], qa_dir / "extent_lsoa_audit.csv")
        write_csv(support["demand_audit"], qa_dir / "demand_completeness_audit.csv")
        write_csv(support["conservation_audit"], qa_dir / "harmonisation_conservation_audit.csv")
        write_csv(providers["screen_audit"], qa_dir / "external_provider_network_screen_audit.csv")
        write_csv(providers["audit"], qa_dir / "provider_completeness_audit.csv")
        write_csv(overlap_audit, qa_dir / "provider_demand_overlap_audit.csv")
        write_csv(duplicate_audit, qa_dir / "duplicates_audit.csv")
        write_csv(road_inventory, qa_dir / "road_source_files_inventory_sha256.csv", float_format=None)

        source_manifest = snapshot(required)
        source_manifest["source_role"] = source_manifest["path"].map(
            lambda value: (
                "Data Spine national geocoded care-strict stage"
                if value == str(CHARITY_STAGE)
                else (
                    "authoritative internal final charity input"
                    if value in {str(path) for path in CHARITY_FINAL_PATHS.values()}
                    else (
                        "December-2021 OS MasterMap Highways source"
                        if str(ROAD_SOURCE_DIR) in value
                        else "frozen project input"
                    )
                )
            )
        )
        write_csv(source_manifest, qa_dir / "source_manifest_sha256.csv", float_format=None)

        protected_after = snapshot(protected_files)
        protected = protected_before.merge(
            protected_after,
            on="path",
            suffixes=("_before", "_after"),
            validate="one_to_one",
        )
        protected["unchanged"] = (
            protected["size_bytes_before"].eq(protected["size_bytes_after"])
            & protected["mtime_ns_before"].eq(protected["mtime_ns_after"])
            & protected["sha256_before"].eq(protected["sha256_after"])
        )
        if not protected["unchanged"].all():
            changed = protected.loc[~protected["unchanged"], "path"].tolist()
            raise AssertionError(f"Protected distance workflow files changed: {changed}")
        write_csv(protected, qa_dir / "protected_files_unchanged_audit.csv", float_format=None)

        nonexecution = pd.DataFrame(
            [
                {
                    "od_matrix_files_created": 0,
                    "e2sfca_result_files_created": 0,
                    "bounded_external_provider_reachability_screen_run": True,
                    "pairwise_provider_demand_matrix_materialised": False,
                    "step1_to_step5_files_modified": int((~protected["unchanged"]).sum()),
                    "status": "data_and_road_preparation_only",
                    "pass": True,
                }
            ]
        )
        write_csv(nonexecution, qa_dir / "analysis_nonexecution_audit.csv")

        method_manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "data_and_travel_time_road_preparation_only_not_od_not_e2sfca",
            "output_root": str(OUTPUT_ROOT),
            "main_catchment_minutes": MAIN_CATCHMENT_MIN,
            "time_bands_minutes": list(TIME_BANDS_MIN),
            "primary_weights_reserved_for_future_e2sfca": list(PRIMARY_WEIGHTS),
            "maximum_dft_default_speed_kmh": MAX_CAR_SPEED_KMH,
            "external_provider_candidate_distance_m": PROVIDER_CANDIDATE_DISTANCE_M,
            "provider_centred_demand_candidate_radius_m": PROVIDER_CENTRED_DEMAND_DISTANCE_M,
            "routing_margin_m": ROUTING_MARGIN_M,
            "off_network_connector_speed_kmh": CONNECTOR_SPEED_KMH,
            "charity_authority": str(CHARITY_PACKAGE),
            "charity_rules": [
                "authoritative internal 2001/2011/2021 final CSVs",
                "national care_strict stage for external candidates",
                "historical provider coordinates at each Census date",
                "2021 constant-price income",
                "missing income retained as missing and recorded zero retained",
                "charity-level log1p(income_2021_gbp) before provider-LSOA aggregation",
            ],
            "external_provider_network_screen": [
                "42 km Euclidean candidate filter uses each charity's historical exact point",
                "30-minute network screen uses the assigned fixed-2021 provider-LSOA representative point",
                "internal demand sources are the 3,411 fixed-2021 LSOA representative points",
                "screening reuses the audited 45 km DfT travel-time graph read-only",
                "nearest valid surface-node connectors are converted at 25.6 km/h and included at both ends",
                "one virtual multi-source shortest-path screen is run; no pairwise OD matrix is created",
            ],
            "demand_rules": [
                "Care50 count and population aged 5+ from the existing three Census definitions",
                "2001/2011 source candidates are the minimal topological intersection closure of selected fixed-2021 target polygons",
                "2001 and 2011 positive-area intersections normalised within source LSOA",
                "counts allocated before rates are recalculated",
                "2021 identity crosswalk",
                "external demand candidate areas are 41.9 km buffers around network-reachable external provider-LSOA representative points",
                "the union of provider-centred areas across 2001, 2011 and 2021 defines one fixed support target",
                "no complete 83.8 km demand ring is constructed",
            ],
            "road_rules": [
                "December-2021 OS MasterMap Highways source order 3004697",
                "exact whole-link intersection with the irregular provider-centred routing footprint",
                "road core is the union of study/provider candidate area and provider-centred demand-support areas",
                "a 5 km routing margin is applied to the irregular core; no uniform 90 km study-area buffer is constructed",
                "operationalState=Open",
                "six retained car-access formOfWay categories",
                "FerryLink excluded, matching the current 45 km cleaned network",
                "speed assigned only from routeHierarchy using DfT defaults",
                "Local Road mapped to DfT Local Street at 25.6 km/h by user-confirmed naming correspondence",
                "time_min = length_km / speed_kmh * 60",
                "endpoints keyed to 1 mm plus grade separation",
                "undirected duplicate endpoint pairs retain minimum travel time",
            ],
            "explicitly_not_run": [
                "full provider-demand OD matrix",
                "final E2SFCA OD calculation",
                "E2SFCA",
                "mismatch classification",
                "longitudinal trajectories",
                "Bi-LISA",
                "BYM2",
            ],
        }
        write_json(method_manifest, build_root / "METHOD_MANIFEST.json")

        graph_row = road["graph_audit"].iloc[0]
        report = [
            "# 30-minute travel-time E2SFCA preparation audit",
            "",
            "Status: data and road preparation only; no OD matrix or E2SFCA was run.",
            "",
            "## Extents",
            "",
            f"- External provider candidate halo: {PROVIDER_CANDIDATE_DISTANCE_M/1000:.1f} km.",
            f"- Demand candidates: {PROVIDER_CENTRED_DEMAND_DISTANCE_M/1000:.1f} km around network-reachable external provider LSOA points.",
            f"- Road extent: irregular union footprint plus {ROUTING_MARGIN_M/1000:.1f} km routing margin; no complete 84/90 km ring.",
            "",
            "## Provider and demand preparation",
            "",
        ]
        for row in providers["audit"].itertuples(index=False):
            report.append(
                f"- {row.year}: {row.internal_authoritative_charities:,} internal; "
                f"{row.external_42km_candidate_charities:,} external Euclidean candidates, "
                f"{row.external_network_reachable_charities_30min:,} retained at <=30 min; "
                f"{row.with_income_2021_gbp:,} with usable 2021-price income."
            )
        for row in support["demand_audit"].itertuples(index=False):
            report.append(
                f"- {row.year} demand: {row.native_rows:,} native LSOAs to "
                f"{row.harmonised_rows:,} fixed-2021 support LSOAs; missing counts "
                f"{row.missing_harmonised_counts:,}."
            )
        report.extend(
            [
                "",
                "## Road preparation",
                "",
                f"- Selected Open RoadLinks: {int(graph_row.input_filtered_open_roadlinks):,}.",
                f"- Graph: {int(graph_row.graph_nodes):,} nodes and "
                f"{int(graph_row.graph_edges_undirected):,} undirected edges.",
                f"- Missing routeHierarchy speeds: {int(graph_row.missing_speed_roadlinks):,}.",
                f"- RoadLinks intersecting the 5 km routing margin: "
                f"{int(graph_row.roadlinks_intersecting_routing_margin_5km):,}.",
                "",
                "## Protection and scope",
                "",
                f"- Protected old files SHA-256 checked unchanged: {len(protected):,}.",
                "- No old halo, distance graph, OD cache, Step 1-5 output or Data Spine final CSV was overwritten.",
                "- Only the bounded external-provider reachability screen was run; no full OD matrix or E2SFCA was run.",
                "",
            ]
        )
        report_path = build_root / "DATA_AUDIT_REPORT.md"
        if report_path.exists():
            raise FileExistsError(report_path)
        report_path.write_text("\n".join(report), encoding="utf-8")

        merge_build(build_root)

        manifest_rows: list[dict[str, object]] = []
        for path in sorted(OUTPUT_ROOT.rglob("*")):
            if not path.is_file() or path.name in {".DS_Store", "output_manifest.csv"}:
                continue
            manifest_rows.append(
                {
                    "relative_path": str(path.relative_to(OUTPUT_ROOT)),
                    "full_path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
        write_csv(
            pd.DataFrame(manifest_rows),
            OUTPUT_ROOT / "output_manifest.csv",
            float_format=None,
        )
        shutil.rmtree(build_root, ignore_errors=True)
        print("TRAVEL_TIME_PACKAGE_READY", OUTPUT_ROOT, flush=True)
        print(providers["screen_audit"].to_string(index=False), flush=True)
        print(support["extent_audit"].to_string(index=False), flush=True)
        print(support["demand_audit"].to_string(index=False), flush=True)
        print(providers["audit"].to_string(index=False), flush=True)
        print(road["graph_audit"].to_string(index=False), flush=True)
    except Exception:
        print(f"BUILD_FAILED_PARTIAL_DIRECTORY {build_root}", flush=True)
        raise


if __name__ == "__main__":
    main()
