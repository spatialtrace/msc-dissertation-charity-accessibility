#!/usr/bin/env python3
"""Prepare the isolated external-halo data package for a 20 km E2SFCA.

This script performs data collection/organisation only. It deliberately does
not calculate OD matrices, E2SFCA results, HP-LA outputs, trajectories,
Bi-LISA outputs or BYM2 models.

The implementation reuses the current dissertation definitions:

* seven April-2023 ICB study boundary;
* exact provider points in the external 20 km ring;
* representative-point LSOA inclusion in the external 40 km ring for competing demand;
* the existing E/W Census source tables and Care50/population-5+ definitions;
* count-first, source-normalised areal harmonisation to 2021 LSOAs;
* Data Spine charity rebuild v2 before its internal geography filter;
* its presence, care_strict, historical-address, finance and CPIH decisions;
* charity-level log1p(income_2021_gbp); and
* the Step-1 coordinate-to-2021-LSOA assignment with nearest-boundary fallback;
* the same December-2021 OS MasterMap Highways source, spatially extended to
  the study boundary plus 45 km. The current final road output is then
  post-filtered by ``filter_road_network_45km.py`` to the six user-selected
  Open car-access ``formOfWay`` categories, with no FerryLinks.

Outputs are written to a temporary directory and moved into the requested
location only after all assertions pass. The existing extension package is then
replaced in place; files outside that package are never written.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import shapely
from scipy.sparse import csr_matrix, save_npz
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


# The host enables PROJ network lookups by default. In the restricted runtime a
# missing OSTN15 grid then yields infinite coordinates instead of selecting the
# installed offline EPSG fallback. Disable network lookup explicitly so the
# deterministic installed transformation is used and no external source is
# introduced.
pyproj.network.set_network_enabled(False)


ROOT = Path(os.environ["DISSERTATION_DATA_ROOT"]).expanduser().resolve()
PACKAGE = ROOT / "final_data_and_analysis"
OUTPUT_ROOT = PACKAGE / "halo-20km"
YEARS = (2001, 2011, 2021)
E2SFCA_CATCHMENT_M = 20_000.0
PROVIDER_HALO_DISTANCE_M = 20_000.0
DEMAND_HALO_DISTANCE_M = 40_000.0
ROAD_BUFFER_DISTANCE_M = 45_000.0
MAX_CROSSWALK_EDGE_GAP_M = 1.0
EXPECTED_INTERNAL_LSOAS = {2001: 3230, 2011: 3285, 2021: 3411}

ICB_PATH = ROOT / "分析历史/data/Boundary_file/South_West_ICBs.shp"
LSOA_PATHS = {
    2001: ROOT
    / "分析历史/varaibles/LSOA_boundaries/LSOA_Dec_2001_EW_BFC_2022_2626872798174610206/LSOA_2001_EW_BFC_V2.shp",
    2011: ROOT
    / "分析历史/varaibles/LSOA_boundaries/Lower_layer_Super_Output_Areas_Dec_2011_Boundaries_Full_Clipped_BFC_EW_V3_2022_4674653630540948161/LSOA_2011_EW_BFC_V3.shp",
    2021: ROOT
    / "分析历史/varaibles/LSOA_boundaries/Lower_layer_Super_Output_Areas_December_2021_Boundaries_EW_BSC_V4_-5098480560471456651/LSOA_2021_EW_BSC_V4.shp",
}
LSOA_CODE_COLUMNS = {2001: "LSOA01CD", 2011: "LSOA11CD", 2021: "LSOA21CD"}
LSOA_NAME_COLUMNS = {2001: "LSOA01NM", 2011: "LSOA11NM", 2021: "LSOA21NM"}

CENSUS_PATHS = {
    "2001_care": ROOT / "分析历史/final variables/2001/2001_UV021_LSOA.csv",
    "2001_age": ROOT / "分析历史/final variables/2001/2001_UV004_LSOA.csv",
    "2011_care": ROOT / "分析历史/final variables/2011/2011_QS301EW_LSOA.csv",
    "2011_age": ROOT / "分析历史/final variables/2011/2011_QS103EW_LSOA.csv",
    "2021_care": ROOT / "分析历史/final variables/2021/2021_TS039_LSOA.csv",
    "2021_age": ROOT / "分析历史/final variables/2021/2021_TS007A_LSOA.csv",
}

CHARITY_PACKAGE = PACKAGE / "Data_Spine/charity_rebuild_v2"
CHARITY_STAGE = CHARITY_PACKAGE / "05_geocoding/active_care_geocoded.parquet"
CHARITY_FINAL_PATHS = {
    year: CHARITY_PACKAGE / f"07_final_outputs/{year}charity.csv" for year in YEARS
}
COVARIATE_PATHS = {year: PACKAGE / f"covariates/{year}.csv" for year in YEARS}

ROAD_SOURCE_DIR = (
    ROOT
    / "分析历史/data/Road/Download_OSMM_Highways_2021_GB_3004697"
    / "MasterMap Highways Network_roads_6440382"
)
ROAD_PREPARATION_SCRIPT = ROOT / "分析历史/data/Road/prepare_osmm_highways_2021_sw.py"
ROAD_ORDER_CONTENTS = (
    ROOT
    / "分析历史/data/Road/Download_OSMM_Highways_2021_GB_3004697"
    / "contents_order_3004697.txt"
)
ROAD_ORDER_CITATIONS = (
    ROOT
    / "分析历史/data/Road/Download_OSMM_Highways_2021_GB_3004697"
    / "citations_orders_3004697.txt"
)
ROAD_SOURCE_PATTERNS = {
    "RoadLink": "Highways_Roads_RoadLink_FULL_*.gml.gz",
    "FerryLink": "Highways_Roads_FerryLink_FULL_*.gml.gz",
}
ROAD_GEOMETRY_COLUMNS = {
    "RoadLink": "centrelineGeometry",
    "FerryLink": "centrelineGeometry",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot(paths: list[Path]) -> pd.DataFrame:
    rows = []
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


def write_csv(frame: pd.DataFrame, path: Path, *, float_format: str | None = "%.15g") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    frame.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
        float_format=float_format,
    )


def run_command(args: list[str]) -> None:
    subprocess.run(args, check=True)


def prepare_road_network(
    build_root: Path,
    study_union,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Create the initial 45 km Step-1-faithful extract and routing cache.

    This function preserves the pre-filter extraction lineage. The current
    final extension road files additionally require the documented
    ``filter_road_network_45km.py`` post-filter; direct output from this helper
    is not the final car-access road selection.
    """

    road_dir = build_root / "road_network"
    road_dir.mkdir(parents=True, exist_ok=True)
    staging = road_dir / ".road_bbox_staging.gpkg"
    road_gpkg = road_dir / "OSMM_Highways_2021_SW_45km.gpkg"
    graph_path = road_dir / "road_graph_open_with_ferries_45km.npz"
    nodes_path = road_dir / "road_nodes_xyz_45km.npy"
    buffer_layer = "study_area_buffer_45km"

    road_buffer = study_union.buffer(ROAD_BUFFER_DISTANCE_M)
    if not road_buffer.is_valid:
        road_buffer = road_buffer.make_valid()
    buffer_frame = gpd.GeoDataFrame(
        {
            "road_buffer_distance_m": [ROAD_BUFFER_DISTANCE_M],
            "definition": ["dissolved seven-ICB study boundary plus 45000 m"],
        },
        geometry=[road_buffer],
        crs="EPSG:27700",
    )
    buffer_frame.to_file(staging, layer=buffer_layer, driver="GPKG")
    minx, miny, maxx, maxy = buffer_frame.total_bounds

    source_inventory_rows = []
    for layer, pattern in ROAD_SOURCE_PATTERNS.items():
        files = sorted(ROAD_SOURCE_DIR.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No current OSMM source files for {layer}: {pattern}")
        for index, source in enumerate(files, start=1):
            stat = source.stat()
            source_inventory_rows.append(
                {
                    "source_layer": layer,
                    "source_filename": source.name,
                    "source_full_path": str(source),
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
            if index == 1 or index % 10 == 0 or index == len(files):
                print(
                    f"ROAD_BBOX_STAGE {layer} {index}/{len(files)} {source.name}",
                    flush=True,
                )
            run_command(
                [
                    "ogr2ogr",
                    "--config",
                    "OGR_SQLITE_SYNCHRONOUS",
                    "OFF",
                    "-f",
                    "GPKG",
                    "-update",
                    "-append",
                    "-gt",
                    "65536",
                    str(staging),
                    f"/vsigzip/{source}",
                    layer,
                    "-spat",
                    str(minx),
                    str(miny),
                    str(maxx),
                    str(maxy),
                    "-dim",
                    "XY",
                    "-nln",
                    layer,
                ]
            )

    for index, layer in enumerate(ROAD_SOURCE_PATTERNS):
        args = [
            "ogr2ogr",
            "--config",
            "OGR_SQLITE_SYNCHRONOUS",
            "OFF",
            "-f",
            "GPKG",
        ]
        if index:
            args.extend(["-update", "-append"])
        args.extend(
            [
                "-gt",
                "65536",
                str(road_gpkg),
                str(staging),
                "-dialect",
                "SQLITE",
                "-sql",
                (
                    f'SELECT r.* FROM "{layer}" r, "{buffer_layer}" b '
                    f'WHERE ST_Intersects(r."{ROAD_GEOMETRY_COLUMNS[layer]}", b.geom)'
                ),
                "-nln",
                layer,
            ]
        )
        print(f"ROAD_EXACT_CLIP {layer}", flush=True)
        run_command(args)

    run_command(
        [
            "ogr2ogr",
            "-f",
            "GPKG",
            "-update",
            "-append",
            str(road_gpkg),
            str(staging),
            buffer_layer,
        ]
    )

    links = gpd.read_file(
        road_gpkg,
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
    roadlink_rows = len(links)
    open_links = links.loc[links["operationalState"].eq("Open")].reset_index(drop=True)
    if not len(open_links):
        raise AssertionError("The 45 km road extract has no Open RoadLink rows")

    geometries = open_links.geometry.array
    start = shapely.get_point(geometries, 0)
    end = shapely.get_point(geometries, -1)
    start_key = np.column_stack(
        [
            np.rint(shapely.get_x(start) * 1000).astype(np.int64),
            np.rint(shapely.get_y(start) * 1000).astype(np.int64),
            open_links["startGradeSeparation"].fillna(0).to_numpy(dtype=np.int64),
        ]
    )
    end_key = np.column_stack(
        [
            np.rint(shapely.get_x(end) * 1000).astype(np.int64),
            np.rint(shapely.get_y(end) * 1000).astype(np.int64),
            open_links["endGradeSeparation"].fillna(0).to_numpy(dtype=np.int64),
        ]
    )
    weights = open_links["length"].to_numpy(dtype=float)

    ferries = gpd.read_file(road_gpkg, layer="FerryLink", engine="pyogrio").to_crs(27700)
    ferry_rows = len(ferries)
    if ferry_rows:
        ferry_geometry = ferries.geometry.array
        ferry_start = shapely.get_point(ferry_geometry, 0)
        ferry_end = shapely.get_point(ferry_geometry, -1)
        start_key = np.vstack(
            [
                start_key,
                np.column_stack(
                    [
                        np.rint(shapely.get_x(ferry_start) * 1000).astype(np.int64),
                        np.rint(shapely.get_y(ferry_start) * 1000).astype(np.int64),
                        np.zeros(ferry_rows, dtype=np.int64),
                    ]
                ),
            ]
        )
        end_key = np.vstack(
            [
                end_key,
                np.column_stack(
                    [
                        np.rint(shapely.get_x(ferry_end) * 1000).astype(np.int64),
                        np.rint(shapely.get_y(ferry_end) * 1000).astype(np.int64),
                        np.zeros(ferry_rows, dtype=np.int64),
                    ]
                ),
            ]
        )
        weights = np.concatenate([weights, ferries.length.to_numpy(dtype=float)])

    nodes, inverse = np.unique(
        np.vstack([start_key, end_key]), axis=0, return_inverse=True
    )
    edge_count = len(weights)
    u, v = inverse[:edge_count], inverse[edge_count:]
    keep = (u != v) & np.isfinite(weights) & (weights > 0)
    u, v, weights = u[keep], v[keep], weights[keep]
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
    a, b = a[first], b[first]
    weights = np.minimum.reduceat(weights, first)
    graph = csr_matrix((weights, (a, b)), shape=(len(nodes), len(nodes)))
    graph = graph + graph.T
    save_npz(graph_path, graph)
    np.save(nodes_path, nodes)

    component_count, component_labels = connected_components(graph, directed=False)
    component_sizes = np.bincount(component_labels)
    largest_component = int(component_sizes.argmax())
    outer_40_45 = study_union.buffer(ROAD_BUFFER_DISTANCE_M).difference(
        study_union.buffer(DEMAND_HALO_DISTANCE_M)
    )
    outer_frame = gpd.GeoDataFrame(geometry=[outer_40_45], crs="EPSG:27700")
    outer_links = gpd.read_file(
        road_gpkg,
        layer="RoadLink",
        engine="pyogrio",
        columns=[],
        mask=outer_frame,
    )
    if not len(outer_links):
        raise AssertionError("The road extract contains no RoadLink in the 40-45 km band")

    form_of_way = (
        open_links.groupby("formOfWay", dropna=False)
        .size()
        .rename("open_roadlinks")
        .reset_index()
        .sort_values("formOfWay", na_position="last")
        .reset_index(drop=True)
    )
    form_of_way["included_in_graph"] = True
    form_of_way["selection_rule"] = (
        "all operationalState=Open RoadLink categories, matching executed Step-1 cache"
    )

    bounds = links.total_bounds
    road_audit = pd.DataFrame(
        [
            {
                "road_source": "OS MasterMap Highways Network December 2021 existing local order 3004697",
                "study_boundary_definition": "dissolved seven April-2023 South West ICB polygons",
                "road_buffer_distance_m": ROAD_BUFFER_DISTANCE_M,
                "roadlink_rows_intersecting_buffer": roadlink_rows,
                "open_roadlink_rows_used_in_graph": len(open_links),
                "nonopen_roadlink_rows_not_used_in_graph": roadlink_rows - len(open_links),
                "ferrylink_rows_used_in_graph": ferry_rows,
                "roadlinks_intersecting_outer_40_45km_band": len(outer_links),
                "graph_nodes": graph.shape[0],
                "graph_edges_undirected": graph.nnz // 2,
                "graph_components": component_count,
                "largest_component_id": largest_component,
                "largest_component_nodes": int(component_sizes[largest_component]),
                "surface_grade_zero_nodes": int((nodes[:, 2] == 0).sum()),
                "minimum_easting_m": float(bounds[0]),
                "minimum_northing_m": float(bounds[1]),
                "maximum_easting_m": float(bounds[2]),
                "maximum_northing_m": float(bounds[3]),
                "graph_method": (
                    "Open RoadLink endpoints keyed to 1 mm plus grade separation; FerryLink "
                    "endpoints at grade 0; undirected duplicate pairs retain minimum length"
                ),
                "road_type_rule_status": (
                    "later road_type_filtered GPKG not applied because it was not used by the "
                    "currently executed Step-1 cache"
                ),
                "used_in_od": False,
                "used_in_e2sfca": False,
            }
        ]
    )

    for suffix in ("", "-shm", "-wal"):
        Path(str(staging) + suffix).unlink(missing_ok=True)
    return (
        road_audit,
        form_of_way,
        pd.DataFrame(source_inventory_rows),
        nodes,
    )


def road_snap_audit(
    road_nodes: np.ndarray,
    demand_target: gpd.GeoDataFrame,
    charity_outputs: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    surface = road_nodes[:, 2] == 0
    if not surface.any():
        raise AssertionError("The road graph has no grade-zero nodes")
    tree = cKDTree(road_nodes[surface, :2] / 1000.0)
    rows = []

    demand_points = demand_target.geometry.representative_point()
    demand_xy = np.column_stack([demand_points.x, demand_points.y])
    distance, _ = tree.query(demand_xy, k=1)
    rows.append(
        {
            "spatial_role": "external_competing_demand_2021_lsoa_representative_points",
            "year": 2021,
            "rows": len(demand_target),
            "missing_coordinates": 0,
            "minimum_nearest_surface_node_m": float(np.min(distance)),
            "median_nearest_surface_node_m": float(np.median(distance)),
            "p95_nearest_surface_node_m": float(np.quantile(distance, 0.95)),
            "maximum_nearest_surface_node_m": float(np.max(distance)),
            "used_in_od": False,
        }
    )

    for year in YEARS:
        frame = charity_outputs[year]
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


def reuse_existing_45km_road_network(
    build_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Copy the already generated and audited 45 km road preparation unchanged."""

    source_dir = OUTPUT_ROOT / "road_network"
    source_audit = OUTPUT_ROOT / "qa/road_network_45km_audit.csv"
    source_form = OUTPUT_ROOT / "qa/road_form_of_way_audit.csv"
    source_inventory = OUTPUT_ROOT / "qa/road_source_files_inventory.csv"
    required = [
        source_dir / "OSMM_Highways_2021_SW_45km.gpkg",
        source_dir / "road_graph_open_with_ferries_45km.npz",
        source_dir / "road_nodes_xyz_45km.npy",
        source_audit,
        source_form,
        source_inventory,
    ]
    if missing := [str(path) for path in required if not path.is_file()]:
        raise FileNotFoundError(f"Cannot reuse current 45 km road preparation: {missing}")

    audit = pd.read_csv(source_audit)
    if len(audit) != 1 or not np.isclose(
        pd.to_numeric(audit["road_buffer_distance_m"], errors="raise").iloc[0],
        ROAD_BUFFER_DISTANCE_M,
    ):
        raise AssertionError("Existing extension road audit is not the requested 45 km extract")
    if int(audit["roadlinks_intersecting_outer_40_45km_band"].iloc[0]) <= 0:
        raise AssertionError("Existing extension road audit does not prove 40-45 km coverage")
    if (
        audit["used_in_od"].astype("string").str.lower().eq("true").any()
        or audit["used_in_e2sfca"].astype("string").str.lower().eq("true").any()
    ):
        raise AssertionError("Existing extension road preparation has an unexpected analysis flag")

    destination_dir = build_root / "road_network"
    for source in sorted(source_dir.rglob("*")):
        if not source.is_file():
            continue
        destination = destination_dir / source.relative_to(source_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if sha256(source) != sha256(destination):
            raise AssertionError(f"45 km road reuse copy failed SHA-256: {source}")

    nodes = np.load(source_dir / "road_nodes_xyz_45km.npy")
    return (
        audit,
        pd.read_csv(source_form),
        pd.read_csv(source_inventory),
        nodes,
    )


def repair_lsoa_layer(year: int) -> tuple[gpd.GeoDataFrame, dict[str, int]]:
    code = LSOA_CODE_COLUMNS[year]
    name = LSOA_NAME_COLUMNS[year]
    raw = gpd.read_file(LSOA_PATHS[year]).to_crs("EPSG:27700")
    required = {code, name, "geometry"}
    if missing := required - set(raw.columns):
        raise AssertionError(f"{year} boundary missing fields: {sorted(missing)}")
    layer = raw[[code, name, "geometry"]].rename(
        columns={code: "lsoa_code", name: "lsoa_name"}
    )
    layer["lsoa_code"] = layer["lsoa_code"].astype("string").str.strip()
    invalid_before = int((~layer.geometry.is_valid).sum())
    layer = layer.loc[layer.geometry.notna() & ~layer.geometry.is_empty].copy()
    invalid = ~layer.geometry.is_valid
    if invalid.any():
        layer.loc[invalid, "geometry"] = layer.loc[invalid, "geometry"].make_valid()
    layer = layer.explode(index_parts=False, ignore_index=True)
    layer = layer.loc[layer.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    layer = layer.dissolve(by="lsoa_code", as_index=False, aggfunc="first")
    layer = layer[["lsoa_code", "lsoa_name", "geometry"]].sort_values("lsoa_code")
    layer = layer.reset_index(drop=True)
    if not layer["lsoa_code"].is_unique:
        raise AssertionError(f"{year} repaired boundary codes are not unique")
    if not layer.geometry.is_valid.all():
        raise AssertionError(f"{year} repaired boundary still has invalid geometry")
    return layer, {
        "year": year,
        "raw_rows": len(raw),
        "clean_rows": len(layer),
        "invalid_before": invalid_before,
        "invalid_after": int((~layer.geometry.is_valid).sum()),
    }


def extract_native_demand(year: int, keep: set[str], names: pd.DataFrame) -> pd.DataFrame:
    if year == 2001:
        care_raw = pd.read_csv(CENSUS_PATHS["2001_care"], low_memory=False)
        age_raw = pd.read_csv(CENSUS_PATHS["2001_age"], low_memory=False)
        care = care_raw.loc[
            care_raw["C_CARER_NAME"].eq(
                "Provides 50 or more hours unpaid care a week"
            ),
            ["GEOGRAPHY_CODE", "OBS_VALUE"],
        ].rename(columns={"GEOGRAPHY_CODE": "lsoa_code", "OBS_VALUE": "care50_num"})
        age = age_raw.loc[
            age_raw["C_AGE_NAME"].eq("Age 5 plus"),
            ["GEOGRAPHY_CODE", "OBS_VALUE"],
        ].rename(
            columns={"GEOGRAPHY_CODE": "lsoa_code", "OBS_VALUE": "population_5plus"}
        )
        care_table = "UV021"
        population_table = "UV004"
        population_definition = "C_AGE_NAME == Age 5 plus"
        care_definition = "C_CARER_NAME == Provides 50 or more hours unpaid care a week"
    elif year == 2011:
        care_raw = pd.read_csv(CENSUS_PATHS["2011_care"], low_memory=False)
        age_raw = pd.read_csv(CENSUS_PATHS["2011_age"], low_memory=False)
        care = care_raw[["GeographyCode", "QS301EW0005"]].rename(
            columns={"GeographyCode": "lsoa_code", "QS301EW0005": "care50_num"}
        )
        age = age_raw[["GeographyCode", "QS103EW0001", *[f"QS103EW{i:04d}" for i in range(2, 7)]]].copy()
        age["population_5plus"] = pd.to_numeric(age["QS103EW0001"], errors="raise")
        for column in [f"QS103EW{i:04d}" for i in range(2, 7)]:
            age["population_5plus"] -= pd.to_numeric(age[column], errors="raise")
        age = age[["GeographyCode", "population_5plus"]].rename(
            columns={"GeographyCode": "lsoa_code"}
        )
        care_table = "QS301EW"
        population_table = "QS103EW"
        population_definition = "QS103EW0001 minus QS103EW0002:QS103EW0006"
        care_definition = "QS301EW0005"
    else:
        care_raw = pd.read_csv(CENSUS_PATHS["2021_care"], low_memory=False)
        age_raw = pd.read_csv(CENSUS_PATHS["2021_age"], low_memory=False)
        care_column = (
            "Provision of unpaid care: Provides 50 or more hours unpaid care a week"
        )
        care = care_raw[["geography code", care_column]].rename(
            columns={"geography code": "lsoa_code", care_column: "care50_num"}
        )
        age = age_raw[["geography code", "Age: Total", "Age: Aged 4 years and under"]].copy()
        age["population_5plus"] = pd.to_numeric(age["Age: Total"], errors="raise") - pd.to_numeric(
            age["Age: Aged 4 years and under"], errors="raise"
        )
        age = age[["geography code", "population_5plus"]].rename(
            columns={"geography code": "lsoa_code"}
        )
        care_table = "TS039"
        population_table = "TS007A"
        population_definition = "Age: Total minus Age: Aged 4 years and under"
        care_definition = care_column

    for frame_name, frame in (("care", care), ("age", age)):
        frame["lsoa_code"] = frame["lsoa_code"].astype("string").str.strip()
        if not frame["lsoa_code"].is_unique:
            duplicates = int(frame["lsoa_code"].duplicated().sum())
            raise AssertionError(f"{year} {frame_name} has {duplicates} duplicate LSOA rows")

    selected = names[["lsoa_code", "lsoa_name"]].copy()
    selected = selected.merge(care, on="lsoa_code", how="left", validate="one_to_one")
    selected = selected.merge(age, on="lsoa_code", how="left", validate="one_to_one")
    if set(selected["lsoa_code"]) != keep:
        raise AssertionError(f"{year} native demand key mismatch")
    for column in ("care50_num", "population_5plus"):
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    selected.insert(0, "year", year)
    selected["care50_rate"] = selected["care50_num"] / selected["population_5plus"]
    selected["care50_source_table"] = care_table
    selected["population_5plus_source_table"] = population_table
    selected["care50_definition"] = care_definition
    selected["population_5plus_definition"] = population_definition
    selected["harmonisation_status"] = "native_counts_not_harmonised"
    return selected.sort_values("lsoa_code").reset_index(drop=True)


def build_crosswalk(
    year: int,
    source: gpd.GeoDataFrame,
    target: gpd.GeoDataFrame,
) -> pd.DataFrame:
    target_small = target[["lsoa_code", "geometry"]].rename(
        columns={"lsoa_code": "target_lsoa_2021_code"}
    )
    if year == 2021:
        area = target_small.geometry.area.to_numpy(dtype=float)
        return pd.DataFrame(
            {
                "source_lsoa_code": target_small["target_lsoa_2021_code"].to_numpy(),
                "target_lsoa_2021_code": target_small["target_lsoa_2021_code"].to_numpy(),
                "intersection_area_m2": area,
                "source_overlap_area_m2": area,
                "source_normalized_weight": np.ones(len(target_small), dtype=float),
                "crosswalk_method": "identity_2021",
                "boundary_gap_m": np.zeros(len(target_small), dtype=float),
            }
        )
    source_small = source[["lsoa_code", "geometry"]].rename(
        columns={"lsoa_code": "source_lsoa_code"}
    )
    intersections = gpd.overlay(
        source_small,
        target_small,
        how="intersection",
        keep_geom_type=True,
    )
    intersections = intersections.loc[
        intersections.geometry.notna() & ~intersections.geometry.is_empty
    ].copy()
    intersections["intersection_area_m2"] = intersections.geometry.area
    intersections = intersections.loc[intersections["intersection_area_m2"].gt(0)].copy()
    intersections["source_overlap_area_m2"] = intersections.groupby("source_lsoa_code")[
        "intersection_area_m2"
    ].transform("sum")
    intersections["source_normalized_weight"] = (
        intersections["intersection_area_m2"] / intersections["source_overlap_area_m2"]
    )
    intersections["crosswalk_method"] = "positive_area_source_normalised"
    intersections["boundary_gap_m"] = 0.0
    crosswalk = intersections[
        [
            "source_lsoa_code",
            "target_lsoa_2021_code",
            "intersection_area_m2",
            "source_overlap_area_m2",
            "source_normalized_weight",
            "crosswalk_method",
            "boundary_gap_m",
        ]
    ].copy()
    absent = set(source["lsoa_code"]) - set(crosswalk["source_lsoa_code"])
    if absent:
        missing_source = source_small.loc[
            source_small["source_lsoa_code"].isin(absent)
        ].copy()
        nearest = gpd.sjoin_nearest(
            missing_source,
            target_small,
            how="left",
            distance_col="boundary_gap_m",
        )
        nearest = nearest.sort_values(
            ["source_lsoa_code", "boundary_gap_m", "target_lsoa_2021_code"]
        ).drop_duplicates("source_lsoa_code")
        if nearest["target_lsoa_2021_code"].isna().any():
            raise AssertionError(f"{year} crosswalk cannot resolve an edge source LSOA")
        if nearest["boundary_gap_m"].gt(MAX_CROSSWALK_EDGE_GAP_M).any():
            gaps = nearest.loc[
                nearest["boundary_gap_m"].gt(MAX_CROSSWALK_EDGE_GAP_M),
                ["source_lsoa_code", "boundary_gap_m"],
            ]
            raise AssertionError(
                f"{year} crosswalk has non-topological unmatched sources: {gaps.to_dict('records')}"
            )
        fallback = nearest[
            ["source_lsoa_code", "target_lsoa_2021_code", "boundary_gap_m"]
        ].copy()
        fallback["intersection_area_m2"] = 0.0
        fallback["source_overlap_area_m2"] = 0.0
        fallback["source_normalized_weight"] = 1.0
        fallback["crosswalk_method"] = "submetre_boundary_edge_nearest_target"
        fallback = fallback[crosswalk.columns]
        crosswalk = pd.concat([crosswalk, fallback], ignore_index=True)
    if set(crosswalk["source_lsoa_code"]) != set(source["lsoa_code"]):
        unresolved = set(source["lsoa_code"]) - set(crosswalk["source_lsoa_code"])
        raise AssertionError(f"{year} crosswalk still misses {len(unresolved)} source LSOAs")
    sums = crosswalk.groupby("source_lsoa_code")["source_normalized_weight"].sum()
    if not np.allclose(sums, 1.0, atol=1e-12, rtol=0):
        raise AssertionError(f"{year} crosswalk source weights do not sum to one")
    return crosswalk.sort_values(
        ["source_lsoa_code", "target_lsoa_2021_code"]
    ).reset_index(drop=True)


def harmonise_demand(
    year: int,
    native: pd.DataFrame,
    crosswalk: pd.DataFrame,
    target: gpd.GeoDataFrame,
) -> pd.DataFrame:
    source = native[["lsoa_code", "care50_num", "population_5plus"]].rename(
        columns={"lsoa_code": "source_lsoa_code"}
    )
    allocated = crosswalk.merge(source, on="source_lsoa_code", validate="many_to_one")
    for field in ("care50_num", "population_5plus"):
        allocated[field] = allocated[field] * allocated["source_normalized_weight"]
    totals = allocated.groupby("target_lsoa_2021_code", as_index=False)[
        ["care50_num", "population_5plus"]
    ].sum()
    result = target[["lsoa_code", "lsoa_name"]].rename(
        columns={"lsoa_code": "lsoa_2021_code", "lsoa_name": "lsoa_2021_name"}
    )
    result = result.merge(
        totals.rename(columns={"target_lsoa_2021_code": "lsoa_2021_code"}),
        on="lsoa_2021_code",
        how="left",
        validate="one_to_one",
    )
    result.insert(0, "source_year", year)
    result["care50_rate"] = result["care50_num"] / result["population_5plus"]
    result["harmonisation_method"] = (
        "identity_2021" if year == 2021 else "source_normalised_areal_count_allocation"
    )
    if result[["care50_num", "population_5plus"]].isna().any().any():
        missing = int(result[["care50_num", "population_5plus"]].isna().any(axis=1).sum())
        raise AssertionError(f"{year} harmonised demand misses {missing} target LSOAs")
    if not result["population_5plus"].gt(0).all():
        raise AssertionError(f"{year} harmonised population_5plus is not positive")
    return result.sort_values("lsoa_2021_code").reset_index(drop=True)


def assign_charities_to_target(
    charities: pd.DataFrame,
    target: gpd.GeoDataFrame,
) -> pd.DataFrame:
    work = charities.reset_index(drop=True).copy()
    work["_row_id"] = np.arange(len(work), dtype=int)
    points = gpd.GeoDataFrame(
        work,
        geometry=gpd.points_from_xy(work["long"], work["lat"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:27700")
    target_for_join = target[["lsoa_code", "lsoa_name", "geometry"]].rename(
        columns={
            "lsoa_code": "provider_lsoa_2021_code",
            "lsoa_name": "provider_lsoa_2021_name",
        }
    )
    joined = gpd.sjoin(
        points,
        target_for_join,
        how="left",
        predicate="within",
    )
    if joined["_row_id"].duplicated().any():
        raise AssertionError("A charity point matched multiple 2021 halo LSOAs")
    joined["provider_assignment_method"] = np.where(
        joined["provider_lsoa_2021_code"].notna(), "within", "nearest_boundary_fallback"
    )
    joined["nearest_boundary_distance_m"] = np.where(
        joined["provider_lsoa_2021_code"].notna(), 0.0, np.nan
    )
    unmatched_ids = joined.loc[joined["provider_lsoa_2021_code"].isna(), "_row_id"].tolist()
    if unmatched_ids:
        nearest = gpd.sjoin_nearest(
            points.loc[points["_row_id"].isin(unmatched_ids), ["_row_id", "geometry"]],
            target_for_join,
            how="left",
            distance_col="nearest_boundary_distance_m",
        )
        nearest = nearest.sort_values(
            ["_row_id", "nearest_boundary_distance_m", "provider_lsoa_2021_code"]
        ).drop_duplicates("_row_id")
        lookup = nearest.set_index("_row_id")
        mask = joined["_row_id"].isin(unmatched_ids)
        joined.loc[mask, "provider_lsoa_2021_code"] = joined.loc[mask, "_row_id"].map(
            lookup["provider_lsoa_2021_code"]
        )
        joined.loc[mask, "provider_lsoa_2021_name"] = joined.loc[mask, "_row_id"].map(
            lookup["provider_lsoa_2021_name"]
        )
        joined.loc[mask, "nearest_boundary_distance_m"] = joined.loc[mask, "_row_id"].map(
            lookup["nearest_boundary_distance_m"]
        )
    if joined["provider_lsoa_2021_code"].isna().any():
        raise AssertionError("Provider 2021 LSOA assignment remains incomplete")
    if set(joined["provider_lsoa_2021_code"]) - set(target["lsoa_code"]):
        raise AssertionError("Provider assignment produced a non-halo target code")
    return joined.drop(columns=["geometry", "index_right", "_row_id"], errors="ignore")


def main() -> None:
    if not OUTPUT_ROOT.is_dir():
        raise FileNotFoundError(
            f"Expected the existing extension package to replace: {OUTPUT_ROOT}"
        )

    required_paths = [
        ICB_PATH,
        *LSOA_PATHS.values(),
        *CENSUS_PATHS.values(),
        CHARITY_STAGE,
        *CHARITY_FINAL_PATHS.values(),
        *COVARIATE_PATHS.values(),
        CHARITY_PACKAGE / "rebuild_charity_data.py",
        PACKAGE / "5-Step-Analysis/build_step1_notebook.py",
        ROAD_PREPARATION_SCRIPT,
        ROAD_ORDER_CONTENTS,
        ROAD_ORDER_CITATIONS,
    ]
    for pattern in ROAD_SOURCE_PATTERNS.values():
        required_paths.extend(sorted(ROAD_SOURCE_DIR.glob(pattern)))
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required frozen inputs: {missing}")

    protected_files = []
    workflow = PACKAGE / "5-Step-Analysis"
    for notebook in (
        "01_step1_spatial_foundation_road_od.ipynb",
        "02_step2_three_year_e2sfca_accessibility.ipynb",
        "03_step3_spatial_mismatch_typology.ipynb",
        "04_step3b_bivariate_lisa.ipynb",
        "05_step4_longitudinal_trajectories_and_decomposition.ipynb",
    ):
        protected_files.append(workflow / notebook)
    for step in ("step1", "step2", "step3", "step3b", "step4"):
        protected_files.extend(path for path in (workflow / "outputs" / step).rglob("*") if path.is_file())
    protected_files.extend(COVARIATE_PATHS.values())
    protected_files.extend(CHARITY_FINAL_PATHS.values())
    protected_files.append(CHARITY_STAGE)
    protected_files.extend(
        path
        for path in (PACKAGE / "Road_network").rglob("*")
        if path.is_file()
    )
    protected_files.extend(
        path
        for path in workflow.glob("build_step*_notebook.py")
        if path.is_file()
    )
    protected_before = snapshot(protected_files)

    previous_provider_tables = {}
    for year in YEARS:
        previous_path = OUTPUT_ROOT / f"charities/external_eligible_charities_{year}.csv"
        if not previous_path.is_file():
            raise FileNotFoundError(f"Missing provider baseline in existing extension: {previous_path}")
        previous_provider_tables[year] = pd.read_csv(previous_path, low_memory=False)

    build_root = Path(
        tempfile.mkdtemp(prefix="halo-20km-build-", dir=str(PACKAGE))
    )
    for folder in ("boundaries", "demand", "charities", "road_network", "qa"):
        (build_root / folder).mkdir(parents=True, exist_ok=True)

    try:
        boundaries = {}
        boundary_audit_rows = []
        for year in YEARS:
            boundaries[year], audit = repair_lsoa_layer(year)
            boundary_audit_rows.append(audit)

        icbs = gpd.read_file(ICB_PATH).to_crs("EPSG:27700")
        study_union = icbs.geometry.union_all()
        if not study_union.is_valid:
            study_union = study_union.make_valid()
        provider_expanded = study_union.buffer(PROVIDER_HALO_DISTANCE_M)
        provider_halo_ring = provider_expanded.difference(study_union)
        demand_expanded = study_union.buffer(DEMAND_HALO_DISTANCE_M)
        demand_halo_ring = demand_expanded.difference(study_union)
        if not provider_halo_ring.is_valid:
            provider_halo_ring = provider_halo_ring.make_valid()
        if not demand_halo_ring.is_valid:
            demand_halo_ring = demand_halo_ring.make_valid()

        internal_codes = {}
        provider_halo_boundaries = {}
        demand_halo_boundaries = {}
        provider_halo_list_frames = []
        demand_halo_list_frames = []
        halo_audit_rows = []
        for year in YEARS:
            covariates = pd.read_csv(COVARIATE_PATHS[year], usecols=["lsoa_code"])
            covariates["lsoa_code"] = covariates["lsoa_code"].astype("string").str.strip()
            if len(covariates) != EXPECTED_INTERNAL_LSOAS[year] or not covariates["lsoa_code"].is_unique:
                raise AssertionError(f"{year} internal LSOA input does not match the frozen workflow")
            internal_codes[year] = set(covariates["lsoa_code"])
            layer = boundaries[year].copy()
            representatives = layer.geometry.representative_point()
            distance = representatives.distance(study_union)
            external = ~layer["lsoa_code"].isin(internal_codes[year])
            for role, distance_m, ring_geometry, destination, list_frames in (
                (
                    "external_provider_context",
                    PROVIDER_HALO_DISTANCE_M,
                    provider_halo_ring,
                    provider_halo_boundaries,
                    provider_halo_list_frames,
                ),
                (
                    "external_competing_demand",
                    DEMAND_HALO_DISTANCE_M,
                    demand_halo_ring,
                    demand_halo_boundaries,
                    demand_halo_list_frames,
                ),
            ):
                selected = representatives.within(ring_geometry) & external
                halo = layer.loc[selected].copy()
                halo["representative_easting"] = representatives.loc[selected].x
                halo["representative_northing"] = representatives.loc[selected].y
                halo["representative_distance_from_study_area_m"] = distance.loc[selected]
                halo["country"] = np.where(
                    halo["lsoa_code"].str.startswith("E"), "England", "Wales"
                )
                halo["halo_role"] = role
                halo["halo_distance_m"] = distance_m
                halo["membership_predicate"] = "representative_point_within_buffer_ring"
                if set(halo["lsoa_code"]) & internal_codes[year]:
                    raise AssertionError(f"{year} {role} halo overlaps internal LSOA codes")
                if not halo["representative_distance_from_study_area_m"].gt(0).all():
                    raise AssertionError(f"{year} {role} halo includes an internal point")
                if not halo["representative_distance_from_study_area_m"].lt(distance_m).all():
                    raise AssertionError(f"{year} {role} halo exceeds {distance_m} m")
                destination[year] = halo.sort_values("lsoa_code").reset_index(drop=True)
                listing = halo.drop(columns="geometry").copy()
                listing.insert(0, "year", year)
                listing["halo_definition"] = (
                    "LSOA representative point outside seven-ICB union and within "
                    f"{int(distance_m)} m; current internal LSOA codes excluded"
                )
                list_frames.append(listing)
                halo_audit_rows.append(
                    {
                        "year": year,
                        "halo_role": role,
                        "halo_distance_m": distance_m,
                        "halo_lsoas": len(halo),
                        "england_lsoas": int(halo["country"].eq("England").sum()),
                        "wales_lsoas": int(halo["country"].eq("Wales").sum()),
                        "internal_lsoas_unchanged": len(internal_codes[year]),
                        "internal_code_overlap": len(
                            set(halo["lsoa_code"]) & internal_codes[year]
                        ),
                        "minimum_representative_distance_m": float(
                            halo["representative_distance_from_study_area_m"].min()
                        ),
                        "maximum_representative_distance_m": float(
                            halo["representative_distance_from_study_area_m"].max()
                        ),
                    }
                )
            if not set(provider_halo_boundaries[year]["lsoa_code"]).issubset(
                set(demand_halo_boundaries[year]["lsoa_code"])
            ):
                raise AssertionError(f"{year} provider-context LSOAs are not a subset of demand LSOAs")

        provider_halo_lists_long = pd.concat(provider_halo_list_frames, ignore_index=True)
        demand_halo_lists_long = pd.concat(demand_halo_list_frames, ignore_index=True)
        write_csv(
            demand_halo_lists_long,
            build_root / "boundaries/halo_native_lsoa_list_2001_2011_2021.csv",
        )
        write_csv(
            provider_halo_lists_long,
            build_root / "boundaries/provider_halo_20km_native_lsoa_list_2001_2011_2021.csv",
        )
        write_csv(
            provider_halo_lists_long.loc[
                provider_halo_lists_long["year"].eq(2021)
            ].reset_index(drop=True),
            build_root / "boundaries/halo_20km_2021_lsoa_list.csv",
        )
        write_csv(
            demand_halo_lists_long,
            build_root
            / "boundaries/competing_demand_halo_40km_native_lsoa_list_2001_2011_2021.csv",
        )
        write_csv(
            demand_halo_lists_long.loc[
                demand_halo_lists_long["year"].eq(2021)
            ].reset_index(drop=True),
            build_root / "boundaries/competing_demand_halo_40km_2021_lsoa_list.csv",
        )

        gpkg = build_root / "boundaries/halo_20km_spatial.gpkg"
        provider_ring_frame = gpd.GeoDataFrame(
            {
                "halo_distance_m": [PROVIDER_HALO_DISTANCE_M],
                "spatial_role": ["external_provider"],
                "definition": ["outside seven-ICB study boundary and within 20 km"],
            },
            geometry=[provider_halo_ring],
            crs="EPSG:27700",
        )
        demand_ring_frame = gpd.GeoDataFrame(
            {
                "halo_distance_m": [DEMAND_HALO_DISTANCE_M],
                "spatial_role": ["external_competing_demand"],
                "definition": ["outside seven-ICB study boundary and within 40 km"],
            },
            geometry=[demand_halo_ring],
            crs="EPSG:27700",
        )
        study_frame = gpd.GeoDataFrame(
            {"definition": ["dissolved seven April-2023 ICBs"]},
            geometry=[study_union],
            crs="EPSG:27700",
        )
        provider_ring_frame.to_file(gpkg, layer="halo_ring_20km", driver="GPKG")
        demand_ring_frame.to_file(
            gpkg, layer="competing_demand_ring_40km", driver="GPKG", mode="a"
        )
        study_frame.to_file(gpkg, layer="study_boundary", driver="GPKG", mode="a")
        provider_halo_boundaries[2021].to_file(
            gpkg, layer="halo_lsoa_2021", driver="GPKG", mode="a"
        )
        provider_target_points = provider_halo_boundaries[2021].copy()
        provider_target_points["geometry"] = (
            provider_target_points.geometry.representative_point()
        )
        provider_target_points.to_file(
            gpkg, layer="halo_lsoa_2021_points", driver="GPKG", mode="a"
        )
        demand_halo_boundaries[2021].to_file(
            gpkg,
            layer="competing_demand_lsoa_2021_40km",
            driver="GPKG",
            mode="a",
        )
        demand_target_points = demand_halo_boundaries[2021].copy()
        demand_target_points["geometry"] = demand_target_points.geometry.representative_point()
        demand_target_points.to_file(
            gpkg,
            layer="competing_demand_lsoa_2021_40km_points",
            driver="GPKG",
            mode="a",
        )

        native_demand = {}
        harmonised_demand = {}
        crosswalks = {}
        demand_audit_rows = []
        conservation_rows = []
        for year in YEARS:
            keep = set(demand_halo_boundaries[year]["lsoa_code"])
            native_demand[year] = extract_native_demand(
                year,
                keep,
                demand_halo_boundaries[year][["lsoa_code", "lsoa_name"]],
            )
            native_demand[year]["spatial_role"] = "external_competing_demand"
            native_demand[year]["demand_halo_distance_m"] = DEMAND_HALO_DISTANCE_M
            native_demand[year]["e2sfca_catchment_m"] = E2SFCA_CATCHMENT_M
            missing_care = int(native_demand[year]["care50_num"].isna().sum())
            missing_population = int(native_demand[year]["population_5plus"].isna().sum())
            invalid_population = int(
                native_demand[year]["population_5plus"].fillna(0).le(0).sum()
            )
            invalid_care = int(native_demand[year]["care50_num"].fillna(-1).lt(0).sum())
            if any((missing_care, missing_population, invalid_population, invalid_care)):
                raise AssertionError(
                    f"{year} native demand completeness failure: "
                    f"care missing={missing_care}, population missing={missing_population}, "
                    f"invalid population={invalid_population}, invalid care={invalid_care}"
                )
            write_csv(
                native_demand[year],
                build_root / f"demand/native_external_care50_{year}.csv",
            )
            crosswalks[year] = build_crosswalk(
                year, demand_halo_boundaries[year], demand_halo_boundaries[2021]
            )
            edge_fallback = crosswalks[year]["crosswalk_method"].eq(
                "submetre_boundary_edge_nearest_target"
            )
            write_csv(
                crosswalks[year],
                build_root / f"demand/native_to_2021_external_crosswalk_{year}.csv",
            )
            harmonised_demand[year] = harmonise_demand(
                year,
                native_demand[year],
                crosswalks[year],
                demand_halo_boundaries[2021],
            )
            harmonised_demand[year]["spatial_role"] = "external_competing_demand"
            harmonised_demand[year]["demand_halo_distance_m"] = DEMAND_HALO_DISTANCE_M
            harmonised_demand[year]["e2sfca_catchment_m"] = E2SFCA_CATCHMENT_M
            write_csv(
                harmonised_demand[year],
                build_root
                / f"demand/harmonised_external_care50_to_2021_{year}.csv",
            )
            demand_audit_rows.append(
                {
                    "year": year,
                    "spatial_role": "external_competing_demand",
                    "demand_halo_distance_m": DEMAND_HALO_DISTANCE_M,
                    "expected_native_halo_lsoas": len(keep),
                    "native_rows": len(native_demand[year]),
                    "unique_native_lsoas": native_demand[year]["lsoa_code"].nunique(),
                    "missing_care50_num": missing_care,
                    "missing_population_5plus": missing_population,
                    "invalid_care50_num": invalid_care,
                    "nonpositive_population_5plus": invalid_population,
                    "native_completeness_percent": 100.0
                    * (len(native_demand[year]) - max(missing_care, missing_population))
                    / len(keep),
                    "harmonised_target_rows": len(harmonised_demand[year]),
                    "expected_2021_halo_lsoas": len(demand_halo_boundaries[2021]),
                    "missing_harmonised_counts": int(
                        harmonised_demand[year][["care50_num", "population_5plus"]]
                        .isna()
                        .any(axis=1)
                        .sum()
                    ),
                    "harmonised_completeness_percent": 100.0
                    * harmonised_demand[year][["care50_num", "population_5plus"]]
                    .notna()
                    .all(axis=1)
                    .mean(),
                    "submetre_boundary_edge_fallback_rows": int(edge_fallback.sum()),
                    "maximum_boundary_edge_gap_m": (
                        float(crosswalks[year].loc[edge_fallback, "boundary_gap_m"].max())
                        if edge_fallback.any()
                        else 0.0
                    ),
                    "rates_generated_for_qa_only": True,
                    "used_in_model": False,
                }
            )
            for field in ("care50_num", "population_5plus"):
                native_total = float(native_demand[year][field].sum())
                harmonised_total = float(harmonised_demand[year][field].sum())
                difference = harmonised_total - native_total
                relative_error = abs(difference) / max(abs(native_total), 1.0)
                conservation_rows.append(
                    {
                        "year": year,
                        "variable": field,
                        "native_total": native_total,
                        "harmonised_total": harmonised_total,
                        "signed_difference": difference,
                        "absolute_difference": abs(difference),
                        "relative_error": relative_error,
                        "conservation_pass": relative_error < 1e-12,
                    }
                )
        conservation = pd.DataFrame(conservation_rows)
        if not conservation["conservation_pass"].all():
            raise AssertionError("Demand count harmonisation failed conservation")
        write_csv(
            pd.concat([harmonised_demand[year] for year in YEARS], ignore_index=True),
            build_root / "demand/harmonised_external_care50_to_2021_long.csv",
        )

        charity_stage = pd.read_parquet(CHARITY_STAGE)
        if not charity_stage["care_strict"].fillna(False).all():
            raise AssertionError("The selected Data Spine stage contains a non-care_strict row")
        charity_audit_rows = []
        charity_outputs = {}
        address_breakdown_frames = []
        finance_breakdown_frames = []
        location_frames = []
        finance_frames = []
        assignment_frames = []
        gap_rows = []
        for demand_row in demand_audit_rows:
            if demand_row["submetre_boundary_edge_fallback_rows"]:
                gap_rows.append(
                    {
                        "year": demand_row["year"],
                        "gap_type": "independent_halo_vintage_boundary_edge_match",
                        "affected_rows": demand_row[
                            "submetre_boundary_edge_fallback_rows"
                        ],
                        "halo_specific_count_known": True,
                        "handling": (
                            "assigned to nearest fixed-2021 halo target only where polygon gap "
                            f"was <= {MAX_CROSSWALK_EDGE_GAP_M:g} m; exact gap retained in crosswalk"
                        ),
                    }
                )
        provider_unchanged_rows = []
        for year in YEARS:
            all_year = charity_stage.loc[charity_stage["target_year"].eq(year)].copy()
            native_halo_codes = set(provider_halo_boundaries[year]["lsoa_code"])
            geocoded_year = all_year.loc[all_year["geocoded"].fillna(False)].copy()
            geocoded_points = gpd.GeoDataFrame(
                geocoded_year,
                geometry=gpd.points_from_xy(geocoded_year["long"], geocoded_year["lat"]),
                crs="EPSG:4326",
            ).to_crs("EPSG:27700")
            exact_halo = geocoded_points.geometry.within(provider_halo_ring)
            selected = geocoded_year.loc[exact_halo.to_numpy()].copy()
            if selected["charity_number"].duplicated().any():
                raise AssertionError(f"{year} external charity numbers are not unique")
            internal_charities = pd.read_csv(
                CHARITY_FINAL_PATHS[year], usecols=["charity_number"]
            )
            internal_ids = set(
                pd.to_numeric(internal_charities["charity_number"], errors="raise").astype(int)
            )
            selected_ids = set(
                pd.to_numeric(selected["charity_number"], errors="raise").astype(int)
            )
            overlap = selected_ids & internal_ids
            if overlap:
                raise AssertionError(f"{year} halo charities overlap {len(overlap)} internal charities")
            if not selected["historical_address_flag"].fillna(False).all():
                raise AssertionError(f"{year} selected halo charity lacks the current historical-address flag")
            if selected[["historical_postcode", "lat", "long"]].isna().any().any():
                raise AssertionError(f"{year} selected halo charity has an incomplete location")

            selected["analysis_year"] = year
            selected["registration_date"] = selected["registerdate"]
            selected["removal_date"] = selected["removeddate"]
            selected["postcode"] = selected["historical_postcode"]
            selected["native_lsoa_code"] = selected["lsoa_code"].astype("string")
            selected["native_lsoa_representative_in_halo"] = selected[
                "native_lsoa_code"
            ].isin(native_halo_codes)
            selected[f"lsoa_{year}"] = selected["native_lsoa_code"]
            selected["income_proxy_gbp"] = selected["income_nominal"]
            selected["income_date"] = selected["fye"]
            selected["log1p_income_2021_gbp"] = np.where(
                selected["income_2021_gbp"].notna(),
                np.log1p(pd.to_numeric(selected["income_2021_gbp"], errors="coerce")),
                np.nan,
            )
            if pd.to_numeric(selected["income_2021_gbp"], errors="coerce").dropna().lt(0).any():
                raise AssertionError(f"{year} halo charity has negative income_2021_gbp")

            # Assign to the actual full E/W 2021 LSOA layer. Restricting the join
            # to representative-point-selected halo LSOAs would incorrectly snap
            # exact edge providers by several kilometres when their containing
            # LSOA representative point lies just beyond the ring.
            assigned = assign_charities_to_target(selected, boundaries[2021])
            assigned["provider_lsoa_in_2021_demand_halo"] = assigned[
                "provider_lsoa_2021_code"
            ].isin(set(demand_halo_boundaries[2021]["lsoa_code"]))
            assigned["provider_lsoa_is_current_internal"] = assigned[
                "provider_lsoa_2021_code"
            ].isin(internal_codes[2021])
            if assigned["provider_lsoa_is_current_internal"].any():
                raise AssertionError(
                    f"{year} exact external providers were assigned to current internal 2021 LSOAs"
                )
            point_frame = gpd.GeoDataFrame(
                selected[["charity_number", "lat", "long"]].copy(),
                geometry=gpd.points_from_xy(selected["long"], selected["lat"]),
                crs="EPSG:4326",
            ).to_crs("EPSG:27700")
            point_distance = point_frame.geometry.distance(study_union)
            point_inside_study = point_frame.geometry.within(study_union)
            point_inside_expanded = point_frame.geometry.within(provider_expanded)
            exact_ring_flag = (~point_inside_study) & point_inside_expanded
            if not exact_ring_flag.all():
                raise AssertionError(f"{year} exact provider halo selection is not reproducible")
            exact_ring_lookup = pd.Series(
                exact_ring_flag.to_numpy(),
                index=pd.to_numeric(point_frame["charity_number"], errors="raise").astype(int),
            )
            distance_lookup = pd.Series(
                point_distance.to_numpy(),
                index=pd.to_numeric(point_frame["charity_number"], errors="raise").astype(int),
            )
            assigned_numbers = pd.to_numeric(assigned["charity_number"], errors="raise").astype(int)
            assigned["provider_point_within_exact_20km_ring"] = assigned_numbers.map(
                exact_ring_lookup
            )
            assigned["provider_point_distance_from_study_area_m"] = assigned_numbers.map(
                distance_lookup
            )
            assigned["external_selection_rule"] = (
                "historical provider coordinate outside the seven-ICB study boundary and "
                "within 20000 m; current internal charity IDs excluded"
            )
            assigned["provider_halo_distance_m"] = PROVIDER_HALO_DISTANCE_M
            assigned["competing_demand_envelope_distance_m"] = DEMAND_HALO_DISTANCE_M
            assigned["e2sfca_catchment_m"] = E2SFCA_CATCHMENT_M
            assigned["used_in_e2sfca"] = False
            assigned["used_in_od"] = False
            assigned["log1p_definition"] = "log1p(income_2021_gbp) at charity level"
            assigned = assigned.sort_values("charity_number").reset_index(drop=True)

            ordered = [
                "analysis_year",
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
                f"lsoa_{year}",
                "native_lsoa_code",
                "native_lsoa_representative_in_halo",
                "provider_lsoa_2021_code",
                "provider_lsoa_2021_name",
                "provider_lsoa_in_2021_demand_halo",
                "provider_lsoa_is_current_internal",
                "provider_assignment_method",
                "nearest_boundary_distance_m",
                "provider_point_within_exact_20km_ring",
                "provider_point_distance_from_study_area_m",
                "provider_halo_distance_m",
                "competing_demand_envelope_distance_m",
                "e2sfca_catchment_m",
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
                "income_proxy_gbp",
                "income_date",
                "fy",
                "fys",
                "fye",
                "income_nominal",
                "income_2021_gbp",
                "log1p_income_2021_gbp",
                "selection_method",
                "finance_source",
                "finance_extract",
                "finance_year_offset",
                "fye_offset_days",
                "flagged_1000x",
                "cpih_source_year",
                "cpih_index_source",
                "cpih_index_2021",
                "cpih_multiplier_to_2021",
                "cpih_series",
                "cpih_source_url",
                "external_selection_rule",
                "log1p_definition",
                "used_in_e2sfca",
                "used_in_od",
            ]
            missing_columns = set(ordered) - set(assigned.columns)
            if missing_columns:
                raise AssertionError(f"{year} charity output missing columns: {sorted(missing_columns)}")
            charity_outputs[year] = assigned[ordered].copy()

            previous = previous_provider_tables[year].copy()
            stable_columns = [
                "charity_number",
                "lat",
                "long",
                "income_2021_gbp",
                "log1p_income_2021_gbp",
                "postcode",
                "provider_lsoa_2021_code",
                "provider_point_distance_from_study_area_m",
            ]
            if missing_baseline := set(stable_columns) - set(previous.columns):
                raise AssertionError(
                    f"{year} previous provider baseline lacks {sorted(missing_baseline)}"
                )
            before = previous[stable_columns].sort_values("charity_number").reset_index(drop=True)
            after = charity_outputs[year][stable_columns].sort_values(
                "charity_number"
            ).reset_index(drop=True)
            numeric_stable_columns = [
                "charity_number",
                "lat",
                "long",
                "income_2021_gbp",
                "log1p_income_2021_gbp",
                "provider_point_distance_from_study_area_m",
            ]
            string_stable_columns = ["postcode", "provider_lsoa_2021_code"]
            for column in numeric_stable_columns:
                before[column] = pd.to_numeric(before[column], errors="coerce")
                after[column] = pd.to_numeric(after[column], errors="coerce")
            for column in string_stable_columns:
                before[column] = before[column].astype("string")
                after[column] = after[column].astype("string")
            pd.testing.assert_frame_equal(
                before,
                after,
                check_dtype=False,
                check_exact=False,
                rtol=1e-12,
                atol=1e-9,
            )
            provider_unchanged_rows.append(
                {
                    "year": year,
                    "previous_rows": len(before),
                    "updated_rows": len(after),
                    "same_charity_ids_coordinates_income_and_provider_lsoa": True,
                    "provider_halo_distance_m": PROVIDER_HALO_DISTANCE_M,
                }
            )
            write_csv(
                charity_outputs[year],
                build_root / f"charities/external_eligible_charities_{year}.csv",
            )

            locations = charity_outputs[year][
                [
                    "analysis_year",
                    "uid",
                    "charity_number",
                    "charity_name",
                    "postcode",
                    "native_lsoa_code",
                    "native_lsoa_representative_in_halo",
                    "lat",
                    "long",
                    "address_source",
                    "address_evidence_date",
                    "address_year_offset",
                    "address_method",
                    "address_quality",
                    "historical_address_flag",
                    "provider_point_within_exact_20km_ring",
                ]
            ].copy()
            location_frames.append(locations)
            finances = charity_outputs[year][
                [
                    "analysis_year",
                    "uid",
                    "charity_number",
                    "charity_name",
                    "income_nominal",
                    "income_2021_gbp",
                    "log1p_income_2021_gbp",
                    "selection_method",
                    "finance_source",
                    "finance_extract",
                    "fy",
                    "fys",
                    "fye",
                    "finance_year_offset",
                    "fye_offset_days",
                    "cpih_source_year",
                    "cpih_index_source",
                    "cpih_index_2021",
                    "cpih_multiplier_to_2021",
                    "cpih_series",
                    "cpih_source_url",
                ]
            ].copy()
            finance_frames.append(finances)
            assignments = charity_outputs[year][
                [
                    "analysis_year",
                    "uid",
                    "charity_number",
                    "charity_name",
                    "native_lsoa_code",
                    "native_lsoa_representative_in_halo",
                    "provider_lsoa_2021_code",
                    "provider_lsoa_2021_name",
                    "provider_lsoa_in_2021_demand_halo",
                    "provider_lsoa_is_current_internal",
                    "provider_assignment_method",
                    "nearest_boundary_distance_m",
                    "provider_point_within_exact_20km_ring",
                    "provider_point_distance_from_study_area_m",
                    "lat",
                    "long",
                ]
            ].copy()
            assignment_frames.append(assignments)

            address_breakdown = (
                charity_outputs[year]
                .groupby(["address_method", "address_quality"], dropna=False)
                .size()
                .rename("charities")
                .reset_index()
            )
            address_breakdown.insert(0, "year", year)
            address_breakdown_frames.append(address_breakdown)
            finance_breakdown = (
                charity_outputs[year]
                .groupby(["selection_method", "finance_source"], dropna=False)
                .size()
                .rename("charities")
                .reset_index()
            )
            finance_breakdown.insert(0, "year", year)
            finance_breakdown_frames.append(finance_breakdown)

            reliable_history = int(
                charity_outputs[year]["historical_address_flag"].fillna(False).sum()
            )
            with_income = int(charity_outputs[year]["income_2021_gbp"].notna().sum())
            fallback = charity_outputs[year]["provider_assignment_method"].eq(
                "nearest_boundary_fallback"
            )
            charity_audit_rows.append(
                {
                    "year": year,
                    "national_active_care_strict_rows": len(all_year),
                    "national_historical_postcode_rows": int(
                        all_year["historical_postcode"].notna().sum()
                    ),
                    "national_geocoded_rows": int(all_year["geocoded"].fillna(False).sum()),
                    "national_unresolved_historical_location_rows": int(
                        all_year["historical_postcode"].isna().sum()
                    ),
                    "national_companies_house_followup_possible_rows": int(
                        all_year["companies_house_followup_possible"].fillna(False).sum()
                    ),
                    "external_eligible_charities": len(charity_outputs[year]),
                    "with_income_2021_gbp": with_income,
                    "missing_income_2021_gbp": len(charity_outputs[year]) - with_income,
                    "with_charity_level_log1p_income": int(
                        charity_outputs[year]["log1p_income_2021_gbp"].notna().sum()
                    ),
                    "with_reliable_historical_location_under_current_rule": reliable_history,
                    "with_historical_postcode": int(
                        charity_outputs[year]["postcode"].notna().sum()
                    ),
                    "with_coordinates": int(
                        charity_outputs[year][["lat", "long"]].notna().all(axis=1).sum()
                    ),
                    "point_within_exact_20km_ring": int(
                        charity_outputs[year]["provider_point_within_exact_20km_ring"].sum()
                    ),
                    "native_lsoa_representative_in_halo": int(
                        charity_outputs[year]["native_lsoa_representative_in_halo"].sum()
                    ),
                    "native_edge_lsoa_added_by_exact_provider_point": int(
                        (~charity_outputs[year]["native_lsoa_representative_in_halo"]).sum()
                    ),
                    "provider_lsoa_in_2021_demand_halo": int(
                        charity_outputs[year]["provider_lsoa_in_2021_demand_halo"].sum()
                    ),
                    "provider_lsoa_outside_representative_point_demand_halo": int(
                        (~charity_outputs[year]["provider_lsoa_in_2021_demand_halo"]).sum()
                    ),
                    "provider_lsoa_is_current_internal": int(
                        charity_outputs[year]["provider_lsoa_is_current_internal"].sum()
                    ),
                    "provider_assignment_within": int((~fallback).sum()),
                    "provider_assignment_nearest_fallback": int(fallback.sum()),
                    "maximum_nearest_fallback_distance_m": (
                        float(
                            charity_outputs[year].loc[
                                fallback, "nearest_boundary_distance_m"
                            ].max()
                        )
                        if fallback.any()
                        else 0.0
                    ),
                    "overlap_with_current_internal_charity_ids": len(overlap),
                    "used_in_e2sfca": False,
                    "used_in_od": False,
                }
            )
            gap_rows.append(
                {
                    "year": year,
                    "gap_type": "missing_income",
                    "affected_rows": len(charity_outputs[year]) - with_income,
                    "halo_specific_count_known": True,
                    "handling": "left missing; not converted to zero; log1p left missing",
                }
            )
            gap_rows.append(
                {
                    "year": year,
                    "gap_type": "unresolved_historical_location_before_spatial_filter",
                    "affected_rows": int(all_year["historical_postcode"].isna().sum()),
                    "halo_specific_count_known": False,
                    "handling": (
                        "cannot classify to halo without a defensible dated address; current/undated "
                        "addresses were not back-cast"
                    ),
                }
            )

        write_csv(
            pd.concat(location_frames, ignore_index=True),
            build_root / "charities/external_charity_historical_locations_long.csv",
        )
        write_csv(
            pd.concat(finance_frames, ignore_index=True),
            build_root / "charities/external_charity_finance_log1p_long.csv",
        )
        write_csv(
            pd.concat(assignment_frames, ignore_index=True),
            build_root / "charities/external_provider_2021_lsoa_assignments_long.csv",
        )

        reusable_road = all(
            path.is_file()
            for path in (
                OUTPUT_ROOT / "road_network/OSMM_Highways_2021_SW_45km.gpkg",
                OUTPUT_ROOT / "road_network/road_graph_open_with_ferries_45km.npz",
                OUTPUT_ROOT / "road_network/road_nodes_xyz_45km.npy",
                OUTPUT_ROOT / "qa/road_network_45km_audit.csv",
                OUTPUT_ROOT / "qa/road_form_of_way_audit.csv",
                OUTPUT_ROOT / "qa/road_source_files_inventory.csv",
            )
        )
        if reusable_road:
            road_audit, road_form_of_way, road_source_inventory, road_nodes = (
                reuse_existing_45km_road_network(build_root)
            )
        else:
            road_audit, road_form_of_way, road_source_inventory, road_nodes = (
                prepare_road_network(build_root, study_union)
            )
        snap_audit = road_snap_audit(
            road_nodes,
            demand_halo_boundaries[2021],
            charity_outputs,
        )

        halo_audit = pd.DataFrame(halo_audit_rows)
        demand_audit = pd.DataFrame(demand_audit_rows)
        charity_audit = pd.DataFrame(charity_audit_rows)
        write_csv(pd.DataFrame(boundary_audit_rows), build_root / "qa/boundary_repair_audit.csv")
        write_csv(halo_audit, build_root / "qa/halo_lsoa_audit.csv")
        write_csv(demand_audit, build_root / "qa/demand_completeness_audit.csv")
        write_csv(conservation, build_root / "qa/harmonisation_conservation_audit.csv")
        write_csv(charity_audit, build_root / "qa/charity_completeness_audit.csv")
        write_csv(
            pd.DataFrame(provider_unchanged_rows),
            build_root / "qa/provider_20km_unchanged_audit.csv",
        )
        write_csv(road_audit, build_root / "qa/road_network_45km_audit.csv")
        write_csv(road_form_of_way, build_root / "qa/road_form_of_way_audit.csv")
        write_csv(road_source_inventory, build_root / "qa/road_source_files_inventory.csv")
        write_csv(snap_audit, build_root / "qa/road_snap_preparation_audit.csv")
        existing_filter_audit = OUTPUT_ROOT / "qa/road_filter_exclusion_audit.csv"
        if existing_filter_audit.is_file():
            shutil.copy2(
                existing_filter_audit,
                build_root / "qa/road_filter_exclusion_audit.csv",
            )
        write_csv(
            pd.concat(address_breakdown_frames, ignore_index=True),
            build_root / "qa/address_quality_breakdown.csv",
        )
        write_csv(
            pd.concat(finance_breakdown_frames, ignore_index=True),
            build_root / "qa/finance_method_breakdown.csv",
        )
        write_csv(pd.DataFrame(gap_rows), build_root / "qa/data_gap_register.csv")

        source_manifest = snapshot(required_paths)
        source_manifest["role"] = source_manifest["path"].map(
            lambda path: (
                "Data Spine charity rebuild v2 national geocoded stage"
                if path == str(CHARITY_STAGE)
                else (
                    "existing local December-2021 OS MasterMap Highways source"
                    if str(ROAD_SOURCE_DIR) in path
                    else "frozen project input"
                )
            )
        )
        write_csv(source_manifest, build_root / "qa/source_manifest_sha256.csv", float_format=None)

        method_manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "data_preparation_only_not_analysis",
            "output_root": str(OUTPUT_ROOT),
            "e2sfca_catchment_m": E2SFCA_CATCHMENT_M,
            "provider_halo_distance_m": PROVIDER_HALO_DISTANCE_M,
            "competing_demand_halo_distance_m": DEMAND_HALO_DISTANCE_M,
            "road_buffer_distance_m": ROAD_BUFFER_DISTANCE_M,
            "competing_demand_halo_definition": (
                "Annual LSOAs whose representative points lie outside the dissolved seven-ICB "
                "study boundary and inside its 40 km buffer; current internal LSOA codes excluded."
            ),
            "lsoa_halo_membership_predicate": (
                "representative_point.within(buffer_ring), preserving the original extension "
                "implementation and its existing Shapely default buffer geometry"
            ),
            "provider_halo_definition": (
                "Geocoded historical provider points lying outside the dissolved seven-ICB study "
                "boundary and inside its exact 20 km buffer; current internal charity IDs excluded."
            ),
            "internal_lsoas_untouched": {str(year): len(internal_codes[year]) for year in YEARS},
            "census_sources": {key: str(value) for key, value in CENSUS_PATHS.items()},
            "charity_source": str(CHARITY_STAGE),
            "charity_authority": str(CHARITY_PACKAGE),
            "road_source": str(ROAD_SOURCE_DIR),
            "road_graph_definition": (
                "45 km car-access preparation: retain operationalState=Open RoadLinks only for "
                "Single Carriageway, Dual Carriageway, Roundabout, Slip Road, Traffic Island Link "
                "and Traffic Island Link At Junction; ordinary local/unclassified roads remain "
                "through Single Carriageway; FerryLinks are excluded; endpoints are keyed to 1 mm "
                "plus grade separation and duplicate undirected endpoint pairs retain minimum length."
            ),
            "road_retained_form_of_way": [
                "Single Carriageway",
                "Dual Carriageway",
                "Roundabout",
                "Slip Road",
                "Traffic Island Link",
                "Traffic Island Link At Junction",
            ],
            "road_excluded_form_of_way": [
                "Track",
                "Enclosed Traffic Area",
                "Guided Busway",
                "Layby",
                "Shared Use Carriageway",
            ],
            "road_ferrylink_rule": "excluded",
            "road_filter_script": str(OUTPUT_ROOT / "filter_road_network_45km.py"),
            "road_graph_filename_note": (
                "road_graph_open_with_ferries_45km.npz is retained only for path compatibility; "
                "the current graph contains no FerryLinks."
            ),
            "coordinate_transform_policy": (
                "PROJ network disabled; use the installed offline EPSG:4326 to EPSG:27700 "
                "transformation so no external grid is downloaded"
            ),
            "charity_rules_reused": [
                "Data Spine GB-CHC registration/removal presence at Census date",
                "care_strict care-related charity rule",
                "validated Companies House decisions then dated Charity Commission archive proxy",
                "exact-UID covering period; nearest FYE within 365 days; existing proxy fallback",
                "ONS L522 CPIH to constant 2021 pounds",
                "missing income remains missing; recorded zero retained",
                "charity-level log1p(income_2021_gbp)",
                "coordinate-based assignment to the full repaired 2021 LSOA layer with nearest-boundary fallback",
            ],
            "harmonisation": (
                "2001 and 2011 counts allocated by positive polygon intersection area, "
                "normalised within each source LSOA; rates recomputed after allocation. "
                "2021 is identity. A fixed-target boundary closure is permitted only for an "
                f"otherwise unmatched source polygon within {MAX_CROSSWALK_EDGE_GAP_M:g} m; "
                "the exact gap and method are retained in the crosswalk and QA."
            ),
            "provider_2021_lsoa_assignment": (
                "Historical coordinates spatially joined to the full repaired E/W 2021 LSOA layer, "
                "with the same nearest-boundary fallback used in Step 1. A separate flag states whether "
                "the containing LSOA is in the representative-point-selected demand halo."
            ),
            "extension_geometry_rationale": (
                "A 20 km provider halo requires external competing demand to 40 km because an "
                "external provider can sit 20 km beyond the study boundary and draw demand from "
                "a further 20 km catchment. Road source coverage is study area plus 45 km, retaining "
                "an additional 5 km network envelope outside the demand halo."
            ),
            "explicitly_not_run": [
                "Step 1-4 notebooks",
                "OD",
                "E2SFCA",
                "HP-LA",
                "trajectory",
                "Bi-LISA",
                "BYM2",
            ],
        }
        manifest_path = build_root / "METHOD_MANIFEST.json"
        manifest_path.write_text(
            json.dumps(method_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        protected_after = snapshot(protected_files)
        protected_audit = protected_before.merge(
            protected_after,
            on="path",
            suffixes=("_before", "_after"),
            validate="one_to_one",
        )
        protected_audit["unchanged"] = (
            protected_audit["size_bytes_before"].eq(protected_audit["size_bytes_after"])
            & protected_audit["mtime_ns_before"].eq(protected_audit["mtime_ns_after"])
            & protected_audit["sha256_before"].eq(protected_audit["sha256_after"])
        )
        if not protected_audit["unchanged"].all():
            changed = protected_audit.loc[~protected_audit["unchanged"], "path"].tolist()
            raise AssertionError(f"Protected workflow files changed: {changed}")
        write_csv(
            protected_audit,
            build_root / "qa/protected_files_unchanged_audit.csv",
            float_format=None,
        )

        script_copy = build_root / "prepare_halo_20km.py"
        shutil.copy2(Path(__file__), script_copy)
        filter_script = OUTPUT_ROOT / "filter_road_network_45km.py"
        if filter_script.is_file():
            shutil.copy2(filter_script, build_root / filter_script.name)

        edge_fallback_total = int(
            demand_audit["submetre_boundary_edge_fallback_rows"].sum()
        )
        edge_fallback_max_m = float(demand_audit["maximum_boundary_edge_gap_m"].max())
        report_lines = [
            "# 20 km E2SFCA external-extension data audit",
            "",
            "Status: data preparation only; no OD, E2SFCA or downstream analysis was run.",
            "",
            (
                "Geometry: providers remain in the exact external 20 km ring; competing-demand "
                "LSOAs extend to 40 km and road coverage extends to 45 km."
            ),
            "",
            (
                "2021 provider-context 20 km LSOAs: "
                f"{int(halo_audit.loc[(halo_audit.year.eq(2021)) & (halo_audit.halo_role.eq('external_provider_context')), 'halo_lsoas'].iloc[0]):,}."
            ),
            (
                "2021 external competing-demand 40 km LSOAs: "
                f"{int(halo_audit.loc[(halo_audit.year.eq(2021)) & (halo_audit.halo_role.eq('external_competing_demand')), 'halo_lsoas'].iloc[0]):,}."
            ),
            "",
            "## Demand completeness",
            "",
        ]
        for row in demand_audit.itertuples(index=False):
            report_lines.append(
                f"- {row.year}: native {row.native_rows:,}/{row.expected_native_halo_lsoas:,}; "
                f"harmonised {row.harmonised_target_rows:,}/{row.expected_2021_halo_lsoas:,}; "
                f"missing counts {row.missing_harmonised_counts:,}."
            )
        report_lines.extend(["", "## External charities", ""])
        for row in charity_audit.itertuples(index=False):
            report_lines.append(
                f"- {row.year}: {row.external_eligible_charities:,} charities; "
                f"income {row.with_income_2021_gbp:,}; "
                f"current-rule historical location "
                f"{row.with_reliable_historical_location_under_current_rule:,}."
            )
        road_row = road_audit.iloc[0]
        source_roadlinks = int(
            road_row.get(
                "source_roadlink_rows_intersecting_buffer",
                road_row.get("roadlink_rows_intersecting_buffer", 0),
            )
        )
        retained_open = int(
            road_row.get(
                "retained_open_roadlink_rows",
                road_row.get("open_roadlink_rows_used_in_graph", 0),
            )
        )
        excluded_nonopen = int(road_row.get("excluded_nonopen_roadlink_rows", 0))
        excluded_disallowed = int(
            road_row.get("excluded_open_disallowed_formofway_rows", 0)
        )
        source_ferries = int(
            road_row.get(
                "source_ferrylink_rows",
                road_row.get("ferrylink_rows_used_in_graph", 0),
            )
        )
        used_ferries = int(road_row.get("ferrylink_rows_used_in_graph", 0))
        report_lines.extend(
            [
                "",
                "## Road preparation",
                "",
                (
                    f"- 45 km source extract: {source_roadlinks:,} RoadLinks; "
                    f"{retained_open:,} selected Open RoadLinks and {used_ferries:,} FerryLinks "
                    "represented in the prepared graph."
                ),
                (
                    f"- Road exclusions: {excluded_nonopen:,} non-Open RoadLinks, "
                    f"{excluded_disallowed:,} Open RoadLinks in disallowed formOfWay categories, "
                    f"and {source_ferries - used_ferries:,} FerryLinks."
                ),
                (
                    f"- Prepared graph: {int(road_row['graph_nodes']):,} nodes and "
                    f"{int(road_row['graph_edges_undirected']):,} undirected edges; not used for OD."
                ),
            ]
        )
        report_lines.extend(
            [
                "",
                "## Scope and limitations",
                "",
                "- All selected rows use the current Data Spine charity rebuild v2 and existing Census definitions.",
                "- Current or undated charity addresses were not back-cast. Records without a defensible dated "
                "location cannot be classified as inside or outside the halo, so their halo-specific count is unknown.",
                "- Missing income remains missing and charity-level log1p is not calculated for those rows.",
                "- Census competing-demand membership uses the representative-point rule at 40 km; charity "
                "providers use exact historical coordinates at 20 km.",
                "- 2001/2011 demand is harmonised as counts to 2021 LSOAs; rates are QA-only.",
                (
                    f"- Fixed-target boundary closures: {edge_fallback_total}; maximum exact gap "
                    f"{edge_fallback_max_m:.9f} m. Methods and gaps are retained in the crosswalks, "
                    "and counts remain conserved."
                ),
                "- The 20 km provider records, coordinates, income and provider-LSOA assignments match the "
                "previous extension exactly on the audited stable fields.",
                "- Current internal data, Step 1-4 files and the main Road_network passed before/after SHA-256 checks.",
                "- See output_manifest.csv for every output file and full path.",
                "",
            ]
        )
        (build_root / "DATA_AUDIT_REPORT.md").write_text(
            "\n".join(report_lines), encoding="utf-8"
        )

        manifest_rows = []
        for path in sorted(build_root.rglob("*")):
            if path.is_file() and path.name != "output_manifest.csv":
                relative = path.relative_to(build_root)
                manifest_rows.append(
                    {
                        "relative_path": str(relative),
                        "full_path": str(OUTPUT_ROOT / relative),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                )
        write_csv(
            pd.DataFrame(manifest_rows),
            build_root / "output_manifest.csv",
            float_format=None,
        )

        backup_root = PACKAGE / (
            ".halo-20km-backup-"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
        OUTPUT_ROOT.rename(backup_root)
        try:
            build_root.rename(OUTPUT_ROOT)
        except Exception:
            backup_root.rename(OUTPUT_ROOT)
            raise
        shutil.rmtree(backup_root)
        print("HALO_PACKAGE_REPLACED", OUTPUT_ROOT, flush=True)
        print(halo_audit.to_string(index=False), flush=True)
        print(demand_audit.to_string(index=False), flush=True)
        print(charity_audit.to_string(index=False), flush=True)
    except Exception:
        print(f"BUILD_FAILED_PARTIAL_DIRECTORY {build_root}", flush=True)
        raise


if __name__ == "__main__":
    main()
