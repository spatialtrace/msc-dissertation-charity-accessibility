from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree


def expand_environment_paths(value):
    """Expand environment variables in nested JSON configuration values."""
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [expand_environment_paths(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_environment_paths(item) for key, item in value.items()}
    return value


YEARS = (2001, 2011, 2021)
EXPECTED_INTERNAL_LSOAS = 3411
EXPECTED_INTERNAL_CHARITIES = {2001: 2276, 2011: 3313, 2021: 3996}


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.15g")


class E2SFCAWorkflow:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir).resolve()
        self.config_path = self.run_dir / "run_configuration.json"
        self.config = expand_environment_paths(
            json.loads(self.config_path.read_text(encoding="utf-8"))
        )
        self.version = self.config["version"]
        self.slug = self.config["slug"]
        self.output_dirs = {
            "od": self.run_dir / "od_impedance",
            "results": self.run_dir / "results",
            "qa": self.run_dir / "qa",
            "manifest": self.run_dir / "manifest",
        }
        for directory in self.output_dirs.values():
            directory.mkdir(parents=True, exist_ok=True)
        self.initial_input_manifest: pd.DataFrame | None = None
        self.internal_demand: dict[int, pd.DataFrame] = {}
        self.demand: dict[int, pd.DataFrame] = {}
        self.providers: dict[int, pd.DataFrame] = {}
        self.weights: dict[int, np.ndarray] = {}
        self.provider_ratio: dict[int, np.ndarray] = {}
        self.weighted_demand: dict[int, np.ndarray] = {}
        self.accessibility_all: dict[int, np.ndarray] = {}
        self.accessibility_internal: dict[int, np.ndarray] = {}
        self.results: dict[int, pd.DataFrame] = {}

    @property
    def qa_dir(self) -> Path:
        return self.output_dirs["qa"]

    @property
    def result_dir(self) -> Path:
        return self.output_dirs["results"]

    def _configured_input_paths(self) -> list[Path]:
        paths: list[Path] = [self.config_path]
        for value in self.config["common"].values():
            if isinstance(value, dict):
                paths.extend(Path(v) for v in value.values())
            elif isinstance(value, str):
                paths.append(Path(value))
        for key in ("demand_paths", "provider_paths", "internal_provider_paths", "external_charity_paths"):
            for value in self.config.get(key, {}).values():
                paths.append(Path(value))
        for key in ("support_points_path", "graph_path", "nodes_path"):
            if self.config.get(key):
                paths.append(Path(self.config[key]))
        paths.extend(Path(p) for p in self.config.get("method_authority_paths", []))
        unique = {p.resolve(): p.resolve() for p in paths}
        return sorted(unique.values(), key=lambda p: p.as_posix())

    # 1. Load and audit inputs
    def load_and_audit_inputs(self) -> pd.DataFrame:
        paths = self._configured_input_paths()
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing configured inputs: {missing}")
        rows = []
        for path in paths:
            rows.append(
                {
                    "source_path": str(path),
                    "size_bytes": path.stat().st_size,
                    "mtime_ns": path.stat().st_mtime_ns,
                    "sha256_before": sha256(path),
                    "exists": True,
                }
            )
        self.initial_input_manifest = pd.DataFrame(rows)
        write_csv(self.initial_input_manifest, self.output_dirs["manifest"] / "input_manifest_initial.csv")
        audit = pd.DataFrame(
            [
                {
                    "version": self.version,
                    "configured_inputs": len(paths),
                    "missing_inputs": 0,
                    "python": sys.version.split()[0],
                    "platform": platform.platform(),
                    "run_dir": str(self.run_dir),
                    "pass": True,
                }
            ]
        )
        write_csv(audit, self.qa_dir / "input_availability_audit.csv")
        print(audit.to_string(index=False), flush=True)
        return audit

    # 2. Build/load common 2021 spatial foundation
    def load_common_spatial_foundation(self) -> pd.DataFrame:
        common = self.config["common"]
        points = pd.read_csv(common["points_path"], low_memory=False)
        points["lsoa_code"] = points["lsoa_code"].astype(str).str.strip()
        if len(points) != EXPECTED_INTERNAL_LSOAS or not points["lsoa_code"].is_unique:
            raise AssertionError("Common representative points are not the expected 3,411 unique LSOAs")
        for column in ("easting", "northing"):
            points[column] = pd.to_numeric(points[column], errors="raise")
        self.points = points.sort_values("lsoa_code").reset_index(drop=True)
        code_set = set(self.points["lsoa_code"])

        rows = []
        for year in YEARS:
            frame = pd.read_csv(common["covariate_paths"][str(year)], low_memory=False)
            frame["lsoa_code"] = frame["lsoa_code"].astype(str).str.strip()
            frame["care50_num"] = pd.to_numeric(frame["care50_num"], errors="raise")
            frame["population_5plus"] = pd.to_numeric(frame["population_5plus"], errors="raise")
            if len(frame) != EXPECTED_INTERNAL_LSOAS or not frame["lsoa_code"].is_unique:
                raise AssertionError(f"{year}: internal common geography is not 3,411 unique LSOAs")
            if set(frame["lsoa_code"]) != code_set:
                raise AssertionError(f"{year}: internal common geography code set differs")
            if frame[["care50_num", "population_5plus"]].isna().any().any():
                raise AssertionError(f"{year}: missing Care50 demand")
            if not frame["care50_num"].ge(0).all() or not frame["population_5plus"].gt(0).all():
                raise AssertionError(f"{year}: invalid demand values")
            frame = frame.merge(self.points, on=["lsoa_code", "lsoa_name"], how="left", validate="one_to_one")
            self.internal_demand[year] = frame.sort_values("lsoa_code").reset_index(drop=True)
            rows.append(
                {
                    "year": year,
                    "rows": len(frame),
                    "unique_lsoas": frame["lsoa_code"].nunique(),
                    "same_fixed_2021_code_set": set(frame["lsoa_code"]) == code_set,
                    "care50_missing": int(frame["care50_num"].isna().sum()),
                    "population_5plus_missing": int(frame["population_5plus"].isna().sum()),
                    "pass": True,
                }
            )

        conservation = pd.read_csv(common["harmonisation_audit_path"], low_memory=False)
        if "conservation_pass" in conservation.columns:
            passed = conservation["conservation_pass"].astype(str).str.lower().eq("true")
            if not passed.all():
                raise AssertionError("Existing counts-first common-geography conservation audit failed")
        foundation_path = Path(common["foundation_gpkg"])
        foundation_uri = f"file:{foundation_path.as_posix()}?mode=ro&immutable=1"
        with sqlite3.connect(foundation_uri, uri=True) as connection:
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
            boundary_rows = connection.execute('SELECT COUNT(*) FROM "lsoa_2021"').fetchone()[0]
        if integrity != "ok" or boundary_rows != EXPECTED_INTERNAL_LSOAS:
            raise AssertionError("Common spatial foundation boundary layer is not 3,411 rows")
        audit = pd.DataFrame(rows)
        write_csv(audit, self.qa_dir / "common_spatial_foundation_audit.csv")
        print(audit.to_string(index=False), flush=True)
        return audit

    def _validate_data_spine_capacity(self, year: int, internal_provider: pd.DataFrame) -> dict[str, float | int | bool]:
        path = Path(self.config["common"]["charity_paths"][str(year)])
        charity = pd.read_csv(path, low_memory=False)
        if len(charity) != EXPECTED_INTERNAL_CHARITIES[year] or not charity["charity_number"].is_unique:
            raise AssertionError(f"{year}: authoritative Data Spine charity count/uniqueness changed")
        income = pd.to_numeric(charity["income_2021_gbp"], errors="coerce")
        if not income.dropna().ge(0).all():
            raise AssertionError(f"{year}: negative constant-price income")
        expected_capacity = float(np.log1p(income.dropna().to_numpy(dtype=float)).sum())
        actual_capacity = float(internal_provider["registered_capacity_log1p_income"].sum())
        if not np.isclose(expected_capacity, actual_capacity, atol=1e-8, rtol=1e-12):
            raise AssertionError(f"{year}: internal provider capacity no longer matches Data Spine")
        return {
            "authoritative_charities": len(charity),
            "usable_income_charities": int(income.notna().sum()),
            "data_spine_capacity": expected_capacity,
            "provider_capacity": actual_capacity,
            "capacity_match": True,
        }

    # 3. Load provider capacity and Care50 demand
    def load_provider_capacity_and_demand(self) -> pd.DataFrame:
        if not self.internal_demand:
            raise RuntimeError("Load the common spatial foundation first")
        internal_codes = set(self.points["lsoa_code"])
        audit_rows = []
        prepared_internal_comparison_rows = []

        exact_halo = self.config.get("boundary_support_mode") == "exact_20km_provider_40km_demand"
        if exact_halo:
            support_points = pd.read_csv(self.config["support_points_path"], low_memory=False)
            support_points = support_points.rename(
                columns={
                    "representative_easting": "easting",
                    "representative_northing": "northing",
                }
            )
            support_points["lsoa_code"] = support_points["lsoa_code"].astype(str).str.strip()
            support_points = support_points[["lsoa_code", "lsoa_name", "easting", "northing"]]
            if set(support_points["lsoa_code"]) & internal_codes:
                raise AssertionError("Distance external demand support overlaps internal LSOAs")

        for year in YEARS:
            internal = self.internal_demand[year].copy()
            internal["study_scope"] = "internal_result_area"

            if exact_halo:
                external = pd.read_csv(self.config["demand_paths"][str(year)], low_memory=False).rename(
                    columns={"lsoa_2021_code": "lsoa_code", "lsoa_2021_name": "lsoa_name"}
                )
                external["lsoa_code"] = external["lsoa_code"].astype(str).str.strip()
                external = external.merge(support_points, on=["lsoa_code", "lsoa_name"], how="left", validate="one_to_one")
                external["study_scope"] = "external_competing_demand_support"
            else:
                prepared = pd.read_csv(self.config["demand_paths"][str(year)], low_memory=False).rename(
                    columns={
                        "lsoa_2021_code": "lsoa_code",
                        "lsoa_2021_name": "lsoa_name",
                        "representative_easting": "easting",
                        "representative_northing": "northing",
                    }
                )
                prepared["lsoa_code"] = prepared["lsoa_code"].astype(str).str.strip()
                overlap = prepared.loc[prepared["lsoa_code"].isin(internal_codes)].merge(
                    internal[["lsoa_code", "care50_num", "population_5plus"]],
                    on="lsoa_code",
                    suffixes=("_prepared", "_common"),
                    validate="one_to_one",
                )
                if len(overlap) != EXPECTED_INTERNAL_LSOAS:
                    raise AssertionError(f"{year}: travel-time support does not contain all internal LSOAs")
                for field in ("care50_num", "population_5plus"):
                    prepared_values = overlap[f"{field}_prepared"].to_numpy(dtype=float)
                    common_values = overlap[f"{field}_common"].to_numpy(dtype=float)
                    difference = prepared_values - common_values
                    prepared_internal_comparison_rows.append(
                        {
                            "year": year,
                            "field": field,
                            "rows_compared": len(overlap),
                            "prepared_total": float(prepared_values.sum()),
                            "current_common2021_total": float(common_values.sum()),
                            "signed_total_difference": float(difference.sum()),
                            "maximum_absolute_row_difference": float(np.max(np.abs(difference))),
                            "all_values_match": bool(np.allclose(prepared_values, common_values, atol=1e-8, rtol=1e-12)),
                            "selected_internal_authority": "current unified-lsoa common2021 covariates",
                            "prepared_internal_rows_used_in_e2sfca": False,
                        }
                    )
                external = prepared.loc[~prepared["lsoa_code"].isin(internal_codes)].copy()
                external["study_scope"] = "external_provider_centred_support"

            for field in ("care50_num", "population_5plus", "easting", "northing"):
                external[field] = pd.to_numeric(external[field], errors="raise")
            demand_columns = ["lsoa_code", "lsoa_name", "care50_num", "population_5plus", "easting", "northing", "study_scope"]
            combined_demand = pd.concat([internal[demand_columns], external[demand_columns]], ignore_index=True)
            combined_demand = combined_demand.sort_values("lsoa_code").reset_index(drop=True)
            if not combined_demand["lsoa_code"].is_unique:
                raise AssertionError(f"{year}: duplicate demand support LSOA")
            combined_demand["care50_rate"] = combined_demand["care50_num"] / combined_demand["population_5plus"]
            self.demand[year] = combined_demand

            if exact_halo:
                provider_internal = pd.read_csv(self.config["internal_provider_paths"][str(year)], low_memory=False)
                provider_internal = provider_internal.rename(columns={"lsoa_code": "provider_lsoa_code"})
                provider_internal["provider_scope"] = "internal_authoritative"
                provider_internal = provider_internal.merge(
                    self.points.rename(columns={"lsoa_code": "provider_lsoa_code"}),
                    on="provider_lsoa_code",
                    how="left",
                    validate="one_to_one",
                )
                provider_internal = provider_internal.rename(columns={"easting_x": "easting", "northing_x": "northing"})
                for column in ("easting_y", "northing_y"):
                    if column in provider_internal:
                        provider_internal = provider_internal.drop(columns=column)

                charity_external = pd.read_csv(self.config["external_charity_paths"][str(year)], low_memory=False)
                charity_external["income_2021_gbp"] = pd.to_numeric(charity_external["income_2021_gbp"], errors="coerce")
                stored_log = pd.to_numeric(charity_external["log1p_income_2021_gbp"], errors="coerce")
                recalculated = np.log1p(charity_external["income_2021_gbp"])
                comparable = stored_log.notna() & recalculated.notna()
                if comparable.any() and not np.allclose(stored_log[comparable], recalculated[comparable], atol=1e-10, rtol=1e-12):
                    raise AssertionError(f"{year}: stored external charity log1p capacity differs")
                charity_external["log_capacity"] = recalculated
                usable = charity_external.loc[charity_external["log_capacity"].notna()].copy()
                provider_external = usable.groupby("provider_lsoa_2021_code", as_index=False).agg(
                    registered_capacity_log1p_income=("log_capacity", "sum"),
                    charity_records=("uid", "size"),
                ).rename(columns={"provider_lsoa_2021_code": "provider_lsoa_code"})
                all_records = charity_external.groupby("provider_lsoa_2021_code").size()
                provider_external["all_eligible_charity_records"] = provider_external["provider_lsoa_code"].map(all_records).astype(int)
                provider_external["provider_scope"] = "external_20km_candidate"
                provider_external = provider_external.merge(
                    support_points.rename(columns={"lsoa_code": "provider_lsoa_code"}),
                    on="provider_lsoa_code",
                    how="left",
                    validate="one_to_one",
                )
                if set(provider_internal["provider_lsoa_code"]) & set(provider_external["provider_lsoa_code"]):
                    raise AssertionError(f"{year}: internal/external provider LSOAs overlap")
                provider = pd.concat([provider_internal, provider_external], ignore_index=True, sort=False)
                provider["charity_records_with_usable_income"] = provider["charity_records"].astype(int)
            else:
                provider = pd.read_csv(self.config["provider_paths"][str(year)], low_memory=False).rename(
                    columns={
                        "provider_lsoa_2021_code": "provider_lsoa_code",
                        "provider_lsoa_2021_name": "lsoa_name",
                        "representative_easting": "easting",
                        "representative_northing": "northing",
                    }
                )
                provider["provider_scope"] = np.where(
                    provider["provider_lsoa_code"].isin(internal_codes),
                    "internal_authoritative",
                    "external_network_screened_30min_candidate",
                )
                provider["charity_records"] = provider["charity_records_with_usable_income"]

            provider["provider_lsoa_code"] = provider["provider_lsoa_code"].astype(str).str.strip()
            provider["registered_capacity_log1p_income"] = pd.to_numeric(
                provider["registered_capacity_log1p_income"], errors="raise"
            )
            for field in ("easting", "northing"):
                provider[field] = pd.to_numeric(provider[field], errors="raise")
            if provider["provider_lsoa_code"].duplicated().any():
                raise AssertionError(f"{year}: duplicate provider LSOA")
            if not provider["registered_capacity_log1p_income"].ge(0).all():
                raise AssertionError(f"{year}: negative capacity")
            if not set(provider["provider_lsoa_code"]).issubset(set(combined_demand["lsoa_code"])):
                raise AssertionError(f"{year}: provider LSOA is outside prepared competing-demand support")
            provider = provider.sort_values("provider_lsoa_code").reset_index(drop=True)
            self.providers[year] = provider

            authority = self._validate_data_spine_capacity(
                year, provider.loc[provider["provider_scope"].eq("internal_authoritative")]
            )
            audit_rows.append(
                {
                    "year": year,
                    "internal_result_lsoas": int(combined_demand["study_scope"].eq("internal_result_area").sum()),
                    "external_demand_support_lsoas": int((~combined_demand["study_scope"].eq("internal_result_area")).sum()),
                    "total_demand_support_lsoas": len(combined_demand),
                    "care50_count_total_support": float(combined_demand["care50_num"].sum()),
                    "provider_lsoas": len(provider),
                    "internal_provider_lsoas": int(provider["provider_scope"].eq("internal_authoritative").sum()),
                    "external_provider_lsoas": int((~provider["provider_scope"].eq("internal_authoritative")).sum()),
                    "capacity_log1p_income_total": float(provider["registered_capacity_log1p_income"].sum()),
                    **authority,
                    "pass": True,
                }
            )

        demand_sets = [tuple(frame["lsoa_code"]) for frame in self.demand.values()]
        if not all(item == demand_sets[0] for item in demand_sets[1:]):
            raise AssertionError("Demand-support code/order is not fixed across years")
        audit = pd.DataFrame(audit_rows)
        write_csv(audit, self.qa_dir / "provider_demand_input_audit.csv")
        if prepared_internal_comparison_rows:
            write_csv(
                pd.DataFrame(prepared_internal_comparison_rows),
                self.qa_dir / "prepared_internal_demand_comparison_audit.csv",
            )
        print(audit.to_string(index=False), flush=True)
        return audit

    def _connector_cost(self, snap_m: np.ndarray) -> np.ndarray:
        if self.config["impedance_unit"] == "metres":
            return snap_m.astype(np.float64)
        speed = float(self.config["connector_speed_kmh"])
        return snap_m.astype(np.float64) / 1000.0 / speed * 60.0

    # 4. Load/calculate OD impedance
    def calculate_od_impedance(self) -> pd.DataFrame:
        graph_path = Path(self.config["graph_path"])
        nodes_path = Path(self.config["nodes_path"])
        graph = load_npz(graph_path).tocsr()
        nodes = np.load(nodes_path, mmap_mode="r")
        if graph.shape[0] != graph.shape[1] or graph.shape[0] != len(nodes):
            raise AssertionError("Graph/node shape mismatch")
        component_count, labels = connected_components(graph, directed=False)
        surface_nodes = np.flatnonzero(nodes[:, 2] == 0)
        if not len(surface_nodes):
            raise AssertionError("No surface-grade nodes")
        tree = cKDTree(nodes[surface_nodes, :2] / 1000.0)

        demand_master = self.demand[2001][["lsoa_code", "lsoa_name", "easting", "northing", "study_scope"]].copy()
        provider_frames = []
        for year, frame in self.providers.items():
            x = frame[["provider_lsoa_code", "lsoa_name", "easting", "northing", "provider_scope"]].copy()
            x["year"] = year
            provider_frames.append(x)
        provider_long = pd.concat(provider_frames, ignore_index=True)
        coordinate_check = provider_long.groupby("provider_lsoa_code").agg(
            easting_min=("easting", "min"), easting_max=("easting", "max"),
            northing_min=("northing", "min"), northing_max=("northing", "max"),
        )
        if not np.allclose(coordinate_check["easting_min"], coordinate_check["easting_max"], atol=1e-6, rtol=0):
            raise AssertionError("Provider representative easting changes across years")
        if not np.allclose(coordinate_check["northing_min"], coordinate_check["northing_max"], atol=1e-6, rtol=0):
            raise AssertionError("Provider representative northing changes across years")
        provider_master = (
            provider_long.sort_values(["provider_lsoa_code", "year"])
            .drop_duplicates("provider_lsoa_code")
            .sort_values("provider_lsoa_code")
            .reset_index(drop=True)
        )
        provider_master["provider_scope"] = np.where(
            provider_master["provider_lsoa_code"].isin(set(self.points["lsoa_code"])),
            "internal_authoritative",
            "external_candidate",
        )

        def snap(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            xy = frame[["easting", "northing"]].to_numpy(dtype=float)
            snap_m, positions = tree.query(xy, k=1)
            node = surface_nodes[positions]
            return node.astype(np.int64), snap_m.astype(np.float64), labels[node].astype(np.int64)

        demand_node, demand_snap_m, demand_component = snap(demand_master)
        provider_node, provider_snap_m, provider_component = snap(provider_master)
        demand_connector = self._connector_cost(demand_snap_m)
        provider_connector = self._connector_cost(provider_snap_m)
        max_impedance = float(self.config["max_impedance"])

        started = time.time()
        matrix = np.full((len(demand_master), len(provider_master)), np.inf, dtype=np.float32)
        batch_size = int(self.config.get("dijkstra_batch_size", 16))
        for start in range(0, len(provider_master), batch_size):
            stop = min(start + batch_size, len(provider_master))
            graph_distance = dijkstra(
                graph,
                directed=False,
                indices=provider_node[start:stop],
                limit=max_impedance,
            )
            total = (
                graph_distance[:, demand_node].T
                + demand_connector[:, None]
                + provider_connector[None, start:stop]
            )
            total[total > max_impedance] = np.inf
            matrix[:, start:stop] = total.astype(np.float32)
            if start == 0 or stop == len(provider_master) or stop % 400 < batch_size:
                print(
                    f"{self.version} OD: {stop:,}/{len(provider_master):,} provider LSOAs; "
                    f"{(time.time() - started):.1f} s",
                    flush=True,
                )

        cross_component = demand_component[:, None] != provider_component[None, :]
        if np.isfinite(matrix[cross_component]).any():
            raise AssertionError("Cross-component graph pairs became finite")
        finite = np.isfinite(matrix)
        if finite.any() and float(matrix[finite].max()) > max_impedance:
            raise AssertionError("OD cache exceeds catchment")

        self.graph = graph
        self.nodes = nodes
        self.demand_master = demand_master
        self.provider_master = provider_master
        self.od_master = matrix
        self.demand_node = demand_node
        self.provider_node = provider_node
        self.demand_snap_m = demand_snap_m
        self.provider_snap_m = provider_snap_m
        self.demand_component = demand_component
        self.provider_component = provider_component

        np.savez_compressed(
            self.output_dirs["od"] / f"od_master_{self.slug}.npz",
            impedance=matrix,
            demand_lsoa=demand_master["lsoa_code"].to_numpy(dtype="U9"),
            provider_lsoa=provider_master["provider_lsoa_code"].to_numpy(dtype="U9"),
            max_impedance=np.asarray([max_impedance], dtype=float),
            impedance_unit=np.asarray([self.config["impedance_unit"]]),
            graph_sha256=np.asarray([sha256(graph_path)]),
            nodes_sha256=np.asarray([sha256(nodes_path)]),
            connector_rule=np.asarray([self.config["connector_rule"]]),
        )

        provider_index = {code: i for i, code in enumerate(provider_master["provider_lsoa_code"])}
        year_rows = []
        self.year_od: dict[int, np.ndarray] = {}
        for year in YEARS:
            columns = np.asarray([provider_index[code] for code in self.providers[year]["provider_lsoa_code"]], dtype=int)
            year_matrix = matrix[:, columns]
            self.year_od[year] = year_matrix
            np.savez_compressed(
                self.output_dirs["od"] / f"od_impedance_{year}_{self.slug}.npz",
                impedance=year_matrix,
                demand_lsoa=demand_master["lsoa_code"].to_numpy(dtype="U9"),
                provider_lsoa=self.providers[year]["provider_lsoa_code"].to_numpy(dtype="U9"),
                year=np.asarray([year], dtype=np.int16),
                max_impedance=np.asarray([max_impedance], dtype=float),
                impedance_unit=np.asarray([self.config["impedance_unit"]]),
            )
            finite_year = np.isfinite(year_matrix)
            year_rows.append(
                {
                    "year": year,
                    "demand_support_lsoas": year_matrix.shape[0],
                    "provider_lsoas": year_matrix.shape[1],
                    "all_candidate_pairs": year_matrix.size,
                    "finite_true_network_pairs": int(finite_year.sum()),
                    "unreachable_or_over_catchment_pairs": int((~finite_year).sum()),
                    "finite_min_impedance": float(year_matrix[finite_year].min()) if finite_year.any() else np.nan,
                    "finite_max_impedance": float(year_matrix[finite_year].max()) if finite_year.any() else np.nan,
                    "max_impedance": max_impedance,
                    "impedance_unit": self.config["impedance_unit"],
                    "pass": bool(finite_year.any()),
                }
            )

        od_audit = pd.DataFrame(year_rows)
        write_csv(od_audit, self.qa_dir / "od_impedance_audit.csv")
        snap_audit = pd.DataFrame(
            [
                {
                    "role": role,
                    "locations": len(values),
                    "median_snap_m": float(np.median(values)),
                    "p95_snap_m": float(np.quantile(values, 0.95)),
                    "p99_snap_m": float(np.quantile(values, 0.99)),
                    "max_snap_m": float(np.max(values)),
                    "unique_graph_components": int(len(np.unique(components))),
                }
                for role, values, components in (
                    ("demand", demand_snap_m, demand_component),
                    ("provider", provider_snap_m, provider_component),
                )
            ]
        )
        write_csv(snap_audit, self.qa_dir / "network_snap_audit.csv")
        graph_audit = pd.DataFrame(
            [
                {
                    "graph_nodes": graph.shape[0],
                    "stored_directed_entries": graph.nnz,
                    "graph_components": component_count,
                    "surface_nodes": len(surface_nodes),
                    "graph_symmetric": bool((graph != graph.T).nnz == 0),
                    "cross_component_finite_pairs": int(np.isfinite(matrix[cross_component]).sum()),
                    "graph_sha256": sha256(graph_path),
                    "nodes_sha256": sha256(nodes_path),
                    "pass": True,
                }
            ]
        )
        write_csv(graph_audit, self.qa_dir / "road_graph_execution_audit.csv")
        self._write_cross_sea_audit()
        print(od_audit.to_string(index=False), flush=True)
        return od_audit

    def _write_cross_sea_audit(self) -> None:
        pattern = r"Swansea|Isle of Wight"
        internal_demand = self.demand_master["study_scope"].eq("internal_result_area").to_numpy()
        internal_provider = self.provider_master["provider_scope"].eq("internal_authoritative").to_numpy()
        rows = []
        for role, frame, name_col, matrix, candidate_axis in (
            ("demand", self.demand_master, "lsoa_name", self.od_master, 0),
            ("provider", self.provider_master, "lsoa_name", self.od_master, 1),
        ):
            selected = frame[name_col].astype(str).str.contains(pattern, case=False, regex=True, na=False)
            for position in np.flatnonzero(selected):
                if candidate_axis == 0:
                    values = matrix[position, internal_provider]
                    component = self.demand_component[position]
                    code = frame.iloc[position]["lsoa_code"]
                else:
                    values = matrix[internal_demand, position]
                    component = self.provider_component[position]
                    code = frame.iloc[position]["provider_lsoa_code"]
                finite = values[np.isfinite(values)]
                rows.append(
                    {
                        "role": role,
                        "lsoa_code": code,
                        "lsoa_name": frame.iloc[position][name_col],
                        "graph_component": int(component),
                        "true_network_pairs_to_internal_counterpart": int(len(finite)),
                        "minimum_true_network_impedance": float(finite.min()) if len(finite) else np.nan,
                        "buffer_membership_used_as_pair_eligibility": False,
                        "pair_eligibility_rule": "finite graph path including both connectors and within catchment",
                    }
                )
        if not rows:
            rows.append(
                {
                    "role": "candidate_name_check",
                    "lsoa_code": "not_present",
                    "lsoa_name": "No Swansea or Isle of Wight LSOA in prepared candidate tables",
                    "graph_component": np.nan,
                    "true_network_pairs_to_internal_counterpart": 0,
                    "minimum_true_network_impedance": np.nan,
                    "buffer_membership_used_as_pair_eligibility": False,
                    "pair_eligibility_rule": "finite graph path including both connectors and within catchment",
                }
            )
        write_csv(pd.DataFrame(rows), self.qa_dir / "cross_sea_candidate_network_audit.csv")

    def _impedance_weights(self, impedance: np.ndarray) -> np.ndarray:
        b1, b2, b3 = [float(x) for x in self.config["bands"]]
        w1, w2, w3 = [float(x) for x in self.config["weights"]]
        return np.select(
            [impedance <= b1, impedance <= b2, impedance <= b3],
            [w1, w2, w3],
            default=0.0,
        ).astype(np.float64)

    # 5. E2SFCA Step 1: provider supply-demand ratio
    def e2sfca_step1(self) -> pd.DataFrame:
        rows = []
        for year in YEARS:
            impedance = self.year_od[year]
            weights = self._impedance_weights(impedance)
            if not np.all(weights[~np.isfinite(impedance)] == 0):
                raise AssertionError(f"{year}: unreachable pair received a weight")
            demand_value = self.demand[year]["care50_num"].to_numpy(dtype=float)
            capacity = self.providers[year]["registered_capacity_log1p_income"].to_numpy(dtype=float)
            denominator = demand_value @ weights
            positive_capacity_without_demand = (capacity > 0) & (denominator <= 0)
            if positive_capacity_without_demand.any():
                raise AssertionError(f"{year}: positive provider capacity has no true-network competing demand")
            ratio = np.divide(capacity, denominator, out=np.zeros_like(capacity), where=denominator > 0)
            self.weights[year] = weights
            self.weighted_demand[year] = denominator
            self.provider_ratio[year] = ratio

            external_demand = ~self.demand[year]["study_scope"].eq("internal_result_area").to_numpy()
            ext_weighted = self.demand[year].loc[external_demand, "care50_num"].to_numpy(dtype=float) @ weights[external_demand, :]
            diagnostics = self.providers[year][[
                "provider_lsoa_code", "provider_scope", "registered_capacity_log1p_income"
            ]].copy()
            diagnostics.insert(0, "year", year)
            diagnostics["distance_or_time_weighted_care50_demand"] = denominator
            diagnostics["external_weighted_care50_demand"] = ext_weighted
            diagnostics["provider_supply_demand_ratio_R"] = ratio
            diagnostics["true_network_demand_pairs"] = np.isfinite(impedance).sum(axis=0)
            diagnostics["weighted_demand_pairs"] = (weights > 0).sum(axis=0)
            write_csv(diagnostics, self.result_dir / f"provider_ratio_{year}.csv")
            rows.append(
                {
                    "year": year,
                    "provider_lsoas": len(capacity),
                    "providers_with_positive_capacity": int((capacity > 0).sum()),
                    "positive_capacity_without_demand": int(positive_capacity_without_demand.sum()),
                    "sum_capacity": float(capacity.sum()),
                    "sum_weighted_competing_demand": float(denominator.sum()),
                    "sum_external_weighted_competing_demand": float(ext_weighted.sum()),
                    "pass": True,
                }
            )
        audit = pd.DataFrame(rows)
        write_csv(audit, self.qa_dir / "e2sfca_step1_provider_ratio_audit.csv")
        print(audit.to_string(index=False), flush=True)
        return audit

    # 6. E2SFCA Step 2: LSOA accessibility A
    def e2sfca_step2(self) -> pd.DataFrame:
        rows = []
        for year in YEARS:
            access_all = self.weights[year] @ self.provider_ratio[year]
            if not np.isfinite(access_all).all() or not (access_all >= 0).all():
                raise AssertionError(f"{year}: invalid accessibility")
            internal = self.demand[year]["study_scope"].eq("internal_result_area").to_numpy()
            external_provider = ~self.providers[year]["provider_scope"].eq("internal_authoritative").to_numpy()
            ext_contribution = (
                self.weights[year][internal][:, external_provider]
                @ self.provider_ratio[year][external_provider]
            )
            self.accessibility_all[year] = access_all
            self.accessibility_internal[year] = access_all[internal]
            rows.append(
                {
                    "year": year,
                    "support_lsoas": len(access_all),
                    "internal_result_lsoas": int(internal.sum()),
                    "internal_A_min": float(access_all[internal].min()),
                    "internal_A_median": float(np.median(access_all[internal])),
                    "internal_A_max": float(access_all[internal].max()),
                    "internal_lsoas_with_external_provider_contribution": int((ext_contribution > 0).sum()),
                    "sum_external_provider_contribution_to_internal_A": float(ext_contribution.sum()),
                    "pass": int(internal.sum()) == EXPECTED_INTERNAL_LSOAS,
                }
            )
        audit = pd.DataFrame(rows)
        write_csv(audit, self.qa_dir / "e2sfca_step2_accessibility_audit.csv")
        print(audit.to_string(index=False), flush=True)
        return audit

    # 7. Standardise A to A* with fixed 2001 South West benchmark
    def standardise_accessibility(self) -> pd.DataFrame:
        demand_2001 = self.internal_demand[2001]["care50_num"].to_numpy(dtype=float)
        baseline = float(np.average(self.accessibility_internal[2001], weights=demand_2001))
        if not baseline > 0:
            raise AssertionError("Invalid 2001 South West benchmark")
        self.baseline_2001 = baseline
        rows = []
        self.astar: dict[int, np.ndarray] = {}
        for year in YEARS:
            values = self.accessibility_internal[year] / baseline
            self.astar[year] = values
            weighted_mean = float(np.average(values, weights=self.internal_demand[year]["care50_num"]))
            rows.append(
                {
                    "year": year,
                    "fixed_baseline_A_2001_SW_Care50_weighted": baseline,
                    "care50_weighted_mean_Astar": weighted_mean,
                    "baseline_is_fixed_across_years": True,
                    "pass": year != 2001 or abs(weighted_mean - 1.0) < 1e-12,
                }
            )
        audit = pd.DataFrame(rows)
        if not audit["pass"].all():
            raise AssertionError("A* standardisation failed")
        write_csv(audit, self.qa_dir / "standardisation_audit.csv")
        print(audit.to_string(index=False), flush=True)
        return audit

    # 8. Summary statistics and final 3,411-LSOA files
    def write_descriptive_outputs(self) -> pd.DataFrame:
        summary_rows = []
        for year in YEARS:
            frame = self.internal_demand[year].copy()
            frame["accessibility_A"] = self.accessibility_internal[year]
            frame["accessibility_Astar"] = self.astar[year]
            frame["benchmark_relative_accessibility_deficit"] = 1.0 - frame["accessibility_Astar"]
            frame["below_2001_baseline"] = frame["accessibility_Astar"].lt(1.0).astype(int)
            local_capacity = self.providers[year].set_index("provider_lsoa_code")["registered_capacity_log1p_income"]
            frame["registered_capacity_log1p_income"] = frame["lsoa_code"].map(local_capacity).fillna(0.0)
            if len(frame) != EXPECTED_INTERNAL_LSOAS or not frame["lsoa_code"].is_unique:
                raise AssertionError(f"{year}: final result is not 3,411 unique internal LSOAs")
            self.results[year] = frame
            write_csv(frame, self.result_dir / f"lsoa_accessibility_{year}_{self.slug}.csv")
            D = frame["care50_num"].to_numpy(dtype=float)
            summary_rows.append(
                {
                    "year": year,
                    "lsoas": len(frame),
                    "fixed_baseline_A_2001_SW_Care50_weighted": self.baseline_2001,
                    "mean_A": float(frame["accessibility_A"].mean()),
                    "care50_weighted_mean_A": float(np.average(frame["accessibility_A"], weights=D)),
                    "mean_Astar": float(frame["accessibility_Astar"].mean()),
                    "care50_weighted_mean_Astar": float(np.average(frame["accessibility_Astar"], weights=D)),
                    "median_Astar": float(frame["accessibility_Astar"].median()),
                    "p10_Astar": float(frame["accessibility_Astar"].quantile(0.10)),
                    "p90_Astar": float(frame["accessibility_Astar"].quantile(0.90)),
                    "min_Astar": float(frame["accessibility_Astar"].min()),
                    "max_Astar": float(frame["accessibility_Astar"].max()),
                    "lsoas_below_2001_baseline": int(frame["below_2001_baseline"].sum()),
                }
            )
        summary = pd.DataFrame(summary_rows)
        write_csv(summary, self.result_dir / "accessibility_summary_2001_2011_2021.csv")
        combined = pd.concat([self.results[y] for y in YEARS], ignore_index=True)
        write_csv(combined, self.result_dir / f"lsoa_accessibility_all_years_{self.slug}.csv")
        print(summary.to_string(index=False), flush=True)
        return summary

    # 9. QA / conservation / boundary checks
    def run_final_qa(self) -> pd.DataFrame:
        conservation_rows = []
        boundary_rows = []
        completeness_rows = []
        for year in YEARS:
            demand_value = self.demand[year]["care50_num"].to_numpy(dtype=float)
            capacity = self.providers[year]["registered_capacity_log1p_income"].to_numpy(dtype=float)
            lhs = float(demand_value @ self.accessibility_all[year])
            rhs = float(capacity.sum())
            relative_error = abs(lhs - rhs) / max(abs(rhs), 1.0)
            if relative_error >= 1e-10:
                raise AssertionError(f"{year}: supply-accessibility conservation failed")
            conservation_rows.append(
                {
                    "year": year,
                    "support_demand_sum_D_times_A": lhs,
                    "provider_capacity_sum_S": rhs,
                    "absolute_difference": abs(lhs - rhs),
                    "relative_error": relative_error,
                    "conservation_scope": "all prepared internal plus external competing-demand support",
                    "pass": True,
                }
            )

            impedance = self.year_od[year]
            weights = self.weights[year]
            internal_demand = self.demand[year]["study_scope"].eq("internal_result_area").to_numpy()
            external_demand = ~internal_demand
            internal_provider = self.providers[year]["provider_scope"].eq("internal_authoritative").to_numpy()
            external_provider = ~internal_provider
            if not np.all(weights[~np.isfinite(impedance)] == 0):
                raise AssertionError(f"{year}: unreachable pair not excluded")
            boundary_rows.append(
                {
                    "year": year,
                    "external_demand_candidate_pairs": int(external_demand.sum() * len(internal_provider)),
                    "external_demand_true_network_pairs_to_all_providers": int(np.isfinite(impedance[external_demand, :]).sum()),
                    "external_demand_weighted_pairs_to_all_providers": int((weights[external_demand, :] > 0).sum()),
                    "external_provider_candidate_pairs_to_internal_demand": int(internal_demand.sum() * external_provider.sum()),
                    "external_provider_true_network_pairs_to_internal_demand": int(np.isfinite(impedance[internal_demand, :][:, external_provider]).sum()),
                    "external_provider_weighted_pairs_to_internal_demand": int((weights[internal_demand, :][:, external_provider] > 0).sum()),
                    "unreachable_pairs": int((~np.isfinite(impedance)).sum()),
                    "unreachable_pairs_with_nonzero_weight": int((weights[~np.isfinite(impedance)] > 0).sum()),
                    "buffer_membership_used_as_pair_eligibility": False,
                    "network_rule": "finite shortest path plus both connectors and within catchment",
                    "pass": int((weights[~np.isfinite(impedance)] > 0).sum()) == 0,
                }
            )
            frame = self.results[year]
            completeness_rows.append(
                {
                    "year": year,
                    "rows": len(frame),
                    "unique_lsoas": frame["lsoa_code"].nunique(),
                    "duplicate_lsoas": int(frame["lsoa_code"].duplicated().sum()),
                    "missing_A": int(frame["accessibility_A"].isna().sum()),
                    "missing_Astar": int(frame["accessibility_Astar"].isna().sum()),
                    "external_lsoas_in_final_results": int((~frame["lsoa_code"].isin(set(self.points["lsoa_code"]))).sum()),
                    "pass": len(frame) == EXPECTED_INTERNAL_LSOAS and frame["lsoa_code"].is_unique,
                }
            )

        conservation = pd.DataFrame(conservation_rows)
        boundary = pd.DataFrame(boundary_rows)
        completeness = pd.DataFrame(completeness_rows)
        write_csv(conservation, self.qa_dir / "supply_accessibility_conservation_audit.csv")
        write_csv(boundary, self.qa_dir / "boundary_network_catchment_audit.csv")
        write_csv(completeness, self.qa_dir / "result_completeness_audit.csv")

        if self.initial_input_manifest is None:
            raise RuntimeError("Input manifest is unavailable")
        source_rows = []
        for row in self.initial_input_manifest.itertuples(index=False):
            path = Path(row.source_path)
            current_hash = sha256(path)
            source_rows.append(
                {
                    "source_path": str(path),
                    "size_bytes_before": int(row.size_bytes),
                    "size_bytes_after": path.stat().st_size,
                    "mtime_ns_before": int(row.mtime_ns),
                    "mtime_ns_after": path.stat().st_mtime_ns,
                    "sha256_before": row.sha256_before,
                    "sha256_after": current_hash,
                    "unchanged": bool(
                        row.sha256_before == current_hash
                        and int(row.size_bytes) == path.stat().st_size
                        and int(row.mtime_ns) == path.stat().st_mtime_ns
                    ),
                }
            )
        source_audit = pd.DataFrame(source_rows)
        if not source_audit["unchanged"].all():
            raise AssertionError("A configured source file changed during execution")
        write_csv(source_audit, self.qa_dir / "source_files_unchanged_audit.csv")
        print(conservation.to_string(index=False), flush=True)
        print(boundary.to_string(index=False), flush=True)
        print(completeness.to_string(index=False), flush=True)
        return conservation

    # 10. Prepare downstream longitudinal hand-off, without trajectories
    def prepare_longitudinal_handoff(self) -> pd.DataFrame:
        columns = [
            "year", "lsoa_code", "lsoa_name", "ICB23CD", "ICB23NM",
            "care50_num", "population_5plus", "care50_rate",
            "accessibility_A", "accessibility_Astar",
            "benchmark_relative_accessibility_deficit", "below_2001_baseline",
        ]
        handoff = pd.concat([self.results[y][columns] for y in YEARS], ignore_index=True)
        if len(handoff) != EXPECTED_INTERNAL_LSOAS * len(YEARS):
            raise AssertionError("Longitudinal hand-off is not 10,233 rows")
        if handoff.duplicated(["year", "lsoa_code"]).any():
            raise AssertionError("Duplicate LSOA-year in longitudinal hand-off")
        write_csv(handoff, self.result_dir / f"longitudinal_accessibility_input_{self.slug}.csv")
        (self.run_dir / "LONGITUDINAL_HANDOFF.md").write_text(
            "# Longitudinal hand-off\n\n"
            f"This file set contains only the three annual {self.version} accessibility results "
            "on the fixed 3,411 2021 LSOAs. Trajectory classification, mismatch, Bi-LISA, "
            "regression and BYM2 were not run.\n",
            encoding="utf-8",
        )
        method = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "completed_od_and_e2sfca_only_longitudinal_not_run",
            "version": self.version,
            "run_dir": str(self.run_dir),
            "workflow": [
                "Data collection", "common spatial foundation", "OD/network impedance",
                "E2SFCA", "A/A standardisation", "descriptive outputs",
                "longitudinal analysis preparation",
            ],
            "years": list(YEARS),
            "final_internal_lsoas": EXPECTED_INTERNAL_LSOAS,
            "demand": "Care50 count",
            "capacity": "charity-level log1p(income_2021_gbp), then provider-LSOA aggregation",
            "impedance": {
                "unit": self.config["impedance_unit"],
                "catchment": self.config["max_impedance"],
                "bands": self.config["bands"],
                "weights": self.config["weights"],
                "graph": self.config["graph_path"],
                "connector_rule": self.config["connector_rule"],
                "buffer_membership_is_not_pair_eligibility": True,
            },
            "boundary_correction": {
                "candidate_geometry": "external providers within exact 20 km Euclidean halo; external competing-demand representative points within 40 km halo",
                "external_demand": "enters Step 1 only for finite true-network pairs within the 30-minute catchment",
                "external_provider": "enters internal Step 2 only for finite true-network pairs within the 30-minute catchment",
                "final_results": "internal 3,411 fixed-2021 LSOAs only",
            },
            "standardisation": "A divided by fixed 2001 South West Care50-demand-weighted mean A",
            "network_limitations": self.config["network_limitations"],
            "explicitly_not_run": [
                "trajectory classification", "longitudinal decomposition", "mismatch classification",
                "Bi-LISA", "regression", "BYM2",
            ],
        }
        (self.run_dir / "METHOD_MANIFEST.json").write_text(
            json.dumps(method, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        nonexecution = pd.DataFrame(
            [
                {
                    "od_run": True,
                    "e2sfca_run": True,
                    "trajectory_analysis_run": False,
                    "mismatch_run": False,
                    "bi_lisa_run": False,
                    "regression_or_bym2_run": False,
                    "status": "longitudinal_input_prepared_only",
                    "pass": True,
                }
            ]
        )
        write_csv(nonexecution, self.qa_dir / "downstream_nonexecution_audit.csv")
        audit = pd.DataFrame(
            [
                {
                    "rows": len(handoff),
                    "years": handoff["year"].nunique(),
                    "unique_lsoas": handoff["lsoa_code"].nunique(),
                    "duplicate_lsoa_year": int(handoff.duplicated(["year", "lsoa_code"]).sum()),
                    "trajectory_fields_created": 0,
                    "pass": True,
                }
            ]
        )
        write_csv(audit, self.qa_dir / "longitudinal_handoff_audit.csv")
        print(audit.to_string(index=False), flush=True)
        return audit
