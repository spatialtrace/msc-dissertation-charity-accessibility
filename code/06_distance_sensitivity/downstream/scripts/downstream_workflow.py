from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


YEARS = (2001, 2011, 2021)
INTERVALS = ((2001, 2011), (2011, 2021), (2001, 2021))
EXPECTED_LSOAS = 3411
EXPECTED_CHARITIES = {2001: 2276, 2011: 3313, 2021: 3996}
TRAJECTORY_ORDER = [
    "Persistent HP–LA",
    "Emerging HP–LA",
    "Resolved / improved",
    "Intermittent",
    "Never HP–LA",
]
TRAJECTORY_COLOURS = {
    "Persistent HP–LA": "#b2182b",
    "Emerging HP–LA": "#ef8a62",
    "Resolved / improved": "#4daf4a",
    "Intermittent": "#6a51a3",
    "Never HP–LA": "#d9d9d9",
}
STATE_ORDER = ["HP-LA", "HP-HA", "LP-LA", "LP-HA"]


def expand_environment_paths(value):
    """Expand environment variables in nested JSON configuration values."""
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [expand_environment_paths(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_environment_paths(item) for key, item in value.items()}
    return value


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.15g")


def describe_values(values: pd.Series, year: int, variable: str) -> dict[str, object]:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.dropna().astype(float)
    return {
        "year": year,
        "variable": variable,
        "n": len(numeric),
        "nonmissing": len(valid),
        "missing": int(numeric.isna().sum()),
        "sum": float(valid.sum()),
        "mean": float(valid.mean()),
        "std": float(valid.std(ddof=1)),
        "min": float(valid.min()),
        "p05": float(valid.quantile(0.05)),
        "p10": float(valid.quantile(0.10)),
        "p25": float(valid.quantile(0.25)),
        "median": float(valid.median()),
        "p75": float(valid.quantile(0.75)),
        "p90": float(valid.quantile(0.90)),
        "p95": float(valid.quantile(0.95)),
        "max": float(valid.max()),
    }


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    value_array = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    weight_array = pd.to_numeric(weights, errors="raise").to_numpy(dtype=float)
    return float(np.average(value_array, weights=weight_array))


def save_figure(fig: plt.Figure, figure_dir: Path, stem: str) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(figure_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.show()
    plt.close(fig)


def plot_three_continuous(
    maps: dict[int, gpd.GeoDataFrame],
    column: str,
    title: str,
    legend_title: str,
    cmap: str,
    figure_dir: Path,
    stem: str,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    percent: bool = False,
) -> None:
    pooled = np.concatenate([maps[year][column].to_numpy(dtype=float) for year in YEARS])
    if vmin is None:
        vmin = float(np.nanmin(pooled))
    if vmax is None:
        vmax = float(np.nanquantile(pooled, 0.98))
    if not vmax > vmin:
        vmax = vmin + 1.0
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    fig, axes = plt.subplots(1, 3, figsize=(15.3, 5.8))
    for axis, year in zip(axes, YEARS):
        maps[year].plot(column=column, cmap=cmap, norm=norm, linewidth=0, ax=axis)
        axis.set_title(str(year), fontsize=14, fontweight="bold")
        axis.set_axis_off()
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.045, pad=0.045, aspect=45)
    cbar.set_label(legend_title)
    if percent:
        cbar.ax.set_xticklabels([f"{tick:.1%}" for tick in cbar.get_ticks()])
    fig.suptitle(title, fontsize=17, fontweight="bold", y=0.99)
    fig.text(0.5, 0.015, "Common colour scale; values above the pooled P98 use the darkest colour.", ha="center", fontsize=8.5, color="#555555")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.91, bottom=0.14, wspace=0.015)
    save_figure(fig, figure_dir, stem)


def plot_change_panels(
    geometry: gpd.GeoDataFrame,
    frames: dict[int, pd.DataFrame],
    column: str,
    title: str,
    legend_title: str,
    figure_dir: Path,
    stem: str,
    *,
    scale: float = 1.0,
) -> None:
    change_maps = []
    values = []
    for start, end in INTERVALS:
        start_frame = frames[start][["lsoa_code", column]].rename(columns={column: "start_value"})
        end_frame = frames[end][["lsoa_code", column]].rename(columns={column: "end_value"})
        change = start_frame.merge(end_frame, on="lsoa_code", validate="one_to_one")
        change["change"] = (change["end_value"] - change["start_value"]) * scale
        mapped = geometry.merge(change[["lsoa_code", "change"]], on="lsoa_code", validate="one_to_one")
        change_maps.append(((start, end), mapped))
        values.append(mapped["change"].to_numpy(dtype=float))
    pooled = np.concatenate(values)
    limit = float(np.nanquantile(np.abs(pooled), 0.98))
    if not limit > 0:
        limit = 1.0
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    fig, axes = plt.subplots(1, 3, figsize=(15.3, 5.8))
    for axis, ((start, end), mapped) in zip(axes, change_maps):
        mapped.plot(column="change", cmap="RdBu", norm=norm, linewidth=0, ax=axis)
        axis.set_title(f"{start}–{end}", fontsize=14, fontweight="bold")
        axis.set_axis_off()
    sm = plt.cm.ScalarMappable(norm=norm, cmap="RdBu")
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.045, pad=0.045, aspect=45)
    cbar.set_label(legend_title)
    fig.suptitle(title, fontsize=17, fontweight="bold", y=0.99)
    fig.text(0.5, 0.015, "Symmetric colour scale centred on zero; tails beyond pooled P98 absolute change are saturated.", ha="center", fontsize=8.5, color="#555555")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.91, bottom=0.14, wspace=0.015)
    save_figure(fig, figure_dir, stem)


class BaseWorkflow:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir).resolve()
        self.config_path = self.run_dir / "run_configuration.json"
        self.config = expand_environment_paths(
            json.loads(self.config_path.read_text(encoding="utf-8"))
        )
        self.table_dir = self.run_dir / "tables"
        self.figure_dir = self.run_dir / "figures"
        self.qa_dir = self.run_dir / "qa"
        self.manifest_dir = self.run_dir / "manifest"
        for directory in (self.table_dir, self.figure_dir, self.qa_dir, self.manifest_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self.initial_manifest: pd.DataFrame | None = None

    def configured_input_paths(self) -> list[Path]:
        return sorted({Path(item).resolve() for item in self.config["input_paths"]}, key=lambda p: p.as_posix())

    def load_and_audit_inputs(self) -> pd.DataFrame:
        paths = self.configured_input_paths() + [self.config_path]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing configured inputs: {missing}")
        rows = [
            {
                "source_path": str(path),
                "size_bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
                "sha256_before": sha256(path),
                "exists": True,
            }
            for path in paths
        ]
        self.initial_manifest = pd.DataFrame(rows)
        write_csv(self.initial_manifest, self.manifest_dir / "input_manifest_initial.csv")
        audit = pd.DataFrame(
            [{
                "analysis_type": self.config["analysis_type"],
                "configured_inputs": len(paths),
                "missing_inputs": 0,
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "run_dir": str(self.run_dir),
                "pass": True,
            }]
        )
        write_csv(audit, self.qa_dir / "input_availability_audit.csv")
        print(audit.to_string(index=False), flush=True)
        return audit

    def verify_sources_unchanged(self) -> pd.DataFrame:
        if self.initial_manifest is None:
            raise RuntimeError("Initial manifest is unavailable")
        rows = []
        for item in self.initial_manifest.itertuples(index=False):
            path = Path(item.source_path)
            current = sha256(path)
            rows.append(
                {
                    "source_path": str(path),
                    "size_bytes_before": int(item.size_bytes),
                    "size_bytes_after": path.stat().st_size,
                    "mtime_ns_before": int(item.mtime_ns),
                    "mtime_ns_after": path.stat().st_mtime_ns,
                    "sha256_before": item.sha256_before,
                    "sha256_after": current,
                    "unchanged": bool(
                        item.sha256_before == current
                        and int(item.size_bytes) == path.stat().st_size
                        and int(item.mtime_ns) == path.stat().st_mtime_ns
                    ),
                }
            )
        audit = pd.DataFrame(rows)
        if not audit["unchanged"].all():
            raise AssertionError("A configured source changed during downstream analysis")
        write_csv(audit, self.qa_dir / "source_files_unchanged_audit.csv")
        return audit


class DescriptiveWorkflow(BaseWorkflow):
    def load_common_inputs(self) -> pd.DataFrame:
        geometry_path = Path(self.config["boundary_gpkg"])
        self.geometry = gpd.read_file(geometry_path, layer="lsoa_2021").to_crs("EPSG:27700")
        if len(self.geometry) != EXPECTED_LSOAS or not self.geometry["lsoa_code"].is_unique:
            raise AssertionError("Boundary is not 3,411 unique LSOAs")
        code_set = set(self.geometry["lsoa_code"].astype(str))
        self.frames: dict[int, pd.DataFrame] = {}
        self.assignments: dict[int, pd.DataFrame] = {}
        audit_rows = []
        for year in YEARS:
            covariates = pd.read_csv(self.config["covariate_paths"][str(year)], low_memory=False)
            covariates["lsoa_code"] = covariates["lsoa_code"].astype(str)
            if len(covariates) != EXPECTED_LSOAS or not covariates["lsoa_code"].is_unique:
                raise AssertionError(f"{year}: invalid common covariates")
            if set(covariates["lsoa_code"]) != code_set:
                raise AssertionError(f"{year}: common code set differs")
            if not np.allclose(
                covariates["care50_rate"],
                covariates["care50_num"] / covariates["population_5plus"],
                atol=1e-12,
                rtol=1e-12,
            ):
                raise AssertionError(f"{year}: Care50 rate definition changed")

            assignment = pd.read_csv(self.config["assignment_paths"][str(year)], low_memory=False)
            assignment["common2021_lsoa_code"] = assignment["common2021_lsoa_code"].astype(str)
            if len(assignment) != EXPECTED_CHARITIES[year] or not assignment["charity_number"].is_unique:
                raise AssertionError(f"{year}: Data Spine assignment count/uniqueness changed")
            if not set(assignment["common2021_lsoa_code"]).issubset(code_set):
                raise AssertionError(f"{year}: assigned charity outside internal geography")
            income = pd.to_numeric(assignment["income_2021_gbp"], errors="coerce")
            if not income.dropna().ge(0).all():
                raise AssertionError(f"{year}: negative income")
            assignment["income_2021_gbp"] = income
            assignment["log1p_income_2021_gbp"] = np.log1p(income)
            grouped = assignment.groupby("common2021_lsoa_code", as_index=False).agg(
                charity_count_all_eligible=("charity_number", "size"),
                charities_with_usable_income=("income_2021_gbp", "count"),
                registered_capacity_log1p_income=("log1p_income_2021_gbp", "sum"),
            ).rename(columns={"common2021_lsoa_code": "lsoa_code"})
            grouped["charity_presence_all_eligible"] = grouped["charity_count_all_eligible"].gt(0).astype(int)

            provider = pd.read_csv(self.config["provider_paths"][str(year)], low_memory=False)
            provider["lsoa_code"] = provider["lsoa_code"].astype(str)
            check = grouped.merge(
                provider[["lsoa_code", "charity_records", "registered_capacity_log1p_income"]],
                on="lsoa_code",
                how="outer",
                suffixes=("_assignment", "_provider"),
            ).fillna(0)
            if not np.allclose(
                check["charities_with_usable_income"], check["charity_records"], atol=0, rtol=0
            ) or not np.allclose(
                check["registered_capacity_log1p_income_assignment"],
                check["registered_capacity_log1p_income_provider"],
                atol=1e-9,
                rtol=1e-12,
            ):
                raise AssertionError(f"{year}: assignment/provider capacity mismatch")

            frame = covariates.merge(grouped, on="lsoa_code", how="left", validate="one_to_one")
            fill_columns = [
                "charity_count_all_eligible",
                "charities_with_usable_income",
                "registered_capacity_log1p_income",
                "charity_presence_all_eligible",
            ]
            frame[fill_columns] = frame[fill_columns].fillna(0)
            for column in ("charity_count_all_eligible", "charities_with_usable_income", "charity_presence_all_eligible"):
                frame[column] = frame[column].astype(int)
            if frame["charity_count_all_eligible"].sum() != EXPECTED_CHARITIES[year]:
                raise AssertionError(f"{year}: charity count conservation failed")
            if frame["charities_with_usable_income"].sum() != int(income.notna().sum()):
                raise AssertionError(f"{year}: usable income count conservation failed")
            self.frames[year] = frame.sort_values("lsoa_code").reset_index(drop=True)
            self.assignments[year] = assignment
            write_csv(self.frames[year], self.table_dir / f"lsoa_descriptive_inputs_{year}.csv")
            audit_rows.append(
                {
                    "year": year,
                    "lsoas": len(frame),
                    "unique_lsoas": frame["lsoa_code"].nunique(),
                    "care50_missing": int(frame["care50_num"].isna().sum()),
                    "eligible_charities": len(assignment),
                    "mapped_eligible_charities": int(frame["charity_count_all_eligible"].sum()),
                    "usable_income_charities": int(income.notna().sum()),
                    "mapped_usable_income_charities": int(frame["charities_with_usable_income"].sum()),
                    "capacity_sum": float(frame["registered_capacity_log1p_income"].sum()),
                    "provider_capacity_sum": float(provider["registered_capacity_log1p_income"].sum()),
                    "pass": True,
                }
            )
        audit = pd.DataFrame(audit_rows)
        write_csv(audit, self.qa_dir / "descriptive_input_conservation_audit.csv")
        print(audit.to_string(index=False), flush=True)
        return audit

    def write_statistics(self) -> pd.DataFrame:
        variables = [
            "care50_num",
            "care50_rate",
            "charity_presence_all_eligible",
            "charity_count_all_eligible",
            "charities_with_usable_income",
            "registered_capacity_log1p_income",
        ]
        rows = [describe_values(self.frames[year][variable], year, variable) for year in YEARS for variable in variables]
        statistics = pd.DataFrame(rows)
        write_csv(statistics, self.table_dir / "lsoa_descriptive_statistics.csv")

        income_rows = []
        for year in YEARS:
            assignment = self.assignments[year]
            income = assignment["income_2021_gbp"]
            valid = income.dropna().astype(float)
            income_rows.append(
                {
                    "year": year,
                    "eligible_charities": len(assignment),
                    "usable_income_charities": len(valid),
                    "missing_income_charities": int(income.isna().sum()),
                    "recorded_zero_income_charities": int(valid.eq(0).sum()),
                    "income_sum_2021_gbp": float(valid.sum()),
                    "income_mean_2021_gbp": float(valid.mean()),
                    "income_median_2021_gbp": float(valid.median()),
                    "income_p10_2021_gbp": float(valid.quantile(0.10)),
                    "income_p90_2021_gbp": float(valid.quantile(0.90)),
                    "income_p99_2021_gbp": float(valid.quantile(0.99)),
                    "income_max_2021_gbp": float(valid.max()),
                    "capacity_sum_log1p_income": float(np.log1p(valid).sum()),
                }
            )
        income_summary = pd.DataFrame(income_rows)
        write_csv(income_summary, self.table_dir / "charity_income_capacity_summary.csv")
        print(statistics.to_string(index=False), flush=True)
        print(income_summary.to_string(index=False), flush=True)
        return statistics

    def write_regional_icb_summaries(self) -> pd.DataFrame:
        regional_rows = []
        icb_rows = []
        for year in YEARS:
            frame = self.frames[year]
            regional_rows.append(
                {
                    "year": year,
                    "lsoas": len(frame),
                    "care50_num": float(frame["care50_num"].sum()),
                    "population_5plus": float(frame["population_5plus"].sum()),
                    "regional_care50_rate": float(frame["care50_num"].sum() / frame["population_5plus"].sum()),
                    "eligible_charities": int(frame["charity_count_all_eligible"].sum()),
                    "lsoas_with_charity_presence": int(frame["charity_presence_all_eligible"].sum()),
                    "charities_with_usable_income": int(frame["charities_with_usable_income"].sum()),
                    "registered_capacity_log1p_income": float(frame["registered_capacity_log1p_income"].sum()),
                }
            )
            for (icb_code, icb_name), group in frame.groupby(["ICB23CD", "ICB23NM"], sort=True):
                icb_rows.append(
                    {
                        "year": year,
                        "ICB23CD": icb_code,
                        "ICB23NM": icb_name,
                        "lsoas": len(group),
                        "care50_num": float(group["care50_num"].sum()),
                        "population_5plus": float(group["population_5plus"].sum()),
                        "care50_rate": float(group["care50_num"].sum() / group["population_5plus"].sum()),
                        "eligible_charities": int(group["charity_count_all_eligible"].sum()),
                        "lsoas_with_charity_presence": int(group["charity_presence_all_eligible"].sum()),
                        "charities_with_usable_income": int(group["charities_with_usable_income"].sum()),
                        "registered_capacity_log1p_income": float(group["registered_capacity_log1p_income"].sum()),
                    }
                )
        regional = pd.DataFrame(regional_rows)
        icb = pd.DataFrame(icb_rows)
        write_csv(regional, self.table_dir / "regional_descriptive_summary.csv")
        write_csv(icb, self.table_dir / "icb_descriptive_summary.csv")
        print(regional.to_string(index=False), flush=True)
        return regional

    def make_level_maps(self) -> pd.DataFrame:
        maps = {year: self.geometry.merge(self.frames[year], on="lsoa_code", validate="one_to_one") for year in YEARS}
        plot_three_continuous(
            maps, "care50_num", "High-intensity unpaid care counts", "Care50 count", "Blues",
            self.figure_dir, "three_year_care50_count_maps", vmin=0,
        )
        plot_three_continuous(
            maps, "care50_rate", "High-intensity unpaid care rates", "Care50 rate", "YlOrRd",
            self.figure_dir, "three_year_care50_rate_maps", vmin=0, percent=True,
        )

        fig, axes = plt.subplots(1, 3, figsize=(15.3, 5.8))
        presence_colours = ListedColormap(["#eeeeee", "#238b45"])
        presence_norm = BoundaryNorm([-0.5, 0.5, 1.5], presence_colours.N)
        for axis, year in zip(axes, YEARS):
            maps[year].plot(column="charity_presence_all_eligible", cmap=presence_colours, norm=presence_norm, linewidth=0, ax=axis)
            axis.set_title(str(year), fontsize=14, fontweight="bold")
            axis.set_axis_off()
        handles = [Patch(facecolor="#eeeeee", label="No registered charity"), Patch(facecolor="#238b45", label="One or more registered charities")]
        fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, title="Registered care-related charity presence")
        fig.suptitle("Registered care-related charity presence", fontsize=17, fontweight="bold", y=0.99)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.91, bottom=0.13, wspace=0.015)
        save_figure(fig, self.figure_dir, "three_year_charity_presence_maps")

        plot_three_continuous(
            maps, "charity_count_all_eligible", "Registered care-related charity counts", "Eligible charity count", "Greens",
            self.figure_dir, "three_year_charity_count_maps", vmin=0,
        )
        plot_three_continuous(
            maps, "registered_capacity_log1p_income", "Registered-charity financial capacity", "Sum log(1 + income in 2021 GBP)", "YlGn",
            self.figure_dir, "three_year_charity_capacity_maps", vmin=0,
        )
        audit = pd.DataFrame([{"figures_created_png": 5, "figures_created_pdf": 5, "mapped_lsoas_each_year": EXPECTED_LSOAS, "pass": True}])
        write_csv(audit, self.qa_dir / "descriptive_level_map_audit.csv")
        return audit

    def write_change_outputs(self) -> pd.DataFrame:
        variables = ["care50_num", "care50_rate", "charity_count_all_eligible", "registered_capacity_log1p_income"]
        rows = []
        change_frames: dict[tuple[int, int], pd.DataFrame] = {}
        for start, end in INTERVALS:
            change = self.frames[start][["lsoa_code"]].copy()
            for variable in variables:
                start_values = self.frames[start].set_index("lsoa_code")[variable]
                end_values = self.frames[end].set_index("lsoa_code")[variable]
                delta = end_values - start_values
                change[f"delta_{variable}_{start}_{end}"] = change["lsoa_code"].map(delta)
                rows.append(
                    {
                        "start_year": start,
                        "end_year": end,
                        "variable": variable,
                        "mean_change": float(delta.mean()),
                        "median_change": float(delta.median()),
                        "p05_change": float(delta.quantile(0.05)),
                        "p95_change": float(delta.quantile(0.95)),
                        "min_change": float(delta.min()),
                        "max_change": float(delta.max()),
                        "lsoas_increase": int(delta.gt(0).sum()),
                        "lsoas_no_change": int(np.isclose(delta, 0, atol=1e-12).sum()),
                        "lsoas_decrease": int(delta.lt(0).sum()),
                    }
                )
            change_frames[(start, end)] = change
            write_csv(change, self.table_dir / f"lsoa_descriptive_changes_{start}_{end}.csv")
        summary = pd.DataFrame(rows)
        write_csv(summary, self.table_dir / "descriptive_change_summary.csv")
        plot_change_panels(self.geometry, self.frames, "care50_num", "Change in high-intensity unpaid care counts", "Change in Care50 count", self.figure_dir, "care50_count_change_maps")
        plot_change_panels(self.geometry, self.frames, "care50_rate", "Change in high-intensity unpaid care rates", "Percentage-point change", self.figure_dir, "care50_rate_change_maps", scale=100.0)
        plot_change_panels(self.geometry, self.frames, "charity_count_all_eligible", "Change in registered charity counts", "Change in eligible charity count", self.figure_dir, "charity_count_change_maps")
        plot_change_panels(self.geometry, self.frames, "registered_capacity_log1p_income", "Change in registered-charity financial capacity", "Change in sum log(1 + income)", self.figure_dir, "charity_capacity_change_maps")
        print(summary.to_string(index=False), flush=True)
        return summary

    def final_qa_and_method(self) -> pd.DataFrame:
        source_audit = self.verify_sources_unchanged()
        checks = []
        for year in YEARS:
            frame = self.frames[year]
            checks.extend(
                [
                    {"year": year, "check": "3,411 unique LSOAs", "value": len(frame), "pass": len(frame) == EXPECTED_LSOAS and frame["lsoa_code"].is_unique},
                    {"year": year, "check": "Care50 rate definition", "value": float(np.max(np.abs(frame["care50_rate"] - frame["care50_num"] / frame["population_5plus"]))), "pass": True},
                    {"year": year, "check": "eligible charity count conserved", "value": int(frame["charity_count_all_eligible"].sum()), "pass": int(frame["charity_count_all_eligible"].sum()) == EXPECTED_CHARITIES[year]},
                    {"year": year, "check": "capacity nonnegative", "value": float(frame["registered_capacity_log1p_income"].min()), "pass": frame["registered_capacity_log1p_income"].ge(0).all()},
                ]
            )
        qa = pd.DataFrame(checks)
        if not qa["pass"].all() or not source_audit["unchanged"].all():
            raise AssertionError("Descriptive downstream QA failed")
        write_csv(qa, self.qa_dir / "descriptive_final_qa.csv")
        method = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "analysis_type": "shared_descriptive_mapping_and_summary",
            "years": list(YEARS),
            "geography": "fixed 3,411 2021 LSOAs",
            "care50_rate": "care50_num / population_5plus",
            "charity_count": "all eligible Data Spine rebuild v2 charity assignments to common 2021 LSOAs",
            "capacity": "charity-level log1p(income_2021_gbp), summed within common 2021 LSOA; missing income excluded and recorded zero retained",
            "maps": "common scale across years; level maps saturate beyond pooled P98; change maps use symmetric scales centred on zero and saturate beyond pooled P98 absolute change",
            "explicitly_not_run": ["E2SFCA", "HP-LA", "trajectory analysis", "regression", "BYM2"],
            "source_files_verified_unchanged": len(source_audit),
        }
        (self.run_dir / "METHOD_MANIFEST.json").write_text(json.dumps(method, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(qa.to_string(index=False), flush=True)
        return qa


class AccessibilityLongitudinalWorkflow(BaseWorkflow):
    def load_accessibility_outputs(self) -> pd.DataFrame:
        self.geometry = gpd.read_file(self.config["boundary_gpkg"], layer="lsoa_2021").to_crs("EPSG:27700")
        if len(self.geometry) != EXPECTED_LSOAS or not self.geometry["lsoa_code"].is_unique:
            raise AssertionError("Boundary is not 3,411 unique LSOAs")
        code_set = set(self.geometry["lsoa_code"].astype(str))
        self.frames: dict[int, pd.DataFrame] = {}
        rows = []
        for year in YEARS:
            frame = pd.read_csv(self.config["result_paths"][str(year)], low_memory=False)
            frame["lsoa_code"] = frame["lsoa_code"].astype(str)
            if len(frame) != EXPECTED_LSOAS or not frame["lsoa_code"].is_unique or set(frame["lsoa_code"]) != code_set:
                raise AssertionError(f"{year}: invalid E2SFCA result geography")
            required = ["care50_num", "population_5plus", "care50_rate", "accessibility_A", "accessibility_Astar", "below_2001_baseline"]
            if frame[required].isna().any().any():
                raise AssertionError(f"{year}: missing outcome values")
            if not np.allclose(frame["care50_rate"], frame["care50_num"] / frame["population_5plus"], atol=1e-12, rtol=1e-12):
                raise AssertionError(f"{year}: Care50 rate definition changed")
            if not frame[["accessibility_A", "accessibility_Astar"]].ge(0).all().all():
                raise AssertionError(f"{year}: invalid accessibility")
            self.frames[year] = frame.sort_values("lsoa_code").reset_index(drop=True)
            rows.append(
                {
                    "year": year,
                    "rows": len(frame),
                    "unique_lsoas": frame["lsoa_code"].nunique(),
                    "missing_A": int(frame["accessibility_A"].isna().sum()),
                    "missing_Astar": int(frame["accessibility_Astar"].isna().sum()),
                    "care50_rate_definition_match": True,
                    "pass": True,
                }
            )
        audit = pd.DataFrame(rows)
        write_csv(audit, self.qa_dir / "accessibility_input_audit.csv")
        print(audit.to_string(index=False), flush=True)
        return audit

    def write_accessibility_statistics(self) -> pd.DataFrame:
        rows = []
        for year in YEARS:
            frame = self.frames[year]
            for variable in ("accessibility_A", "accessibility_Astar"):
                row = describe_values(frame[variable], year, variable)
                row["care50_weighted_mean"] = weighted_mean(frame[variable], frame["care50_num"])
                rows.append(row)
        statistics = pd.DataFrame(rows)
        write_csv(statistics, self.table_dir / "accessibility_summary_statistics.csv")

        benchmark_rows = []
        for year in YEARS:
            frame = self.frames[year]
            total_care = float(frame["care50_num"].sum())
            for status, mask in (
                ("below", frame["accessibility_Astar"].lt(1.0)),
                ("equal", np.isclose(frame["accessibility_Astar"], 1.0, atol=1e-12)),
                ("above", frame["accessibility_Astar"].gt(1.0)),
            ):
                count = int(np.asarray(mask).sum())
                care = float(frame.loc[np.asarray(mask), "care50_num"].sum())
                benchmark_rows.append(
                    {
                        "year": year,
                        "benchmark_status": status,
                        "lsoas": count,
                        "share_of_lsoas": count / len(frame),
                        "care50_num": care,
                        "share_of_care50": care / total_care,
                    }
                )
        benchmark = pd.DataFrame(benchmark_rows)
        write_csv(benchmark, self.table_dir / "benchmark_status_summary.csv")
        print(statistics.to_string(index=False), flush=True)
        return statistics

    def make_accessibility_maps(self) -> pd.DataFrame:
        maps = {year: self.geometry.merge(self.frames[year][["lsoa_code", "accessibility_Astar"]], on="lsoa_code", validate="one_to_one") for year in YEARS}
        bins = [-np.inf, 0.50, 0.75, 1.00, 1.25, 1.50, np.inf]
        colours = ["#762a83", "#af8dc3", "#e7d4e8", "#d9f0d3", "#7fbf7b", "#1b7837"]
        labels = ["<0.50", "0.50–0.75", "0.75–1.00", "1.00–1.25", "1.25–1.50", ">1.50"]
        cmap = ListedColormap(colours)
        norm = BoundaryNorm(bins, cmap.N)
        fig, axes = plt.subplots(1, 3, figsize=(15.3, 5.8))
        for axis, year in zip(axes, YEARS):
            maps[year].plot(column="accessibility_Astar", cmap=cmap, norm=norm, linewidth=0, ax=axis)
            axis.set_title(str(year), fontsize=14, fontweight="bold")
            axis.set_axis_off()
        handles = [Patch(facecolor=colour, label=label) for colour, label in zip(colours, labels)]
        fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, title="A* relative to fixed 2001 South West benchmark")
        fig.suptitle(f"{self.config['display_name']} E2SFCA accessibility", fontsize=17, fontweight="bold", y=0.99)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.91, bottom=0.14, wspace=0.015)
        save_figure(fig, self.figure_dir, "three_year_Astar_maps")

        fig, axis = plt.subplots(figsize=(8.4, 5.0))
        pooled = np.concatenate([self.frames[year]["accessibility_Astar"].to_numpy(dtype=float) for year in YEARS])
        display_limit = float(np.quantile(pooled, 0.99))
        for year, colour in zip(YEARS, ["#3b6fb6", "#d17a22", "#3a8f5b"]):
            values = np.sort(self.frames[year]["accessibility_Astar"].to_numpy(dtype=float))
            cumulative = np.arange(1, len(values) + 1) / len(values)
            visible = values <= display_limit
            axis.step(values[visible], cumulative[visible], where="post", linewidth=2.1, label=str(year), color=colour)
        axis.axvline(1.0, color="#222222", linestyle="--", linewidth=1.2, label="Fixed 2001 benchmark")
        axis.set(xlim=(0, display_limit), ylim=(0, 1), xlabel="Standardised accessibility A*", ylabel="Cumulative share of LSOAs")
        axis.set_title(f"{self.config['display_name']} A* distribution", fontweight="bold")
        axis.grid(alpha=0.18)
        axis.legend(frameon=False)
        fig.tight_layout()
        save_figure(fig, self.figure_dir, "three_year_Astar_distribution")
        audit = pd.DataFrame([{"mapped_lsoas_each_year": EXPECTED_LSOAS, "png_figures": 2, "pdf_figures": 2, "pass": True}])
        write_csv(audit, self.qa_dir / "accessibility_map_audit.csv")
        return audit

    def write_changes(self) -> pd.DataFrame:
        rows = []
        for start, end in INTERVALS:
            change = self.frames[start][["lsoa_code", "lsoa_name", "ICB23CD", "ICB23NM"]].copy()
            for variable in ("accessibility_A", "accessibility_Astar"):
                start_values = self.frames[start].set_index("lsoa_code")[variable]
                end_values = self.frames[end].set_index("lsoa_code")[variable]
                delta = end_values - start_values
                change[f"delta_{variable}_{start}_{end}"] = change["lsoa_code"].map(delta)
                rows.append(
                    {
                        "start_year": start,
                        "end_year": end,
                        "variable": variable,
                        "mean_change": float(delta.mean()),
                        "median_change": float(delta.median()),
                        "p05_change": float(delta.quantile(0.05)),
                        "p95_change": float(delta.quantile(0.95)),
                        "min_change": float(delta.min()),
                        "max_change": float(delta.max()),
                        "lsoas_increase": int(delta.gt(0).sum()),
                        "lsoas_no_change": int(np.isclose(delta, 0, atol=1e-12).sum()),
                        "lsoas_decrease": int(delta.lt(0).sum()),
                    }
                )
            write_csv(change, self.table_dir / f"lsoa_accessibility_changes_{start}_{end}.csv")
        summary = pd.DataFrame(rows)
        write_csv(summary, self.table_dir / "accessibility_change_summary.csv")
        plot_change_panels(self.geometry, self.frames, "accessibility_Astar", f"{self.config['display_name']} change in A*", "Change in A*", self.figure_dir, "Astar_change_maps")
        print(summary.to_string(index=False), flush=True)
        return summary

    def write_icb_summary(self) -> pd.DataFrame:
        rows = []
        for year in YEARS:
            frame = self.frames[year]
            for (icb_code, icb_name), group in frame.groupby(["ICB23CD", "ICB23NM"], sort=True):
                below = group["accessibility_Astar"].lt(1.0)
                rows.append(
                    {
                        "year": year,
                        "ICB23CD": icb_code,
                        "ICB23NM": icb_name,
                        "lsoas": len(group),
                        "care50_num": float(group["care50_num"].sum()),
                        "population_5plus": float(group["population_5plus"].sum()),
                        "registered_capacity_log1p_income": float(group["registered_capacity_log1p_income"].sum()),
                        "mean_A": float(group["accessibility_A"].mean()),
                        "median_A": float(group["accessibility_A"].median()),
                        "care50_weighted_A": weighted_mean(group["accessibility_A"], group["care50_num"]),
                        "mean_Astar": float(group["accessibility_Astar"].mean()),
                        "median_Astar": float(group["accessibility_Astar"].median()),
                        "care50_weighted_Astar": weighted_mean(group["accessibility_Astar"], group["care50_num"]),
                        "care50_weighted_relative_accessibility_deficit": weighted_mean(1.0 - group["accessibility_Astar"], group["care50_num"]),
                        "lsoas_below_baseline": int(below.sum()),
                        "share_lsoas_below_baseline": float(below.mean()),
                        "care50_in_below_baseline_lsoas": float(group.loc[below, "care50_num"].sum()),
                    }
                )
        summary = pd.DataFrame(rows)
        write_csv(summary, self.table_dir / "icb_accessibility_summary.csv")
        print(summary.to_string(index=False), flush=True)
        return summary

    @staticmethod
    def classify_state(high_pressure: pd.Series, low_access: pd.Series) -> np.ndarray:
        return np.select(
            [high_pressure & low_access, high_pressure & ~low_access, ~high_pressure & low_access, ~high_pressure & ~low_access],
            STATE_ORDER,
            default="Unclassified",
        )

    def classify_annual_hpla(self) -> pd.DataFrame:
        threshold_rows = []
        self.annual_status: dict[int, pd.DataFrame] = {}
        for year in YEARS:
            frame = self.frames[year].copy()
            pressure_median = float(frame["care50_rate"].median())
            access_median = float(frame["accessibility_Astar"].median())
            frame["annual_median_care50_rate"] = pressure_median
            frame["annual_median_Astar"] = access_median
            frame["within_year_high_pressure"] = frame["care50_rate"].gt(pressure_median)
            frame["within_year_low_access"] = frame["accessibility_Astar"].lt(access_median)
            frame["within_year_relative_state_code"] = self.classify_state(frame["within_year_high_pressure"], frame["within_year_low_access"])
            frame["within_year_relative_hpla"] = frame["within_year_relative_state_code"].eq("HP-LA")
            if not frame["within_year_relative_state_code"].isin(STATE_ORDER).all():
                raise AssertionError(f"{year}: unclassified state")
            self.annual_status[year] = frame
            write_csv(frame, self.table_dir / f"annual_hp_la_status_{year}.csv")
            threshold_rows.append(
                {
                    "year": year,
                    "annual_median_care50_rate": pressure_median,
                    "annual_median_Astar": access_median,
                    "high_pressure_lsoas": int(frame["within_year_high_pressure"].sum()),
                    "low_access_lsoas": int(frame["within_year_low_access"].sum()),
                    "hpla_lsoas": int(frame["within_year_relative_hpla"].sum()),
                    "hpla_share": float(frame["within_year_relative_hpla"].mean()),
                }
            )
        thresholds = pd.DataFrame(threshold_rows)
        write_csv(thresholds, self.table_dir / "annual_hp_la_thresholds_counts.csv")

        fig, axes = plt.subplots(1, 3, figsize=(15.3, 5.8))
        cmap = ListedColormap(["#eeeeee", "#b2182b"])
        norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
        for axis, year in zip(axes, YEARS):
            mapped = self.geometry.merge(
                self.annual_status[year][["lsoa_code", "within_year_relative_hpla"]],
                on="lsoa_code",
                validate="one_to_one",
            )
            mapped["hpla"] = mapped["within_year_relative_hpla"].astype(int)
            mapped.plot(column="hpla", cmap=cmap, norm=norm, linewidth=0, ax=axis)
            axis.set_title(f"{year}: {int(mapped['hpla'].sum()):,} HP–LA", fontsize=14, fontweight="bold")
            axis.set_axis_off()
        handles = [Patch(facecolor="#eeeeee", label="Not HP–LA"), Patch(facecolor="#b2182b", label="High demand–low accessibility")]
        fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False)
        fig.suptitle(f"{self.config['display_name']} annual relative HP–LA", fontsize=17, fontweight="bold", y=0.99)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.91, bottom=0.13, wspace=0.015)
        save_figure(fig, self.figure_dir, "three_year_hp_la_maps")
        print(thresholds.to_string(index=False), flush=True)
        return thresholds

    def build_trajectories(self) -> pd.DataFrame:
        if not hasattr(self, "annual_status"):
            raise RuntimeError("Classify annual HP-LA first")
        base = self.annual_status[2021][["lsoa_code", "lsoa_name", "ICB23CD", "ICB23NM"]].copy()
        trajectory = base
        for year in YEARS:
            fields = self.annual_status[year][[
                "lsoa_code", "care50_num", "population_5plus", "care50_rate", "accessibility_A", "accessibility_Astar",
                "within_year_relative_state_code", "within_year_relative_hpla",
            ]].copy()
            fields = fields.rename(columns={column: f"{column}_{year}" for column in fields.columns if column != "lsoa_code"})
            trajectory = trajectory.merge(fields, on="lsoa_code", validate="one_to_one")
        trajectory["hpla_pattern"] = (
            trajectory["within_year_relative_hpla_2001"].astype(int).astype(str)
            + trajectory["within_year_relative_hpla_2011"].astype(int).astype(str)
            + trajectory["within_year_relative_hpla_2021"].astype(int).astype(str)
        )
        mapping = {
            "111": "Persistent HP–LA",
            "001": "Emerging HP–LA", "011": "Emerging HP–LA",
            "100": "Resolved / improved", "110": "Resolved / improved",
            "010": "Intermittent", "101": "Intermittent",
            "000": "Never HP–LA",
        }
        trajectory["trajectory_category"] = trajectory["hpla_pattern"].map(mapping)
        if trajectory["trajectory_category"].isna().any() or len(trajectory) != EXPECTED_LSOAS:
            raise AssertionError("Trajectory classification failed")
        self.trajectory = trajectory
        write_csv(trajectory, self.table_dir / "lsoa_hp_la_trajectories_2001_2021.csv")

        summary_rows = []
        for category in TRAJECTORY_ORDER:
            group = trajectory.loc[trajectory["trajectory_category"].eq(category)]
            summary_rows.append(
                {
                    "trajectory_category": category,
                    "lsoas": len(group),
                    "share_of_lsoas": len(group) / len(trajectory),
                    "care50_num_2021": float(group["care50_num_2021"].sum()),
                    "share_of_2021_care50": float(group["care50_num_2021"].sum() / trajectory["care50_num_2021"].sum()),
                }
            )
        summary = pd.DataFrame(summary_rows)
        write_csv(summary, self.table_dir / "hp_la_trajectory_summary.csv")

        pattern_summary = trajectory.groupby(["hpla_pattern", "trajectory_category"], as_index=False).agg(lsoas=("lsoa_code", "size"))
        pattern_summary["share_of_lsoas"] = pattern_summary["lsoas"] / len(trajectory)
        write_csv(pattern_summary, self.table_dir / "hp_la_status_sequence_summary.csv")

        icb = trajectory.groupby(["ICB23CD", "ICB23NM", "trajectory_category"], as_index=False).agg(
            lsoas=("lsoa_code", "size"), care50_num_2021=("care50_num_2021", "sum")
        )
        icb["icb_total_lsoas"] = icb.groupby(["ICB23CD", "ICB23NM"])["lsoas"].transform("sum")
        icb["share_of_icb_lsoas"] = icb["lsoas"] / icb["icb_total_lsoas"]
        icb["icb_total_care50_2021"] = icb.groupby(["ICB23CD", "ICB23NM"])["care50_num_2021"].transform("sum")
        icb["share_of_icb_care50_2021"] = icb["care50_num_2021"] / icb["icb_total_care50_2021"]
        write_csv(icb, self.table_dir / "icb_hp_la_trajectory_summary.csv")

        mapped = self.geometry.merge(trajectory[["lsoa_code", "trajectory_category"]], on="lsoa_code", validate="one_to_one")
        fig, axis = plt.subplots(figsize=(11.2, 8.8))
        for category in TRAJECTORY_ORDER:
            subset = mapped.loc[mapped["trajectory_category"].eq(category)]
            subset.plot(ax=axis, color=TRAJECTORY_COLOURS[category], edgecolor="white", linewidth=0.10)
        axis.set_axis_off()
        axis.set_title(f"{self.config['display_name']} HP–LA trajectories, 2001–2021", loc="left", fontsize=18, fontweight="bold", pad=12)
        handles = [Line2D([0], [0], marker="s", linestyle="", markersize=10, markerfacecolor=TRAJECTORY_COLOURS[c], markeredgecolor="none", label=c) for c in TRAJECTORY_ORDER]
        axis.legend(handles=handles, title="Trajectory category", loc="lower right", frameon=True)
        save_figure(fig, self.figure_dir, "hp_la_trajectory_map_2001_2021")
        print(summary.to_string(index=False), flush=True)
        return summary

    def write_transition_statistics(self) -> pd.DataFrame:
        if not hasattr(self, "trajectory"):
            raise RuntimeError("Build trajectories first")
        binary_rows = []
        for start, end in INTERVALS:
            start_hpla = self.trajectory[f"within_year_relative_hpla_{start}"].astype(int)
            end_hpla = self.trajectory[f"within_year_relative_hpla_{end}"].astype(int)
            for start_status, end_status in ((0, 0), (0, 1), (1, 0), (1, 1)):
                count = int(((start_hpla == start_status) & (end_hpla == end_status)).sum())
                binary_rows.append(
                    {
                        "start_year": start,
                        "end_year": end,
                        "start_hpla": start_status,
                        "end_hpla": end_status,
                        "transition": f"{start_status}→{end_status}",
                        "lsoas": count,
                        "share_of_lsoas": count / EXPECTED_LSOAS,
                    }
                )

            counts = pd.crosstab(
                self.trajectory[f"within_year_relative_state_code_{start}"],
                self.trajectory[f"within_year_relative_state_code_{end}"],
            ).reindex(index=STATE_ORDER, columns=STATE_ORDER, fill_value=0)
            row_percent = counts.div(counts.sum(axis=1), axis=0)
            counts.index.name = f"state_{start}"
            row_percent.index.name = f"state_{start}"
            counts.reset_index().to_csv(self.table_dir / f"four_state_transition_counts_{start}_{end}.csv", index=False, encoding="utf-8-sig")
            row_percent.reset_index().to_csv(self.table_dir / f"four_state_transition_row_percent_{start}_{end}.csv", index=False, encoding="utf-8-sig", float_format="%.15g")
            if int(counts.to_numpy().sum()) != EXPECTED_LSOAS:
                raise AssertionError("Transition matrix total failed")
        binary = pd.DataFrame(binary_rows)
        write_csv(binary, self.table_dir / "hp_la_binary_transition_summary.csv")
        print(binary.to_string(index=False), flush=True)
        return binary

    def final_qa_and_method(self) -> pd.DataFrame:
        source_audit = self.verify_sources_unchanged()
        checks = []
        for year in YEARS:
            frame = self.annual_status[year]
            checks.extend(
                [
                    {"year": year, "check": "3,411 unique annual LSOAs", "value": len(frame), "pass": len(frame) == EXPECTED_LSOAS and frame["lsoa_code"].is_unique},
                    {"year": year, "check": "annual states exhaustive", "value": int(frame["within_year_relative_state_code"].isin(STATE_ORDER).sum()), "pass": frame["within_year_relative_state_code"].isin(STATE_ORDER).all()},
                    {"year": year, "check": "strict high-pressure rule", "value": int(frame["within_year_high_pressure"].sum()), "pass": frame["within_year_high_pressure"].equals(frame["care50_rate"].gt(frame["annual_median_care50_rate"]))},
                    {"year": year, "check": "strict low-access rule", "value": int(frame["within_year_low_access"].sum()), "pass": frame["within_year_low_access"].equals(frame["accessibility_Astar"].lt(frame["annual_median_Astar"]))},
                ]
            )
        checks.extend(
            [
                {"year": "all", "check": "trajectory rows", "value": len(self.trajectory), "pass": len(self.trajectory) == EXPECTED_LSOAS},
                {"year": "all", "check": "trajectory categories complete", "value": int(self.trajectory["trajectory_category"].notna().sum()), "pass": self.trajectory["trajectory_category"].notna().all()},
            ]
        )
        qa = pd.DataFrame(checks)
        if not qa["pass"].all() or not source_audit["unchanged"].all():
            raise AssertionError("Accessibility/longitudinal downstream QA failed")
        write_csv(qa, self.qa_dir / "downstream_final_qa.csv")
        nonexecution = pd.DataFrame(
            [{
                "source_e2sfca_recalculated": False,
                "annual_hpla_run": True,
                "trajectory_classification_run": True,
                "bi_lisa_run": False,
                "regression_run": False,
                "bym2_run": False,
                "pass": True,
            }]
        )
        write_csv(nonexecution, self.qa_dir / "downstream_nonexecution_audit.csv")
        method = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "analysis_type": "accessibility_mapping_and_longitudinal_hp_la",
            "specification": self.config["display_name"],
            "source_e2sfca_run": self.config["source_run"],
            "source_e2sfca_recalculated": False,
            "years": list(YEARS),
            "geography": "fixed 3,411 2021 LSOAs",
            "A_and_Astar": "read unchanged from the executed source E2SFCA run",
            "standardisation": "source A* retains fixed 2001 South West Care50-weighted mean benchmark",
            "annual_hpla": "care50_rate > annual median and A* < annual median, using strict inequalities",
            "trajectory_mapping": {
                "111": "Persistent HP–LA", "001/011": "Emerging HP–LA", "100/110": "Resolved / improved",
                "010/101": "Intermittent", "000": "Never HP–LA",
            },
            "icb_accessibility": "Care50-weighted A/A* plus unweighted descriptive summaries",
            "icb_trajectory": "local LSOA composition and 2021 Care50 composition; not an ICB weighted trajectory",
            "explicitly_not_run": ["E2SFCA recalculation", "Bi-LISA", "regression", "BYM2"],
            "source_files_verified_unchanged": len(source_audit),
        }
        (self.run_dir / "METHOD_MANIFEST.json").write_text(json.dumps(method, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(qa.to_string(index=False), flush=True)
        return qa
