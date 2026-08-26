from __future__ import annotations

import os
from pathlib import Path

import geopandas as gpd
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from libpysal.weights import fuzzy_contiguity
from matplotlib.patches import Patch


PROJECT_ROOT = Path(os.environ["DISSERTATION_DATA_ROOT"]).expanduser().resolve()
ANALYSIS_ROOT = (
    PROJECT_ROOT
    / "final_data_and_analysis"
    / "Downstream_Analysis_Descriptive_E2SFCA_Longitudinal_20260820"
)
SOURCE_DIR = ANALYSIS_ROOT / "04_TravelTime_E2SFCA_30min_ExactHalo20_40"
OUTPUT_DIR = Path(
    os.environ.get(
        "GLOBAL_BB_PARTICIPATION_OUTPUT_DIR",
        ANALYSIS_ROOT / "05_Global_BB_Join_Count_ExactHalo20_40",
    )
).expanduser().resolve()
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
BOUNDARY_GPKG = (
    PROJECT_ROOT
    / "final_data_and_analysis"
    / "unified-lsoa"
    / "common2021_lsoa_spatial_foundation.gpkg"
)

YEARS = (2001, 2011, 2021)
EXPECTED_LSOAS = 3411
EXPECTED_HP_LA = {2001: 999, 2011: 982, 2021: 981}
EXPECTED_BB = {2001: 1803, 2011: 1740, 2021: 1626}
HP_LA_FIELD = "within_year_relative_hpla"

CATEGORY_ORDER = (
    "Not HP–LA",
    "Isolated HP–LA",
    "1–2 HP–LA neighbours",
    "3–4 HP–LA neighbours",
    "5+ HP–LA neighbours",
)
CATEGORY_COLOURS = {
    "Not HP–LA": "#EEEEEE",
    "Isolated HP–LA": "#FDDDCB",
    "1–2 HP–LA neighbours": "#F4A582",
    "3–4 HP–LA neighbours": "#D6604D",
    "5+ HP–LA neighbours": "#B2182B",
}
ICB_SHORT_NAMES = {
    "E54000036": "Cornwall & IoS",
    "E54000037": "Devon",
    "E54000038": "Somerset",
    "E54000039": "Bristol area",
    "E54000040": "Bath, Swindon\n& Wiltshire",
    "E54000041": "Dorset",
    "E54000043": "Gloucestershire",
}


def as_binary(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype("int8")
    converted = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": 1, "false": 0, "1": 1, "0": 0})
    )
    if converted.isna().any():
        raise ValueError("HP–LA field contains values other than True/False or 1/0.")
    return converted.astype("int8")


def classify_participation(hp_la: int, same_status_neighbours: int) -> str:
    if hp_la == 0:
        return "Not HP–LA"
    if same_status_neighbours == 0:
        return "Isolated HP–LA"
    if same_status_neighbours <= 2:
        return "1–2 HP–LA neighbours"
    if same_status_neighbours <= 4:
        return "3–4 HP–LA neighbours"
    return "5+ HP–LA neighbours"


def build_map_data() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame]:
    geometry = gpd.read_file(BOUNDARY_GPKG, layer="lsoa_2021").to_crs("EPSG:27700")
    geometry["lsoa_code"] = geometry["lsoa_code"].astype(str)
    if len(geometry) != EXPECTED_LSOAS or not geometry["lsoa_code"].is_unique:
        raise ValueError("Fixed 2021 geometry is not the expected 3,411 unique LSOAs.")

    indexed_geometry = geometry.set_index("lsoa_code", drop=False)
    weights = fuzzy_contiguity(
        indexed_geometry,
        predicate="intersects",
        buffering=False,
        silence_warnings=True,
    )
    weights.transform = "b"
    if weights.sparse.nnz // 2 != 9607:
        raise ValueError("Queen-like graph does not contain the expected 9,607 links.")

    annual_map_frames: list[gpd.GeoDataFrame] = []
    category_rows: list[dict[str, object]] = []
    icb_rows: list[dict[str, object]] = []
    icb_lookup: pd.DataFrame | None = None

    for year in YEARS:
        status_path = SOURCE_DIR / "tables" / f"annual_hp_la_status_{year}.csv"
        status = pd.read_csv(status_path, encoding="utf-8-sig", dtype={"lsoa_code": str})
        status[HP_LA_FIELD] = as_binary(status[HP_LA_FIELD])
        if len(status) != EXPECTED_LSOAS or not status["lsoa_code"].is_unique:
            raise ValueError(f"{year} status file is not the expected 3,411 unique LSOAs.")

        if icb_lookup is None:
            icb_lookup = status[["lsoa_code", "ICB23CD", "ICB23NM"]].copy()

        values = status.set_index("lsoa_code")[HP_LA_FIELD].to_dict()
        local_counts = {
            lsoa: (
                sum(values[neighbour] for neighbour in weights.neighbors[lsoa])
                if values[lsoa] == 1
                else 0
            )
            for lsoa in weights.id_order
        }
        observed_bb = int(sum(local_counts.values()) // 2)
        hp_la_total = int(status[HP_LA_FIELD].sum())
        if hp_la_total != EXPECTED_HP_LA[year] or observed_bb != EXPECTED_BB[year]:
            raise ValueError(
                f"{year} did not reconcile: HP–LA={hp_la_total}, BB={observed_bb}."
            )

        status["hp_la_neighbour_count"] = status["lsoa_code"].map(local_counts).astype(int)
        status["bb_join_endpoint_contribution"] = status["hp_la_neighbour_count"] / 2
        status["bb_participation_class"] = [
            classify_participation(int(hp), int(count))
            for hp, count in zip(status[HP_LA_FIELD], status["hp_la_neighbour_count"])
        ]

        mapped = geometry.merge(
            status[
                [
                    "lsoa_code",
                    "ICB23CD",
                    "ICB23NM",
                    HP_LA_FIELD,
                    "hp_la_neighbour_count",
                    "bb_join_endpoint_contribution",
                    "bb_participation_class",
                ]
            ],
            on="lsoa_code",
            how="left",
            validate="one_to_one",
        )
        mapped["year"] = year
        annual_map_frames.append(mapped)

        counts = mapped["bb_participation_class"].value_counts()
        for category in CATEGORY_ORDER:
            category_rows.append(
                {
                    "year": year,
                    "category": category,
                    "lsoa_count": int(counts.get(category, 0)),
                    "hp_la_total": hp_la_total,
                    "observed_bb": observed_bb,
                }
            )

        icb_summary = (
            mapped.groupby(["ICB23CD", "ICB23NM"], observed=True)
            .agg(
                total_lsoas=("lsoa_code", "size"),
                hp_la_lsoas=(HP_LA_FIELD, "sum"),
                clustered_hp_la_lsoas=(
                    "hp_la_neighbour_count",
                    lambda values_: int((values_ > 0).sum()),
                ),
                isolated_hp_la_lsoas=(
                    "bb_participation_class",
                    lambda values_: int((values_ == "Isolated HP–LA").sum()),
                ),
                bb_join_endpoint_contribution=("bb_join_endpoint_contribution", "sum"),
            )
            .reset_index()
        )
        icb_summary["year"] = year
        icb_summary["hp_la_share_percent"] = 100 * icb_summary["hp_la_lsoas"] / icb_summary["total_lsoas"]
        icb_rows.extend(icb_summary.to_dict("records"))

    all_maps = gpd.GeoDataFrame(
        pd.concat(annual_map_frames, ignore_index=True),
        geometry="geometry",
        crs=geometry.crs,
    )
    assert icb_lookup is not None
    icb_geometry = geometry.merge(icb_lookup, on="lsoa_code", how="left", validate="one_to_one")
    icb_boundaries = icb_geometry.dissolve(by=["ICB23CD", "ICB23NM"], as_index=False)
    return all_maps, icb_boundaries, pd.DataFrame(category_rows), pd.DataFrame(icb_rows)


def plot_maps(
    all_maps: gpd.GeoDataFrame,
    icb_boundaries: gpd.GeoDataFrame,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 5.6), constrained_layout=False)
    bounds = all_maps.total_bounds
    x_pad = 0.02 * (bounds[2] - bounds[0])
    y_pad = 0.02 * (bounds[3] - bounds[1])
    label_points = icb_boundaries.copy()
    label_points["geometry"] = label_points.geometry.representative_point()

    for ax, year in zip(axes, YEARS):
        annual = all_maps.loc[all_maps["year"].eq(year)]
        for category in CATEGORY_ORDER:
            subset = annual.loc[annual["bb_participation_class"].eq(category)]
            subset.plot(
                ax=ax,
                color=CATEGORY_COLOURS[category],
                edgecolor="none",
                linewidth=0,
                rasterized=False,
            )

        icb_boundaries.boundary.plot(ax=ax, color="#4D4D4D", linewidth=0.55, zorder=4)
        for row in label_points.itertuples(index=False):
            label = ICB_SHORT_NAMES.get(row.ICB23CD, row.ICB23NM)
            text = ax.text(
                row.geometry.x,
                row.geometry.y,
                label,
                ha="center",
                va="center",
                fontsize=6.5,
                color="#333333",
                zorder=5,
            )
            text.set_path_effects(
                [path_effects.withStroke(linewidth=2.0, foreground="white", alpha=0.9)]
            )

        ax.set_xlim(bounds[0] - x_pad, bounds[2] + x_pad)
        ax.set_ylim(bounds[1] - y_pad, bounds[3] + y_pad)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(
            f"{year}\nObserved BB = {EXPECTED_BB[year]:,}; global p = 0.001",
            fontsize=10.5,
            pad=5,
        )

    handles = [
        Patch(facecolor=CATEGORY_COLOURS[category], edgecolor="none", label=category)
        for category in CATEGORY_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=5,
        frameon=False,
        fontsize=8.5,
        handlelength=1.5,
        columnspacing=1.25,
    )
    fig.subplots_adjust(left=0.015, right=0.985, top=0.90, bottom=0.13, wspace=0.02)
    return fig


def export_geopackage(
    all_maps: gpd.GeoDataFrame,
    icb_boundaries: gpd.GeoDataFrame,
) -> Path:
    gpkg_path = OUTPUT_DIR / "annual_hp_la_bb_join_participation.gpkg"
    class_codes = {
        category: code for code, category in enumerate(CATEGORY_ORDER)
    }

    for index, year in enumerate(YEARS):
        layer = all_maps.loc[all_maps["year"].eq(year)].copy()
        layer["bb_class_code"] = layer["bb_participation_class"].map(class_codes).astype("int8")
        layer = layer.rename(
            columns={
                HP_LA_FIELD: "hp_la",
                "hp_la_neighbour_count": "hp_la_neighbours",
                "bb_join_endpoint_contribution": "bb_join_contribution",
                "bb_participation_class": "bb_class",
            }
        )
        layer = layer[
            [
                "year",
                "lsoa_code",
                "lsoa_name",
                "ICB23CD",
                "ICB23NM",
                "hp_la",
                "hp_la_neighbours",
                "bb_join_contribution",
                "bb_class_code",
                "bb_class",
                "geometry",
            ]
        ]
        layer.to_file(
            gpkg_path,
            layer=f"bb_participation_{year}",
            driver="GPKG",
            mode="w" if index == 0 else "a",
            index=False,
        )

    icb_layer = icb_boundaries.rename(
        columns={"ICB23CD": "icb_code", "ICB23NM": "icb_name"}
    )[["icb_code", "icb_name", "geometry"]]
    icb_layer.to_file(
        gpkg_path,
        layer="icb_boundaries_2023",
        driver="GPKG",
        mode="a",
        index=False,
    )
    return gpkg_path


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    all_maps, icb_boundaries, category_summary, icb_summary = build_map_data()

    category_summary.to_csv(
        TABLE_DIR / "annual_bb_join_participation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    icb_summary.to_csv(
        TABLE_DIR / "icb_bb_join_participation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    gpkg_path = export_geopackage(all_maps, icb_boundaries)

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica"],
            "font.size": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    ):
        fig = plot_maps(all_maps, icb_boundaries)
        output_stem = FIGURE_DIR / "annual_hp_la_bb_join_participation_maps"
        fig.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
        fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)

    print(category_summary.to_string(index=False))
    print()
    print(
        icb_summary.sort_values(
            ["year", "bb_join_endpoint_contribution"], ascending=[True, False]
        ).to_string(index=False)
    )
    print(f"Saved: {output_stem.with_suffix('.png')}")
    print(f"Saved: {output_stem.with_suffix('.pdf')}")
    print(f"Saved: {gpkg_path}")


if __name__ == "__main__":
    main()
