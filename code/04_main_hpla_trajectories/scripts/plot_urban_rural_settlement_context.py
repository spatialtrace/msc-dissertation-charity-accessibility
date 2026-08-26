from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


PROJECT_ROOT = Path(os.environ["DISSERTATION_DATA_ROOT"]).expanduser().resolve()
RUN_DIR = (
    PROJECT_ROOT
    / "final_data_and_analysis"
    / "Downstream_Analysis_Descriptive_E2SFCA_Longitudinal_20260820"
    / "04_TravelTime_E2SFCA_30min_ExactHalo20_40"
)
TABLE_DIR = RUN_DIR / "tables"
FIGURE_DIR = RUN_DIR / "figures"

YEARS = (2001, 2011, 2021)
SETTLEMENT_ORDER = ("Urban", "Rural")
SETTLEMENT_COLOURS = {"Rural": "#2C7FB8", "Urban": "#D95F0E"}
TRAJECTORY_ORDER = (
    "Persistent HP–LA",
    "Emerging HP–LA",
    "Resolved / improved",
    "Intermittent",
    "Never HP–LA",
)
TRAJECTORY_COLOURS = {
    "Persistent HP–LA": "#B2182B",
    "Emerging HP–LA": "#EF8A62",
    "Resolved / improved": "#4DAF4A",
    "Intermittent": "#6A51A3",
    "Never HP–LA": "#D9D9D9",
}


def load_exact_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    annual_frames = []
    for year in YEARS:
        frame = pd.read_csv(
            TABLE_DIR / f"annual_hp_la_status_{year}.csv",
            encoding="utf-8-sig",
            usecols=[
                "year",
                "lsoa_code",
                "rural_binary",
                "accessibility_Astar",
                "within_year_relative_hpla",
            ],
        )
        frame["settlement_type"] = np.where(
            frame["rural_binary"].eq(1), "Rural", "Urban"
        )
        annual_frames.append(frame)

    annual = pd.concat(annual_frames, ignore_index=True)
    if annual.groupby("year")["lsoa_code"].nunique().to_dict() != {
        2001: 3411,
        2011: 3411,
        2021: 3411,
    }:
        raise ValueError("Annual exact-halo outputs do not contain 3,411 unique LSOAs per year.")

    summary = (
        annual.groupby(["year", "settlement_type"], observed=True)
        .agg(
            n_lsoas=("lsoa_code", "nunique"),
            mean_Astar=("accessibility_Astar", "mean"),
            annual_relative_hpla_count=("within_year_relative_hpla", "sum"),
            annual_relative_hpla_percent=("within_year_relative_hpla", "mean"),
        )
        .reset_index()
    )
    summary["annual_relative_hpla_percent"] *= 100

    trajectories = pd.read_csv(
        TABLE_DIR / "lsoa_hp_la_trajectories_2001_2021.csv",
        encoding="utf-8-sig",
        usecols=["lsoa_code", "trajectory_category"],
    )
    settlement_lookup = annual.loc[
        annual["year"].eq(2021), ["lsoa_code", "settlement_type"]
    ]
    trajectories = trajectories.merge(
        settlement_lookup, on="lsoa_code", how="left", validate="one_to_one"
    )
    if trajectories["settlement_type"].isna().any() or len(trajectories) != 3411:
        raise ValueError("Trajectory-to-settlement join is incomplete.")

    composition = (
        trajectories.groupby(
            ["settlement_type", "trajectory_category"], observed=True
        )
        .size()
        .rename("trajectory_count")
        .reset_index()
    )
    full_index = pd.MultiIndex.from_product(
        [SETTLEMENT_ORDER, TRAJECTORY_ORDER],
        names=["settlement_type", "trajectory_category"],
    ).to_frame(index=False)
    composition = full_index.merge(
        composition,
        on=["settlement_type", "trajectory_category"],
        how="left",
    )
    composition["trajectory_count"] = (
        composition["trajectory_count"].fillna(0).astype(int)
    )
    composition["group_lsoas"] = composition.groupby("settlement_type")[
        "trajectory_count"
    ].transform("sum")
    composition["trajectory_percent"] = (
        100 * composition["trajectory_count"] / composition["group_lsoas"]
    )
    return summary, composition


def style_axis(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#555555")
        spine.set_linewidth(0.8)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#000000", labelsize=9)


def plot(summary: pd.DataFrame, composition: pd.DataFrame) -> plt.Figure:
    fig = plt.figure(figsize=(10.8, 7.6), facecolor="white")
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 0.60],
        left=0.08,
        right=0.985,
        bottom=0.16,
        top=0.975,
        hspace=0.34,
        wspace=0.30,
    )
    ax_access = fig.add_subplot(grid[0, 0])
    ax_hpla = fig.add_subplot(grid[0, 1])
    ax_trajectory = fig.add_subplot(grid[1, :])

    panels = (
        (ax_access, "mean_Astar", "Mean A*", (1.00, 1.38)),
        (
            ax_hpla,
            "annual_relative_hpla_percent",
            "Annual relative HP–LA (%)",
            (25.0, 37.0),
        ),
    )
    for ax, value_col, ylabel, ylim in panels:
        style_axis(ax)
        for settlement in SETTLEMENT_ORDER:
            group = summary.loc[
                summary["settlement_type"].eq(settlement)
            ].sort_values("year")
            colour = SETTLEMENT_COLOURS[settlement]
            values = group[value_col].to_numpy()
            ax.plot(
                group["year"],
                values,
                color=colour,
                linewidth=2.0,
                marker="o",
                markersize=5.0,
                markerfacecolor=colour,
                markeredgewidth=0.8,
                markeredgecolor=colour,
                label=settlement,
                zorder=3,
            )
            for year, value in zip(group["year"], values):
                label = (
                    f"{value:.3f}"
                    if value_col == "mean_Astar"
                    else f"{value:.1f}%"
                )
                x_offset = 0
                if value_col == "mean_Astar":
                    y_offset = 14 if settlement == "Urban" else -16
                    if settlement == "Rural" and year == 2001:
                        x_offset, y_offset = 12, -10
                else:
                    y_offset = 7 if settlement == "Urban" else -14
                ax.annotate(
                    label,
                    (year, value),
                    xytext=(x_offset, y_offset),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8.5,
                    color="#000000",
                )
        ax.set_xticks(YEARS)
        ax.set_xlim(1999.2, 2022.8)
        ax.set_ylim(*ylim)
        ax.set_xlabel("Year", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(loc="upper left", frameon=False, fontsize=9, handlelength=2.2)

    ax_access.set_title(
        "(a) Urban–rural mean accessibility from 2001 to 2021",
        fontsize=10.5,
        pad=10,
    )
    ax_hpla.set_title(
        "(b) Urban–rural HP–LA prevalence from 2001 to 2021",
        fontsize=10.5,
        pad=10,
    )

    style_axis(ax_trajectory)
    ax_trajectory.grid(False)
    ax_trajectory.spines[["left", "right", "top"]].set_visible(False)
    y_positions = {
        settlement: len(SETTLEMENT_ORDER) - position - 1
        for position, settlement in enumerate(SETTLEMENT_ORDER)
    }
    for settlement in SETTLEMENT_ORDER:
        left = 0.0
        subset = composition.loc[
            composition["settlement_type"].eq(settlement)
        ].set_index("trajectory_category")
        for category in TRAJECTORY_ORDER:
            value = float(subset.loc[category, "trajectory_percent"])
            ax_trajectory.barh(
                y_positions[settlement],
                value,
                left=left,
                height=0.50,
                color=TRAJECTORY_COLOURS[category],
                edgecolor="white",
                linewidth=0.8,
            )
            if value >= 4.0:
                ax_trajectory.text(
                    left + value / 2,
                    y_positions[settlement],
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    fontweight="bold",
                    color=(
                        "#000000" if category == "Never HP–LA" else "#FFFFFF"
                    ),
                )
            left += value

    ax_trajectory.set_xlim(0, 100)
    ax_trajectory.set_xticks(np.arange(0, 101, 20))
    ax_trajectory.set_yticks(
        [y_positions[settlement] for settlement in SETTLEMENT_ORDER],
        list(SETTLEMENT_ORDER),
    )
    ax_trajectory.set_xlabel("Share of LSOAs (%)", fontsize=10)
    ax_trajectory.set_title(
        "(c) Three-year HP–LA trajectory composition",
        fontsize=10.5,
        loc="left",
        pad=10,
    )
    handles = [
        Patch(facecolor=TRAJECTORY_COLOURS[category], label=category)
        for category in TRAJECTORY_ORDER
    ]
    ax_trajectory.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
        ncol=5,
        frameon=False,
        fontsize=8.5,
        handlelength=1.4,
        columnspacing=1.2,
    )

    return fig


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    summary, composition = load_exact_outputs()
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica"],
            "font.size": 10,
            "text.color": "#000000",
            "axes.labelcolor": "#000000",
            "axes.titlecolor": "#000000",
            "xtick.color": "#000000",
            "ytick.color": "#000000",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    ):
        fig = plot(summary, composition)
        output_stem = FIGURE_DIR / "urban_rural_settlement_context_three_panel"
        fig.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
        fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)

    print(summary.to_string(index=False))
    print()
    print(composition.to_string(index=False))
    print(f"Saved: {output_stem.with_suffix('.png')}")
    print(f"Saved: {output_stem.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
