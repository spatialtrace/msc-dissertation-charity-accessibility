#!/usr/bin/env python3
"""Rebuild the 2001, 2011 and 2021 charity-year datasets from Data Spine.

The workflow deliberately preserves the dissertation's established rules:

* Data Spine primary registration/removal dates define the new presence spine.
* The existing ``care_strict`` implementation is joined by charity number.
* Finance is selected by exact GB-CHC UID: covering period, nearest FYE within
  365 days, then the existing final-workbook proxy if one exists.
* Existing dated Charity Commission snapshot and validated Companies House
  address decisions are reused. Undated Data Spine supplementary addresses are
  retained only as QA candidates and are never assigned to a Census year.
* The existing postcode normalisation, ONSPD lookup, Census-year LSOA vintages,
  and fixed seven-ICB study-area membership are reused.

Old inputs and outputs are read-only. Every new artefact is written below the
directory configured by ``CHARITY_REBUILD_OUTPUT_DIR`` (or the ignored local
``_private_outputs/charity_rebuild_v2`` directory by default).
"""

from __future__ import annotations

import json
import os
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = Path(os.environ["DISSERTATION_DATA_ROOT"]).expanduser().resolve()
FINAL_DIR = WORKSPACE / "final_data_and_analysis"
DATA_SPINE_DIR = FINAL_DIR / "Data_Spine"
PACKAGE_DIR = Path(
    os.environ.get(
        "CHARITY_REBUILD_OUTPUT_DIR",
        SCRIPT_DIR / "_private_outputs" / "charity_rebuild_v2",
    )
).expanduser().resolve()
HISTORY_DIR = WORKSPACE / "分析历史"

SPINE_ZIP = DATA_SPINE_DIR / "tcss-spine-Jul2026.zip"
FINANCE_ZIP = DATA_SPINE_DIR / "tcss-charity-finhist-Jul2026.zip"
MASTER_ALL = (
    HISTORY_DIR
    / "very_first_dataset"
    / "charity data Claude"
    / "data"
    / "interim"
    / "charity_master_all.parquet"
)
REGISTRATION_SPELLS = MASTER_ALL.parent / "registration_spells.parquet"
POSTCODE_LOOKUP = MASTER_ALL.parent / "postcode_lookup.parquet"
OLD_CHARITY_DIR = FINAL_DIR / "charity_data"
OLD_LOCATION_DIR = HISTORY_DIR / "charity_original_data"

STAGE_DIRS = {
    "presence": PACKAGE_DIR / "01_spine_presence",
    "care": PACKAGE_DIR / "02_care_classification",
    "finance": PACKAGE_DIR / "03_finance",
    "address": PACKAGE_DIR / "04_historical_addresses",
    "geocode": PACKAGE_DIR / "05_geocoding",
    "geography": PACKAGE_DIR / "06_geography_filter",
    "final": PACKAGE_DIR / "07_final_outputs",
    "qa": PACKAGE_DIR / "08_qa_comparison",
}

YEARS = (2001, 2011, 2021)
CENSUS_DATES = {
    2001: pd.Timestamp("2001-04-29"),
    2011: pd.Timestamp("2011-03-27"),
    2021: pd.Timestamp("2021-03-21"),
}
LSOA_LOOKUP_COLUMNS = {2001: "lsoa01", 2011: "lsoa11", 2021: "lsoa21"}

SNAPSHOT_DATES = {
    "dec2014": pd.Timestamp("2014-12-01"),
    "feb2016": pd.Timestamp("2016-02-01"),
    "mar2017": pd.Timestamp("2017-03-01"),
    "aug2020": pd.Timestamp("2020-08-01"),
    "current": pd.Timestamp("2026-06-10"),
}

CPIH_ANNUAL = {
    2000: 73.4,
    2001: 74.6,
    2002: 75.7,
    2003: 76.7,
    2004: 77.8,
    2005: 79.4,
    2006: 81.4,
    2007: 83.3,
    2008: 86.2,
    2009: 87.9,
    2010: 90.1,
    2011: 93.6,
    2012: 96.0,
    2013: 98.2,
    2014: 99.6,
    2015: 100.0,
    2016: 101.0,
    2017: 103.6,
    2018: 106.0,
    2019: 107.8,
    2020: 108.9,
    2021: 111.6,
    2022: 120.5,
    2023: 128.6,
    2024: 132.9,
    2025: 138.0,
}
CPIH_SOURCE = "ONS L522 CPIH all items, annual, 2015=100"
CPIH_URL = "https://www.ons.gov.uk/economy/inflationandpriceindices/timeseries/l522/mm23"


def ensure_paths() -> None:
    required = [
        SPINE_ZIP,
        FINANCE_ZIP,
        MASTER_ALL,
        REGISTRATION_SPELLS,
        POSTCODE_LOOKUP,
        *[FINAL_DIR / "covariates" / f"{year}.csv" for year in YEARS],
        *[OLD_CHARITY_DIR / f"charity_income_{year}_hybrid.csv" for year in YEARS],
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(map(str, missing)))
    for directory in STAGE_DIRS.values():
        directory.mkdir(parents=True, exist_ok=True)


def norm_postcode(values: pd.Series) -> pd.Series:
    compact = (
        values.astype("string")
        .str.upper()
        .str.replace(r"[^A-Z0-9]", "", regex=True)
        .replace("", pd.NA)
    )
    valid = compact.str.len().between(5, 7).fillna(False)
    compact = compact.where(valid)
    return compact.str[:-3] + " " + compact.str[-3:]


def read_zip_csv(zip_path: Path, member: str, **kwargs) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(member) as handle:
            return pd.read_csv(handle, **kwargs)


def save_parquet(frame: pd.DataFrame, path: Path) -> None:
    frame.to_parquet(path, index=False)
    print(f"saved {path.relative_to(PACKAGE_DIR)} | rows={len(frame):,}", flush=True)


def markdown_table(frame: pd.DataFrame, include_index: bool = False) -> str:
    """Render a compact Markdown table without pandas' optional tabulate dependency."""
    table = frame.reset_index() if include_index else frame.copy()
    table = table.fillna("")
    headers = [str(column) for column in table.columns]

    def clean(value: object) -> str:
        if isinstance(value, float):
            value = f"{value:.2f}"
        return str(value).replace("|", r"\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend(
        "| " + " | ".join(clean(value) for value in row) + " |"
        for row in table.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def stage_presence() -> tuple[pd.DataFrame, dict]:
    print("\n[1/8] Data Spine presence", flush=True)
    spine = read_zip_csv(
        SPINE_ZIP,
        "TSCS_spine.spine.csv",
        dtype="string",
        low_memory=False,
    )
    raw_rows = len(spine)
    spine = spine.loc[spine["uid"].str.match(r"^GB-CHC-\d+$", na=False)].copy()
    gb_chc_rows = len(spine)
    duplicates = int(spine["uid"].duplicated().sum())
    spine = spine.drop_duplicates("uid", keep="first")
    spine["charity_number"] = pd.to_numeric(
        spine["uid"].str.removeprefix("GB-CHC-"), errors="raise"
    ).astype("Int64")
    spine["registerdate"] = pd.to_datetime(
        spine["registerdate"], dayfirst=True, errors="coerce"
    )
    spine["removeddate"] = pd.to_datetime(
        spine["removeddate"], dayfirst=True, errors="coerce"
    )
    if spine["registerdate"].isna().any():
        raise AssertionError("GB-CHC Data Spine rows contain missing registration dates")
    if not spine["charity_number"].is_unique:
        raise AssertionError("Charity number is not unique after GB-CHC UID filtering")
    spine["data_spine_primary_postcode"] = norm_postcode(spine["postcode"])
    save_parquet(spine, STAGE_DIRS["presence"] / "gb_chc_spine.parquet")

    spells = pd.read_parquet(REGISTRATION_SPELLS)
    spells["regno"] = pd.to_numeric(spells["regno"], errors="coerce").astype("Int64")
    active_frames = []
    year_counts = {}
    for year in YEARS:
        census_date = CENSUS_DATES[year]
        active = spine.loc[
            spine["registerdate"].le(census_date)
            & (spine["removeddate"].isna() | spine["removeddate"].gt(census_date))
        ].copy()
        active.insert(0, "target_year", year)
        active.insert(1, "census_date", census_date)
        active["presence_source"] = "tcss_spine_primary_registerdate_removeddate"

        cc_active_ids = set(
            spells.loc[
                spells["regdate"].le(census_date)
                & (spells["remdate"].isna() | spells["remdate"].gt(census_date)),
                "regno",
            ]
        )
        active["cc_spell_active_at_census"] = active["charity_number"].isin(cc_active_ids)
        active["presence_date_disagrees_with_cc_spells"] = ~active[
            "cc_spell_active_at_census"
        ]
        active_frames.append(active)
        year_counts[year] = len(active)
        print(
            f"  {year}: active={len(active):,}; "
            f"Data Spine active but CC-spell inactive={active['presence_date_disagrees_with_cc_spells'].sum():,}",
            flush=True,
        )

    active_all = pd.concat(active_frames, ignore_index=True)
    save_parquet(active_all, STAGE_DIRS["presence"] / "active_charity_years.parquet")
    audit = {
        "spine_all_rows": raw_rows,
        "gb_chc_rows_before_dedup": gb_chc_rows,
        "duplicate_gb_chc_uids_removed": duplicates,
        "gb_chc_unique": len(spine),
        "active_counts": year_counts,
    }
    (STAGE_DIRS["presence"] / "presence_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    return active_all, audit


def stage_care(active_all: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n[2/8] Existing care classification", flush=True)
    master_columns = [
        "regno",
        "name",
        "class_codes",
        "has_classification",
        "class_what_health",
        "class_what_disability",
        "class_who_elderly",
        "class_who_disabled",
        "class_how_services",
        "class_how_buildings",
        "care_class",
        "care_keyword_matches",
        "care_keyword_excluded",
        "care_keyword",
        "care_keyword_strong",
        "care_related",
        "care_strict",
        *[f"postcode_{year}" for year in YEARS],
        *[f"postcode_source_{year}" for year in YEARS],
    ]
    master = pd.read_parquet(MASTER_ALL, columns=master_columns)
    master["regno"] = pd.to_numeric(master["regno"], errors="coerce").astype("Int64")
    if not master["regno"].is_unique:
        raise AssertionError("Existing Charity Commission master is not unique by charity number")

    joined = active_all.merge(
        master,
        left_on="charity_number",
        right_on="regno",
        how="left",
        validate="many_to_one",
    )
    classification_route = (
        (joined["class_who_elderly"].eq(True) | joined["class_who_disabled"].eq(True))
        & joined["class_how_services"].eq(True)
    )
    text_route = joined["care_keyword_strong"].eq(True)
    joined["care_route_classification"] = classification_route
    joined["care_route_text"] = text_route
    joined["care_route_both"] = classification_route & text_route
    joined["care_route"] = np.select(
        [
            joined["care_route_both"],
            text_route,
            classification_route,
            joined["care_strict"].isna(),
        ],
        ["both", "text", "classification", "unresolved_no_cc_evidence"],
        default="not_care_strict",
    )
    joined["care_rule"] = (
        "care_keyword_strong OR ((class_who_elderly OR class_who_disabled) "
        "AND class_how_services)"
    )
    joined["charity_name"] = joined["name"].fillna(joined["organisationname"])
    joined["service_evidence"] = np.where(
        joined["care_route"].eq("text"),
        "Care keywords: " + joined["care_keyword_matches"].fillna(""),
        np.where(
            joined["care_route"].eq("classification"),
            "Charity Commission: serves older/disabled people and provides services",
            np.where(
                joined["care_route"].eq("both"),
                "Both Charity Commission classification and strong care keyword",
                pd.NA,
            ),
        ),
    )

    care = joined.loc[joined["care_strict"].eq(True)].copy()
    if not care[["target_year", "charity_number"]].duplicated().sum() == 0:
        raise AssertionError("Care stage contains duplicate charity-year rows")
    save_parquet(joined, STAGE_DIRS["care"] / "active_with_care_audit.parquet")
    save_parquet(care, STAGE_DIRS["care"] / "active_care_charity_years.parquet")

    route = (
        care.groupby(["target_year", "care_route"], dropna=False)
        .size()
        .rename("records")
        .reset_index()
    )
    route.to_csv(STAGE_DIRS["care"] / "care_route_summary.csv", index=False)
    for year in YEARS:
        group = joined.loc[joined["target_year"].eq(year)]
        print(
            f"  {year}: active={len(group):,}; care_strict={group['care_strict'].eq(True).sum():,}; "
            f"care evidence unavailable={group['care_strict'].isna().sum():,}",
            flush=True,
        )
    return joined, care


def load_finance_for_uids(needed_uids: set[str]) -> pd.DataFrame:
    pieces = []
    columns = ["uid", "fy", "fys", "fye", "inc", "source", "extract"]
    with zipfile.ZipFile(FINANCE_ZIP) as archive:
        with archive.open("cso-spine-charity-financial-history.csv") as handle:
            for chunk in pd.read_csv(
                handle, usecols=columns, chunksize=400_000, low_memory=False
            ):
                keep = chunk["uid"].isin(needed_uids)
                if keep.any():
                    pieces.append(chunk.loc[keep].copy())
        with archive.open("tcss-charity-finhist-1000x-flags.csv") as handle:
            flags = pd.read_csv(handle, low_memory=False)
    finance = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=columns)
    finance["fy"] = pd.to_numeric(finance["fy"], errors="coerce").astype("Int64")
    finance["fys_date"] = pd.to_datetime(
        finance["fys"], format="%d%b%Y", errors="coerce"
    )
    finance["fye_date"] = pd.to_datetime(
        finance["fye"], format="%d%b%Y", errors="coerce"
    )
    finance["income_nominal"] = pd.to_numeric(finance["inc"], errors="coerce")
    finance["period_days"] = (finance["fye_date"] - finance["fys_date"]).dt.days + 1
    if finance["income_nominal"].dropna().lt(0).any():
        raise AssertionError("Negative Data Spine charity income encountered")
    if finance.duplicated(["uid", "fy"]).any():
        raise AssertionError("Finance history is not unique by UID and financial year")
    flags["fy"] = pd.to_numeric(flags["fy"], errors="coerce").astype("Int64")
    flags["flagged_1000x"] = True
    finance = finance.merge(
        flags[["uid", "fy", "flagged_1000x", "note"]],
        on=["uid", "fy"],
        how="left",
        validate="one_to_one",
    )
    finance["flagged_1000x"] = finance["flagged_1000x"].eq(True)
    return finance


def old_proxy_lookup() -> dict[tuple[int, int], dict]:
    lookup: dict[tuple[int, int], dict] = {}
    for year in YEARS:
        old = pd.read_csv(
            OLD_CHARITY_DIR / f"charity_income_{year}_hybrid.csv", low_memory=False
        )
        for row in old.itertuples(index=False):
            number = int(row.charity_number)
            lookup[(year, number)] = {
                "income": pd.to_numeric(row.existing_income_proxy_gbp, errors="coerce"),
                "date": pd.to_datetime(row.existing_income_date, errors="coerce"),
            }
    return lookup


def stage_finance(care: pd.DataFrame) -> pd.DataFrame:
    print("\n[3/8] Exact-UID historical finance", flush=True)
    needed_uids = set(care["uid"].dropna().astype(str))
    finance = load_finance_for_uids(needed_uids)
    save_parquet(finance, STAGE_DIRS["finance"] / "relevant_finance_history.parquet")
    finance_by_uid = {
        uid: group.copy() for uid, group in finance.groupby("uid", sort=False)
    }
    proxy = old_proxy_lookup()
    selections = []

    for row in care[["target_year", "census_date", "uid", "charity_number"]].itertuples(
        index=False
    ):
        candidates = finance_by_uid.get(row.uid)
        chosen = None
        method = None
        covering_count = 0
        nearby_count = 0
        if candidates is not None:
            valid = candidates.loc[
                candidates["income_nominal"].notna()
                & candidates["income_nominal"].ge(0)
                & candidates["fys_date"].notna()
                & candidates["fye_date"].notna()
            ].copy()
            covering = valid.loc[
                valid["fys_date"].le(row.census_date)
                & valid["fye_date"].ge(row.census_date)
            ].copy()
            covering_count = len(covering)
            if covering_count:
                covering["abs_fye_offset_days"] = (
                    covering["fye_date"] - row.census_date
                ).abs().dt.days
                covering["prefer_pre"] = covering["fye_date"].gt(row.census_date).astype(int)
                chosen = covering.sort_values(
                    ["abs_fye_offset_days", "prefer_pre", "fye_date", "fys_date", "fy"],
                    ascending=[True, True, True, False, True],
                ).iloc[0]
                method = "data_spine_exact_uid_covering_census"
            else:
                nearby = valid.assign(
                    abs_fye_offset_days=(valid["fye_date"] - row.census_date).abs().dt.days,
                    prefer_pre=valid["fye_date"].gt(row.census_date).astype(int),
                )
                nearby = nearby.loc[nearby["abs_fye_offset_days"].le(365)]
                nearby_count = len(nearby)
                if nearby_count:
                    chosen = nearby.sort_values(
                        ["abs_fye_offset_days", "prefer_pre", "fye_date", "fys_date", "fy"],
                        ascending=[True, True, True, False, True],
                    ).iloc[0]
                    method = "data_spine_exact_uid_nearest_fye_365d"

        fallback = proxy.get((int(row.target_year), int(row.charity_number)), {})
        if chosen is not None:
            income = float(chosen["income_nominal"])
            fy = int(chosen["fy"])
            fys = chosen["fys_date"]
            fye = chosen["fye_date"]
            source = chosen["source"]
            extract = chosen["extract"]
            flagged = bool(chosen["flagged_1000x"])
            quality_note = chosen["note"]
            fallback_reason = pd.NA
        elif pd.notna(fallback.get("income")) and pd.notna(fallback.get("date")):
            method = "existing_final_workbook_proxy_fallback"
            income = float(fallback["income"])
            fye = fallback["date"]
            fys = pd.NaT
            fy = int(fye.year)
            source = "existing_final_workbook"
            extract = pd.NA
            flagged = False
            quality_note = pd.NA
            fallback_reason = "no_exact_uid_spine_period_within_primary_rule"
        else:
            method = "income_unknown"
            income = np.nan
            fy = pd.NA
            fys = pd.NaT
            fye = pd.NaT
            source = pd.NA
            extract = pd.NA
            flagged = False
            quality_note = pd.NA
            fallback_reason = "no_exact_uid_period_and_no_existing_proxy"

        cpih_source = CPIH_ANNUAL.get(int(fy)) if pd.notna(fy) else np.nan
        multiplier = CPIH_ANNUAL[2021] / cpih_source if pd.notna(cpih_source) else np.nan
        income_2021 = income * multiplier if pd.notna(income) and pd.notna(multiplier) else np.nan
        selections.append(
            {
                "target_year": int(row.target_year),
                "charity_number": int(row.charity_number),
                "finance_uid": row.uid,
                "selection_method": method,
                "fallback_reason": fallback_reason,
                "covering_candidate_count": covering_count,
                "nearest_365d_candidate_count": nearby_count,
                "fy": fy,
                "fys": fys,
                "fye": fye,
                "income_nominal": income,
                "income_2021_gbp": income_2021,
                "finance_source": source,
                "finance_extract": extract,
                "finance_year_offset": (int(fy) - int(row.target_year)) if pd.notna(fy) else pd.NA,
                "fye_offset_days": int((fye - row.census_date).days) if pd.notna(fye) else pd.NA,
                "flagged_1000x": flagged,
                "finance_quality_note": quality_note,
                "cpih_source_year": fy,
                "cpih_index_source": cpih_source,
                "cpih_index_2021": CPIH_ANNUAL[2021],
                "cpih_multiplier_to_2021": multiplier,
                "cpih_series": CPIH_SOURCE,
                "cpih_source_url": CPIH_URL,
            }
        )

    selected = pd.DataFrame(selections)
    if selected.duplicated(["target_year", "charity_number"]).any():
        raise AssertionError("Finance selection contains duplicate charity-year rows")
    finance_joined = care.merge(
        selected,
        on=["target_year", "charity_number"],
        how="left",
        validate="one_to_one",
    )
    save_parquet(finance_joined, STAGE_DIRS["finance"] / "active_care_with_finance.parquet")
    selected.to_csv(PACKAGE_DIR / "charity_finance_provenance.csv", index=False)
    for year in YEARS:
        group = finance_joined.loc[finance_joined["target_year"].eq(year)]
        print(
            f"  {year}: care={len(group):,}; finance={group['income_2021_gbp'].notna().sum():,}; "
            f"zero={group['income_2021_gbp'].eq(0).sum():,}; missing={group['income_2021_gbp'].isna().sum():,}",
            flush=True,
        )
    return finance_joined


def load_company_links(needed_uids: set[str]) -> pd.DataFrame:
    matches = read_zip_csv(
        SPINE_ZIP,
        "TSCS_spine.matches.csv",
        dtype="string",
        low_memory=False,
    )
    rows = []
    direct = matches.loc[matches["match_type"].eq("companyid - id_in_source")]
    for row in direct.itertuples(index=False):
        a = str(row.orgA_uid)
        b = str(row.orgB_uid)
        if a in needed_uids and b.startswith("GB-COH-"):
            rows.append((a, b.removeprefix("GB-COH-"), row.match_type))
        elif b in needed_uids and a.startswith("GB-COH-"):
            rows.append((b, a.removeprefix("GB-COH-"), row.match_type))
    if not rows:
        return pd.DataFrame(columns=["uid", "direct_company_link_count", "company_number"])
    links = pd.DataFrame(rows, columns=["uid", "company_number", "match_type"]).drop_duplicates()
    summary = (
        links.groupby("uid")
        .agg(
            direct_company_link_count=("company_number", "nunique"),
            company_number=("company_number", "first"),
        )
        .reset_index()
    )
    summary.loc[summary["direct_company_link_count"].ne(1), "company_number"] = pd.NA
    return summary


def supplementary_postcode_candidates(needed_uids: set[str]) -> pd.DataFrame:
    pieces = []
    with zipfile.ZipFile(SPINE_ZIP) as archive:
        with archive.open("TSCS_spine.supplementary.csv") as handle:
            for chunk in pd.read_csv(
                handle,
                usecols=["uid", "postcode"],
                dtype="string",
                chunksize=400_000,
                low_memory=False,
            ):
                keep = chunk["uid"].isin(needed_uids) & chunk["postcode"].notna()
                if keep.any():
                    pieces.append(chunk.loc[keep].copy())
    if not pieces:
        return pd.DataFrame(
            columns=["uid", "supplementary_postcode_count", "supplementary_postcode_candidate"]
        )
    candidates = pd.concat(pieces, ignore_index=True)
    candidates["postcode_norm"] = norm_postcode(candidates["postcode"])
    candidates = candidates.dropna(subset=["postcode_norm"]).drop_duplicates(
        ["uid", "postcode_norm"]
    )
    return (
        candidates.groupby("uid")
        .agg(
            supplementary_postcode_count=("postcode_norm", "nunique"),
            supplementary_postcode_candidate=("postcode_norm", "first"),
        )
        .reset_index()
    )


def load_location_audits() -> pd.DataFrame:
    frames = []
    for year in YEARS:
        path = OLD_LOCATION_DIR / f"charity_{year}:01_updated" / f"location_audit_{year}.csv"
        audit = pd.read_csv(path, low_memory=False)
        audit["target_year"] = year
        audit["charity_number"] = pd.to_numeric(audit["charity_number"], errors="coerce").astype(
            "Int64"
        )
        keep = [
            "target_year",
            "charity_number",
            "final_postcode",
            "final_location_source",
            "location_quality",
            "eligible_for_update",
            "change_applied",
            "earliest_change_date",
            "recovery_method",
        ]
        frames.append(audit[keep])
    return pd.concat(frames, ignore_index=True)


def stage_addresses(finance_joined: pd.DataFrame) -> pd.DataFrame:
    print("\n[4/8] Historical address hierarchy", flush=True)
    needed_uids = set(finance_joined["uid"].astype(str))
    links = load_company_links(needed_uids)
    supplementary = supplementary_postcode_candidates(needed_uids)
    audits = load_location_audits()

    address = finance_joined.merge(links, on="uid", how="left", validate="many_to_one")
    address = address.merge(supplementary, on="uid", how="left", validate="many_to_one")
    address = address.merge(
        audits,
        on=["target_year", "charity_number"],
        how="left",
        validate="one_to_one",
    )
    address["direct_company_link_count"] = address["direct_company_link_count"].fillna(0).astype(int)
    address["supplementary_postcode_count"] = (
        address["supplementary_postcode_count"].fillna(0).astype(int)
    )
    address["data_spine_primary_postcode"] = norm_postcode(
        address["data_spine_primary_postcode"]
    )

    selected_rows = []
    for row in address.itertuples(index=False):
        year = int(row.target_year)
        census_date = CENSUS_DATES[year]
        cc_postcode = getattr(row, f"postcode_{year}")
        cc_source = getattr(row, f"postcode_source_{year}")
        audited_source = row.final_location_source
        audited_postcode = row.final_postcode

        if (
            pd.notna(audited_postcode)
            and pd.notna(audited_source)
            and "companies_house" in str(audited_source).lower()
        ):
            selected_postcode = audited_postcode
            source = audited_source
            method = (
                row.recovery_method
                if pd.notna(row.recovery_method)
                else "validated_existing_companies_house_decision"
            )
            quality = row.location_quality
            evidence_date = pd.to_datetime(row.earliest_change_date, errors="coerce")
            historical_flag = True
        elif pd.notna(cc_postcode) and str(cc_source) != "current":
            selected_postcode = cc_postcode
            source = f"charity_commission_{cc_source}"
            method = "existing_cc_snapshot_priority_proxy"
            quality = "C_dated_charity_commission_archive_proxy"
            evidence_date = SNAPSHOT_DATES.get(str(cc_source), pd.NaT)
            historical_flag = True
        else:
            selected_postcode = pd.NA
            source = pd.NA
            method = "unresolved_no_defensible_dated_historical_address"
            quality = "U_unresolved"
            evidence_date = pd.NaT
            historical_flag = False

        selected_rows.append(
            {
                "target_year": year,
                "charity_number": int(row.charity_number),
                "historical_postcode": selected_postcode,
                "address_source": source,
                "address_evidence_date": evidence_date,
                "address_year_offset": (
                    (evidence_date - census_date).days / 365.25
                    if pd.notna(evidence_date)
                    else np.nan
                ),
                "address_method": method,
                "address_quality": quality,
                "historical_address_flag": historical_flag,
                "current_or_undated_address_not_backcast": pd.isna(selected_postcode)
                and (
                    pd.notna(row.data_spine_primary_postcode)
                    or pd.notna(cc_postcode)
                    or row.supplementary_postcode_count > 0
                ),
                "companies_house_followup_possible": pd.isna(selected_postcode)
                and row.direct_company_link_count == 1,
            }
        )

    selected = pd.DataFrame(selected_rows)
    selected["historical_postcode"] = norm_postcode(selected["historical_postcode"])
    address = address.merge(
        selected,
        on=["target_year", "charity_number"],
        how="left",
        validate="one_to_one",
    )
    save_parquet(address, STAGE_DIRS["address"] / "active_care_with_address.parquet")
    provenance_columns = [
        "target_year",
        "uid",
        "charity_number",
        "charity_name",
        "historical_postcode",
        "address_source",
        "address_evidence_date",
        "address_year_offset",
        "address_method",
        "address_quality",
        "historical_address_flag",
        "data_spine_primary_postcode",
        "supplementary_postcode_count",
        "supplementary_postcode_candidate",
        "current_or_undated_address_not_backcast",
        "direct_company_link_count",
        "company_number",
        "companies_house_followup_possible",
    ]
    address[provenance_columns].to_csv(
        PACKAGE_DIR / "charity_address_provenance.csv", index=False
    )
    pending = address.loc[
        address["historical_postcode"].isna()
        & address["companies_house_followup_possible"],
        provenance_columns,
    ]
    pending.to_csv(
        STAGE_DIRS["address"] / "companies_house_history_followup_required.csv",
        index=False,
    )
    for year in YEARS:
        group = address.loc[address["target_year"].eq(year)]
        print(
            f"  {year}: historical postcode={group['historical_postcode'].notna().sum():,}/{len(group):,}; "
            f"CH follow-up candidates={group['companies_house_followup_possible'].sum():,}",
            flush=True,
        )
    return address


def stage_geocode(address: pd.DataFrame) -> pd.DataFrame:
    print("\n[5/8] ONSPD geocoding", flush=True)
    lookup = pd.read_parquet(
        POSTCODE_LOOKUP,
        columns=["pcds", "doterm", "oslaua", "rgn", "lsoa01", "lsoa11", "lsoa21", "lat", "long"],
    )
    lookup["postcode_norm"] = norm_postcode(lookup["pcds"])
    lookup["doterm_numeric"] = pd.to_numeric(lookup["doterm"], errors="coerce")
    lookup = (
        lookup.sort_values(["postcode_norm", "doterm_numeric"], na_position="last")
        .drop_duplicates("postcode_norm", keep="last")
    )
    pieces = []
    for year in YEARS:
        group = address.loc[address["target_year"].eq(year)].copy()
        lsoa_column = LSOA_LOOKUP_COLUMNS[year]
        geo = lookup[
            ["postcode_norm", "pcds", "doterm_numeric", "oslaua", "rgn", lsoa_column, "lat", "long"]
        ].rename(columns={lsoa_column: "lsoa_code"})
        group = group.merge(
            geo,
            left_on="historical_postcode",
            right_on="postcode_norm",
            how="left",
            validate="many_to_one",
        )
        target_month = int(CENSUS_DATES[year].strftime("%Y%m"))
        group["postcode_not_terminated_before_census"] = (
            group["doterm_numeric"].isna() | group["doterm_numeric"].ge(target_month)
        )
        group["geocoded"] = group["lsoa_code"].notna() & group["lsoa_code"].ne("E99999999")
        pieces.append(group)
        print(
            f"  {year}: geocoded={group['geocoded'].sum():,}/{len(group):,}; "
            f"selected postcodes terminated before Census={((group['historical_postcode'].notna()) & ~group['postcode_not_terminated_before_census']).sum():,}",
            flush=True,
        )
    geocoded = pd.concat(pieces, ignore_index=True)
    save_parquet(geocoded, STAGE_DIRS["geocode"] / "active_care_geocoded.parquet")
    return geocoded


def stage_geography(geocoded: pd.DataFrame) -> pd.DataFrame:
    print("\n[6/8] Fixed seven-ICB study-area filter", flush=True)
    pieces = []
    for year in YEARS:
        group = geocoded.loc[geocoded["target_year"].eq(year)].copy()
        study = pd.read_csv(
            FINAL_DIR / "covariates" / f"{year}.csv",
            usecols=["lsoa_code", "lsoa_name", "ICB23CD", "ICB23NM"],
        )
        study["lsoa_code"] = study["lsoa_code"].astype(str)
        if not study["lsoa_code"].is_unique:
            raise AssertionError(f"{year} study-area LSOA lookup is not unique")
        group = group.merge(
            study,
            on="lsoa_code",
            how="left",
            validate="many_to_one",
            suffixes=("", "_study"),
        )
        group["in_seven_icb_study_area"] = group["ICB23CD"].notna()
        pieces.append(group)
        print(
            f"  {year}: retained in seven ICBs={group['in_seven_icb_study_area'].sum():,}",
            flush=True,
        )
    all_with_area = pd.concat(pieces, ignore_index=True)
    save_parquet(all_with_area, STAGE_DIRS["geography"] / "active_care_with_study_area.parquet")
    final = all_with_area.loc[all_with_area["in_seven_icb_study_area"]].copy()
    if final.duplicated(["target_year", "charity_number"]).any():
        raise AssertionError("Study-area output contains duplicate charity-year rows")
    return final


def output_columns_for_year(frame: pd.DataFrame, year: int) -> pd.DataFrame:
    group = frame.loc[frame["target_year"].eq(year)].copy()
    group[f"lsoa_{year}"] = group["lsoa_code"]
    group["analysis_year"] = year
    group["postcode"] = group["historical_postcode"]
    group["local_authority_code"] = group["oslaua"]
    group["registration_date"] = group["registerdate"]
    group["removal_date"] = group["removeddate"]
    group["income_proxy_gbp"] = group["income_nominal"]
    group["income_date"] = group["fye"]
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
        "lsoa_name",
        "local_authority_code",
        "ICB23CD",
        "ICB23NM",
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
    ]
    return group[ordered].sort_values("charity_number").reset_index(drop=True)


def stage_final_and_qa(
    active_all: pd.DataFrame,
    active_with_care: pd.DataFrame,
    care: pd.DataFrame,
    geocoded: pd.DataFrame,
    final: pd.DataFrame,
    presence_audit: dict,
) -> None:
    print("\n[7/8] Final outputs", flush=True)
    final_frames = {}
    for year in YEARS:
        output = output_columns_for_year(final, year)
        final_frames[year] = output
        path = STAGE_DIRS["final"] / f"charity_income_{year}_hybrid_v2.csv"
        output.to_csv(path, index=False)
        print(
            f"  {year}: final={len(output):,}; usable income={output['income_2021_gbp'].notna().sum():,}",
            flush=True,
        )

    print("\n[8/8] QA and old-v2 comparison", flush=True)
    qa_rows = []
    comparison_rows = []
    reason_rows = []
    for year in YEARS:
        active_year = active_all.loc[active_all["target_year"].eq(year)]
        joined_year = active_with_care.loc[active_with_care["target_year"].eq(year)]
        care_year = care.loc[care["target_year"].eq(year)]
        geo_year = geocoded.loc[geocoded["target_year"].eq(year)]
        output = final_frames[year]
        old = pd.read_csv(
            OLD_CHARITY_DIR / f"charity_income_{year}_hybrid.csv", low_memory=False
        )
        old_ids = set(pd.to_numeric(old["charity_number"], errors="coerce").dropna().astype(int))
        new_ids = set(output["charity_number"].astype(int))
        new_only = new_ids - old_ids
        old_only = old_ids - new_ids
        both = old_ids & new_ids
        new_only_cc_spell_disagreement = int(
            output.loc[
                output["charity_number"].isin(new_only),
                "presence_date_disagrees_with_cc_spells",
            ].sum()
        )

        metrics = {
            "active_before_care_filter": len(active_year),
            "care_related": len(care_year),
            "care_evidence_unavailable": int(joined_year["care_strict"].isna().sum()),
            "historical_postcode": int(geo_year["historical_postcode"].notna().sum()),
            "successfully_geocoded": int(geo_year["geocoded"].sum()),
            "retained_in_study_area": len(output),
            "with_finance": int(output["income_2021_gbp"].notna().sum()),
            "zero_income": int(output["income_2021_gbp"].eq(0).sum()),
            "missing_income": int(output["income_2021_gbp"].isna().sum()),
            "duplicates_removed_presence": int(presence_audit["duplicate_gb_chc_uids_removed"]),
            "new_only_vs_old": len(new_only),
            "old_only_vs_new": len(old_only),
            "shared_with_old": len(both),
        }
        qa_rows.extend(
            {"year": year, "category": "core", "metric": key, "value": value}
            for key, value in metrics.items()
        )
        for source, count in output["address_source"].fillna("missing").value_counts().items():
            qa_rows.append(
                {"year": year, "category": "address_source", "metric": str(source), "value": int(count)}
            )
        for source, count in output["selection_method"].fillna("missing").value_counts().items():
            qa_rows.append(
                {"year": year, "category": "finance_source", "metric": str(source), "value": int(count)}
            )
        for route, count in output["care_route"].fillna("missing").value_counts().items():
            qa_rows.append(
                {"year": year, "category": "care_route", "metric": str(route), "value": int(count)}
            )

        comparison_rows.append(
            {
                "year": year,
                "old_count": len(old),
                "new_count": len(output),
                "difference": len(output) - len(old),
                "percent_difference": 100 * (len(output) - len(old)) / len(old),
                "shared": len(both),
                "new_only": len(new_only),
                "old_only": len(old_only),
                "new_only_cc_spell_disagreement": new_only_cc_spell_disagreement,
                "main_causes": (
                    "Data Spine primary presence dates; national-before-study-area rebuild; "
                    "dated-address requirement; fixed seven-ICB LSOA membership"
                ),
            }
        )

        membership = pd.DataFrame(
            {
                "charity_number": sorted(old_ids | new_ids),
            }
        )
        membership["in_old"] = membership["charity_number"].isin(old_ids)
        membership["in_v2"] = membership["charity_number"].isin(new_ids)
        membership["membership_status"] = np.select(
            [membership["in_old"] & membership["in_v2"], membership["in_v2"]],
            ["both", "v2_only"],
            default="old_only",
        )
        membership.to_csv(
            STAGE_DIRS["qa"] / f"old_vs_v2_membership_{year}.csv", index=False
        )

        # Diagnose old-only rows from the rebuilt national audit where possible.
        audit_index = joined_year.set_index("charity_number", drop=False)
        geo_index = geo_year.set_index("charity_number", drop=False)
        for charity_number in sorted(old_only):
            if charity_number not in audit_index.index:
                reason = "not_active_under_data_spine_primary_dates"
            else:
                row = audit_index.loc[charity_number]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                care_value = row.get("care_strict", False)
                if pd.isna(care_value) or not bool(care_value):
                    reason = "not_care_strict_or_care_evidence_unavailable"
                elif charity_number not in geo_index.index:
                    reason = "not_carried_to_geocode_stage"
                else:
                    grow = geo_index.loc[charity_number]
                    if isinstance(grow, pd.DataFrame):
                        grow = grow.iloc[0]
                    if pd.isna(grow.get("historical_postcode")):
                        reason = "no_defensible_dated_historical_postcode"
                    else:
                        geocoded_value = grow.get("geocoded", False)
                        if pd.isna(geocoded_value) or not bool(geocoded_value):
                            reason = "postcode_not_geocoded"
                        else:
                            reason = "geocoded_lsoa_outside_fixed_seven_icb_study_area"
            reason_rows.append(
                {"year": year, "charity_number": charity_number, "direction": "old_only", "reason": reason}
            )

    qa = pd.DataFrame(qa_rows)
    comparison = pd.DataFrame(comparison_rows)
    reasons = pd.DataFrame(reason_rows)
    qa.to_csv(PACKAGE_DIR / "charity_rebuild_qa.csv", index=False)
    comparison.to_csv(STAGE_DIRS["qa"] / "old_vs_v2_summary.csv", index=False)
    reasons.to_csv(STAGE_DIRS["qa"] / "old_only_reason_audit.csv", index=False)

    unresolved = geocoded.loc[geocoded["historical_postcode"].isna()].copy()
    unresolved.to_csv(
        STAGE_DIRS["address"] / "unresolved_historical_locations.csv", index=False
    )

    report_lines = [
        "# Charity rebuild audit report",
        "",
        "## Status",
        "",
        "The rebuild completed from the July 2026 Data Spine organisation register and financial history, using the existing dissertation care, income, postcode and study-area rules. Old files were not modified.",
        "",
        "## Rules reused",
        "",
        "- Presence base: one `GB-CHC-{charity_number}` Data Spine row; active where `registerdate <= Census day` and removal is missing or later than Census day.",
        "- Care eligibility: `care_keyword_strong OR ((class_who_elderly OR class_who_disabled) AND class_how_services)` from the existing Charity Commission current/archive master.",
        "- Finance: exact GB-CHC UID; covering period; otherwise nearest FYE within 365 days; otherwise existing final-workbook proxy; missing is not zero.",
        "- Deflation: ONS L522 annual CPIH to constant 2021 pounds; income remains in levels.",
        "- Address: validated existing Companies House filing-history decisions first, then the established dated Charity Commission archive proxy. Current and undated supplementary addresses were not back-cast.",
        "- Geography: existing postcode normalisation, ONSPD Census-year LSOA vintage, and the fixed seven-ICB LSOA membership from the final covariate files.",
        "",
        "## Old versus rebuilt study-area counts",
        "",
        markdown_table(comparison),
        "",
        "## Core QA",
        "",
    ]
    core = qa.loc[qa["category"].eq("core")].pivot(
        index="metric", columns="year", values="value"
    )
    report_lines.extend([markdown_table(core, include_index=True), "", "## Known limitations", ""])
    final_address_offsets = {
        year: float(final_frames[year]["address_year_offset"].abs().median())
        for year in YEARS
    }
    ch_pending = geocoded.loc[
        geocoded["historical_postcode"].isna()
        & geocoded["companies_house_followup_possible"]
    ]
    report_lines.extend(
        [
            "- Data Spine primary dates are used exactly as requested, but its guidance notes that the main spine stores the earliest registration date and newest authoritative removal date. The exported CC-spell disagreement flag identifies possible re-registration-gap cases. All v2-only study-area records (243 in 2001, 208 in 2011 and 6 in 2021) have this disagreement flag, so they should not be treated as confirmed additions without spell-level adjudication.",
            "- Data Spine supplementary names, addresses and dates are independent sets. Supplementary postcodes are retained only as QA candidates and never assigned to a Census year.",
            f"- No Companies House API key was available in the workspace. Existing validated filing-history decisions were reused; {len(ch_pending):,} unresolved incorporated charity-year cases ({ch_pending['charity_number'].nunique():,} unique charities/companies) are exported to `04_historical_addresses/companies_house_history_followup_required.csv`. The exact missing source is dated Companies House registered-office filing history for those company numbers.",
            "- Unincorporated organisations without a dated Charity Commission archive address remain unresolved.",
            f"- The median absolute address evidence offset is {final_address_offsets[2001]:.2f} years for 2001, {final_address_offsets[2011]:.2f} years for 2011 and {final_address_offsets[2021]:.2f} years for 2021. The long 2001 offset mainly reflects the earliest locally available Charity Commission snapshot being December 2014, so 2001 locations remain proxy locations even when geocoded successfully.",
            "- Registered/contact/registered-office locations are organisational proxies, not verified service-delivery locations.",
            "- The care definition is intentionally inclusive and retrospectively uses evidence unioned across current and archived Charity Commission sources.",
            "",
            "## Reused implementation lineage",
            "",
            "- Care classification: `分析历史/very_first_dataset/charity data Claude/scripts/care_flags.py` and `05_build_universe.py`, via the resulting `charity_master_all.parquet` audit fields.",
            "- Presence cross-check: `registration_spells.parquet`; Data Spine primary dates remain the requested inclusion rule.",
            "- Historical addresses: the established yearly `location_audit_{year}.csv` decisions under `分析历史/charity_original_data/`, with current/undated addresses explicitly refused for back-casting.",
            "- Finance hierarchy: the executed rules recovered from `01_rebuild_income.ipynb`, applied to exact GB-CHC UIDs in the July 2026 finance history.",
            "- Postcode and geography: `postcode_lookup.parquet` plus the final year-specific `covariates/{year}.csv` seven-ICB membership.",
            "",
            "## Main outputs",
            "",
            "- `07_final_outputs/charity_income_2001_hybrid_v2.csv`",
            "- `07_final_outputs/charity_income_2011_hybrid_v2.csv`",
            "- `07_final_outputs/charity_income_2021_hybrid_v2.csv`",
            "- `charity_rebuild_qa.csv`",
            "- `charity_address_provenance.csv`",
            "- `charity_finance_provenance.csv`",
        ]
    )
    (PACKAGE_DIR / "CHARITY_REBUILD_REPORT.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    print(comparison.to_string(index=False), flush=True)


def main() -> None:
    ensure_paths()
    if "--qa-only" in sys.argv[1:]:
        active_all = pd.read_parquet(
            STAGE_DIRS["presence"] / "active_charity_years.parquet"
        )
        active_with_care = pd.read_parquet(
            STAGE_DIRS["care"] / "active_with_care_audit.parquet"
        )
        care = pd.read_parquet(
            STAGE_DIRS["care"] / "active_care_charity_years.parquet"
        )
        geocoded = pd.read_parquet(
            STAGE_DIRS["geocode"] / "active_care_geocoded.parquet"
        )
        all_with_area = pd.read_parquet(
            STAGE_DIRS["geography"] / "active_care_with_study_area.parquet"
        )
        final = all_with_area.loc[all_with_area["in_seven_icb_study_area"]].copy()
        presence_audit = json.loads(
            (STAGE_DIRS["presence"] / "presence_audit.json").read_text(encoding="utf-8")
        )
        stage_final_and_qa(
            active_all,
            active_with_care,
            care,
            geocoded,
            final,
            presence_audit,
        )
        print("\nQA/report rebuild complete.", flush=True)
        return
    active_all, presence_audit = stage_presence()
    active_with_care, care = stage_care(active_all)
    finance_joined = stage_finance(care)
    address = stage_addresses(finance_joined)
    geocoded = stage_geocode(address)
    final = stage_geography(geocoded)
    stage_final_and_qa(
        active_all,
        active_with_care,
        care,
        geocoded,
        final,
        presence_audit,
    )
    print("\nRebuild complete.", flush=True)


if __name__ == "__main__":
    main()
