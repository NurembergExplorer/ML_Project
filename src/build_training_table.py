#!/usr/bin/env python3
"""
Build a composition training table and write report-friendly summaries.

This keeps the original behavior but adds:
- train_table_summary.csv
- train_target_distribution.csv
- train_dominant_class_distribution.csv
- train_valid_frac_distribution.csv
- data_issues_support.csv
"""

from __future__ import annotations

import argparse
import os
from typing import List

import geopandas as gpd
import numpy as np
import pandas as pd

FEATURE_COLS = [
    "B2_med",
    "B3_med",
    "B4_med",
    "B8_med",
    "ndvi_med",
    "ndvi_std",
    "ndwi_med",
    "ndwi_std",
    "valid_frac",
]

TARGET_COLS = ["built_prop", "veg_prop", "water_prop", "other_prop"]


def _ensure_dir_for_file(filepath: str) -> None:
    d = os.path.dirname(os.path.abspath(filepath))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _load_grid_block_ids(grid_path: str, grid_layer: str | None) -> pd.DataFrame:
    gdf = gpd.read_file(grid_path, layer=grid_layer) if grid_layer else gpd.read_file(grid_path)
    if gdf.empty:
        raise ValueError("Grid file is empty.")
    for col in ["cell_id", "block_id"]:
        if col not in gdf.columns:
            raise ValueError(f"Grid must include '{col}'. Re-run make_grid.py to ensure block_id exists.")
    df = gdf[["cell_id", "block_id"]].copy()
    df["cell_id"] = df["cell_id"].astype(np.int64)
    df["block_id"] = df["block_id"].astype(np.int64)
    return df


def _load_features(features_path: str, year: int) -> pd.DataFrame:
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found for year={year}: {features_path}")
    df = pd.read_parquet(features_path)
    if "cell_id" not in df.columns:
        raise ValueError(f"Features missing 'cell_id': {features_path}")
    if "year" not in df.columns:
        df["year"] = year
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Features file missing columns {missing}: {features_path}")
    df["cell_id"] = df["cell_id"].astype(np.int64)
    df["year"] = df["year"].astype(np.int32)
    out = df[["cell_id", "year"] + FEATURE_COLS].copy()
    if out.duplicated(subset=["cell_id", "year"]).any():
        raise ValueError(f"Features file contains duplicate cell_id/year rows: {features_path}")
    return out


def _load_labels(labels_path: str, year: int) -> pd.DataFrame:
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found for year={year}: {labels_path}")
    df = pd.read_parquet(labels_path)
    if "cell_id" not in df.columns:
        raise ValueError(f"Labels missing 'cell_id': {labels_path}")
    if "year" not in df.columns:
        df["year"] = year
    missing = [c for c in TARGET_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Labels file missing columns {missing}: {labels_path}")
    df["cell_id"] = df["cell_id"].astype(np.int64)
    df["year"] = df["year"].astype(np.int32)
    out = df[["cell_id", "year"] + TARGET_COLS].copy()
    if out.duplicated(subset=["cell_id", "year"]).any():
        raise ValueError(f"Labels file contains duplicate cell_id/year rows: {labels_path}")
    return out


def _clip_props(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in TARGET_COLS:
        out[c] = out[c].clip(lower=0.0, upper=1.0)
    s = out[TARGET_COLS].sum(axis=1)
    mask = s > 0
    out.loc[mask, TARGET_COLS] = out.loc[mask, TARGET_COLS].div(s[mask], axis=0)
    return out


def _write_report_support(train: pd.DataFrame, output_path: str) -> None:
    base = os.path.splitext(output_path)[0]
    summary_path = base + "_summary.csv"

    summary = (
        train.groupby("year")[TARGET_COLS + ["valid_frac"]]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    summary.to_csv(summary_path, index=False)

    target_dist = train.groupby("year")[TARGET_COLS].mean().reset_index()
    target_dist.to_csv(os.path.join(os.path.dirname(output_path), "train_target_distribution.csv"), index=False)

    dominant = train[TARGET_COLS].idxmax(axis=1).str.replace("_prop", "", regex=False)
    dom_df = (
        pd.DataFrame({"year": train["year"], "dominant_class": dominant})
        .value_counts()
        .rename("n_cells")
        .reset_index()
    )
    dom_df["share"] = dom_df.groupby("year")["n_cells"].transform(lambda s: s / s.sum())
    dom_df.to_csv(os.path.join(os.path.dirname(output_path), "train_dominant_class_distribution.csv"), index=False)

    valid_bins = pd.cut(
        train["valid_frac"],
        bins=[-0.001, 0.25, 0.50, 0.75, 0.90, 1.0],
        labels=["0-0.25", "0.25-0.50", "0.50-0.75", "0.75-0.90", "0.90-1.00"],
    )
    valid_df = (
        pd.DataFrame({"year": train["year"], "valid_frac_bin": valid_bins})
        .value_counts()
        .rename("n_cells")
        .reset_index()
    )
    valid_df["share"] = valid_df.groupby("year")["n_cells"].transform(lambda s: s / s.sum())
    valid_df.to_csv(os.path.join(os.path.dirname(output_path), "train_valid_frac_distribution.csv"), index=False)

    mixed_cell_share = float((train[TARGET_COLS].max(axis=1) < 0.8).mean())
    low_valid_share = float((train["valid_frac"] < 0.8).mean())
    rare_water_share = float((train["water_prop"] > 0.5).mean())
    other_dominant_share = float((dominant == "other").mean())

    issues = pd.DataFrame(
        [
            {
                "issue": "low_valid_fraction",
                "evidence_value": low_valid_share,
                "explanation": "Cells with low valid pixel fraction may reflect clouds, masking, or missingness.",
            },
            {
                "issue": "mixed_cells",
                "evidence_value": mixed_cell_share,
                "explanation": "Cells whose dominant class is below 80% are mixed and harder to model cleanly.",
            },
            {
                "issue": "rare_water_dominance",
                "evidence_value": rare_water_share,
                "explanation": "Water-dominant cells are relatively uncommon, so class-specific performance may vary.",
            },
            {
                "issue": "other_class_ambiguity",
                "evidence_value": other_dominant_share,
                "explanation": "'Other' aggregates several heterogeneous classes and is intrinsically noisy.",
            },
        ]
    )
    issues.to_csv(os.path.join(os.path.dirname(output_path), "data_issues_support.csv"), index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build training table by merging features and labels.")
    p.add_argument("--grid", type=str, required=True, help="Path to base grid gpkg (must include block_id).")
    p.add_argument("--grid-layer", type=str, default=None, help="Optional grid layer name for gpkg.")
    p.add_argument("--features-dir", type=str, required=True, help="Directory containing features_YYYY.parquet.")
    p.add_argument("--labels-dir", type=str, required=True, help="Directory containing labels_YYYY.parquet.")
    p.add_argument("--years", type=int, nargs="+", required=True, help="Label years to include.")
    p.add_argument("--features-pattern", type=str, default="features_{year}.parquet")
    p.add_argument("--labels-pattern", type=str, default="labels_{year}.parquet")
    p.add_argument("--output", type=str, default="data/processed/tables/train_table.parquet")
    p.add_argument("--min-valid-frac", type=float, default=0.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not 0.0 <= float(args.min_valid_frac) <= 1.0:
        raise ValueError("--min-valid-frac must be between 0 and 1 inclusive.")

    years: List[int] = sorted({int(y) for y in args.years})
    grid_df = _load_grid_block_ids(args.grid, args.grid_layer)

    tables = []
    for y in years:
        feat_path = os.path.join(args.features_dir, args.features_pattern.format(year=y))
        lab_path = os.path.join(args.labels_dir, args.labels_pattern.format(year=y))

        feats = _load_features(feat_path, y)
        labs = _load_labels(lab_path, y)

        df = feats.merge(labs, on=["cell_id", "year"], how="inner")
        df = df.merge(grid_df, on="cell_id", how="left")

        if df["block_id"].isna().any():
            raise ValueError("Some rows are missing block_id after merge.")

        df["block_id"] = df["block_id"].astype(np.int64)

        if float(args.min_valid_frac) > 0:
            df = df.loc[df["valid_frac"] >= float(args.min_valid_frac)].copy()

        df = _clip_props(df)
        needed = ["cell_id", "year", "block_id"] + FEATURE_COLS + TARGET_COLS
        df = df.dropna(subset=needed).copy()
        tables.append(df)

    if not tables:
        raise ValueError("No data loaded. Check input years and file paths.")

    train = pd.concat(tables, ignore_index=True)
    _ensure_dir_for_file(args.output)
    train.to_parquet(args.output, index=False)
    _write_report_support(train, args.output)

    print("✅ Training table built successfully")
    print(f"Rows: {len(train):,}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()