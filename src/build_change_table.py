#!/usr/bin/env python3
"""
Build an explicit supervised change table for direct change modeling.

Outputs include:
- delta_built
- delta_veg
- delta_water
- delta_other
- change_binary

Feature design:
- t1 features
- t2 features
- feature deltas (t2 - t1)

This is the assignment-required direct change dataset.
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
DELTA_TARGETS = ["delta_built", "delta_veg", "delta_water", "delta_other"]


def _ensure_dir_for_file(filepath: str) -> None:
    d = os.path.dirname(os.path.abspath(filepath))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _load_grid(grid_path: str, grid_layer: str | None) -> pd.DataFrame:
    gdf = gpd.read_file(grid_path, layer=grid_layer) if grid_layer else gpd.read_file(grid_path)
    needed = ["cell_id", "block_id"]
    missing = [c for c in needed if c not in gdf.columns]
    if missing:
        raise ValueError(f"Grid missing columns: {missing}")
    out = gdf[needed].copy()
    out["cell_id"] = out["cell_id"].astype(np.int64)
    out["block_id"] = out["block_id"].astype(np.int64)
    return out


def _load_year_table(path: str, cols: List[str], year: int) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    if "cell_id" not in df.columns:
        raise ValueError(f"Missing cell_id in {path}")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    out = df[["cell_id"] + cols].copy()
    out["cell_id"] = out["cell_id"].astype(np.int64)
    out["year"] = int(year)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build direct change modeling table.")
    p.add_argument("--grid", required=True, type=str)
    p.add_argument("--grid-layer", default=None, type=str)
    p.add_argument("--features-dir", required=True, type=str)
    p.add_argument("--labels-dir", required=True, type=str)
    p.add_argument("--year-pairs", nargs="+", required=True, help="Pairs like 2020:2021 2021:2022")
    p.add_argument("--features-pattern", default="features_{year}.parquet", type=str)
    p.add_argument("--labels-pattern", default="labels_{year}.parquet", type=str)
    p.add_argument("--change-threshold", default=0.05, type=float)
    p.add_argument("--output", default="data/processed/tables/change_table.parquet", type=str)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= float(args.change_threshold) <= 1.0:
        raise ValueError("--change-threshold must be in [0,1].")

    grid_df = _load_grid(args.grid, args.grid_layer)
    tables = []

    for pair in args.year_pairs:
        if ":" not in pair:
            raise ValueError(f"Invalid pair '{pair}'. Expected format YEAR1:YEAR2")
        year_t1, year_t2 = [int(x) for x in pair.split(":")]

        feat_t1 = _load_year_table(
            os.path.join(args.features_dir, args.features_pattern.format(year=year_t1)),
            FEATURE_COLS,
            year_t1,
        ).drop(columns=["year"])
        feat_t2 = _load_year_table(
            os.path.join(args.features_dir, args.features_pattern.format(year=year_t2)),
            FEATURE_COLS,
            year_t2,
        ).drop(columns=["year"])

        lab_t1 = _load_year_table(
            os.path.join(args.labels_dir, args.labels_pattern.format(year=year_t1)),
            TARGET_COLS,
            year_t1,
        ).drop(columns=["year"])
        lab_t2 = _load_year_table(
            os.path.join(args.labels_dir, args.labels_pattern.format(year=year_t2)),
            TARGET_COLS,
            year_t2,
        ).drop(columns=["year"])

        df = (
            grid_df.merge(feat_t1, on="cell_id", how="inner")
            .merge(feat_t2, on="cell_id", how="inner", suffixes=("_t1", "_t2"))
            .merge(lab_t1, on="cell_id", how="inner")
            .merge(lab_t2, on="cell_id", how="inner", suffixes=("_t1", "_t2"))
        )

        for c in FEATURE_COLS:
            df[f"{c}_d"] = df[f"{c}_t2"] - df[f"{c}_t1"]

        df["delta_built"] = df["built_prop_t2"] - df["built_prop_t1"]
        df["delta_veg"] = df["veg_prop_t2"] - df["veg_prop_t1"]
        df["delta_water"] = df["water_prop_t2"] - df["water_prop_t1"]
        df["delta_other"] = df["other_prop_t2"] - df["other_prop_t1"]

        max_abs_delta = df[DELTA_TARGETS].abs().max(axis=1)
        df["change_binary"] = (max_abs_delta >= float(args.change_threshold)).astype(int)

        df["year_t1"] = year_t1
        df["year_t2"] = year_t2
        df["pair_id"] = f"{year_t1}_{year_t2}"

        keep_cols = (
            ["cell_id", "block_id", "year_t1", "year_t2", "pair_id"]
            + [f"{c}_t1" for c in FEATURE_COLS]
            + [f"{c}_t2" for c in FEATURE_COLS]
            + [f"{c}_d" for c in FEATURE_COLS]
            + [f"{c}_t1" for c in TARGET_COLS]
            + [f"{c}_t2" for c in TARGET_COLS]
            + DELTA_TARGETS
            + ["change_binary"]
        )
        df = df[keep_cols].dropna().copy()
        tables.append(df)

    if not tables:
        raise ValueError("No change tables were built.")

    out = pd.concat(tables, ignore_index=True)
    _ensure_dir_for_file(args.output)
    out.to_parquet(args.output, index=False)

    change_dist = (
        out.groupby("pair_id")[DELTA_TARGETS + ["change_binary"]]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    change_dist.to_csv(os.path.join(os.path.dirname(args.output), "change_distribution.csv"), index=False)

    print("✅ Change table built successfully")
    print(f"Rows: {len(out):,}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()