#!/usr/bin/env python3
"""
Predict composition for all requested years and direct change for year pairs.

Composition outputs:
- pred_YYYY.parquet
  includes Ridge + LightGBM predictions and LightGBM uncertainty

Change outputs:
- change_pred_YYYY_YYYY.parquet
  includes direct delta predictions from Ridge + LightGBM
  includes binary change probabilities from LogisticRegression + RandomForest
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import Booster

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
TARGETS = ["built_prop", "veg_prop", "water_prop", "other_prop"]
CHANGE_TARGETS = ["delta_built", "delta_veg", "delta_water", "delta_other"]
CHANGE_FEATURE_COLS = (
    [f"{c}_t1" for c in FEATURE_COLS]
    + [f"{c}_t2" for c in FEATURE_COLS]
    + [f"{c}_d" for c in FEATURE_COLS]
)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_features(path: str, year: int) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    missing = [c for c in ["cell_id"] + FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    out = df[["cell_id"] + FEATURE_COLS].copy()
    out["cell_id"] = out["cell_id"].astype(np.int64)
    out["year"] = int(year)
    return out


@dataclass
class LGBMBundle:
    booster: Booster
    feature_cols: List[str]
    median: np.ndarray


def _load_lgbm_bundle(txt_path: str, meta_path: str) -> LGBMBundle:
    booster = Booster(model_file=txt_path)
    meta = json.loads(open(meta_path, "r", encoding="utf-8").read())
    return LGBMBundle(
        booster=booster,
        feature_cols=list(meta["feature_cols"]),
        median=np.array(meta["imputer_median"], dtype=np.float32),
    )


def _impute(X: np.ndarray, median: np.ndarray) -> np.ndarray:
    return np.where(np.isnan(X), median[None, :], X).astype(np.float32, copy=False)


def _clip_and_renorm(preds: np.ndarray) -> np.ndarray:
    p = np.clip(preds, 0.0, 1.0)
    s = p.sum(axis=1, keepdims=True)
    mask = s.squeeze() > 0
    p[mask] = p[mask] / s[mask]
    return p


def _load_ensemble_members(ens_dir: str) -> Tuple[List[Booster], Optional[np.ndarray], Optional[List[str]]]:
    if not os.path.isdir(ens_dir):
        return [], None, None
    member_paths = sorted(glob.glob(os.path.join(ens_dir, "member_*.txt")))
    if not member_paths:
        return [], None, None
    meta = json.loads(open(os.path.join(ens_dir, "imputer_median.json"), "r", encoding="utf-8").read())
    members = [Booster(model_file=p) for p in member_paths]
    return members, np.array(meta["median"], dtype=np.float32), list(meta["feature_cols"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict composition and direct change.")
    p.add_argument("--features-dir", required=True, type=str)
    p.add_argument("--years", nargs="+", required=True, type=int)
    p.add_argument("--model-dir", default="models", type=str)
    p.add_argument("--output-dir", default="data/processed/predictions", type=str)
    p.add_argument("--features-pattern", default="features_{year}.parquet", type=str)
    p.add_argument("--include-uncertainty", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_dir(args.output_dir)

    # composition loaders
    ridge_models = {
        target: joblib.load(os.path.join(args.model_dir, "baseline", f"ridge_{target}.pkl"))
        for target in TARGETS
    }
    lgbm_models = {
        target: _load_lgbm_bundle(
            os.path.join(args.model_dir, "lightgbm", f"lgbm_{target}.txt"),
            os.path.join(args.model_dir, "lightgbm", f"lgbm_{target}_features.json"),
        )
        for target in TARGETS
    }

    lgbm_ensembles: Dict[str, Tuple[List[Booster], Optional[np.ndarray], Optional[List[str]]]] = {}
    if args.include_uncertainty:
        for target in TARGETS:
            lgbm_ensembles[target] = _load_ensemble_members(os.path.join(args.model_dir, "lightgbm", f"ensemble_{target}"))

    feats_by_year: Dict[int, pd.DataFrame] = {}
    for year in args.years:
        fp = os.path.join(args.features_dir, args.features_pattern.format(year=year))
        feats_by_year[int(year)] = _load_features(fp, int(year))

    # composition prediction
    for year, feats in feats_by_year.items():
        out = pd.DataFrame({"cell_id": feats["cell_id"], "year": feats["year"], "valid_frac": feats["valid_frac"]})

        ridge_matrix = []
        lgbm_matrix = []
        std_matrix = []

        for short, target in zip(["built", "veg", "water", "other"], TARGETS):
            pred_ridge = ridge_models[target].predict(feats[FEATURE_COLS])
            out[f"{short}_pred_ridge"] = pred_ridge.astype(np.float32)
            ridge_matrix.append(pred_ridge.astype(np.float32))

            bundle = lgbm_models[target]
            X = feats[bundle.feature_cols].to_numpy(dtype=np.float32)
            pred_lgbm = bundle.booster.predict(_impute(X, bundle.median)).astype(np.float32)
            out[f"{short}_pred_lgbm"] = pred_lgbm
            lgbm_matrix.append(pred_lgbm)

            if args.include_uncertainty:
                members, med, cols = lgbm_ensembles.get(target, ([], None, None))
                if members and med is not None and cols is not None:
                    X_e = feats[cols].to_numpy(dtype=np.float32)
                    X_ei = _impute(X_e, med)
                    member_preds = np.stack([m.predict(X_ei).astype(np.float32) for m in members], axis=0)
                    std = member_preds.std(axis=0).astype(np.float32)
                else:
                    std = np.full(len(feats), np.nan, dtype=np.float32)
                out[f"{short}_std_lgbm"] = std
                std_matrix.append(std)

        ridge_matrix = _clip_and_renorm(np.stack(ridge_matrix, axis=1))
        lgbm_matrix = _clip_and_renorm(np.stack(lgbm_matrix, axis=1))

        for idx, short in enumerate(["built", "veg", "water", "other"]):
            out[f"{short}_pred_ridge"] = ridge_matrix[:, idx]
            out[f"{short}_pred_lgbm"] = lgbm_matrix[:, idx]

        out.to_parquet(os.path.join(args.output_dir, f"pred_{year}.parquet"), index=False)
        print(f"✅ Saved composition predictions for {year}")

    # change model loaders
    change_ridge = {
        target: joblib.load(os.path.join(args.model_dir, "change", "regression", "baseline", f"ridge_{target}.pkl"))
        for target in CHANGE_TARGETS
    }
    change_lgbm = {
        target: _load_lgbm_bundle(
            os.path.join(args.model_dir, "change", "regression", "lightgbm", f"lgbm_{target}.txt"),
            os.path.join(args.model_dir, "change", "regression", "lightgbm", f"lgbm_{target}_features.json"),
        )
        for target in CHANGE_TARGETS
    }
    change_ensembles: Dict[str, Tuple[List[Booster], Optional[np.ndarray], Optional[List[str]]]] = {}
    if args.include_uncertainty:
        for target in CHANGE_TARGETS:
            change_ensembles[target] = _load_ensemble_members(
                os.path.join(args.model_dir, "change", "regression", "lightgbm", f"ensemble_{target}")
            )

    logreg = joblib.load(os.path.join(args.model_dir, "change", "classification", "logreg_change_binary.pkl"))
    rf_bundle = joblib.load(os.path.join(args.model_dir, "change", "classification", "rf_change_binary.pkl"))

    years_sorted = sorted(feats_by_year)
    for year_t1, year_t2 in zip(years_sorted[:-1], years_sorted[1:]):
        f1 = feats_by_year[year_t1].copy()
        f2 = feats_by_year[year_t2].copy()

        df = f1.merge(f2, on="cell_id", suffixes=("_t1", "_t2"), how="inner")
        for c in FEATURE_COLS:
            df[f"{c}_d"] = df[f"{c}_t2"] - df[f"{c}_t1"]

        out = pd.DataFrame(
            {
                "cell_id": df["cell_id"],
                "year_t1": int(year_t1),
                "year_t2": int(year_t2),
                "pair_id": f"{year_t1}_{year_t2}",
                "valid_frac_t1": df["valid_frac_t1"],
                "valid_frac_t2": df["valid_frac_t2"],
            }
        )

        for target in CHANGE_TARGETS:
            pred_ridge = change_ridge[target].predict(df[CHANGE_FEATURE_COLS])
            out[f"{target}_pred_ridge"] = pred_ridge.astype(np.float32)

            bundle = change_lgbm[target]
            X = df[bundle.feature_cols].to_numpy(dtype=np.float32)
            pred_lgbm = bundle.booster.predict(_impute(X, bundle.median)).astype(np.float32)
            out[f"{target}_pred_lgbm"] = pred_lgbm

            if args.include_uncertainty:
                members, med, cols = change_ensembles.get(target, ([], None, None))
                if members and med is not None and cols is not None:
                    X_e = df[cols].to_numpy(dtype=np.float32)
                    X_ei = _impute(X_e, med)
                    member_preds = np.stack([m.predict(X_ei).astype(np.float32) for m in members], axis=0)
                    std = member_preds.std(axis=0).astype(np.float32)
                else:
                    std = np.full(len(df), np.nan, dtype=np.float32)
                out[f"{target}_std_lgbm"] = std

        out["change_binary_prob_logreg"] = logreg.predict_proba(df[CHANGE_FEATURE_COLS])[:, 1].astype(np.float32)
        out["change_binary_pred_logreg"] = (out["change_binary_prob_logreg"] >= 0.5).astype(int)

        X_rf = df[rf_bundle["feature_cols"]].to_numpy(dtype=float)
        X_rf = np.where(np.isnan(X_rf), rf_bundle["median"], X_rf)
        out["change_binary_prob_rf"] = rf_bundle["model"].predict_proba(X_rf)[:, 1].astype(np.float32)
        out["change_binary_pred_rf"] = (out["change_binary_prob_rf"] >= 0.5).astype(int)

        out.to_parquet(os.path.join(args.output_dir, f"change_pred_{year_t1}_{year_t2}.parquet"), index=False)
        print(f"✅ Saved change predictions for {year_t1}->{year_t2}")

    print("✅ All predictions complete")


if __name__ == "__main__":
    main()