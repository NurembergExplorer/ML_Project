#!/usr/bin/env python3
"""
Train composition and change models for the assignment.

Composition:
- Ridge (interpretable)
- LightGBM (nonlinear)

Change regression:
- Ridge (interpretable)
- LightGBM (nonlinear)

Binary change:
- LogisticRegression (interpretable)
- RandomForestClassifier (nonlinear)

Also writes:
- spatial hold-out metrics
- temporal swap metrics
- change-specific metrics
- stress test summary
- feature importance
- ridge coefficients
- failure analysis
- uncertainty summary
- report notes
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from src.metrics import (
        compute_binary_change_metrics,
        compute_change_regression_metrics,
        compute_metrics,
        to_snakecase_metrics,
    )
except ModuleNotFoundError:
    from metrics import (
        compute_binary_change_metrics,
        compute_change_regression_metrics,
        compute_metrics,
        to_snakecase_metrics,
    )

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

CHANGE_FEATURE_COLS = (
    [f"{c}_t1" for c in FEATURE_COLS]
    + [f"{c}_t2" for c in FEATURE_COLS]
    + [f"{c}_d" for c in FEATURE_COLS]
)
CHANGE_TARGETS = ["delta_built", "delta_veg", "delta_water", "delta_other"]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_table(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def _spatial_splits(groups: np.ndarray, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    gkf = GroupKFold(n_splits=n_splits)
    dummy = np.zeros(len(groups))
    return list(gkf.split(dummy, groups=groups))


def _ridge_pipeline(alpha: float, feature_cols: List[str]) -> Pipeline:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    pre = ColumnTransformer([("num", numeric, feature_cols)], remainder="drop")
    model = Ridge(alpha=float(alpha), random_state=42)
    return Pipeline([("pre", pre), ("model", model)])


def _logreg_pipeline(c_value: float, feature_cols: List[str]) -> Pipeline:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    pre = ColumnTransformer([("num", numeric, feature_cols)], remainder="drop")
    model = LogisticRegression(C=float(c_value), random_state=42, max_iter=1000, class_weight="balanced")
    return Pipeline([("pre", pre), ("model", model)])


def _default_lgbm_params(seed: int = 42) -> Dict[str, Any]:
    return {
        "n_estimators": 800,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 40,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "random_state": seed,
        "n_jobs": -1,
        "objective": "regression",
        "verbosity": -1,
    }


def _default_rf_params(seed: int = 42) -> Dict[str, Any]:
    return {
        "n_estimators": 300,
        "max_depth": 12,
        "min_samples_leaf": 5,
        "random_state": seed,
        "n_jobs": -1,
        "class_weight": "balanced_subsample",
    }


def _fit_lgbm(X_train: pd.DataFrame, y_train: np.ndarray, params: Dict[str, Any]) -> Tuple[LGBMRegressor, np.ndarray]:
    X_arr = X_train.to_numpy(dtype=np.float32)
    med = np.nanmedian(X_arr, axis=0)
    X_imp = np.where(np.isnan(X_arr), med, X_arr)
    model = LGBMRegressor(**params)
    model.fit(X_imp, y_train)
    return model, med


def _predict_lgbm(model: LGBMRegressor, median: np.ndarray, X: pd.DataFrame) -> np.ndarray:
    X_arr = X.to_numpy(dtype=np.float32)
    X_imp = np.where(np.isnan(X_arr), median, X_arr)
    return model.predict(X_imp)


def _save_lgbm_bundle(model: LGBMRegressor, median: np.ndarray, feature_cols: List[str], path_txt: str, meta_path: str) -> None:
    _ensure_dir(os.path.dirname(path_txt))
    model.booster_.save_model(path_txt)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_cols": list(feature_cols),
                "imputer_median": median.tolist(),
            },
            f,
            indent=2,
        )


def _train_ensemble(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_cols: List[str],
    params: Dict[str, Any],
    ensemble_size: int,
    outdir: str,
) -> None:
    if ensemble_size <= 1:
        return

    _ensure_dir(outdir)
    X_arr = X[feature_cols].to_numpy(dtype=np.float32)
    median = np.nanmedian(X_arr, axis=0)
    X_imp = np.where(np.isnan(X_arr), median, X_arr)

    with open(os.path.join(outdir, "imputer_median.json"), "w", encoding="utf-8") as f:
        json.dump({"feature_cols": feature_cols, "median": median.tolist()}, f, indent=2)

    for i in range(ensemble_size):
        params_i = dict(params)
        params_i["random_state"] = 1000 + i
        model = LGBMRegressor(**params_i)
        model.fit(X_imp, y)
        model.booster_.save_model(os.path.join(outdir, f"member_{i:02d}.txt"))


def _composition_temporal_swap(df: pd.DataFrame, target: str, ridge_alpha: float, lgbm_params: Dict[str, Any]) -> Dict[str, float]:
    years = sorted(df["year"].unique().tolist())
    if len(years) != 2:
        return {"target": target}

    y0, y1 = years
    df0 = df[df["year"] == y0].copy()
    df1 = df[df["year"] == y1].copy()

    ridge_0 = _ridge_pipeline(ridge_alpha, FEATURE_COLS)
    ridge_0.fit(df0[FEATURE_COLS], df0[target].to_numpy())
    pred_1 = ridge_0.predict(df1[FEATURE_COLS])

    ridge_1 = _ridge_pipeline(ridge_alpha, FEATURE_COLS)
    ridge_1.fit(df1[FEATURE_COLS], df1[target].to_numpy())
    pred_0 = ridge_1.predict(df0[FEATURE_COLS])

    lgbm_0, med_0 = _fit_lgbm(df0[FEATURE_COLS], df0[target].to_numpy(), lgbm_params)
    lgbm_1, med_1 = _fit_lgbm(df1[FEATURE_COLS], df1[target].to_numpy(), lgbm_params)

    pred_1_lgbm = _predict_lgbm(lgbm_0, med_0, df1[FEATURE_COLS])
    pred_0_lgbm = _predict_lgbm(lgbm_1, med_1, df0[FEATURE_COLS])

    ridge_0to1 = to_snakecase_metrics(compute_metrics(df1[target].to_numpy(), pred_1, n_features=len(FEATURE_COLS)))
    ridge_1to0 = to_snakecase_metrics(compute_metrics(df0[target].to_numpy(), pred_0, n_features=len(FEATURE_COLS)))
    lgbm_0to1 = to_snakecase_metrics(compute_metrics(df1[target].to_numpy(), pred_1_lgbm, n_features=len(FEATURE_COLS)))
    lgbm_1to0 = to_snakecase_metrics(compute_metrics(df0[target].to_numpy(), pred_0_lgbm, n_features=len(FEATURE_COLS)))

    row = {"target": target, "year0": y0, "year1": y1}
    row.update({f"ridge_0to1_{k}": v for k, v in ridge_0to1.items()})
    row.update({f"ridge_1to0_{k}": v for k, v in ridge_1to0.items()})
    row.update({f"lgbm_0to1_{k}": v for k, v in lgbm_0to1.items()})
    row.update({f"lgbm_1to0_{k}": v for k, v in lgbm_1to0.items()})
    return row


def _dominant_class(df: pd.DataFrame) -> pd.Series:
    return df[TARGET_COLS].idxmax(axis=1).str.replace("_prop", "", regex=False)


def _save_report_notes(path: str) -> None:
    content = """# Report Notes

## Helpful explanation
- High NDVI and low SWIR-proxy water stress features are plausibly useful for vegetation prediction.
- Ridge coefficients are useful for directionality under the current feature representation.
- LightGBM gain importance is useful for ranking which input features the model relied on most.

## Misleading explanation
- Feature importance is **not causal**.
- A high importance score for NDVI does not prove vegetation causes the prediction in a causal sense.
- Mixed cells, label aggregation, seasonal effects, and WorldCover noise can all make an explanation look convincing but still be misleading.

## Data issues
1. Low valid_frac cells likely reflect cloud, masking, or incomplete observations.
2. WorldCover labels are noisy, especially in transition zones and mixed-use urban cells.
3. 100 m grid cells aggregate multiple land uses, so composition targets may smooth real edges.
4. Temporal comparability is imperfect because yearly composites may still differ in acquisition mix and seasonal conditions.
5. The 'other' class groups heterogeneous land covers and is harder to interpret.

## One issue not fixed
- The project still inherits label uncertainty from ESA WorldCover and mixed-pixel aggregation. This is acknowledged but not fully solved.

## Arguing Against ChatGPT – Case 1
- Reject the suggestion to use CNNs or segmentation models.
- The assignment explicitly asks for tabular ML from satellite-derived features, not end-to-end image models.
- Using Ridge/LightGBM is both compliant and more defensible.

## Arguing Against ChatGPT – Case 2
- Reject the claim that a highlighted change tile guarantees real-world development.
- Predicted change is model-based and may reflect label noise, seasonal differences, or feature instability.
- High uncertainty or low valid_frac should weaken confidence in the interpretation.
"""
    Path(path).write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train composition and change models.")
    p.add_argument("--train", required=True, type=str, help="Composition training table parquet.")
    p.add_argument("--change-train", required=True, type=str, help="Change training table parquet.")
    p.add_argument("--outdir", default="models", type=str)
    p.add_argument("--spatial-folds", default=5, type=int)
    p.add_argument("--ridge-alpha", default=1.0, type=float)
    p.add_argument("--logistic-c", default=1.0, type=float)
    p.add_argument("--ensemble-size", default=5, type=int)
    p.add_argument("--change-threshold", default=0.05, type=float)
    p.add_argument("--noise-std-fraction", default=0.05, type=float)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    outdir = args.outdir
    _ensure_dir(outdir)
    _ensure_dir(os.path.join(outdir, "baseline"))
    _ensure_dir(os.path.join(outdir, "lightgbm"))
    _ensure_dir(os.path.join(outdir, "change", "regression", "baseline"))
    _ensure_dir(os.path.join(outdir, "change", "regression", "lightgbm"))
    _ensure_dir(os.path.join(outdir, "change", "classification"))

    comp = _load_table(args.train).copy()
    change = _load_table(args.change_train).copy()

    comp_splits = _spatial_splits(comp["block_id"].to_numpy(), args.spatial_folds)
    change_splits = _spatial_splits(change["block_id"].to_numpy(), args.spatial_folds)

    lgbm_params = _default_lgbm_params()
    rf_params = _default_rf_params()

    spatial_rows: List[Dict[str, Any]] = []
    temporal_rows: List[Dict[str, Any]] = []
    change_rows: List[Dict[str, Any]] = []
    change_binary_rows: List[Dict[str, Any]] = []
    feature_importance_rows: List[Dict[str, Any]] = []
    ridge_coef_rows: List[Dict[str, Any]] = []
    worst_error_rows: List[Dict[str, Any]] = []
    error_by_valid_rows: List[Dict[str, Any]] = []
    error_by_dom_rows: List[Dict[str, Any]] = []
    stress_rows: List[Dict[str, Any]] = []
    uncertainty_rows: List[Dict[str, Any]] = []

    # --------------------------
    # Composition models
    # --------------------------
    comp_oof = comp[["cell_id", "block_id", "year", "valid_frac"] + TARGET_COLS].copy()
    comp_oof["dominant_class"] = _dominant_class(comp)

    for target in TARGET_COLS:
        ridge_pipe = _ridge_pipeline(args.ridge_alpha, FEATURE_COLS)
        ridge_oof = np.full(len(comp), np.nan, dtype=float)

        lgbm_oof = np.full(len(comp), np.nan, dtype=float)

        for fold, (tr, va) in enumerate(comp_splits):
            X_tr = comp.iloc[tr][FEATURE_COLS]
            y_tr = comp.iloc[tr][target].to_numpy()
            X_va = comp.iloc[va][FEATURE_COLS]
            y_va = comp.iloc[va][target].to_numpy()

            ridge_fold = clone(ridge_pipe)
            ridge_fold.fit(X_tr, y_tr)
            pred_ridge = ridge_fold.predict(X_va)
            ridge_oof[va] = pred_ridge

            row = {"model": "ridge", "target": target, "fold": fold}
            row.update(to_snakecase_metrics(compute_metrics(y_va, pred_ridge, n_features=len(FEATURE_COLS))))
            spatial_rows.append(row)

            lgbm_fold, med = _fit_lgbm(X_tr, y_tr, lgbm_params)
            pred_lgbm = _predict_lgbm(lgbm_fold, med, X_va)
            lgbm_oof[va] = pred_lgbm

            row = {"model": "lgbm", "target": target, "fold": fold}
            row.update(to_snakecase_metrics(compute_metrics(y_va, pred_lgbm, n_features=len(FEATURE_COLS))))
            spatial_rows.append(row)

            noise_scale = np.nanstd(X_tr.to_numpy(dtype=float), axis=0) * float(args.noise_std_fraction)
            X_va_noisy = X_va.copy()
            noise = np.random.default_rng(42 + fold).normal(0.0, noise_scale, size=X_va_noisy.shape)
            X_va_noisy.loc[:, FEATURE_COLS] = X_va_noisy[FEATURE_COLS].to_numpy(dtype=float) + noise
            pred_lgbm_noisy = _predict_lgbm(lgbm_fold, med, X_va_noisy)
            base_rmse = float(np.sqrt(np.mean((y_va - pred_lgbm) ** 2)))
            stress_rmse = float(np.sqrt(np.mean((y_va - pred_lgbm_noisy) ** 2)))
            stress_rows.append(
                {
                    "task": "composition",
                    "target": target,
                    "model": "lgbm",
                    "fold": fold,
                    "stress_test": "gaussian_noise",
                    "baseline_rmse": base_rmse,
                    "stress_rmse": stress_rmse,
                    "rmse_drop": stress_rmse - base_rmse,
                }
            )

        comp_oof[f"{target}_ridge_oof"] = ridge_oof
        comp_oof[f"{target}_lgbm_oof"] = lgbm_oof
        comp_oof[f"{target}_lgbm_abs_error"] = np.abs(comp[target] - lgbm_oof)

        ridge_pipe.fit(comp[FEATURE_COLS], comp[target].to_numpy())
        ridge_path = os.path.join(outdir, "baseline", f"ridge_{target}.pkl")
        joblib.dump(ridge_pipe, ridge_path)

        coef = ridge_pipe.named_steps["model"].coef_.reshape(-1)
        for feat, value in zip(FEATURE_COLS, coef):
            ridge_coef_rows.append(
                {"task": "composition", "target": target, "model": "ridge", "feature": feat, "coefficient": float(value)}
            )

        lgbm_final, med_final = _fit_lgbm(comp[FEATURE_COLS], comp[target].to_numpy(), lgbm_params)
        lgbm_txt = os.path.join(outdir, "lightgbm", f"lgbm_{target}.txt")
        lgbm_meta = os.path.join(outdir, "lightgbm", f"lgbm_{target}_features.json")
        _save_lgbm_bundle(lgbm_final, med_final, FEATURE_COLS, lgbm_txt, lgbm_meta)

        for feat, value in zip(FEATURE_COLS, lgbm_final.feature_importances_):
            feature_importance_rows.append(
                {"task": "composition", "target": target, "model": "lgbm", "feature": feat, "importance": float(value)}
            )

        _train_ensemble(
            comp[FEATURE_COLS],
            comp[target].to_numpy(),
            FEATURE_COLS,
            lgbm_params,
            args.ensemble_size,
            os.path.join(outdir, "lightgbm", f"ensemble_{target}"),
        )

        temporal_rows.append(_composition_temporal_swap(comp, target, args.ridge_alpha, lgbm_params))

    # --------------------------
    # Change regression
    # --------------------------
    change_oof = change[
        ["cell_id", "block_id", "pair_id", "year_t1", "year_t2", "valid_frac_t1", "valid_frac_t2"] + CHANGE_TARGETS
    ].copy()
    change_oof["valid_frac_min"] = change[["valid_frac_t1", "valid_frac_t2"]].min(axis=1)

    for target in CHANGE_TARGETS:
        ridge_pipe = _ridge_pipeline(args.ridge_alpha, CHANGE_FEATURE_COLS)
        ridge_oof = np.full(len(change), np.nan, dtype=float)
        lgbm_oof = np.full(len(change), np.nan, dtype=float)

        for fold, (tr, va) in enumerate(change_splits):
            X_tr = change.iloc[tr][CHANGE_FEATURE_COLS]
            y_tr = change.iloc[tr][target].to_numpy()
            X_va = change.iloc[va][CHANGE_FEATURE_COLS]
            y_va = change.iloc[va][target].to_numpy()

            ridge_fold = clone(ridge_pipe)
            ridge_fold.fit(X_tr, y_tr)
            pred_ridge = ridge_fold.predict(X_va)
            ridge_oof[va] = pred_ridge

            row = {"model": "ridge", "target": target, "fold": fold}
            row.update(
                to_snakecase_metrics(
                    compute_change_regression_metrics(
                        y_va,
                        pred_ridge,
                        threshold=args.change_threshold,
                        n_features=len(CHANGE_FEATURE_COLS),
                    )
                )
            )
            change_rows.append(row)

            lgbm_fold, med = _fit_lgbm(X_tr, y_tr, lgbm_params)
            pred_lgbm = _predict_lgbm(lgbm_fold, med, X_va)
            lgbm_oof[va] = pred_lgbm

            row = {"model": "lgbm", "target": target, "fold": fold}
            row.update(
                to_snakecase_metrics(
                    compute_change_regression_metrics(
                        y_va,
                        pred_lgbm,
                        threshold=args.change_threshold,
                        n_features=len(CHANGE_FEATURE_COLS),
                    )
                )
            )
            change_rows.append(row)

            noise_scale = np.nanstd(X_tr.to_numpy(dtype=float), axis=0) * float(args.noise_std_fraction)
            X_va_noisy = X_va.copy()
            noise = np.random.default_rng(142 + fold).normal(0.0, noise_scale, size=X_va_noisy.shape)
            X_va_noisy.loc[:, CHANGE_FEATURE_COLS] = X_va_noisy[CHANGE_FEATURE_COLS].to_numpy(dtype=float) + noise
            pred_lgbm_noisy = _predict_lgbm(lgbm_fold, med, X_va_noisy)
            base_rmse = float(np.sqrt(np.mean((y_va - pred_lgbm) ** 2)))
            stress_rmse = float(np.sqrt(np.mean((y_va - pred_lgbm_noisy) ** 2)))
            stress_rows.append(
                {
                    "task": "change_regression",
                    "target": target,
                    "model": "lgbm",
                    "fold": fold,
                    "stress_test": "gaussian_noise",
                    "baseline_rmse": base_rmse,
                    "stress_rmse": stress_rmse,
                    "rmse_drop": stress_rmse - base_rmse,
                }
            )

        change_oof[f"{target}_ridge_oof"] = ridge_oof
        change_oof[f"{target}_lgbm_oof"] = lgbm_oof
        change_oof[f"{target}_lgbm_abs_error"] = np.abs(change[target] - lgbm_oof)

        ridge_pipe.fit(change[CHANGE_FEATURE_COLS], change[target].to_numpy())
        ridge_path = os.path.join(outdir, "change", "regression", "baseline", f"ridge_{target}.pkl")
        joblib.dump(ridge_pipe, ridge_path)

        coef = ridge_pipe.named_steps["model"].coef_.reshape(-1)
        for feat, value in zip(CHANGE_FEATURE_COLS, coef):
            ridge_coef_rows.append(
                {"task": "change_regression", "target": target, "model": "ridge", "feature": feat, "coefficient": float(value)}
            )

        lgbm_final, med_final = _fit_lgbm(change[CHANGE_FEATURE_COLS], change[target].to_numpy(), lgbm_params)
        lgbm_txt = os.path.join(outdir, "change", "regression", "lightgbm", f"lgbm_{target}.txt")
        lgbm_meta = os.path.join(outdir, "change", "regression", "lightgbm", f"lgbm_{target}_features.json")
        _save_lgbm_bundle(lgbm_final, med_final, CHANGE_FEATURE_COLS, lgbm_txt, lgbm_meta)

        for feat, value in zip(CHANGE_FEATURE_COLS, lgbm_final.feature_importances_):
            feature_importance_rows.append(
                {"task": "change_regression", "target": target, "model": "lgbm", "feature": feat, "importance": float(value)}
            )

        _train_ensemble(
            change[CHANGE_FEATURE_COLS],
            change[target].to_numpy(),
            CHANGE_FEATURE_COLS,
            lgbm_params,
            args.ensemble_size,
            os.path.join(outdir, "change", "regression", "lightgbm", f"ensemble_{target}"),
        )

    # --------------------------
    # Binary change classification
    # --------------------------
    logreg_pipe = _logreg_pipeline(args.logistic_c, CHANGE_FEATURE_COLS)
    rf_model = RandomForestClassifier(**rf_params)

    logreg_oof = np.full(len(change), np.nan, dtype=float)
    logreg_prob_oof = np.full(len(change), np.nan, dtype=float)
    rf_oof = np.full(len(change), np.nan, dtype=float)
    rf_prob_oof = np.full(len(change), np.nan, dtype=float)

    for fold, (tr, va) in enumerate(change_splits):
        X_tr = change.iloc[tr][CHANGE_FEATURE_COLS]
        y_tr = change.iloc[tr]["change_binary"].to_numpy(dtype=int)
        X_va = change.iloc[va][CHANGE_FEATURE_COLS]
        y_va = change.iloc[va]["change_binary"].to_numpy(dtype=int)

        logreg_fold = clone(logreg_pipe)
        logreg_fold.fit(X_tr, y_tr)
        pred = logreg_fold.predict(X_va)
        prob = logreg_fold.predict_proba(X_va)[:, 1]
        logreg_oof[va] = pred
        logreg_prob_oof[va] = prob

        row = {"model": "logreg", "target": "change_binary", "fold": fold}
        row.update(to_snakecase_metrics(compute_binary_change_metrics(y_va, pred, y_prob=prob)))
        change_binary_rows.append(row)

        med = np.nanmedian(X_tr.to_numpy(dtype=float), axis=0)
        X_tr_imp = np.where(np.isnan(X_tr.to_numpy(dtype=float)), med, X_tr.to_numpy(dtype=float))
        X_va_imp = np.where(np.isnan(X_va.to_numpy(dtype=float)), med, X_va.to_numpy(dtype=float))
        rf_fold = clone(rf_model)
        rf_fold.fit(X_tr_imp, y_tr)
        pred_rf = rf_fold.predict(X_va_imp)
        prob_rf = rf_fold.predict_proba(X_va_imp)[:, 1]
        rf_oof[va] = pred_rf
        rf_prob_oof[va] = prob_rf

        row = {"model": "random_forest", "target": "change_binary", "fold": fold}
        row.update(to_snakecase_metrics(compute_binary_change_metrics(y_va, pred_rf, y_prob=prob_rf)))
        change_binary_rows.append(row)

    logreg_pipe.fit(change[CHANGE_FEATURE_COLS], change["change_binary"].to_numpy(dtype=int))
    joblib.dump(logreg_pipe, os.path.join(outdir, "change", "classification", "logreg_change_binary.pkl"))

    med = np.nanmedian(change[CHANGE_FEATURE_COLS].to_numpy(dtype=float), axis=0)
    rf_final = clone(rf_model)
    X_imp = np.where(np.isnan(change[CHANGE_FEATURE_COLS].to_numpy(dtype=float)), med, change[CHANGE_FEATURE_COLS].to_numpy(dtype=float))
    rf_final.fit(X_imp, change["change_binary"].to_numpy(dtype=int))
    joblib.dump({"model": rf_final, "median": med, "feature_cols": CHANGE_FEATURE_COLS}, os.path.join(outdir, "change", "classification", "rf_change_binary.pkl"))

    for feat, value in zip(CHANGE_FEATURE_COLS, rf_final.feature_importances_):
        feature_importance_rows.append(
            {"task": "change_binary", "target": "change_binary", "model": "random_forest", "feature": feat, "importance": float(value)}
        )

    logreg_coef = logreg_pipe.named_steps["model"].coef_.reshape(-1)
    for feat, value in zip(CHANGE_FEATURE_COLS, logreg_coef):
        ridge_coef_rows.append(
            {"task": "change_binary", "target": "change_binary", "model": "logreg", "feature": feat, "coefficient": float(value)}
        )

    change_oof["change_binary_true"] = change["change_binary"].to_numpy(dtype=int)
    change_oof["change_binary_logreg_pred"] = logreg_oof
    change_oof["change_binary_logreg_prob"] = logreg_prob_oof
    change_oof["change_binary_rf_pred"] = rf_oof
    change_oof["change_binary_rf_prob"] = rf_prob_oof

    # --------------------------
    # Failure analysis
    # --------------------------
    for target in TARGET_COLS:
        df_target = comp_oof[["cell_id", "block_id", "year", "valid_frac", "dominant_class", target, f"{target}_lgbm_oof", f"{target}_lgbm_abs_error"]].copy()
        df_target["task"] = "composition"
        df_target["target"] = target
        df_target = df_target.rename(columns={target: "y_true", f"{target}_lgbm_oof": "y_pred", f"{target}_lgbm_abs_error": "abs_error"})
        worst_error_rows.extend(df_target.sort_values("abs_error", ascending=False).head(25).to_dict("records"))

        bins = pd.cut(df_target["valid_frac"], bins=[-0.001, 0.25, 0.50, 0.75, 0.90, 1.0])
        error_by_valid = (
            df_target.groupby(bins, observed=False)["abs_error"]
            .agg(["count", "mean", "median"])
            .reset_index()
        )
        for _, row in error_by_valid.iterrows():
            error_by_valid_rows.append(
                {
                    "task": "composition",
                    "target": target,
                    "bin": str(row["valid_frac"]),
                    "count": int(row["count"]),
                    "mean_abs_error": float(row["mean"]),
                    "median_abs_error": float(row["median"]),
                }
            )

        error_by_dom = df_target.groupby("dominant_class")["abs_error"].agg(["count", "mean", "median"]).reset_index()
        for _, row in error_by_dom.iterrows():
            error_by_dom_rows.append(
                {
                    "task": "composition",
                    "target": target,
                    "group": row["dominant_class"],
                    "count": int(row["count"]),
                    "mean_abs_error": float(row["mean"]),
                    "median_abs_error": float(row["median"]),
                }
            )

    for target in CHANGE_TARGETS:
        df_target = change_oof[["cell_id", "block_id", "pair_id", "valid_frac_min", target, f"{target}_lgbm_oof", f"{target}_lgbm_abs_error"]].copy()
        df_target["task"] = "change_regression"
        df_target["target"] = target
        df_target = df_target.rename(columns={target: "y_true", f"{target}_lgbm_oof": "y_pred", f"{target}_lgbm_abs_error": "abs_error"})
        worst_error_rows.extend(df_target.sort_values("abs_error", ascending=False).head(25).to_dict("records"))

        bins = pd.cut(df_target["valid_frac_min"], bins=[-0.001, 0.25, 0.50, 0.75, 0.90, 1.0])
        error_by_valid = (
            df_target.groupby(bins, observed=False)["abs_error"]
            .agg(["count", "mean", "median"])
            .reset_index()
        )
        for _, row in error_by_valid.iterrows():
            error_by_valid_rows.append(
                {
                    "task": "change_regression",
                    "target": target,
                    "bin": str(row["valid_frac_min"]),
                    "count": int(row["count"]),
                    "mean_abs_error": float(row["mean"]),
                    "median_abs_error": float(row["median"]),
                }
            )

    for target in TARGET_COLS:
        disagreement = np.abs(comp_oof[f"{target}_ridge_oof"] - comp_oof[f"{target}_lgbm_oof"])
        uncertainty_rows.append(
            {
                "task": "composition",
                "target": target,
                "proxy": "ridge_lgbm_disagreement_oof",
                "mean_uncertainty": float(np.nanmean(disagreement)),
                "p90_uncertainty": float(np.nanpercentile(disagreement, 90)),
            }
        )

    for target in CHANGE_TARGETS:
        disagreement = np.abs(change_oof[f"{target}_ridge_oof"] - change_oof[f"{target}_lgbm_oof"])
        uncertainty_rows.append(
            {
                "task": "change_regression",
                "target": target,
                "proxy": "ridge_lgbm_disagreement_oof",
                "mean_uncertainty": float(np.nanmean(disagreement)),
                "p90_uncertainty": float(np.nanpercentile(disagreement, 90)),
            }
        )

    spatial_df = pd.DataFrame(spatial_rows)
    spatial_df.to_csv(os.path.join(outdir, "metrics_spatial_cv.csv"), index=False)

    spatial_summary = spatial_df.groupby(["model", "target"]).agg(
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        r2_mean=("r2", "mean"),
        r2_std=("r2", "std"),
    ).reset_index()
    spatial_summary.to_csv(os.path.join(outdir, "metrics_spatial_cv_summary.csv"), index=False)

    pd.DataFrame(temporal_rows).to_csv(os.path.join(outdir, "metrics_temporal_swap.csv"), index=False)

    change_df = pd.DataFrame(change_rows)
    change_df.to_csv(os.path.join(outdir, "change_metrics_spatial_cv.csv"), index=False)

    change_summary = change_df.groupby(["model", "target"]).agg(
        mae_mean=("mae", "mean"),
        rmse_mean=("rmse", "mean"),
        r2_mean=("r2", "mean"),
        sign_accuracy_mean=("sign_accuracy", "mean"),
        threshold_f1_mean=("threshold_f1", "mean"),
    ).reset_index()
    change_summary.to_csv(os.path.join(outdir, "change_metrics_spatial_cv_summary.csv"), index=False)

    pd.DataFrame(change_binary_rows).to_csv(os.path.join(outdir, "change_binary_metrics_spatial_cv.csv"), index=False)
    pd.DataFrame(stress_rows).to_csv(os.path.join(outdir, "stress_test_summary.csv"), index=False)
    pd.DataFrame(feature_importance_rows).to_csv(os.path.join(outdir, "feature_importance.csv"), index=False)
    pd.DataFrame(ridge_coef_rows).to_csv(os.path.join(outdir, "ridge_coefficients.csv"), index=False)
    pd.DataFrame(worst_error_rows).to_csv(os.path.join(outdir, "worst_errors.csv"), index=False)
    pd.DataFrame(error_by_valid_rows).to_csv(os.path.join(outdir, "error_by_valid_frac.csv"), index=False)
    pd.DataFrame(error_by_dom_rows).to_csv(os.path.join(outdir, "error_by_dominant_class.csv"), index=False)
    pd.DataFrame(uncertainty_rows).to_csv(os.path.join(outdir, "uncertainty_summary.csv"), index=False)

    _save_report_notes(os.path.join(outdir, "report_notes.md"))

    print("✅ Training complete")
    print(f"Saved outputs under: {outdir}")


if __name__ == "__main__":
    main()