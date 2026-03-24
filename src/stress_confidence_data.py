from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import math
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

TARGETS = ["built", "veg", "water", "other"]
CHANGE_TARGETS = ["delta_built", "delta_veg", "delta_water", "delta_other"]

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


@dataclass
class TestResult:
    name: str
    passed: bool
    expected: str
    actual: str
    reason: str
    duration_ms: float


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _find_project_root(project_root: Optional[str]) -> Path:
    if project_root:
        root = Path(project_root).resolve()
    else:
        root = Path(__file__).resolve().parents[1]

    if not root.exists():
        raise FileNotFoundError(f"Project root does not exist: {root}")
    return root


def _prediction_dir(root: Path) -> Path:
    return root / "data" / "processed" / "predictions"


def _features_dir(root: Path) -> Path:
    return root / "data" / "processed" / "features"


def _population_dir(root: Path) -> Path:
    return root / "data" / "processed" / "population"


def _models_dir(root: Path) -> Path:
    return root / "models"


def _as_numeric(series: pd.Series, fill_value: Optional[float] = None) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    if fill_value is not None:
        out = out.fillna(fill_value)
    return out


def _pick_existing_columns(df: pd.DataFrame, candidates: Sequence[str]) -> List[str]:
    return [c for c in candidates if c in df.columns]


def _compose_confidence_label(score_pct: pd.Series) -> pd.Series:
    score = _as_numeric(score_pct, fill_value=0.0)
    labels = np.where(
        score >= 80.0,
        "High Confidence",
        np.where(score >= 60.0, "Medium Confidence", "Low Confidence"),
    )
    return pd.Series(labels, index=score.index, dtype="object")


def _confidence_color(label: pd.Series) -> pd.Series:
    mapping = {
        "High Confidence": "green",
        "Medium Confidence": "yellow",
        "Low Confidence": "red",
    }
    return label.map(mapping).fillna("yellow")


def _bounded_interval(
    center: np.ndarray,
    half_width: np.ndarray,
    lower: float,
    upper: float,
) -> Tuple[np.ndarray, np.ndarray]:
    lo = np.clip(center - half_width, lower, upper)
    hi = np.clip(center + half_width, lower, upper)
    return lo, hi


def _normalize_score_from_halfwidth(half_width: np.ndarray, max_reasonable_width: float) -> np.ndarray:
    width = np.asarray(half_width, dtype=float)
    max_reasonable_width = max(float(max_reasonable_width), 1e-8)
    score = 100.0 * (1.0 - np.clip(width / max_reasonable_width, 0.0, 1.0))
    return np.clip(score, 0.0, 100.0)


def _parse_year_from_population_file_name(path: Path) -> Optional[int]:
    stem = path.stem
    digits = "".join(ch if ch.isdigit() else " " for ch in stem).split()
    for token in digits:
        if len(token) == 4 and token.isdigit():
            year = int(token)
            if 1900 <= year <= 2100:
                return year
    return None


def _parse_year_from_population_col(columns: Sequence[str]) -> Tuple[str, int]:
    if "cell_id" not in columns:
        raise ValueError("Population file must contain 'cell_id'.")

    pop_cols = [c for c in columns if c != "cell_id"]
    if len(pop_cols) != 1:
        raise ValueError("Population file must contain exactly one non-cell_id population column.")

    pop_col = pop_cols[0]
    digits = "".join(ch for ch in pop_col if ch.isdigit())
    if len(digits) < 4:
        raise ValueError(f"Could not infer year from population column: {pop_col}")

    return pop_col, int(digits[:4])


def discover_population_files(root: Path) -> List[str]:
    pop_dir = _population_dir(root)
    if not pop_dir.exists():
        return []
    return [str(p) for p in sorted(pop_dir.glob("pop_*_by_cell_id.csv"))]


def load_population_files(files: Sequence[str]) -> Dict[int, pd.DataFrame]:
    """
    Expected folder:
      data/processed/population/

    Expected files:
      pop_2019_by_cell_id.csv
      pop_2020_by_cell_id.csv
      ...

    Expected schema:
      cell_id,pop_2019
      or
      cell_id,<single_population_column_with_year_in_name>
    """
    out: Dict[int, pd.DataFrame] = {}

    for fp in files:
        path = Path(fp)
        df = pd.read_csv(path)

        pop_col, year_from_col = _parse_year_from_population_col(df.columns.tolist())
        year_from_name = _parse_year_from_population_file_name(path)
        year = year_from_name if year_from_name is not None else year_from_col

        clean = df[["cell_id", pop_col]].copy()
        clean["cell_id"] = pd.to_numeric(clean["cell_id"], errors="coerce")
        clean[pop_col] = pd.to_numeric(clean[pop_col], errors="coerce")

        clean = clean.dropna(subset=["cell_id"]).copy()
        clean["cell_id"] = clean["cell_id"].astype(np.int64)
        clean[pop_col] = clean[pop_col].fillna(0.0)

        clean = clean.groupby("cell_id", as_index=False)[pop_col].sum()
        clean = clean.rename(columns={pop_col: "population_sum"})
        clean["population_year"] = int(year)

        out[int(year)] = clean

    return out


def normalize_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    required = ["cell_id", *FEATURE_COLS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Feature table missing required columns: {missing}")

    out = df[required].copy()
    out["cell_id"] = pd.to_numeric(out["cell_id"], errors="coerce")

    for col in FEATURE_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["cell_id"]).copy()
    out["cell_id"] = out["cell_id"].astype(np.int64)
    out = out.groupby("cell_id", as_index=False).mean(numeric_only=True)
    out["valid_frac"] = _as_numeric(out["valid_frac"], fill_value=0.0).clip(0.0, 1.0)
    return out


def _build_population_summary_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "population_sum" not in out.columns:
        out["population_sum"] = np.nan
    out["population_sum"] = _as_numeric(out["population_sum"], fill_value=0.0)

    out["population_density_proxy"] = out["population_sum"]
    out["population_bucket"] = pd.cut(
        out["population_sum"],
        bins=[-np.inf, 0, 50, 250, 1000, np.inf],
        labels=["zero", "very_low", "low", "medium", "high"],
    ).astype(str)

    return out


def compute_composition_confidence(df: pd.DataFrame, pop_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Builds confidence outputs for composition predictions.

    Expected prediction columns from repo:
      built_pred_lgbm, veg_pred_lgbm, water_pred_lgbm, other_pred_lgbm
      built_pred_ridge, veg_pred_ridge, ...
      optional built_std_lgbm, veg_std_lgbm, ...

    Confidence logic:
      half_width = 1.96 * std_lgbm + 0.50 * |ridge - lgbm|
      confidence_pct = inverse of half_width relative to a practical width

    Output columns include:
      *_confidence_pct
      *_confidence_half_width
      *_confidence_lower
      *_confidence_upper
      *_confidence_display
      *_confidence_label
      *_confidence_color
      dominant_class
      overall_confidence_pct
      overall_confidence_label
      overall_confidence_color
    """
    out = df.copy()

    if "cell_id" not in out.columns:
        raise ValueError("Prediction table must contain 'cell_id'.")

    out["cell_id"] = pd.to_numeric(out["cell_id"], errors="coerce")
    out = out.dropna(subset=["cell_id"]).copy()
    out["cell_id"] = out["cell_id"].astype(np.int64)

    if pop_df is not None:
        pop_copy = pop_df.copy()
        pop_copy["cell_id"] = pd.to_numeric(pop_copy["cell_id"], errors="coerce")
        pop_copy = pop_copy.dropna(subset=["cell_id"]).copy()
        pop_copy["cell_id"] = pop_copy["cell_id"].astype(np.int64)
        out = out.merge(pop_copy[["cell_id", "population_sum", "population_year"]], on="cell_id", how="left")

    out = _build_population_summary_columns(out)

    dominant_prob_cols: List[str] = []
    dominant_score_cols: List[str] = []

    for target in TARGETS:
        pred_col = f"{target}_pred_lgbm"
        alt_col = f"{target}_pred_ridge"
        std_col = f"{target}_std_lgbm"

        if pred_col not in out.columns:
            raise ValueError(f"Missing prediction column: {pred_col}")

        center = _as_numeric(out[pred_col], fill_value=0.0).clip(0.0, 1.0).to_numpy(dtype=float)

        if alt_col in out.columns:
            alt = _as_numeric(out[alt_col], fill_value=0.0).clip(0.0, 1.0).to_numpy(dtype=float)
        else:
            alt = center.copy()

        if std_col in out.columns:
            std = _as_numeric(out[std_col], fill_value=0.0).clip(lower=0.0).to_numpy(dtype=float)
        else:
            std = np.zeros(len(out), dtype=float)

        disagreement = np.abs(center - alt)
        valid_frac = _as_numeric(out.get("valid_frac", pd.Series(1.0, index=out.index)), fill_value=1.0).clip(0.0, 1.0).to_numpy(dtype=float)

        half_width = np.clip(1.96 * std + 0.50 * disagreement + 0.02 * (1.0 - valid_frac), 0.005, 0.50)
        lo, hi = _bounded_interval(center, half_width, 0.0, 1.0)

        score = _normalize_score_from_halfwidth(half_width, max_reasonable_width=0.20)

        out[f"{target}_confidence_pct"] = score
        out[f"{target}_confidence_half_width"] = half_width
        out[f"{target}_confidence_lower"] = lo
        out[f"{target}_confidence_upper"] = hi
        out[f"{target}_confidence_display"] = [
            f"{c * 100:.1f}% ± {h * 100:.1f}%"
            for c, h in zip(center, half_width)
        ]
        out[f"{target}_confidence_label"] = _compose_confidence_label(out[f"{target}_confidence_pct"])
        out[f"{target}_confidence_color"] = _confidence_color(out[f"{target}_confidence_label"])

        dominant_prob_cols.append(pred_col)
        dominant_score_cols.append(f"{target}_confidence_pct")

    pred_matrix = out[dominant_prob_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    conf_matrix = out[dominant_score_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    dominant_idx = pred_matrix.argmax(axis=1)

    out["dominant_class"] = [TARGETS[i] for i in dominant_idx]
    out["dominant_class_probability"] = pred_matrix[np.arange(len(out)), dominant_idx]
    out["overall_confidence_pct"] = conf_matrix[np.arange(len(out)), dominant_idx]
    out["overall_confidence_label"] = _compose_confidence_label(out["overall_confidence_pct"])
    out["overall_confidence_color"] = _confidence_color(out["overall_confidence_label"])

    return out


def compute_change_confidence(df: pd.DataFrame, pop_df_t1: Optional[pd.DataFrame] = None, pop_df_t2: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Builds confidence outputs for change prediction files.

    Handles:
      delta_*_pred_lgbm
      delta_*_pred_ridge
      optional delta_*_std_lgbm
      change_binary_prob_logreg
      change_binary_prob_rf

    Also attaches population for t1 and t2 when available.
    """
    out = df.copy()

    if "cell_id" not in out.columns:
        raise ValueError("Change prediction table must contain 'cell_id'.")

    out["cell_id"] = pd.to_numeric(out["cell_id"], errors="coerce")
    out = out.dropna(subset=["cell_id"]).copy()
    out["cell_id"] = out["cell_id"].astype(np.int64)

    if pop_df_t1 is not None:
        p1 = pop_df_t1.copy()
        p1["cell_id"] = pd.to_numeric(p1["cell_id"], errors="coerce")
        p1 = p1.dropna(subset=["cell_id"]).copy()
        p1["cell_id"] = p1["cell_id"].astype(np.int64)
        p1 = p1.rename(columns={"population_sum": "population_sum_t1", "population_year": "population_year_t1"})
        out = out.merge(p1[["cell_id", "population_sum_t1", "population_year_t1"]], on="cell_id", how="left")

    if pop_df_t2 is not None:
        p2 = pop_df_t2.copy()
        p2["cell_id"] = pd.to_numeric(p2["cell_id"], errors="coerce")
        p2 = p2.dropna(subset=["cell_id"]).copy()
        p2["cell_id"] = p2["cell_id"].astype(np.int64)
        p2 = p2.rename(columns={"population_sum": "population_sum_t2", "population_year": "population_year_t2"})
        out = out.merge(p2[["cell_id", "population_sum_t2", "population_year_t2"]], on="cell_id", how="left")

    if "population_sum_t1" not in out.columns:
        out["population_sum_t1"] = np.nan
    if "population_sum_t2" not in out.columns:
        out["population_sum_t2"] = np.nan

    out["population_sum_t1"] = _as_numeric(out["population_sum_t1"], fill_value=0.0)
    out["population_sum_t2"] = _as_numeric(out["population_sum_t2"], fill_value=0.0)
    out["population_delta"] = out["population_sum_t2"] - out["population_sum_t1"]

    empirical_scale_parts = []
    for target in CHANGE_TARGETS:
        col = f"{target}_pred_lgbm"
        if col in out.columns:
            empirical_scale_parts.append(_as_numeric(out[col], fill_value=0.0).abs())

    if empirical_scale_parts:
        concat_scale = pd.concat(empirical_scale_parts, axis=0)
        global_scale = max(float(concat_scale.quantile(0.90)), 0.02)
    else:
        global_scale = 0.02

    for target in CHANGE_TARGETS:
        pred_col = f"{target}_pred_lgbm"
        alt_col = f"{target}_pred_ridge"
        std_col = f"{target}_std_lgbm"

        if pred_col not in out.columns:
            continue

        center = _as_numeric(out[pred_col], fill_value=0.0).to_numpy(dtype=float)
        if alt_col in out.columns:
            alt = _as_numeric(out[alt_col], fill_value=0.0).to_numpy(dtype=float)
        else:
            alt = center.copy()

        if std_col in out.columns:
            std = _as_numeric(out[std_col], fill_value=0.0).clip(lower=0.0).to_numpy(dtype=float)
        else:
            std = np.zeros(len(out), dtype=float)

        valid_frac_proxy = np.ones(len(out), dtype=float)
        for candidate in ["valid_frac", "valid_frac_t1", "valid_frac_t2"]:
            if candidate in out.columns:
                valid_frac_proxy = np.minimum(
                    valid_frac_proxy,
                    _as_numeric(out[candidate], fill_value=1.0).clip(0.0, 1.0).to_numpy(dtype=float),
                )

        disagreement = np.abs(center - alt)
        half_width = np.clip(1.96 * std + 0.50 * disagreement + 0.02 * (1.0 - valid_frac_proxy), 0.001, global_scale)

        lo = center - half_width
        hi = center + half_width
        score = _normalize_score_from_halfwidth(half_width, max_reasonable_width=global_scale)

        out[f"{target}_confidence_pct"] = score
        out[f"{target}_confidence_half_width"] = half_width
        out[f"{target}_confidence_lower"] = lo
        out[f"{target}_confidence_upper"] = hi
        out[f"{target}_confidence_display"] = [
            f"{c * 100:.2f}pp ± {h * 100:.2f}pp"
            for c, h in zip(center, half_width)
        ]
        out[f"{target}_confidence_label"] = _compose_confidence_label(out[f"{target}_confidence_pct"])
        out[f"{target}_confidence_color"] = _confidence_color(out[f"{target}_confidence_label"])

    for prob_col in ["change_binary_prob_logreg", "change_binary_prob_rf"]:
        if prob_col in out.columns:
            prob = _as_numeric(out[prob_col], fill_value=0.5).clip(0.0, 1.0)

            entropy = -(
                prob * np.log(np.clip(prob, 1e-8, 1.0))
                + (1.0 - prob) * np.log(np.clip(1.0 - prob, 1e-8, 1.0))
            )
            entropy_norm = entropy / math.log(2.0)
            score = 100.0 * (1.0 - entropy_norm)

            prefix = prob_col.replace("_prob", "")
            half_width = np.clip((1.0 - score / 100.0) * 0.20, 0.01, 0.20)
            lo, hi = _bounded_interval(prob.to_numpy(dtype=float), half_width.to_numpy(dtype=float), 0.0, 1.0)

            out[f"{prefix}_confidence_pct"] = score
            out[f"{prefix}_confidence_half_width"] = half_width
            out[f"{prefix}_confidence_lower"] = lo
            out[f"{prefix}_confidence_upper"] = hi
            out[f"{prefix}_confidence_display"] = [
                f"{p * 100:.1f}% ± {h * 100:.1f}%"
                for p, h in zip(prob, half_width)
            ]
            out[f"{prefix}_confidence_label"] = _compose_confidence_label(out[f"{prefix}_confidence_pct"])
            out[f"{prefix}_confidence_color"] = _confidence_color(out[f"{prefix}_confidence_label"])

    if {"change_binary_logreg_confidence_pct", "change_binary_rf_confidence_pct"}.issubset(out.columns):
        out["overall_confidence_pct"] = (
            _as_numeric(out["change_binary_logreg_confidence_pct"], fill_value=0.0)
            + _as_numeric(out["change_binary_rf_confidence_pct"], fill_value=0.0)
        ) / 2.0
    else:
        conf_cols = [c for c in out.columns if c.endswith("_confidence_pct") and any(t in c for t in CHANGE_TARGETS)]
        if conf_cols:
            out["overall_confidence_pct"] = out[conf_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        else:
            out["overall_confidence_pct"] = 0.0

    out["overall_confidence_label"] = _compose_confidence_label(out["overall_confidence_pct"])
    out["overall_confidence_color"] = _confidence_color(out["overall_confidence_label"])

    return out


def fit_platt_scaler(y_true: Sequence[int], y_prob: Sequence[float]) -> LogisticRegression:
    y = np.asarray(y_true, dtype=int).reshape(-1)
    p = np.asarray(y_prob, dtype=float).reshape(-1)
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    logits = np.log(p / (1.0 - p)).reshape(-1, 1)

    model = LogisticRegression(solver="lbfgs")
    model.fit(logits, y)
    return model


def apply_platt_scaler(model: LogisticRegression, y_prob: Sequence[float]) -> np.ndarray:
    p = np.asarray(y_prob, dtype=float).reshape(-1)
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    logits = np.log(p / (1.0 - p)).reshape(-1, 1)
    return model.predict_proba(logits)[:, 1]


def fit_isotonic_scaler(y_true: Sequence[int], y_prob: Sequence[float]) -> IsotonicRegression:
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(
        np.asarray(y_prob, dtype=float).reshape(-1),
        np.asarray(y_true, dtype=int).reshape(-1),
    )
    return model


def save_enriched_predictions(root: Path, population_files: Sequence[str]) -> Dict[str, Path]:
    """
    Reads prediction parquet files and writes enriched confidence parquet files:
      pred_YYYY_confidence.parquet
      change_pred_YYYY_YYYY_confidence.parquet

    Also writes:
      models/confidence_summary_enhanced.csv
    """
    pred_dir = _prediction_dir(root)
    models_dir = _models_dir(root)

    _safe_mkdir(pred_dir)
    _safe_mkdir(models_dir)

    pop_by_year = load_population_files(population_files) if population_files else {}
    saved: Dict[str, Path] = {}
    summary_rows: List[Dict[str, object]] = []

    for pred_path in sorted(pred_dir.glob("pred_*.parquet")):
        if pred_path.stem.endswith("_confidence"):
            continue

        year_token = pred_path.stem.split("_")[-1]
        if not year_token.isdigit():
            continue
        year = int(year_token)

        df = pd.read_parquet(pred_path)
        enriched = compute_composition_confidence(df, pop_by_year.get(year))

        out_path = pred_dir / f"{pred_path.stem}_confidence.parquet"
        enriched.to_parquet(out_path, index=False)

        saved[f"composition_{year}"] = out_path

        row: Dict[str, object] = {
            "artifact": out_path.name,
            "rows": int(len(enriched)),
            "year": year,
            "population_attached": bool("population_sum" in enriched.columns),
            "overall_confidence_mean": float(_as_numeric(enriched["overall_confidence_pct"], fill_value=0.0).mean()),
            "overall_confidence_p90": float(_as_numeric(enriched["overall_confidence_pct"], fill_value=0.0).quantile(0.90)),
        }

        if "population_sum" in enriched.columns:
            row["population_total"] = float(_as_numeric(enriched["population_sum"], fill_value=0.0).sum())
            row["population_mean"] = float(_as_numeric(enriched["population_sum"], fill_value=0.0).mean())

        summary_rows.append(row)

    for change_path in sorted(pred_dir.glob("change_pred_*.parquet")):
        if change_path.stem.endswith("_confidence"):
            continue

        parts = change_path.stem.split("_")
        if len(parts) < 4:
            continue

        year_tokens = [p for p in parts if p.isdigit()]
        if len(year_tokens) < 2:
            continue

        year_t1, year_t2 = int(year_tokens[-2]), int(year_tokens[-1])

        df = pd.read_parquet(change_path)
        enriched = compute_change_confidence(
            df,
            pop_df_t1=pop_by_year.get(year_t1),
            pop_df_t2=pop_by_year.get(year_t2),
        )

        out_path = pred_dir / f"{change_path.stem}_confidence.parquet"
        enriched.to_parquet(out_path, index=False)

        saved[change_path.stem] = out_path

        row = {
            "artifact": out_path.name,
            "rows": int(len(enriched)),
            "year_t1": year_t1,
            "year_t2": year_t2,
            "overall_confidence_mean": float(_as_numeric(enriched["overall_confidence_pct"], fill_value=0.0).mean()),
        }

        if "population_delta" in enriched.columns:
            row["population_delta_mean"] = float(_as_numeric(enriched["population_delta"], fill_value=0.0).mean())

        for target in CHANGE_TARGETS:
            col = f"{target}_confidence_pct"
            if col in enriched.columns:
                row[f"{target}_mean"] = float(_as_numeric(enriched[col], fill_value=0.0).mean())

        summary_rows.append(row)

    summary_path = models_dir / "confidence_summary_enhanced.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    saved["confidence_summary"] = summary_path

    return saved


def _run_case(name: str, expected: str, fn: Callable[[], Tuple[bool, str, str]]) -> TestResult:
    t0 = _now_ms()
    try:
        passed, actual, reason = fn()
        return TestResult(
            name=name,
            passed=passed,
            expected=expected,
            actual=actual,
            reason=reason,
            duration_ms=_now_ms() - t0,
        )
    except Exception as exc:
        return TestResult(
            name=name,
            passed=False,
            expected=expected,
            actual=f"EXCEPTION: {type(exc).__name__}: {exc}",
            reason="Unhandled exception",
            duration_ms=_now_ms() - t0,
        )


def _case_expect_exception(fn: Callable[[], object]) -> Tuple[bool, str, str]:
    try:
        fn()
        return False, "No exception raised", "Expected a validation exception but function succeeded."
    except Exception as exc:
        return True, f"Raised {type(exc).__name__}", "Validation failed safely as expected."


def _case_wrong_dtype_coercion(df: pd.DataFrame) -> Tuple[bool, str, str]:
    sample = df.head(32).copy()
    sample["B2_med"] = sample["B2_med"].astype(str)
    sample["valid_frac"] = 1.7

    clean = normalize_feature_table(sample)
    passed = clean["B2_med"].dtype.kind in "fi" and clean["valid_frac"].between(0.0, 1.0).all()

    return (
        passed,
        f"dtype={clean['B2_med'].dtype}, valid_frac_max={clean['valid_frac'].max():.3f}",
        "Type coercion and clipping should keep data usable.",
    )


def _case_duplicates_aggregated(df: pd.DataFrame) -> Tuple[bool, str, str]:
    sample = pd.concat([df.head(20), df.head(20)], ignore_index=True)
    clean = normalize_feature_table(sample)
    passed = len(clean) == sample["cell_id"].nunique() and clean["cell_id"].is_unique

    return (
        passed,
        f"rows={len(clean)}, unique_cell_id={clean['cell_id'].is_unique}",
        "Duplicate cell_id rows should collapse to one row per cell.",
    )


def _case_null_cleaning(df: pd.DataFrame) -> Tuple[bool, str, str]:
    sample = df.head(50).copy()
    sample.loc[sample.index[:10], "ndvi_med"] = np.nan

    clean = normalize_feature_table(sample)
    passed = len(clean) == 50 and clean["cell_id"].notna().all()

    return (
        passed,
        f"rows={len(clean)}, null_ndvi={int(clean['ndvi_med'].isna().sum())}",
        "Null feature values are allowed; schema integrity must remain intact.",
    )


def _case_single_row(df: pd.DataFrame) -> Tuple[bool, str, str]:
    clean = normalize_feature_table(df.head(1))
    passed = len(clean) == 1
    return passed, f"rows={len(clean)}", "One valid row should remain valid."


def _case_full_table(df: pd.DataFrame) -> Tuple[bool, str, str]:
    clean = normalize_feature_table(df)
    passed = len(clean) == df["cell_id"].nunique()
    return passed, f"rows={len(clean)}", "Whole-table normalization should finish successfully."


def _case_regression_consistency(comp_df: pd.DataFrame, pop_df: Optional[pd.DataFrame]) -> Tuple[bool, str, str]:
    a = compute_composition_confidence(comp_df.copy(), pop_df=pop_df)
    b = compute_composition_confidence(comp_df.copy(), pop_df=pop_df)

    cols = [
        c
        for c in a.columns
        if c.endswith("confidence_pct")
        or c.startswith("overall_confidence")
        or c.endswith("confidence_half_width")
        or c == "dominant_class"
    ]
    identical = a[cols].equals(b[cols])

    return (
        identical,
        f"checked_columns={len(cols)}",
        "Deterministic computations must stay identical across repeated runs.",
    )


def _case_concurrent_benchmark(comp_df: pd.DataFrame, pop_df: Optional[pd.DataFrame], concurrency: int, iterations: int) -> Tuple[bool, str, str]:
    tracemalloc.start()
    start = time.perf_counter()

    def worker(_: int) -> float:
        t0 = time.perf_counter()
        _ = compute_composition_confidence(comp_df.copy(), pop_df=pop_df)
        return time.perf_counter() - t0

    with futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        durations = list(ex.map(worker, range(iterations)))

    elapsed = time.perf_counter() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    throughput = iterations / max(elapsed, 1e-9)
    passed = throughput > 0 and peak >= current

    actual = (
        f"throughput={throughput:.2f}/s, "
        f"mean_latency_ms={np.mean(durations) * 1000:.1f}, "
        f"peak_mem_mb={peak / 1024 / 1024:.2f}"
    )

    return passed, actual, "Concurrent execution should complete and emit benchmark metrics."


def run_data_handling_stress_tests(
    root: Path,
    sample_year: Optional[int] = None,
    concurrency: int = 8,
    iterations: int = 24,
) -> pd.DataFrame:
    """
    Stress tests the repo-compatible data handling and confidence path.

    Writes:
      models/stress_test_data_handling.csv
      models/stress_test_data_handling.json
    """
    pred_dir = _prediction_dir(root)
    feature_dir = _features_dir(root)
    models_dir = _models_dir(root)

    _safe_mkdir(models_dir)

    population_files = discover_population_files(root)
    pop_by_year = load_population_files(population_files) if population_files else {}

    if sample_year is None:
        feature_years = sorted(
            int(p.stem.split("_")[-1])
            for p in feature_dir.glob("features_*.parquet")
            if p.stem.split("_")[-1].isdigit()
        )
        if not feature_years:
            raise FileNotFoundError(f"No features_*.parquet files found in {feature_dir}")
        sample_year = feature_years[0]

    feature_path = feature_dir / f"features_{sample_year}.parquet"
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing sample feature file: {feature_path}")

    feature_df = pd.read_parquet(feature_path)
    normalized = normalize_feature_table(feature_df)

    results: List[TestResult] = []

    results.append(
        _run_case(
            name="empty_dataframe_rejected",
            expected="Function raises ValueError for empty or schema-invalid input.",
            fn=lambda: _case_expect_exception(lambda: normalize_feature_table(pd.DataFrame())),
        )
    )

    results.append(
        _run_case(
            name="missing_required_columns_rejected",
            expected="Function raises ValueError when feature columns are missing.",
            fn=lambda: _case_expect_exception(lambda: normalize_feature_table(normalized[["cell_id", "B2_med"]])),
        )
    )

    results.append(
        _run_case(
            name="wrong_dtypes_coerced",
            expected="Numeric strings are coerced and valid_frac is clipped to [0,1].",
            fn=lambda: _case_wrong_dtype_coercion(normalized),
        )
    )

    results.append(
        _run_case(
            name="duplicate_cell_ids_aggregated",
            expected="Duplicate cell_id rows are grouped without crashing.",
            fn=lambda: _case_duplicates_aggregated(normalized),
        )
    )

    results.append(
        _run_case(
            name="null_values_survive_with_cleaning",
            expected="Null numeric values remain processable after coercion.",
            fn=lambda: _case_null_cleaning(normalized),
        )
    )

    results.append(
        _run_case(
            name="boundary_minimum_single_row",
            expected="Single valid row remains valid after normalization.",
            fn=lambda: _case_single_row(normalized),
        )
    )

    results.append(
        _run_case(
            name="boundary_maximum_full_table",
            expected="Full feature table processes without error.",
            fn=lambda: _case_full_table(normalized),
        )
    )

    comp_candidates = [
        p for p in sorted(pred_dir.glob("pred_*.parquet"))
        if not p.stem.endswith("_confidence")
    ]

    if comp_candidates:
        comp_path = comp_candidates[0]
        year_token = comp_path.stem.split("_")[-1]
        pop_df = pop_by_year.get(int(year_token)) if year_token.isdigit() else None
        comp_df = pd.read_parquet(comp_path)

        results.append(
            _run_case(
                name="confidence_regression_consistency",
                expected="Repeated identical confidence computations match exactly.",
                fn=lambda: _case_regression_consistency(comp_df, pop_df=pop_df),
            )
        )

        results.append(
            _run_case(
                name="concurrent_load_benchmark",
                expected="Concurrent confidence computation completes with throughput > 0.",
                fn=lambda: _case_concurrent_benchmark(comp_df, pop_df=pop_df, concurrency=concurrency, iterations=iterations),
            )
        )

    results_df = pd.DataFrame([asdict(r) for r in results])
    results_df["status"] = np.where(results_df["passed"], "PASS", "FAIL")

    csv_path = models_dir / "stress_test_data_handling.csv"
    json_path = models_dir / "stress_test_data_handling.json"

    results_df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(results_df.to_dict(orient="records"), indent=2), encoding="utf-8")

    return results_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-file data handling, population merge, confidence generation, calibration helpers, and stress tests."
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Repo root. Defaults to the parent of src/.",
    )
    parser.add_argument(
        "--mode",
        choices=["confidence", "stress", "all"],
        default="all",
        help="Run confidence generation, stress tests, or both.",
    )
    parser.add_argument(
        "--population-files",
        nargs="*",
        default=[],
        help="Optional explicit population CSV paths. If omitted, files are auto-discovered from data/processed/population/.",
    )
    parser.add_argument(
        "--sample-year",
        type=int,
        default=None,
        help="Feature year to use for stress testing. Default: first available feature year.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Concurrent workers for the load benchmark.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=24,
        help="Number of repeated concurrent computations for the load benchmark.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = _find_project_root(args.project_root)
    artifacts: Dict[str, str] = {}

    population_files = args.population_files or discover_population_files(root)

    if args.mode in {"confidence", "all"}:
        saved = save_enriched_predictions(root, population_files)
        artifacts.update({k: str(v) for k, v in saved.items()})

        print("[confidence] wrote:")
        for key, value in saved.items():
            print(f"  - {key}: {value}")

        if population_files:
            print("[confidence] population files:")
            for fp in population_files:
                print(f"  - {fp}")
        else:
            print("[confidence] no population files found")

    if args.mode in {"stress", "all"}:
        results_df = run_data_handling_stress_tests(
            root=root,
            sample_year=args.sample_year,
            concurrency=args.concurrency,
            iterations=args.iterations,
        )

        artifacts["stress_csv"] = str(_models_dir(root) / "stress_test_data_handling.csv")
        artifacts["stress_json"] = str(_models_dir(root) / "stress_test_data_handling.json")

        print("[stress] summary:")
        print(results_df[["name", "status", "actual"]].to_string(index=False))

    print("\n[done] artifacts:")
    print(json.dumps(artifacts, indent=2))


if __name__ == "__main__":
    main()