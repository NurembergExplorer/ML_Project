from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    explained_variance_score,
    f1_score,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
)

EPS = 1e-8


def _coerce_1d_pair(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true_arr = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred_arr = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true_arr.size == 0 or y_pred_arr.size == 0:
        raise ValueError("y_true and y_pred must be non-empty.")
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, got {y_true_arr.shape} vs {y_pred_arr.shape}."
        )
    if not np.isfinite(y_true_arr).all() or not np.isfinite(y_pred_arr).all():
        raise ValueError("y_true and y_pred must contain only finite numeric values.")
    return y_true_arr, y_pred_arr


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_arr, y_pred_arr = _coerce_1d_pair(y_true, y_pred)
    denom = np.where(np.abs(y_true_arr) < EPS, EPS, np.abs(y_true_arr))
    return float(np.mean(np.abs((y_true_arr - y_pred_arr) / denom)) * 100.0)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_arr, y_pred_arr = _coerce_1d_pair(y_true, y_pred)
    denom = (np.abs(y_true_arr) + np.abs(y_pred_arr)) / 2.0
    denom = np.where(denom < EPS, EPS, denom)
    return float(np.mean(np.abs(y_true_arr - y_pred_arr) / denom) * 100.0)


def adjusted_r2(r2: float, n: int, p: int) -> float:
    if p < 0:
        raise ValueError("p must be non-negative.")
    if n <= p + 1:
        return float("nan")
    return float(1.0 - (1.0 - r2) * (n - 1) / (n - p - 1))


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_features: Optional[int] = None,
) -> Dict[str, float]:
    y_true_arr, y_pred_arr = _coerce_1d_pair(y_true, y_pred)

    mse = float(mean_squared_error(y_true_arr, y_pred_arr))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    med_ae = float(median_absolute_error(y_true_arr, y_pred_arr))
    max_err = float(max_error(y_true_arr, y_pred_arr))
    r2 = float(r2_score(y_true_arr, y_pred_arr))
    exp_var = float(explained_variance_score(y_true_arr, y_pred_arr))

    adj = float("nan")
    if n_features is not None:
        adj = adjusted_r2(r2, n=len(y_true_arr), p=int(n_features))

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MedianAE": med_ae,
        "MaxError": max_err,
        "R2": r2,
        "AdjustedR2": adj,
        "ExplainedVariance": exp_var,
        "MAPE": safe_mape(y_true_arr, y_pred_arr),
        "sMAPE": smape(y_true_arr, y_pred_arr),
    }


def sign_accuracy(y_true: np.ndarray, y_pred: np.ndarray, zero_band: float = 1e-6) -> float:
    y_true_arr, y_pred_arr = _coerce_1d_pair(y_true, y_pred)

    def _sign(arr: np.ndarray) -> np.ndarray:
        out = np.zeros_like(arr, dtype=int)
        out[arr > zero_band] = 1
        out[arr < -zero_band] = -1
        return out

    return float(np.mean(_sign(y_true_arr) == _sign(y_pred_arr)))


def threshold_binary(arr: np.ndarray, threshold: float) -> np.ndarray:
    x = np.asarray(arr, dtype=float).reshape(-1)
    return (np.abs(x) >= float(threshold)).astype(int)


def compute_change_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    threshold: float,
    n_features: Optional[int] = None,
) -> Dict[str, float]:
    base = compute_metrics(y_true, y_pred, n_features=n_features)
    y_true_arr, y_pred_arr = _coerce_1d_pair(y_true, y_pred)

    true_bin = threshold_binary(y_true_arr, threshold)
    pred_bin = threshold_binary(y_pred_arr, threshold)

    precision = float(precision_score(true_bin, pred_bin, zero_division=0))
    recall = float(recall_score(true_bin, pred_bin, zero_division=0))
    f1 = float(f1_score(true_bin, pred_bin, zero_division=0))

    fp = int(((pred_bin == 1) & (true_bin == 0)).sum())
    fn = int(((pred_bin == 0) & (true_bin == 1)).sum())
    tn = int(((pred_bin == 0) & (true_bin == 0)).sum())
    tp = int(((pred_bin == 1) & (true_bin == 1)).sum())

    false_change_rate = float(fp / max(fp + tn, 1))
    missed_change_rate = float(fn / max(fn + tp, 1))
    stability_score = float(np.mean((pred_bin == 0) & (true_bin == 0)))

    return {
        **base,
        "SignAccuracy": sign_accuracy(y_true_arr, y_pred_arr),
        "Threshold": float(threshold),
        "ThresholdPrecision": precision,
        "ThresholdRecall": recall,
        "ThresholdF1": f1,
        "FalseChangeRate": false_change_rate,
        "MissedChangeRate": missed_change_rate,
        "StabilityScore": stability_score,
    }


def compute_binary_change_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    yt = np.asarray(y_true, dtype=int).reshape(-1)
    yp = np.asarray(y_pred, dtype=int).reshape(-1)

    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    precision = float(precision_score(yt, yp, zero_division=0))
    recall = float(recall_score(yt, yp, zero_division=0))
    f1 = float(f1_score(yt, yp, zero_division=0))
    acc = float(accuracy_score(yt, yp))

    fp = int(((yp == 1) & (yt == 0)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum())
    tn = int(((yp == 0) & (yt == 0)).sum())
    tp = int(((yp == 1) & (yt == 1)).sum())

    false_change_rate = float(fp / max(fp + tn, 1))
    missed_change_rate = float(fn / max(fn + tp, 1))
    stability_score = float(np.mean((yp == 0) & (yt == 0)))

    out = {
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FalseChangeRate": false_change_rate,
        "MissedChangeRate": missed_change_rate,
        "StabilityScore": stability_score,
        "Positives": int(yt.sum()),
        "Negatives": int((yt == 0).sum()),
    }

    if y_prob is not None:
        prob = np.asarray(y_prob, dtype=float).reshape(-1)
        if prob.shape != yt.shape:
            raise ValueError("y_prob must have the same shape as y_true.")
        out["MeanPredictedProbability"] = float(np.nanmean(prob))

    return out


def to_snakecase_metrics(m: Dict[str, float]) -> Dict[str, float]:
    mapping = {
        "MSE": "mse",
        "RMSE": "rmse",
        "MAE": "mae",
        "MedianAE": "median_ae",
        "MaxError": "max_error",
        "R2": "r2",
        "AdjustedR2": "adj_r2",
        "ExplainedVariance": "explained_var",
        "MAPE": "mape",
        "sMAPE": "smape",
        "SignAccuracy": "sign_accuracy",
        "Threshold": "threshold",
        "ThresholdPrecision": "threshold_precision",
        "ThresholdRecall": "threshold_recall",
        "ThresholdF1": "threshold_f1",
        "FalseChangeRate": "false_change_rate",
        "MissedChangeRate": "missed_change_rate",
        "StabilityScore": "stability_score",
        "Accuracy": "accuracy",
        "Precision": "precision",
        "Recall": "recall",
        "F1": "f1",
        "Positives": "positives",
        "Negatives": "negatives",
        "MeanPredictedProbability": "mean_predicted_probability",
    }
    return {mapping.get(k, k): float(v) for k, v in m.items()}