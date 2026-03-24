# src/overfit_report.py

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, learning_curve

try:
    from src.metrics import compute_metrics, to_snakecase_metrics
except ModuleNotFoundError:
    from metrics import compute_metrics, to_snakecase_metrics


def plot_learning_curve(model, X, y, out_png: str, scoring: str = "r2", cv_splits: int = 5) -> None:
    cv_splits = max(2, min(int(cv_splits), len(y)))
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)

    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        train_sizes=np.linspace(0.2, 1.0, 5),
        n_jobs=-1,
    )

    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    plt.figure()
    plt.plot(train_sizes, train_mean, label="Train")
    plt.plot(train_sizes, val_mean, label="Validation")
    plt.xlabel("Training samples")
    plt.ylabel(scoring)
    plt.title("Learning curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def residual_plots(y_true: np.ndarray, y_pred: np.ndarray, out_scatter: str, out_hist: str) -> None:
    residuals = y_true - y_pred

    plt.figure()
    plt.scatter(y_pred, residuals, s=8, alpha=0.6)
    plt.axhline(0, linewidth=1)
    plt.xlabel("Predicted")
    plt.ylabel("Residual (true - pred)")
    plt.title("Residual plot")
    plt.tight_layout()
    plt.savefig(out_scatter, dpi=160)
    plt.close()

    plt.figure()
    plt.hist(residuals, bins=50)
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.title("Residual distribution")
    plt.tight_layout()
    plt.savefig(out_hist, dpi=160)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate overfit diagnostics: learning curve + residual plots + metrics.")
    ap.add_argument("--train-parquet", required=True, help="Parquet with features + target")
    ap.add_argument("--target", required=True, help="Target column name")
    ap.add_argument("--outdir", default="reports/overfit", help="Output directory")
    ap.add_argument("--model", default="ridge", choices=["ridge"], help="Which model to use for diagnostics (default ridge)")
    ap.add_argument("--cv", type=int, default=5, help="Cross-validation splits for learning curve (default 5)")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.train_parquet)
    if args.target not in df.columns:
        raise ValueError(f"Target not found: {args.target}")

    feat = df.select_dtypes(include="number").drop(columns=[args.target], errors="ignore")
    mask = df[args.target].notna()
    feat = feat.loc[mask].copy()
    y = df.loc[mask, args.target].to_numpy()

    if feat.empty:
        raise ValueError("No numeric feature columns found after excluding the target.")
    if len(y) < 2:
        raise ValueError("Need at least 2 rows with non-null target values.")

    X = feat.to_numpy()

    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0, random_state=42)),
        ]
    )

    if len(y) >= 3:
        plot_learning_curve(model, X, y, str(out / "learning_curve.png"), cv_splits=args.cv)
    else:
        print("Skipping learning curve: need at least 3 rows for a meaningful CV plot.")

    model.fit(X, y)
    pred = model.predict(X)

    residual_plots(y, pred, str(out / "residual_plot.png"), str(out / "residual_hist.png"))

    m = to_snakecase_metrics(compute_metrics(y, pred, n_features=X.shape[1]))
    metrics_path = out / "train_fit_metrics.json"
    metrics_path.write_text(json.dumps(m, indent=2), encoding="utf-8")

    print("Wrote:")
    if len(y) >= 3:
        print(f" - {out / 'learning_curve.png'}")
    print(f" - {out / 'residual_plot.png'}")
    print(f" - {out / 'residual_hist.png'}")
    print(f" - {metrics_path}")


if __name__ == "__main__":
    main()