# src/evaluate.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

try:
    from src.metrics import compute_metrics
except ModuleNotFoundError:
    from metrics import compute_metrics


def _print_table(df: pd.DataFrame, title: str = "Evaluation") -> None:
    if df.empty:
        print(f"{title}: (no rows)")
        return
    print("\n" + title)
    print("-" * len(title))
    with pd.option_context("display.max_columns", 200, "display.width", 140):
        print(df.round(6).to_string(index=False))


def evaluate_file(
    pred_path: str,
    label_path: str,
    *,
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
    id_col: Optional[str] = None,
    n_features: Optional[int] = None,
) -> Dict[str, float]:
    pred_df = pd.read_csv(pred_path)
    label_df = pd.read_csv(label_path)

    if y_true_col not in label_df.columns:
        raise ValueError(f"Label file missing required column '{y_true_col}': {label_path}")

    if y_pred_col not in pred_df.columns:
        raise ValueError(f"Pred file missing required column '{y_pred_col}': {pred_path}")

    if id_col:
        if id_col not in pred_df.columns or id_col not in label_df.columns:
            raise ValueError(f"Both files must contain id column '{id_col}' when --id-col is used.")
        merged = label_df[[id_col, y_true_col]].merge(pred_df[[id_col, y_pred_col]], on=id_col, how="inner")
        if merged.empty:
            raise ValueError(f"No overlapping rows found on id column '{id_col}'.")
        y_true = merged[y_true_col].to_numpy()
        y_pred = merged[y_pred_col].to_numpy()
    else:
        if len(label_df) != len(pred_df):
            raise ValueError(
                "Prediction and label files have different row counts. "
                "Either align them first or pass --id-col for key-based matching."
            )
        y_true = label_df[y_true_col].to_numpy()
        y_pred = pred_df[y_pred_col].to_numpy()

    return compute_metrics(y_true, y_pred, n_features=n_features)


def evaluate_directory(
    pred_dir: str,
    label_dir: str,
    *,
    pattern: str = "*.csv",
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
    id_col: Optional[str] = None,
    output_csv: Optional[str] = None,
    n_features: Optional[int] = None,
) -> pd.DataFrame:
    pred_dir_p = Path(pred_dir)
    label_dir_p = Path(label_dir)

    if not pred_dir_p.exists():
        raise FileNotFoundError(f"pred_dir not found: {pred_dir}")
    if not label_dir_p.exists():
        raise FileNotFoundError(f"label_dir not found: {label_dir}")

    rows = []
    for pred_file in sorted(pred_dir_p.glob(pattern)):
        label_file = label_dir_p / pred_file.name
        if not label_file.exists():
            continue

        m = evaluate_file(
            str(pred_file),
            str(label_file),
            y_true_col=y_true_col,
            y_pred_col=y_pred_col,
            id_col=id_col,
            n_features=n_features,
        )
        rows.append({"file": pred_file.name, **m})

    df = pd.DataFrame(rows)
    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)

    _print_table(df, title="Directory evaluation summary")
    return df


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate predictions against labels with an expanded metric suite.")
    p.add_argument("--pred", type=str, help="Prediction CSV file path")
    p.add_argument("--label", type=str, help="Label CSV file path")
    p.add_argument("--pred-dir", type=str, help="Directory of prediction CSVs")
    p.add_argument("--label-dir", type=str, help="Directory of label CSVs")
    p.add_argument("--pattern", type=str, default="*.csv", help="Glob pattern inside pred-dir (default: *.csv)")
    p.add_argument("--y-true-col", type=str, default="y_true", help="Label column name (default: y_true)")
    p.add_argument("--y-pred-col", type=str, default="y_pred", help="Prediction column name (default: y_pred)")
    p.add_argument("--id-col", type=str, default=None, help="Optional key column for aligning predictions and labels")
    p.add_argument("--output-csv", type=str, default=None, help="Optional output CSV path")
    p.add_argument("--n-features", type=int, default=None, help="Optional #features for Adjusted R2")
    args = p.parse_args()

    if args.pred and args.label:
        m = evaluate_file(
            args.pred,
            args.label,
            y_true_col=args.y_true_col,
            y_pred_col=args.y_pred_col,
            id_col=args.id_col,
            n_features=args.n_features,
        )
        df = pd.DataFrame([m])
        _print_table(df, title="Single file evaluation")
        if args.output_csv:
            Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(args.output_csv, index=False)
        return

    if args.pred_dir and args.label_dir:
        evaluate_directory(
            args.pred_dir,
            args.label_dir,
            pattern=args.pattern,
            y_true_col=args.y_true_col,
            y_pred_col=args.y_pred_col,
            id_col=args.id_col,
            output_csv=args.output_csv,
            n_features=args.n_features,
        )
        return

    raise SystemExit("Provide either --pred + --label OR --pred-dir + --label-dir")


if __name__ == "__main__":
    main()