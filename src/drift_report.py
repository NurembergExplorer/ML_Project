# src/drift_report.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


DEFAULT_ID_COLS = ["cell_id", "block_id", "year"]


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    PSI: Population Stability Index.
    Uses histograms with equal-width bins over combined min/max to be consistent.
    """
    if bins < 2:
        raise ValueError("bins must be >= 2")

    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if expected.size == 0 or actual.size == 0:
        return float("nan")

    lo = float(np.min([expected.min(), actual.min()]))
    hi = float(np.max([expected.max(), actual.max()]))
    if hi <= lo:
        return 0.0

    edges = np.linspace(lo, hi, bins + 1)
    e, _ = np.histogram(expected, bins=edges)
    a, _ = np.histogram(actual, bins=edges)

    e = e / max(1, expected.size)
    a = a / max(1, actual.size)

    e = np.where(e == 0, 1e-6, e)
    a = np.where(a == 0, 1e-6, a)

    return float(np.sum((e - a) * np.log(e / a)))


def cdf_gap(expected: np.ndarray, actual: np.ndarray, grid: int = 200) -> float:
    """
    Max absolute difference between empirical CDFs over a shared grid (KS-like).
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if expected.size == 0 or actual.size == 0:
        return float("nan")

    lo = float(np.min([expected.min(), actual.min()]))
    hi = float(np.max([expected.max(), actual.max()]))
    if hi <= lo:
        return 0.0

    xs = np.linspace(lo, hi, grid)

    e_sorted = np.sort(expected)
    a_sorted = np.sort(actual)

    e_cdf = np.searchsorted(e_sorted, xs, side="right") / e_sorted.size
    a_cdf = np.searchsorted(a_sorted, xs, side="right") / a_sorted.size

    return float(np.max(np.abs(e_cdf - a_cdf)))


def drift_between_years(
    parquet_a: str,
    parquet_b: str,
    *,
    id_cols: Sequence[str] | None = None,
    bins: int = 10,
) -> pd.DataFrame:
    df_a = pd.read_parquet(parquet_a)
    df_b = pd.read_parquet(parquet_b)

    id_cols = list(DEFAULT_ID_COLS if id_cols is None else id_cols)

    num_a = df_a.select_dtypes(include="number")
    num_b = df_b.select_dtypes(include="number")

    common = [c for c in num_a.columns if c in num_b.columns and c not in id_cols]
    rows = []

    for col in common:
        a = num_a[col].to_numpy()
        b = num_b[col].to_numpy()
        rows.append(
            {
                "feature": col,
                "psi": psi(a, b, bins=bins),
                "cdf_gap": cdf_gap(a, b),
                "mean_a": float(np.nanmean(a)),
                "mean_b": float(np.nanmean(b)),
                "std_a": float(np.nanstd(a)),
                "std_b": float(np.nanstd(b)),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["feature", "psi", "cdf_gap", "mean_a", "mean_b", "std_a", "std_b"])

    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute feature drift between two parquets (PSI + CDF gap).")
    ap.add_argument("--a", required=True, help="Parquet A (baseline)")
    ap.add_argument("--b", required=True, help="Parquet B (comparison)")
    ap.add_argument("--out", default="reports/drift_report.csv", help="Output CSV")
    ap.add_argument("--bins", type=int, default=10, help="Bins for PSI (default 10)")
    ap.add_argument(
        "--id-cols",
        nargs="*",
        default=DEFAULT_ID_COLS,
        help="Numeric identifier columns to exclude from drift metrics (default: cell_id block_id year)",
    )
    args = ap.parse_args()

    df = drift_between_years(args.a, args.b, bins=args.bins, id_cols=args.id_cols)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.head(40).to_string(index=False))
    print(f"\nWrote drift report -> {args.out} (rows={len(df)})")


if __name__ == "__main__":
    main()