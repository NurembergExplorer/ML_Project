# src/eda_report.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def generate_eda_report(parquet_path: str, output_dir: str = "reports/eda") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(parquet_path)

    # Markdown summary
    md = out / "eda_summary.md"
    with open(md, "w", encoding="utf-8") as f:
        f.write("# EDA Summary\n\n")
        f.write(f"**Source:** `{parquet_path}`\n\n")
        f.write(f"**Shape:** {df.shape}\n\n")
        f.write("## Missing Values\n\n")
        f.write(df.isna().sum().to_string() + "\n\n")
        f.write("## Describe (numeric)\n\n")
        f.write(df.describe(numeric_only=True).to_string() + "\n")

    # Missingness plot
    miss = df.isna().mean().sort_values(ascending=False)
    plt.figure()
    miss.plot(kind="bar")
    plt.title("Missingness fraction by column")
    plt.ylabel("Fraction missing")
    plt.tight_layout()
    plt.savefig(out / "missingness.png", dpi=160)
    plt.close()

    # Histograms (numeric)
    num = df.select_dtypes(include="number")
    if not num.empty:
        ax = num.hist(figsize=(14, 10), bins=40)
        plt.suptitle("Feature distributions (numeric)")
        plt.tight_layout()
        plt.savefig(out / "feature_distributions.png", dpi=160)
        plt.close()

        # Correlation matrix
        corr = num.corr()
        plt.figure(figsize=(10, 8))
        plt.imshow(corr, cmap="coolwarm", aspect="auto")
        plt.colorbar()
        plt.title("Correlation matrix (numeric)")
        plt.tight_layout()
        plt.savefig(out / "correlation_matrix.png", dpi=160)
        plt.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Generate an EDA report from a parquet dataset.")
    p.add_argument("--parquet", required=True, type=str, help="Input parquet path")
    p.add_argument("--outdir", default="reports/eda", type=str, help="Output directory")
    args = p.parse_args()
    generate_eda_report(args.parquet, args.outdir)


if __name__ == "__main__":
    main()