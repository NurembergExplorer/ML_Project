# src/validate_inputs.py

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

MANIFEST_COLUMNS = ["path", "size_bytes", "mtime_epoch", "sha256"]


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def validate_directory(data_dir: str, output_csv: str, with_hash: bool = False) -> pd.DataFrame:
    p = Path(data_dir)
    if not p.exists():
        raise FileNotFoundError(f"Directory not found: {data_dir}")
    if not p.is_dir():
        raise NotADirectoryError(f"Expected a directory: {data_dir}")

    rows = []
    for fp in sorted(p.rglob("*")):
        if not fp.is_file():
            continue
        stat = fp.stat()
        rows.append(
            {
                "path": str(fp.relative_to(p)),
                "size_bytes": int(stat.st_size),
                "mtime_epoch": float(stat.st_mtime),
                "sha256": _sha256(fp) if with_hash else "",
            }
        )

    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate raw inputs by writing a manifest of files/sizes/timestamps.")
    ap.add_argument("--dir", required=True, help="Directory to validate")
    ap.add_argument("--out", default="reports/input_manifest.csv", help="Output CSV")
    ap.add_argument("--hash", action="store_true", help="Compute sha256 for each file (slower)")
    args = ap.parse_args()

    df = validate_directory(args.dir, args.out, with_hash=args.hash)
    print(df.head(50).to_string(index=False))
    print(f"\nWrote manifest -> {args.out} (rows={len(df)})")


if __name__ == "__main__":
    main()