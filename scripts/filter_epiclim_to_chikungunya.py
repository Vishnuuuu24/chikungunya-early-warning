#!/usr/bin/env python3
"""Filter Epiclim raw CSV down to chikungunya-related rows.

This repo stores multiple diseases in `data/raw/Epiclim_Final_data.csv`, but the
pipeline focuses on chikungunya. This script keeps only rows whose `Disease`
contains the substring "chik" (case-insensitive), writes a timestamped backup
of the original file, then overwrites the original path.

Default behavior preserves all chikungunya-related categories, e.g.:
- Chikungunya
- Dengue And Chikungunya
- Suspected Chikungunya

Usage:
  python scripts/filter_epiclim_to_chikungunya.py \
    --input data/raw/Epiclim_Final_data.csv

Optional strict mode:
  --strict  (keeps only rows where Disease == "Chikungunya" after trimming)
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter Epiclim dataset to chikungunya rows")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/Epiclim_Final_data.csv",
        help="Path to Epiclim_Final_data.csv",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help='Keep only rows where Disease == "Chikungunya" (trimmed, case-insensitive).',
    )
    args = parser.parse_args()

    src = Path(args.input)
    if not src.exists():
        raise SystemExit(f"Input file not found: {src}")

    df = pd.read_csv(src, low_memory=False)
    if "Disease" not in df.columns:
        raise SystemExit("Column 'Disease' not found in input CSV")

    disease = df["Disease"].astype(str).str.strip()

    if args.strict:
        mask = disease.str.casefold().eq("chikungunya")
        mode = "strict == chikungunya"
    else:
        mask = disease.str.casefold().str.contains("chik")
        mode = "contains 'chik'"

    filtered = df.loc[mask].copy()

    if len(filtered) == 0:
        raise SystemExit(f"Filter produced 0 rows (mode: {mode}). Aborting.")

    # Backup original
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = src.with_suffix(f".csv.bak_{stamp}")
    shutil.copy2(src, backup)

    # Overwrite original
    filtered.to_csv(src, index=False)

    # Report
    print("Epiclim cleanup complete")
    print(f"  Input path:   {src}")
    print(f"  Backup path:  {backup}")
    print(f"  Filter mode:  {mode}")
    print(f"  Rows before:  {len(df)}")
    print(f"  Rows after:   {len(filtered)}")
    print("  Disease values after:")
    print(disease.loc[mask].value_counts().to_string())


if __name__ == "__main__":
    main()
