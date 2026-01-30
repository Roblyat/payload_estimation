#!/usr/bin/env python3
"""
Combine UR3_Load0_DataSet*.csv (wide format) into ONE wide CSV, make trajectory IDs unique,
and sort the resulting rows by timestamp.

Output (timestamped):
  /workspace/shared/data/raw/delan_UR3_Load0_combined_26_<YYYYmmdd_HHMMSS>.csv

Usage:
  python3 payload_estimation/services/preprocess/tests/build_ur3_load0_26_combined_csv_timestamped.py
"""

import os
import re
import glob
from datetime import datetime

import pandas as pd


RAW_DIR = "/workspace/shared/data/raw"

# Column that defines trajectory id in your wide dataset
ID_COL = "ID"

# Column used for ordering rows across all combined files
TIME_COL = "t1"

# Only keep columns used by the wide-loader / pipeline
REQUIRED = (
    ["t1"]
    + [f"q{i}" for i in range(1, 7)]
    + [f"dq{i}" for i in range(1, 7)]
    + [f"Iq{i}" for i in range(1, 7)]
    + ["ID"]
)


def natural_key_dataset_num(path: str) -> int:
    """Sort DataSet1, DataSet2, ... DataSet26 numerically."""
    m = re.search(r"DataSet(\d+)\.csv$", os.path.basename(path))
    return int(m.group(1)) if m else 10**9


def main():
    pattern = os.path.join(RAW_DIR, "UR3_Load0_DataSet*.csv")
    files = sorted(glob.glob(pattern), key=natural_key_dataset_num)

    if not files:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")

    print(f"Found {len(files)} files.")
    for f in files:
        print("  -", os.path.basename(f))

    combined_parts = []
    id_offset = 0

    for path in files:
        df = pd.read_csv(path, skipinitialspace=True)

        if ID_COL not in df.columns:
            raise ValueError(f"{os.path.basename(path)} is missing required column '{ID_COL}'")

        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"{os.path.basename(path)} missing required columns: {missing}")

        # Keep only required columns in stable order (ignore extra columns like I, JV*, T*)
        df = df[REQUIRED].copy()

        # Offset ID so trajectories from different files don't collide
        df[ID_COL] = pd.to_numeric(df[ID_COL], errors="coerce").fillna(0).astype(int) + id_offset

        new_max = int(df[ID_COL].max())
        print(
            f"[{os.path.basename(path)}] rows={len(df):,} "
            f"ID_range=({df[ID_COL].min()}..{df[ID_COL].max()}) "
            f"(offset was {id_offset})"
        )
        id_offset = new_max + 1

        combined_parts.append(df)

    df_all = pd.concat(combined_parts, ignore_index=True)

    if TIME_COL not in df_all.columns:
        raise ValueError(f"Combined dataframe is missing required column '{TIME_COL}'")

    df_all[TIME_COL] = pd.to_numeric(df_all[TIME_COL], errors="coerce")
    df_all.sort_values([TIME_COL, ID_COL], kind="mergesort", na_position="last", inplace=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(RAW_DIR, f"delan_UR3_Load0_combined_26_{stamp}.csv")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_all.to_csv(out_csv, index=False)

    print("\n=== Combined WIDE CSV written (timestamped) ===")
    print("Output:", out_csv)
    print(f"Rows: {len(df_all):,}")
    print(f"Unique trajectories (unique ID): {df_all[ID_COL].nunique():,}")
    print("Done.")


if __name__ == "__main__":
    main()

