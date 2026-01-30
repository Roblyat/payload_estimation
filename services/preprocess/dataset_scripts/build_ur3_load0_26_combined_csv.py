#!/usr/bin/env python3
"""
Combine UR3_Load0_DataSet*.csv (wide format) into ONE wide CSV and make trajectory IDs unique.

Output:
  /workspace/shared/data/raw/delan_UR3_Load0_combined_26.csv

Usage:
  python3 scripts/build_ur3_load0_26_combined_csv.py
"""

import os
import re
import glob
import pandas as pd


RAW_DIR = "/workspace/shared/data/raw"
OUT_CSV = os.path.join(RAW_DIR, "delan_UR3_Load0_combined_26.csv")

# Column that defines trajectory id in your wide dataset
ID_COL = "ID"


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

        # Ensure consistent column order across files
        # if base_columns is None:
        #     base_columns = list(df.columns)
        # else:
        #     if list(df.columns) != base_columns:
        #         raise ValueError(
        #             f"Column mismatch in {os.path.basename(path)}.\n"
        #             f"Expected: {base_columns}\n"
        #             f"Got:      {list(df.columns)}"
        #         )
            
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"{os.path.basename(path)} missing required columns: {missing}")

        # Keep only required columns in stable order (ignore extra columns like I, JV*, T*)
        df = df[REQUIRED].copy()

        # Offset ID so trajectories from different files don't collide
        df = df.copy()
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

    # Write combined CSV
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df_all.to_csv(OUT_CSV, index=False)

    print("\n=== Combined WIDE CSV written ===")
    print("Output:", OUT_CSV)
    print(f"Rows: {len(df_all):,}")
    print(f"Unique trajectories (unique ID): {df_all[ID_COL].nunique():,}")
    print("Done.")


if __name__ == "__main__":
    main()
