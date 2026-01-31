#!/usr/bin/env python3
"""
Combine every `UR*_Load*` dataset folder under `<RAW_DIR>/library/` into its own timestamped
wide CSV by collecting rows into trajectories keyed by (source file, raw ID).

Important behavior (per dataset folder):
  - Raw IDs from different source files are NEVER matched/merged (e.g. each file can have ID=1).
  - Row order *within each trajectory* is preserved exactly as read (no within-trajectory sorting).
  - Trajectories are ordered by timestamp (t1) using each trajectory's FIRST t1 value (as-read).
  - Output trajectories are re-numbered sequentially in the ID column (1..N) in timestamp order.

Output (timestamped, written inside each dataset folder):
  <RAW_DIR>/library/<URx_Loady>/delan_<URx_Loady>_groupby_id_<YYYYmmdd_HHMMSS>.csv

Usage:
  python3 payload_estimation/services/preprocess/dataset_scripts/build_ur3_load0_26_combined_csv_groupby_id_timestamped.py
  # Optionally set PAYLOAD_ESTIMATION_RAW_DIR to override <RAW_DIR> discovery.
"""

import os
import re
import glob
from datetime import datetime
from pathlib import Path

import pandas as pd


def resolve_raw_dir() -> str:
    env = os.environ.get("PAYLOAD_ESTIMATION_RAW_DIR")
    if env:
        return env
    if os.path.isdir("/workspace/shared/data/raw"):
        return "/workspace/shared/data/raw"
    # repo-local fallback
    return str(Path(__file__).resolve().parents[3] / "shared" / "data" / "raw")


RAW_DIR = resolve_raw_dir()

# Where the per-dataset folders live (UR3_Load0, UR10_Load1, ...)
LIBRARY_DIR = Path(RAW_DIR) / "library"

# Column that defines trajectory id in your wide dataset
ID_COL = "ID"

# Column used for ordering trajectories
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


def combine_dataset_dir(dataset_dir: Path):
    dataset_name = dataset_dir.name
    pattern = dataset_dir / f"{dataset_name}_DataSet*.csv"
    files = sorted(glob.glob(str(pattern)), key=natural_key_dataset_num)
    files = [p for p in files if not re.search(r"DataSet22\.csv$", os.path.basename(p))]

    if not files:
        print(f"[SKIP] {dataset_name}: no files matched {pattern}")
        return

    print(f"\n=== {dataset_name} ===")
    print(f"RAW_DIR: {RAW_DIR}")
    print(f"Found {len(files)} files.")
    for f in files:
        print("  -", os.path.basename(f))

    # A trajectory is uniquely identified by (file_idx, raw_id) so IDs from different files
    # are never merged together.
    parts_by_traj: dict[tuple[int, object], list[pd.DataFrame]] = {}
    traj_first_seen: dict[tuple[int, object], int] = {}
    next_seen_index = 0

    total_rows = 0
    for file_idx, path in enumerate(files):
        df = pd.read_csv(path, skipinitialspace=True)

        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"{os.path.basename(path)} missing required columns: {missing}")

        df = df[REQUIRED].copy()
        total_rows += len(df)

        # Keep row order as-is, but collect trajectory chunks per-ID.
        for raw_id, traj_part in df.groupby(ID_COL, sort=False):
            traj_key = (file_idx, raw_id)
            if traj_key not in traj_first_seen:
                traj_first_seen[traj_key] = next_seen_index
                next_seen_index += 1
            parts_by_traj.setdefault(traj_key, []).append(traj_part)

        print(
            f"[{os.path.basename(path)}] rows={len(df):,} unique_IDs={df[ID_COL].nunique():,}"
        )

    if not parts_by_traj:
        raise ValueError(f"{dataset_name}: No trajectory data collected (no IDs found).")

    # Materialize trajectories in original row order.
    trajectories: dict[tuple[int, object], pd.DataFrame] = {}
    trajectory_sort_keys: dict[tuple[int, object], tuple[float, int]] = {}

    for traj_key, parts in parts_by_traj.items():
        traj_df = pd.concat(parts, ignore_index=True)
        trajectories[traj_key] = traj_df

        # Order trajectories by the first timestamp as-read (do not sort within a trajectory).
        t = pd.to_numeric(traj_df[TIME_COL], errors="coerce")
        if len(t) == 0 or pd.isna(t.iloc[0]):
            t0 = float("inf")
        else:
            t0 = float(t.iloc[0])
        trajectory_sort_keys[traj_key] = (t0, traj_first_seen[traj_key])

    ordered_traj_keys = sorted(parts_by_traj.keys(), key=lambda key: trajectory_sort_keys[key])
    ordered_frames = []
    for new_id, traj_key in enumerate(ordered_traj_keys, start=1):
        traj_df = trajectories[traj_key]
        traj_df = traj_df.copy()
        traj_df[ID_COL] = int(new_id)
        ordered_frames.append(traj_df)

    df_all = pd.concat(ordered_frames, ignore_index=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = dataset_dir / f"delan_{dataset_name}_groupby_id_{stamp}.csv"

    os.makedirs(dataset_dir, exist_ok=True)
    df_all.to_csv(out_csv, index=False)

    print("=== Combined WIDE CSV written (grouped by ID, timestamp-ordered trajectories) ===")
    print("Output:", out_csv)
    print(f"Input rows: {total_rows:,}")
    print(f"Output rows: {len(df_all):,}")
    print(f"Trajectories (unique ID): {df_all[ID_COL].nunique():,}")
    print("Done.")


def main():
    if not LIBRARY_DIR.exists():
        raise FileNotFoundError(f"Library directory not found: {LIBRARY_DIR}")

    dataset_dirs = [p for p in sorted(LIBRARY_DIR.iterdir()) if p.is_dir() and re.match(r"UR\d+_Load\d+", p.name)]

    if not dataset_dirs:
        raise FileNotFoundError(f"No dataset folders matching 'UR*_Load*' under {LIBRARY_DIR}")

    for dataset_dir in dataset_dirs:
        combine_dataset_dir(dataset_dir)


if __name__ == "__main__":
    main()
