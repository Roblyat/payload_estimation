#!/usr/bin/env python3
import os
import re
import sys
import glob
from pathlib import Path

import pandas as pd

# Ensure preprocess package import works when running as a script
# (same pattern as your build_delan_dataset.py)
if "/workspace/preprocess/src" not in sys.path:
    sys.path.insert(0, "/workspace/preprocess/src")

from preprocess_delan.config import DelanPreprocessConfig
from preprocess_delan.io_csv import RawCSVLoader
from preprocess_delan.joints import JointSelector
from preprocess_delan.pivot import WidePivotBuilder
from preprocess_delan.dataset import TrajectoryDatasetBuilder, NPZDatasetWriter
from preprocess_delan.split import TrajectorySplitter


def natural_key_dataset_num(p: str) -> int:
    """
    Extract dataset number from filenames like UR3_Load0_DataSet16.csv
    so files sort in numeric order.
    """
    m = re.search(r"DataSet(\d+)\.csv$", os.path.basename(p))
    return int(m.group(1)) if m else 10**9


def main():
    raw_dir = "/workspace/shared/data/raw"
    out_dir = "/workspace/shared/data/preprocessed"
    out_path = os.path.join(out_dir, "delan_UR3_Load0_combined_26_dataset.npz")

    pattern = os.path.join(raw_dir, "UR3_Load0_DataSet*.csv")
    files = sorted(glob.glob(pattern), key=natural_key_dataset_num)

    if not files:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")

    print(f"Found {len(files)} files.")
    for f in files:
        print("  -", os.path.basename(f))

    # Config: wide input; derive acceleration from dq if desired
    cfg = DelanPreprocessConfig(
        input_format="wide",
        derive_qdd_from_dq=True,
        # splitter settings (change if you want)
        test_fraction=0.2,
        random_seed=0,
        # segmenter settings are irrelevant because trajectory_id is provided by ID and preserved
    )

    loader = RawCSVLoader()
    selector = JointSelector(cfg.dof_joints, cfg.col_joint)
    pivot = WidePivotBuilder(cfg)
    builder = TrajectoryDatasetBuilder(cfg, pivot)
    splitter = TrajectorySplitter(cfg.test_fraction, cfg.random_seed)
    writer = NPZDatasetWriter()

    dfs = []
    traj_offset = 0

    for path in files:
        # 1) Load wide -> long using your dataset1 adapter
        df = loader.load_dataset1(path, cfg)

        if "trajectory_id" not in df.columns:
            raise RuntimeError(
                f"{os.path.basename(path)}: loader did not produce trajectory_id. "
                f"Expected wide dataset with ID column."
            )

        # 2) Offset trajectory IDs to prevent collisions across files
        df = df.copy()
        df["trajectory_id"] = df["trajectory_id"].astype(int) + traj_offset

        max_id = int(df["trajectory_id"].max())
        traj_offset = max_id + 1

        dfs.append(df)

        print(
            f"[{os.path.basename(path)}] rows={len(df):,} "
            f"traj_id_range=({df['trajectory_id'].min()}..{df['trajectory_id'].max()})"
        )

    # 3) Combine all long frames
    df_all = pd.concat(dfs, ignore_index=True)

    # Optional safety: keep only configured joints (should already match)
    df_all = selector.filter(df_all)

    # 4) Build trajectories (pivot per trajectory_id)
    trajs = builder.build(df_all)

    # 5) Split trajectories into train/test
    train, test = splitter.split(trajs)

    # 6) Ensure output dir exists and write NPZ
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer.write(out_path, train, test)

    print("\n=== Combined dataset written ===")
    print("Output:", out_path)
    print(f"Trajectories: total={len(trajs)} train={len(train)} test={len(test)}")
    print("Done.")


if __name__ == "__main__":
    main()
