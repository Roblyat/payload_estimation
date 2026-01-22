"""
Build a DeLaN-ready trajectory dataset (.npz) from long-format robot logs.

Expected CSV columns:
Time, Joint Name, Position, Velocity, Acceleration, Effort
"""
import argparse
import os
import sys
from pathlib import Path

# Allow running as a script from services/preprocess/scripts
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SERVICE_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
SRC_DIR = os.path.join(SERVICE_DIR, "src")
sys.path.insert(0, SRC_DIR)

from preprocess_delan.config import DelanPreprocessConfig
from preprocess_delan.pipeline import DelanPreprocessPipeline

def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdd", type=str2bool, default=True)
    ap.add_argument("--col_format", type=str, default="long")
    ap.add_argument("--raw_csv", type=str,
        default=os.environ.get("RAW_CSV", "/workspace/shared/data/raw/raw_data.csv"),
        help="Path to raw CSV file.")
    ap.add_argument("--out_npz", type=str,
        default=os.environ.get("OUT_NPZ", "/workspace/shared/data/preprocessed/delan_ur5_dataset.npz"),
        help="Path to output NPZ dataset file.")
    
    ap.add_argument("--trajectory_amount", type=int, default=0,
                    help="If >0: randomly sample this many trajectories (seeded) before splitting.")
    ap.add_argument("--test_fraction", type=float, default=0.2,
                    help="Fraction of trajectories used for test split.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Random seed for trajectory sampling and splitting.")

    args = ap.parse_args()

    col_format = args.col_format
    derive_qdd = args.qdd

    cfg = DelanPreprocessConfig(
        input_format=col_format,
        derive_qdd_from_dq=derive_qdd,
        test_fraction=args.test_fraction,
        random_seed=args.seed,
        trajectory_amount=(None if args.trajectory_amount <= 0 else args.trajectory_amount),
    )

    raw_csv = args.raw_csv 
    out_npz = args.out_npz

    # If out_npz is a directory, auto-name the file inside it
    if os.path.isdir(out_npz):
        ds_stem = Path(raw_csv).stem
        out_npz = os.path.join(out_npz, f"delan_{ds_stem}_dataset.npz")
    if not out_npz.endswith(".npz"):
        out_npz += ".npz"
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)

    # If user passes a directory, write a dataset-named file inside it.
    if os.path.isdir(out_npz):
        ds_stem = Path(raw_csv).stem
        out_npz = os.path.join(out_npz, f"delan_{ds_stem}_dataset.npz")
    # If user passes a path without .npz, add it.
    if not out_npz.endswith(".npz"):
        out_npz = out_npz + ".npz"

    pipe = DelanPreprocessPipeline(cfg)
    train, test = pipe.run(raw_csv_path=raw_csv, out_npz_path=out_npz)

    print(f"Saved: {out_npz}")
    print(f"Trajectories: train={len(train)} test={len(test)}")

    print("Exists:", os.path.exists(out_npz))