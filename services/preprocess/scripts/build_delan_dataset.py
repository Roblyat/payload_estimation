"""
Build a DeLaN-ready trajectory dataset (.npz) from long-format robot logs.

Expected CSV columns:
Time, Joint Name, Position, Velocity, Acceleration, Effort
"""
import os
import sys

# Allow running as a script from services/preprocess/scripts
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SERVICE_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
SRC_DIR = os.path.join(SERVICE_DIR, "src")
sys.path.insert(0, SRC_DIR)

from preprocess_delan.config import DelanPreprocessConfig
from preprocess_delan.pipeline import DelanPreprocessPipeline

if __name__ == "__main__":
    cfg = DelanPreprocessConfig()

    raw_csv = os.environ.get("RAW_CSV", "/workspace/shared/data/raw/raw_data.csv")
    out_npz = os.environ.get("OUT_NPZ", "/workspace/shared/data/preprocessed/delan_ur5_dataset.npz")

    pipe = DelanPreprocessPipeline(cfg)
    train, test = pipe.run(raw_csv_path=raw_csv, out_npz_path=out_npz)

    print(f"Saved: {out_npz}")
    print(f"Trajectories: train={len(train)} test={len(test)}")
