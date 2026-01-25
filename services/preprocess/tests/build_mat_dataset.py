#!/usr/bin/env python3
"""
Convert inverse dynamics .mat datasets to CSV with project-style path handling.

Writes into RAW_DIR/inverse_dynamics_csv/<dataset_name>/...

Usage:
  python3 scripts/mat_inverse_to_csv.py
"""

import os
import numpy as np
import pandas as pd
import scipy.io


RAW_DIR = "/workspace/shared/data/raw"
OUT_DIR = os.path.join(RAW_DIR, "inverse_dynamics_csv")

# Put the .mat files here (in RAW_DIR)
MAT_FILES = [
    os.path.join(RAW_DIR, "BaxterRand.mat"),
    os.path.join(RAW_DIR, "BaxterRhythmic.mat"),
    os.path.join(RAW_DIR, "URpickNplace.mat"),
]


def load_mat(path: str) -> dict:
    d = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    return {k: v for k, v in d.items() if not k.startswith("__")}


def x_colnames(nj: int):
    return [*(f"q{i}" for i in range(1, nj + 1)),
            *(f"qd{i}" for i in range(1, nj + 1)),
            *(f"qdd{i}" for i in range(1, nj + 1))]


def y_colnames(nj: int):
    return [f"tau{i}" for i in range(1, nj + 1)]


def save_csv(df: pd.DataFrame, out_csv: str):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} shape={df.shape}")


def handle_baxter(d: dict, dataset_name: str):
    # Expected keys: X_train/X_test/Y_train/Y_test
    X_train, X_test = d["X_train"], d["X_test"]
    Y_train, Y_test = d["Y_train"], d["Y_test"]

    nj = X_train.shape[1] // 3  # Baxter: 21 -> 7 joints
    X_cols = x_colnames(nj)
    Y_cols = y_colnames(nj)

    base = os.path.join(OUT_DIR, dataset_name)

    save_csv(pd.DataFrame(X_train, columns=X_cols), os.path.join(base, "X_train.csv"))
    save_csv(pd.DataFrame(X_test,  columns=X_cols), os.path.join(base, "X_test.csv"))
    save_csv(pd.DataFrame(Y_train, columns=Y_cols), os.path.join(base, "Y_train.csv"))
    save_csv(pd.DataFrame(Y_test,  columns=Y_cols), os.path.join(base, "Y_test.csv"))

    # convenience combined
    save_csv(pd.DataFrame(np.hstack([X_train, Y_train]), columns=X_cols + Y_cols),
             os.path.join(base, "XY_train.csv"))
    save_csv(pd.DataFrame(np.hstack([X_test, Y_test]), columns=X_cols + Y_cols),
             os.path.join(base, "XY_test.csv"))


def handle_ur(d: dict, dataset_name: str):
    # Expected keys like: urPicknPlace, urPicknPlaceHyper (both N x 24)
    base = os.path.join(OUT_DIR, dataset_name)

    nj = 6
    X_cols = x_colnames(nj)  # 18
    Y_cols = y_colnames(nj)  # 6

    for key, arr in d.items():
        if not (isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] == 24):
            continue

        X = arr[:, :3 * nj]             # first 18 cols
        Y = arr[:, 3 * nj:3 * nj + nj]  # last 6 cols

        save_csv(pd.DataFrame(X, columns=X_cols), os.path.join(base, f"{key}_X.csv"))
        save_csv(pd.DataFrame(Y, columns=Y_cols), os.path.join(base, f"{key}_Y.csv"))
        save_csv(pd.DataFrame(arr, columns=X_cols + Y_cols), os.path.join(base, f"{key}_XY.csv"))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for mat_path in MAT_FILES:
        if not os.path.isfile(mat_path):
            raise FileNotFoundError(f"Missing .mat file: {mat_path}")

        dataset_name = os.path.splitext(os.path.basename(mat_path))[0]
        d = load_mat(mat_path)

        if {"X_train", "X_test", "Y_train", "Y_test"}.issubset(d.keys()):
            handle_baxter(d, dataset_name)
        else:
            handle_ur(d, dataset_name)

    print("\nDone. Output directory:", OUT_DIR)


if __name__ == "__main__":
    main()
