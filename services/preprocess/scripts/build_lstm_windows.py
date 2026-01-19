"""
Build Stage-2 LSTM window dataset from trajectory-wise residual NPZ.

Input NPZ (trajectory-wise object arrays):
  train_q, train_qd, train_qdd, train_tau_hat, train_r_tau
  test_q,  test_qd,  test_qdd,  test_tau_hat,  test_r_tau

Output NPZ (dense arrays):
  X_train: (N, H, 4*n_dof)
  Y_train: (N, n_dof)
  X_test, Y_test
  H, n_dof, feature_dim
"""
import os
import argparse
import numpy as np

import json, time, traceback
from pathlib import Path

import sys
if "/workspace/shared/src" not in sys.path:
    sys.path.insert(0, "/workspace/shared/src")

from path_helpers import resolve_npz_path, normalize_out_npz
from feature_builders import build_features, FEATURE_CHOICES, FEATURE_MODES, feature_dim

def build_windows_for_split(q_list, qd_list, qdd_list, tau_hat_list, r_tau_list, H: int, feature_mode: str):
    Xs, Ys = [], []

    for i in range(len(q_list)):
        q   = np.asarray(q_list[i], dtype=np.float32)
        qd  = np.asarray(qd_list[i], dtype=np.float32)
        qdd = np.asarray(qdd_list[i], dtype=np.float32)
        th  = np.asarray(tau_hat_list[i], dtype=np.float32)
        r   = np.asarray(r_tau_list[i], dtype=np.float32)

        T, n_dof = q.shape
        if T < H:
            continue

        # Features per time step: (T, feature_dim)
        feat = build_features(q, qd, qdd, th, mode=feature_mode)

        # Sliding windows
        for k in range(H - 1, T):
            Xs.append(feat[k - H + 1: k + 1, :])   # (H, feature_dim)
            Ys.append(r[k, :])                     # (n_dof,)

    if len(Xs) == 0:
        # Return empty arrays with consistent ranks.
        return (
            np.zeros((0, H, 0), dtype=np.float32),
            np.zeros((0, 0), dtype=np.float32),
        )

    X = np.stack(Xs, axis=0).astype(np.float32)
    Y = np.stack(Ys, axis=0).astype(np.float32)
    return X, Y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_npz", type=str, required=True)
    ap.add_argument("--out_npz", type=str, required=True)
    ap.add_argument("--H", type=int, default=50)
    ap.add_argument("--features", type=str, default="full", choices=list(FEATURE_CHOICES),
        help="Feature mode: full|tau_hat|state|state_tauhat",)


    args = ap.parse_args()

    print("Building LSTM windows")
    print(f"  in_npz      = {args.in_npz}")
    print(f"  out_npz     = {args.out_npz}")
    print(f"  H           = {args.H}")
    print(f"  feature_mode= {args.features}")

    in_npz = resolve_npz_path(args.in_npz)
    out_npz = normalize_out_npz(args.out_npz)

    out_dir = os.path.dirname(out_npz)
    stem = Path(out_npz).stem
    out_json = os.path.join(out_dir, f"{stem}.json")

    meta = {
        "status": "started",
        "timestamp_unix": time.time(),
        "in_npz_arg": args.in_npz,
        "in_npz": in_npz,
        "out_npz_arg": args.out_npz,
        "out_npz": out_npz,
        "out_json": out_json,
        "H": int(args.H),
        "feature_mode": args.features,
    }

    try:
        if not os.path.exists(in_npz):
            raise FileNotFoundError(f"Input NPZ not found: {in_npz} (arg was {args.in_npz})")

        d = np.load(in_npz, allow_pickle=True)

        X_train, Y_train = build_windows_for_split(
            d["train_q"], d["train_qd"], d["train_qdd"], d["train_tau_hat"], d["train_r_tau"],
            args.H, args.features
        )
        X_test, Y_test = build_windows_for_split(
            d["test_q"], d["test_qd"], d["test_qdd"], d["test_tau_hat"], d["test_r_tau"],
            args.H, args.features
        )

        n_dof = int(np.asarray(d["train_q"][0]).shape[1]) if len(d["train_q"]) > 0 else int(np.asarray(d["test_q"][0]).shape[1])
        feature_dim = int(X_train.shape[-1]) if X_train.ndim == 3 else 0

        os.makedirs(out_dir, exist_ok=True)
        np.savez(
            out_npz,
            X_train=X_train,
            Y_train=Y_train,
            X_test=X_test,
            Y_test=Y_test,
            H=np.int32(args.H),
            n_dof=np.int32(n_dof),
            feature_dim=np.int32(feature_dim),
            feature_mode=np.array(args.features, dtype=object),
        )

        meta.update({
            "status": "ok",
            "n_dof": int(n_dof),
            "feature_dim": int(feature_dim),
            "X_train_shape": list(X_train.shape),
            "Y_train_shape": list(Y_train.shape),
            "X_test_shape": list(X_test.shape),
            "Y_test_shape": list(Y_test.shape),
            "keys": list(d.files),
        })

        with open(out_json, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"Saved: {out_npz}")
        print(f"Saved: {out_json}")
        print(f"H={args.H}, n_dof={n_dof}, feature_dim={feature_dim}")
        print(f"X_train: {X_train.shape}  Y_train: {Y_train.shape}")
        print(f"X_test : {X_test.shape}  Y_test : {Y_test.shape}")
        print(f"feature_mode={args.features}")

    except Exception as e:
        meta.update({
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        })
        os.makedirs(out_dir, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(meta, f, indent=2)
        raise

if __name__ == "__main__":
    main()