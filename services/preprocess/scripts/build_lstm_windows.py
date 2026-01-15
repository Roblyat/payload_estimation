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

import sys
sys.path.append("/workspace/shared/src")
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

    d = np.load(args.in_npz, allow_pickle=True)

    # Train split
    X_train, Y_train = build_windows_for_split(
        d["train_q"], d["train_qd"], d["train_qdd"], d["train_tau_hat"], d["train_r_tau"], args.H, args.features
    )
    if X_train.shape[0] == 0:
        raise RuntimeError(
            f"No windows created. Likely H={args.H} is too large for all trajectories. "
            f"Try a smaller H (e.g. 50)."
    )

    # Test split
    X_test, Y_test = build_windows_for_split(
        d["test_q"], d["test_qd"], d["test_qdd"], d["test_tau_hat"], d["test_r_tau"], args.H, args.features
    )

    # Metadata (robust even if train split empty)
    if len(d["train_q"]) > 0:
        n_dof = int(np.asarray(d["train_q"][0]).shape[1])
    else:
        n_dof = int(np.asarray(d["test_q"][0]).shape[1])

    # Compute feature_dim from the chosen mode
    feature_dim_val = int(feature_dim(n_dof, args.features))

    os.makedirs(os.path.dirname(args.out_npz), exist_ok=True)
    np.savez(
        args.out_npz,
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        Y_test=Y_test,
        H=np.int32(args.H),
        n_dof=np.int32(n_dof),
        feature_dim=np.int32(feature_dim_val),
        feature_mode=np.str_(args.features),
    )

    print(f"Saved: {args.out_npz}")
    print(f"H={args.H}, n_dof={n_dof}, feature_dim={feature_dim_val}")
    print(f"X_train: {X_train.shape}  Y_train: {Y_train.shape}")
    print(f"X_test : {X_test.shape}  Y_test : {Y_test.shape}")
    print(f"feature_mode={args.features}")

if __name__ == "__main__":
    main()