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


def build_windows_for_split(q_list, qd_list, qdd_list, tau_hat_list, r_tau_list, H: int):
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

        # Features per time step: (T, 4*n_dof)
        feat = np.concatenate([q, qd, qdd, th], axis=1)  # (T, 24)

        # Sliding windows
        for k in range(H - 1, T):
            Xs.append(feat[k - H + 1: k + 1, :])   # (H, 24)
            Ys.append(r[k, :])                     # (6,)

    if len(Xs) == 0:
        return np.zeros((0, H, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32)

    X = np.stack(Xs, axis=0).astype(np.float32)
    Y = np.stack(Ys, axis=0).astype(np.float32)
    return X, Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_npz", type=str, required=True)
    ap.add_argument("--out_npz", type=str, required=True)
    ap.add_argument("--H", type=int, default=50)
    args = ap.parse_args()

    d = np.load(args.in_npz, allow_pickle=True)

    # Train split
    X_train, Y_train = build_windows_for_split(
        d["train_q"], d["train_qd"], d["train_qdd"], d["train_tau_hat"], d["train_r_tau"], args.H
    )

    # Test split
    X_test, Y_test = build_windows_for_split(
        d["test_q"], d["test_qd"], d["test_qdd"], d["test_tau_hat"], d["test_r_tau"], args.H
    )

    # Metadata
    n_dof = int(np.asarray(d["train_q"][0]).shape[1])
    feature_dim = int(4 * n_dof)

    os.makedirs(os.path.dirname(args.out_npz), exist_ok=True)
    np.savez(
        args.out_npz,
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        Y_test=Y_test,
        H=np.int32(args.H),
        n_dof=np.int32(n_dof),
        feature_dim=np.int32(feature_dim),
    )

    print(f"Saved: {args.out_npz}")
    print(f"H={args.H}, n_dof={n_dof}, feature_dim={feature_dim}")
    print(f"X_train: {X_train.shape}  Y_train: {Y_train.shape}")
    print(f"X_test : {X_test.shape}  Y_test : {Y_test.shape}")


if __name__ == "__main__":
    main()