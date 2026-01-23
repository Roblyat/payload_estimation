import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import json
import csv
from datetime import datetime

import sys
sys.path.append("/workspace/shared/src")
from feature_builders import build_features, FEATURE_MODES

from path_helpers import resolve_npz_path

def apply_x_scaler_feat(feat: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray):
    # feat: (T, D)
    return ((feat - x_mean[None, :]) / x_std[None, :]).astype(np.float32)

def invert_y_scaler(y_scaled: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray):
    return (y_scaled * y_std[None, :] + y_mean[None, :]).astype(np.float32)

def build_windows(feat: np.ndarray, H: int) -> np.ndarray:
    """
    feat: (T, D)
    returns X: (T-H+1, H, D), where each window ends at time k (k>=H-1)
    """
    T, D = feat.shape
    if T < H:
        return np.zeros((0, H, D), dtype=np.float32)

    X = np.zeros((T - H + 1, H, D), dtype=np.float32)
    for i, k in enumerate(range(H - 1, T)):
        X[i] = feat[k - H + 1 : k + 1]
    return X


def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def per_joint_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))


def concat_valid_across_trajs(traj_list, H):
    """
    Given list of (T_i, dof) arrays, concatenate only valid indices k>=H-1.
    Returns concatenated array shape (sum_i (T_i-H+1), dof).
    """
    chunks = []
    for a in traj_list:
        a = np.asarray(a, dtype=np.float32)
        if a.shape[0] >= H:
            chunks.append(a[H - 1 :])
    if not chunks:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack(chunks).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--residual_npz", required=True,
                    help="Trajectory-wise residual NPZ (ur5_residual_traj.npz)")
    ap.add_argument("--model", required=True,
                    help="Path to trained keras model (best.keras)")
    ap.add_argument("--out_dir", default="/workspace/shared/models/lstm/eval_combined_H50")
    ap.add_argument("--H", type=int, default=50)
    ap.add_argument("--split", choices=["test", "train"], default="test")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--max_plot_samples", type=int, default=800)
    ap.add_argument("--save_pred_npz", action="store_true",
                    help="Save per-trajectory predictions to NPZ in out_dir")
    ap.add_argument("--scalers", required=True, help="scalers_H50.npz from training")
    ap.add_argument("--features", type=str, default="full", choices=list(FEATURE_MODES),
        help="Feature mode used for LSTM input windows (must match training).",)
    ap.add_argument("--metrics_csv", type=str, default="", help="Append summary row to this CSV path")
    ap.add_argument("--metrics_json", type=str, default="", help="Append summary row to this JSON path")
    # Optional sweep metadata for logging
    ap.add_argument("--K", type=int, default=None)
    ap.add_argument("--test_fraction", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--delan_seed", type=int, default=None)
    ap.add_argument("--delan_epochs", type=int, default=None)
    ap.add_argument("--hp_preset", type=str, default=None)
    ap.add_argument("--delan_rmse_val", type=float, default=None)
    ap.add_argument("--delan_rmse_test", type=float, default=None)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    args.residual_npz = resolve_npz_path(args.residual_npz)
    residual_npz = resolve_npz_path(args.residual_npz)
    d = np.load(residual_npz, allow_pickle=True)

    split = args.split
    H = args.H

    q_list       = list(d[f"{split}_q"])
    qd_list      = list(d[f"{split}_qd"])
    qdd_list     = list(d[f"{split}_qdd"])
    tau_list     = list(d[f"{split}_tau"])
    tau_hat_list = list(d[f"{split}_tau_hat"])
    r_tau_list   = list(d[f"{split}_r_tau"])

    n_traj = len(q_list)
    n_dof = int(np.asarray(q_list[0]).shape[1])
    feature_dim = int(build_features(
        np.asarray(q_list[0], dtype=np.float32),
        np.asarray(qd_list[0], dtype=np.float32),
        np.asarray(qdd_list[0], dtype=np.float32),
        np.asarray(tau_hat_list[0], dtype=np.float32),
        mode=args.features
    ).shape[1])


    print("################################################")
    print("Evaluate & Combine")
    print(f" residual_npz = {args.residual_npz}")
    print(f" model        = {args.model}")
    print(f" split        = {split}")
    print(f" H            = {H}")
    print(f" n_traj       = {n_traj}")
    print(f" n_dof        = {n_dof}")
    print(f" feature_dim  = {feature_dim}")
    print("################################################")

    model = tf.keras.models.load_model(args.model)

    sc = np.load(args.scalers)
    x_mean = sc["x_mean"].astype(np.float32)
    x_std  = sc["x_std"].astype(np.float32)
    y_mean = sc["y_mean"].astype(np.float32)
    y_std  = sc["y_std"].astype(np.float32)
    
    # Store per-trajectory predicted residuals aligned to original time
    r_hat_traj = []
    tau_rg_traj = []

    # Also build concatenated arrays (valid region only) for global metrics/plots
    tau_gt_valid_all = []
    tau_delan_valid_all = []
    tau_rg_valid_all = []
    r_gt_valid_all = []
    r_hat_valid_all = []

    for i in range(n_traj):
        q   = np.asarray(q_list[i], dtype=np.float32)
        qd  = np.asarray(qd_list[i], dtype=np.float32)
        qdd = np.asarray(qdd_list[i], dtype=np.float32)
        tau = np.asarray(tau_list[i], dtype=np.float32)
        tau_hat = np.asarray(tau_hat_list[i], dtype=np.float32)
        r_gt = np.asarray(r_tau_list[i], dtype=np.float32)

        T = q.shape[0]
        if T < H:
            # not enough length for a single window
            r_hat_traj.append(np.full((T, n_dof), np.nan, dtype=np.float32))
            tau_rg_traj.append(np.full((T, n_dof), np.nan, dtype=np.float32))
            continue

        feat = build_features(q, qd, qdd, tau_hat, mode=args.features)  # (T, D)
        feat_n = apply_x_scaler_feat(feat, x_mean, x_std)
        X = build_windows(feat_n, H)  # windows are normalized

        r_hat_valid_scaled = model.predict(X, batch_size=args.batch, verbose=0).astype(np.float32)
        r_hat_valid = invert_y_scaler(r_hat_valid_scaled, y_mean, y_std)  # back to physical units


        # Align predicted residuals to full timeline (first H-1 undefined)
        r_hat_full = np.full((T, n_dof), np.nan, dtype=np.float32)
        r_hat_full[H - 1 :] = r_hat_valid

        tau_rg_full = np.full((T, n_dof), np.nan, dtype=np.float32)
        tau_rg_full[H - 1 :] = tau_hat[H - 1 :] + r_hat_valid

        r_hat_traj.append(r_hat_full)
        tau_rg_traj.append(tau_rg_full)

        # Collect valid regions for global metrics
        tau_gt_valid_all.append(tau[H - 1 :])
        tau_delan_valid_all.append(tau_hat[H - 1 :])
        tau_rg_valid_all.append(tau_rg_full[H - 1 :])

        r_gt_valid_all.append(r_gt[H - 1 :])
        r_hat_valid_all.append(r_hat_valid)

        if (i + 1) % 25 == 0 or (i + 1) == n_traj:
            print(f"  done {i+1}/{n_traj}", flush=True)

    tau_gt_valid_all = np.vstack(tau_gt_valid_all).astype(np.float32)
    tau_delan_valid_all = np.vstack(tau_delan_valid_all).astype(np.float32)
    tau_rg_valid_all = np.vstack(tau_rg_valid_all).astype(np.float32)
    r_gt_valid_all = np.vstack(r_gt_valid_all).astype(np.float32)
    r_hat_valid_all = np.vstack(r_hat_valid_all).astype(np.float32)

    # ---- Metrics ----
    delan_mse = mse(tau_gt_valid_all, tau_delan_valid_all)
    delan_rmse = rmse(tau_gt_valid_all, tau_delan_valid_all)

    rg_mse = mse(tau_gt_valid_all, tau_rg_valid_all)
    rg_rmse = rmse(tau_gt_valid_all, tau_rg_valid_all)

    r_mse = mse(r_gt_valid_all, r_hat_valid_all)
    r_rmse = rmse(r_gt_valid_all, r_hat_valid_all)

    delan_joint = per_joint_rmse(tau_gt_valid_all, tau_delan_valid_all)
    rg_joint = per_joint_rmse(tau_gt_valid_all, tau_rg_valid_all)
    r_joint = per_joint_rmse(r_gt_valid_all, r_hat_valid_all)

    print("\n################################################")
    print(f"Stage-2 Evaluation ({split}, valid k>=H-1):")
    print(f"DeLaN torque:   MSE={delan_mse:.6e}  RMSE={delan_rmse:.6e}")
    print("  per-joint RMSE:", " ".join([f"{x:.4f}" for x in delan_joint]))
    print(f"Residual LSTM:  MSE={r_mse:.6e}      RMSE={r_rmse:.6e}")
    print("  per-joint RMSE:", " ".join([f"{x:.4f}" for x in r_joint]))
    print(f"Combined torque MSE={rg_mse:.6e}     RMSE={rg_rmse:.6e}")
    print("  per-joint RMSE:", " ".join([f"{x:.4f}" for x in rg_joint]))
    print("################################################\n")

    # ---- Save metrics ----
    metrics_path = os.path.join(args.out_dir, f"metrics_{split}_H{H}.txt")
    with open(metrics_path, "w") as f:
        f.write(f"split={split}\nH={H}\n")
        f.write(f"delan_mse={delan_mse}\ndelan_rmse={delan_rmse}\n")
        f.write(f"res_mse={r_mse}\nres_rmse={r_rmse}\n")
        f.write(f"rg_mse={rg_mse}\nrg_rmse={rg_rmse}\n")
        f.write("delan_joint_rmse=" + " ".join(map(str, delan_joint.tolist())) + "\n")
        f.write("res_joint_rmse=" + " ".join(map(str, r_joint.tolist())) + "\n")
        f.write("rg_joint_rmse=" + " ".join(map(str, rg_joint.tolist())) + "\n")
        f.write(f"feature_mode={args.features}\n")
        f.write(f"residual_npz={args.residual_npz}\n")
        f.write(f"lstm_model={args.model}\n")
        f.write(f"lstm_scalers={args.scalers}\n")

    print(f"Saved: {metrics_path}")

        # ---- Save metrics JSON (same content, structured) ----
    metrics_json_path = os.path.join(args.out_dir, f"metrics_{split}_H{H}.json")

    metrics_json = {
        "split": split,
        "H": int(H),
        "n_dof": int(n_dof),
        "feature_mode": args.features,
        "paths": {
            "residual_npz": args.residual_npz,
            "lstm_model": args.model,
            "lstm_scalers": args.scalers,
        },
        "metrics": {
            "delan_mse": float(delan_mse),
            "delan_rmse": float(delan_rmse),
            "res_mse": float(r_mse),
            "res_rmse": float(r_rmse),
            "rg_mse": float(rg_mse),
            "rg_rmse": float(rg_rmse),
            "delan_joint_rmse": [float(x) for x in delan_joint.tolist()],
            "res_joint_rmse": [float(x) for x in r_joint.tolist()],
            "rg_joint_rmse": [float(x) for x in rg_joint.tolist()],
        },
    }

    with open(metrics_json_path, "w") as f:
        json.dump(metrics_json, f, indent=2)

    print(f"Saved: {metrics_json_path}")

    # ---- Append summary CSV/JSON ----
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "split": split,
        "H": int(H),
        "feature_mode": args.features,
        "K": args.K,
        "test_fraction": args.test_fraction,
        "seed": args.seed,
        "delan_seed": args.delan_seed,
        "delan_epochs": args.delan_epochs,
        "hp_preset": args.hp_preset,
        "delan_rmse_val": args.delan_rmse_val,
        "delan_rmse_test": float(delan_rmse),
        "delan_rmse_test_meta": args.delan_rmse_test,
        "residual_rmse": float(r_rmse),
        "rg_rmse": float(rg_rmse),
        "gain": float(delan_rmse - rg_rmse),
        "gain_ratio": float(rg_rmse / delan_rmse) if delan_rmse != 0 else None,
        "residual_npz": args.residual_npz,
        "lstm_model": args.model,
        "lstm_scalers": args.scalers,
        "out_dir": args.out_dir,
    }

    if args.metrics_csv:
        os.makedirs(os.path.dirname(args.metrics_csv), exist_ok=True)
        file_exists = os.path.exists(args.metrics_csv)
        with open(args.metrics_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(summary)
        print(f"Appended CSV: {args.metrics_csv}")

    if args.metrics_json:
        os.makedirs(os.path.dirname(args.metrics_json), exist_ok=True)
        if args.metrics_json.endswith(".jsonl"):
            with open(args.metrics_json, "a") as f:
                f.write(json.dumps(summary) + "\n")
        else:
            if os.path.exists(args.metrics_json):
                try:
                    with open(args.metrics_json, "r") as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        items = data
                    else:
                        items = [data]
                except Exception:
                    items = []
            else:
                items = []
            items.append(summary)
            with open(args.metrics_json, "w") as f:
                json.dump(items, f, indent=2)
        print(f"Appended JSON: {args.metrics_json}")

    # ---- Optional save predictions ----
    if args.save_pred_npz:
        out_npz = os.path.join(args.out_dir, f"combined_predictions_{split}_H{H}.npz")
        np.savez(
            out_npz,
            r_hat=np.asarray(r_hat_traj, dtype=object),
            tau_rg=np.asarray(tau_rg_traj, dtype=object),
        )
        print(f"Saved: {out_npz}")

    # ---- Plots (first K samples of concatenated valid region) ----
    K = min(args.max_plot_samples, tau_gt_valid_all.shape[0])

    # 1) Residual GT vs Pred
    fig = plt.figure(figsize=(14, 8), dpi=120)
    for j in range(n_dof):
        ax = fig.add_subplot(3, 2, j + 1)
        ax.plot(r_gt_valid_all[:K, j], label="GT residual", linewidth=1.0)
        ax.plot(r_hat_valid_all[:K, j], label="LSTM residual", linewidth=1.0, alpha=0.85)
        ax.set_title(f"Residual joint {j}")
        ax.grid(True, alpha=0.2)
        if j == 0:
            ax.legend()
    plt.tight_layout()
    out = os.path.join(args.out_dir, f"residual_gt_vs_pred_{split}_H{H}.png")
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)

    # 2) Torque GT vs DeLaN vs Combined
    fig = plt.figure(figsize=(14, 8), dpi=120)
    for j in range(n_dof):
        ax = fig.add_subplot(3, 2, j + 1)
        ax.plot(tau_gt_valid_all[:K, j], label="GT tau", linewidth=1.0)
        ax.plot(tau_delan_valid_all[:K, j], label="DeLaN tau_hat", linewidth=1.0, alpha=0.85)
        ax.plot(tau_rg_valid_all[:K, j], label="Combined tau_RG", linewidth=1.0, alpha=0.85)
        ax.set_title(f"Torque joint {j}")
        ax.grid(True, alpha=0.2)
        if j == 0:
            ax.legend()
    plt.tight_layout()
    out = os.path.join(args.out_dir, f"torque_gt_vs_delan_vs_combined_{split}_H{H}.png")
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
