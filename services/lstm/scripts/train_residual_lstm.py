import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import json
import io
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def per_joint_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))


def compute_x_scaler(X_train: np.ndarray, eps: float = 1e-8):
    # X_train: (N, H, D)
    flat = X_train.reshape(-1, X_train.shape[-1])  # (N*H, D)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < eps, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32), float(eps)


def apply_x_scaler(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return ((X - mean[None, None, :]) / std[None, None, :]).astype(np.float32)


def compute_y_scaler(Y_train: np.ndarray, eps: float = 1e-8):
    # Y_train: (N, dof)
    mean = Y_train.mean(axis=0)
    std = Y_train.std(axis=0)
    std = np.where(std < eps, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32), float(eps)


def apply_y_scaler(Y: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return ((Y - mean[None, :]) / std[None, :]).astype(np.float32)


def invert_y_scaler(Y_scaled: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (Y_scaled * std[None, :] + mean[None, :]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="ur5_lstm_windows_H50.npz")
    ap.add_argument("--out_dir", default="/workspace/shared/models/lstm/residual_lstm_H50")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--units", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--no_plots", action="store_true")
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--model_name", default="best.keras", help="Filename for the best model checkpoint inside out_dir (e.g. best_seed4_H50.keras)"
)
    args = ap.parse_args()

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    d = np.load(args.npz)
    X_train = d["X_train"].astype(np.float32)  # (N, H, 24)
    Y_train = d["Y_train"].astype(np.float32)  # (N, 6)
    if X_train.size == 0 or Y_train.size == 0:
        raise RuntimeError("Empty training set (X_train/Y_train). Your window NPZ has 0 samples.")

    X_test  = d["X_test"].astype(np.float32)
    Y_test  = d["Y_test"].astype(np.float32)

    H = int(d["H"])
    feature_dim = int(d["feature_dim"])
    n_dof = int(d["n_dof"])

    print("################################################")
    print("LSTM Residual Dataset:")
    print(f"  npz = {args.npz}")
    print(f"   H  = {H}")
    print(f"  din = {feature_dim}")
    print(f" dout = {n_dof}")
    print(f"  X_train = {X_train.shape}, Y_train = {Y_train.shape}")
    print(f"  X_test  = {X_test.shape},  Y_test  = {Y_test.shape}")
    print("################################################")

    # ---------- Run metadata / metrics logging ----------
    run_ts = datetime.now().isoformat(timespec="seconds")
    metrics_path = os.path.join(args.out_dir, f"metrics_train_test_H{H}.json")

    metrics = {
        "timestamp": run_ts,
        "npz": args.npz,
        "out_dir": args.out_dir,
        "H": H,
        "feature_dim": feature_dim,
        "n_dof": n_dof,
        "shapes": {
            "X_train": list(X_train.shape),
            "Y_train": list(Y_train.shape),
            "X_test": list(X_test.shape),
            "Y_test": list(Y_test.shape),
        },
        "args": {
            "epochs": args.epochs,
            "batch": args.batch,
            "val_split": args.val_split,
            "seed": args.seed,
            "units": args.units,
            "dropout": args.dropout,
            "eps": args.eps,
            "model_name": args.model_name,
        },
        "dataset_stats": {
            "Y_train_mean": Y_train.mean(axis=0).tolist(),
            "Y_train_std": Y_train.std(axis=0).tolist(),
            "Y_train_min": Y_train.min(axis=0).tolist(),
            "Y_train_max": Y_train.max(axis=0).tolist(),
            "Y_test_mean": Y_test.mean(axis=0).tolist(),
            "Y_test_std": Y_test.std(axis=0).tolist(),
            "Y_test_min": Y_test.min(axis=0).tolist(),
            "Y_test_max": Y_test.max(axis=0).tolist(),
        },
    }

    # ---------- Scalers (train-only) ----------
    x_mean, x_std, x_eps = compute_x_scaler(X_train, eps=args.eps)
    y_mean, y_std, y_eps = compute_y_scaler(Y_train, eps=args.eps)

    scalers_path = os.path.join(args.out_dir, f"scalers_H{H}.npz")
    np.savez(
        scalers_path,
        x_mean=x_mean, x_std=x_std, x_eps=np.float32(x_eps),
        y_mean=y_mean, y_std=y_std, y_eps=np.float32(y_eps),
        H=np.int32(H), feature_dim=np.int32(feature_dim), n_dof=np.int32(n_dof),
    )
    print(f"\nSaved scalers: {scalers_path}")
    print(f"X mean/std shapes: {x_mean.shape}/{x_std.shape}")
    print(f"Y mean/std shapes: {y_mean.shape}/{y_std.shape}")

    metrics["scalers"] = {
        "scalers_path": scalers_path,
        "x_eps": float(x_eps),
        "y_eps": float(y_eps),
        "x_mean_minmax": [float(x_mean.min()), float(x_mean.max())],
        "x_std_minmax": [float(x_std.min()), float(x_std.max())],
        "y_mean": y_mean.tolist(),
        "y_std": y_std.tolist(),
    }

    # Apply normalization/scaling
    X_train_n = apply_x_scaler(X_train, x_mean, x_std)
    X_test_n  = apply_x_scaler(X_test,  x_mean, x_std)

    Y_train_s = apply_y_scaler(Y_train, y_mean, y_std)
    Y_test_s  = apply_y_scaler(Y_test,  y_mean, y_std)

    # ---------- Model ----------
    model = Sequential()
    model.add(LSTM(units=args.units, return_sequences=True, input_shape=(H, feature_dim)))
    model.add(Dropout(args.dropout))
    model.add(LSTM(units=args.units))
    model.add(Dropout(args.dropout))
    model.add(Dense(n_dof))  # predict scaled residuals

    model.compile(optimizer="adam", loss="mse")

    # Capture model summary text for the metrics file
    buf = io.StringIO()

    # NEW (accept extra kwargs)
    def _summary_print(s, **kwargs):
        buf.write(s + "\n")

    model.summary(print_fn=_summary_print)

    model_summary_txt = buf.getvalue()
    print(model_summary_txt)  # keep showing it in container logs

    metrics["model"] = {
        "summary": model_summary_txt,
        "optimizer": "adam",
        "loss": "mse",
        "params": int(model.count_params()),
    }

    ckpt_path = os.path.join(args.out_dir, args.model_name)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True),
    ]

    history = model.fit(
        X_train_n, Y_train_s,
        epochs=args.epochs,
        batch_size=args.batch,
        validation_split=args.val_split,
        shuffle=True,
        callbacks=callbacks,
        verbose=2
    )

    # ---------- Training history metrics ----------
    train_loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    best_epoch = int(np.argmin(val_loss) + 1) if len(val_loss) > 0 else None
    best_val = float(np.min(val_loss)) if len(val_loss) > 0 else None

    metrics["train"] = {
        "epochs_ran": int(len(train_loss)),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "final_train_loss": float(train_loss[-1]) if len(train_loss) > 0 else None,
        "final_val_loss": float(val_loss[-1]) if len(val_loss) > 0 else None,
    }

    # Save full history to CSV for plotting/inspection
    hist_csv = os.path.join(args.out_dir, f"train_history_H{H}.csv")
    np.savetxt(
        hist_csv,
        np.column_stack([
            np.arange(1, len(train_loss) + 1),
            np.array(train_loss, dtype=np.float32),
            np.array(val_loss, dtype=np.float32) if len(val_loss) == len(train_loss) else np.full(len(train_loss), np.nan),
        ]),
        delimiter=",",
        header="epoch,loss,val_loss",
        comments="",
    )
    metrics["train"]["history_csv"] = hist_csv

    # ---------- Evaluate (invert to physical units) ----------
    Y_pred_s = model.predict(X_test_n, batch_size=args.batch, verbose=0).astype(np.float32)
    Y_pred = invert_y_scaler(Y_pred_s, y_mean, y_std)

    total_rmse = rmse(Y_test, Y_pred)
    joint_rmse = per_joint_rmse(Y_test, Y_pred)

    # ---------- Eval metrics (unscaled) ----------
    mse_total = float(np.mean((Y_test - Y_pred) ** 2))
    mse_per_joint = np.mean((Y_test - Y_pred) ** 2, axis=0).astype(np.float32)

    metrics["eval_test"] = {
        "rmse_total": float(total_rmse),
        "rmse_per_joint": joint_rmse.astype(np.float32).tolist(),
        "mse_total": mse_total,
        "mse_per_joint": mse_per_joint.tolist(),
        "predictions_npz": os.path.join(args.out_dir, "predictions_test.npz"),
        "best_model_path": os.path.join(args.out_dir, args.model_name),
    }

    # Write metrics JSON (single file with everything)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics JSON: {metrics_path}")


    print("\n################################################")
    print("LSTM Residual Evaluation (test, unscaled units):")
    print(f"Total RMSE: {total_rmse:.4f}")
    print("Per-joint RMSE:", " ".join([f"{x:.4f}" for x in joint_rmse]))
    print("################################################\n")

    # Save predictions for later
    np.savez(
        os.path.join(args.out_dir, "predictions_test.npz"),
        Y_test=Y_test,
        Y_pred=Y_pred,
        Y_pred_scaled=Y_pred_s,
        H=np.int32(H),
        feature_dim=np.int32(feature_dim),
        n_dof=np.int32(n_dof),
    )
    print(f"Saved predictions: {os.path.join(args.out_dir, 'predictions_test.npz')}")
    print(f"Saved best model:  {ckpt_path}")

    # ---------- Plots ----------
    if not args.no_plots:
        plt.figure(figsize=(10, 4), dpi=120)
        plt.plot(history.history["loss"], label="train")
        plt.plot(history.history["val_loss"], label="val")
        plt.title("LSTM training loss (scaled residuals)")
        plt.xlabel("epoch")
        plt.ylabel("MSE")
        plt.grid(True, alpha=0.2)
        plt.legend()
        out = os.path.join(args.out_dir, "loss_curve.png")
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        print(f"Saved: {out}")

        K = min(600, Y_test.shape[0])
        fig = plt.figure(figsize=(14, 8), dpi=120)
        for j in range(n_dof):
            ax = fig.add_subplot(3, 2, j + 1)
            ax.plot(Y_test[:K, j], label="GT", linewidth=1.0)
            ax.plot(Y_pred[:K, j], label="LSTM", linewidth=1.0, alpha=0.85)
            ax.set_title(f"Residual torque joint {j}")
            ax.grid(True, alpha=0.2)
            if j == 0:
                ax.legend()
        plt.tight_layout()
        out = os.path.join(args.out_dir, "residual_gt_vs_pred.png")
        plt.savefig(out, dpi=150)
        print(f"Saved: {out}")
        plt.close("all")


if __name__ == "__main__":
    main()