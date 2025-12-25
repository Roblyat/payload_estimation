import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def per_joint_rmse(y_true, y_pred):
    # returns (6,)
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))


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
    args = ap.parse_args()

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    d = np.load(args.npz)
    X_train = d["X_train"].astype(np.float32)  # (N, H, 24)
    Y_train = d["Y_train"].astype(np.float32)  # (N, 6)
    X_test  = d["X_test"].astype(np.float32)
    Y_test  = d["Y_test"].astype(np.float32)

    H = int(d["H"])
    feature_dim = int(d["feature_dim"])
    n_dof = int(d["n_dof"])

    print("################################################")
    print("LSTM Residual Dataset:")
    print(f"  npz = {args.npz}")
    print(f"   H  = {H}")
    print(f"  din = {feature_dim}  (expected 24)")
    print(f" dout = {n_dof}        (expected 6)")
    print(f"  X_train = {X_train.shape}, Y_train = {Y_train.shape}")
    print(f"  X_test  = {X_test.shape},  Y_test  = {Y_test.shape}")
    print("################################################")

    # ---- Model (GfG-style, but output=6 instead of 1) ----
    model = Sequential()
    model.add(LSTM(units=args.units, return_sequences=True, input_shape=(H, feature_dim)))
    model.add(Dropout(args.dropout))
    model.add(LSTM(units=args.units))
    model.add(Dropout(args.dropout))
    model.add(Dense(n_dof))  # predict residual torque vector (6)

    model.compile(optimizer="adam", loss="mse")
    model.summary()

    ckpt_path = os.path.join(args.out_dir, "best.keras")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True),
    ]

    history = model.fit(
        X_train, Y_train,
        epochs=args.epochs,
        batch_size=args.batch,
        validation_split=args.val_split,
        shuffle=True,          # OK: each sample is already a window
        callbacks=callbacks,
        verbose=2
    )

    # ---- Evaluate ----
    Y_pred = model.predict(X_test, batch_size=args.batch, verbose=0).astype(np.float32)

    total_rmse = rmse(Y_test, Y_pred)
    joint_rmse = per_joint_rmse(Y_test, Y_pred)

    print("\n################################################")
    print("LSTM Residual Evaluation (test):")
    print(f"Total RMSE: {total_rmse:.4f}")
    print("Per-joint RMSE:", " ".join([f"{x:.4f}" for x in joint_rmse]))
    print("################################################\n")

    # Save predictions for later combination / analysis
    np.savez(
        os.path.join(args.out_dir, "predictions_test.npz"),
        Y_test=Y_test,
        Y_pred=Y_pred,
        H=np.int32(H),
        feature_dim=np.int32(feature_dim),
        n_dof=np.int32(n_dof),
    )
    print(f"Saved predictions: {os.path.join(args.out_dir, 'predictions_test.npz')}")
    print(f"Saved best model:  {ckpt_path}")

    # ---- Plots ----
    if not args.no_plots:
        # 1) Training curve
        plt.figure(figsize=(10, 4), dpi=120)
        plt.plot(history.history["loss"], label="train")
        plt.plot(history.history["val_loss"], label="val")
        plt.title("LSTM training loss")
        plt.xlabel("epoch")
        plt.ylabel("MSE")
        plt.grid(True, alpha=0.2)
        plt.legend()
        out = os.path.join(args.out_dir, "loss_curve.png")
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        print(f"Saved: {out}")

        # 2) Residual GT vs Pred for each joint on test (first K samples)
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
