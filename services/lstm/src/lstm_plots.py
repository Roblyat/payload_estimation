# services/lstm/src/lstm_plots.py
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt


def save_loss_curve(history, out_dir: str, title: str = "LSTM training loss (scaled residuals)") -> str:
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(10.5, 4.2), dpi=140)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(history.history.get("loss", []), label="train", linewidth=1.6)
    ax.plot(history.history.get("val_loss", []), label="val", linewidth=1.6)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(out_dir, "loss_curve.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def save_residual_overlay(
    Y_test: np.ndarray,
    Y_pred: np.ndarray,
    out_dir: str,
    *,
    max_samples: int = 800,
    suptitle: str = "Residual torque: GT vs LSTM (test)",
) -> str:
    os.makedirs(out_dir, exist_ok=True)

    Y_test = np.asarray(Y_test)
    Y_pred = np.asarray(Y_pred)
    n = min(max_samples, Y_test.shape[0])
    n_dof = Y_test.shape[1]

    fig = plt.figure(figsize=(14.5, 8.5), dpi=140)

    for j in range(n_dof):
        ax = fig.add_subplot(3, 2, j + 1)
        ax.plot(Y_test[:n, j], label="GT", linewidth=1.2, color="C0")
        ax.plot(Y_pred[:n, j], label="LSTM", linewidth=1.2, alpha=0.9, color="C1")
        ax.set_title(f"Joint {j}", fontsize=11, pad=6)
        ax.grid(True, alpha=0.2)

        # show legend once
        if j == 0:
            ax.legend()

        # cleaner tick density
        if j < 4:
            ax.set_xticklabels([])

    fig.suptitle(suptitle, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out = os.path.join(out_dir, "residual_gt_vs_pred.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def save_residual_rmse_time_curve(
    Y_test: np.ndarray,
    Y_pred: np.ndarray,
    out_dir: str,
    *,
    dt: float | None = None,
    max_samples: int | None = None,
    max_joints: int = 6,
    title: str = "Residual RMSE over time (joint-avg)",
) -> str:
    os.makedirs(out_dir, exist_ok=True)

    Y_test = np.asarray(Y_test)
    Y_pred = np.asarray(Y_pred)
    n_dof = Y_test.shape[1]

    K = Y_test.shape[0]
    if max_samples is not None:
        K = min(int(max_samples), K)

    J = min(int(max_joints), n_dof)
    Y_test = Y_test[:K, :J]
    Y_pred = Y_pred[:K, :J]

    err = Y_pred - Y_test
    rmse_t = np.sqrt(np.mean(err ** 2, axis=1))

    if dt is not None and dt > 0:
        x = np.arange(rmse_t.shape[0]) * float(dt)
        x_label = "Time [s]"
    else:
        x = np.arange(rmse_t.shape[0])
        x_label = "Sample"

    fig = plt.figure(figsize=(10.5, 4.2), dpi=140)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, rmse_t, linewidth=1.4)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("RMSE")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    out = os.path.join(out_dir, "residual_rmse_time.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def save_residual_rmse_per_joint_bar(
    Y_test: np.ndarray,
    Y_pred: np.ndarray,
    out_dir: str,
    *,
    max_samples: int | None = None,
    max_joints: int = 6,
    title: str = "Residual RMSE per joint",
) -> str:
    os.makedirs(out_dir, exist_ok=True)

    Y_test = np.asarray(Y_test)
    Y_pred = np.asarray(Y_pred)
    n_dof = Y_test.shape[1]

    K = Y_test.shape[0]
    if max_samples is not None:
        K = min(int(max_samples), K)

    J = min(int(max_joints), n_dof)
    Y_test = Y_test[:K, :J]
    Y_pred = Y_pred[:K, :J]

    err = Y_pred - Y_test
    rmse_per_joint = np.sqrt(np.mean(err ** 2, axis=0))

    fig = plt.figure(figsize=(8.5, 4.2), dpi=140)
    ax = fig.add_subplot(1, 1, 1)
    joints = np.arange(rmse_per_joint.shape[0])
    ax.bar(joints, rmse_per_joint)
    ax.set_title(title)
    ax.set_xlabel("Joint")
    ax.set_ylabel("RMSE")
    ax.set_xticks(joints)
    ax.set_xticklabels([f"joint{j}" for j in joints])
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()

    out = os.path.join(out_dir, "residual_rmse_per_joint.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out
