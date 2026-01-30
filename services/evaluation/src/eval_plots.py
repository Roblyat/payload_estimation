from __future__ import annotations

import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


def _time_axis(n: int, dt: float | None):
    if dt is not None and dt > 0:
        x = np.arange(int(n)) * float(dt)
        x_label = "Time [s]"
    else:
        x = np.arange(int(n))
        x_label = "Sample"
    return x, x_label


def save_residual_overlay_grid(
    r_gt: np.ndarray,
    r_pred: np.ndarray,
    out_dir: str,
    *,
    max_samples: int = 800,
    title: str = "Residual $i_{motor}$: GT vs LSTM (valid region)",
    out_name: str = "residual_gt_vs_pred.png",
) -> str:
    os.makedirs(out_dir, exist_ok=True)

    r_gt = np.asarray(r_gt)
    r_pred = np.asarray(r_pred)
    n = min(int(max_samples), int(r_gt.shape[0]))
    n_dof = int(r_gt.shape[1])

    cols = 2
    rows = int(np.ceil(n_dof / cols))
    fig = plt.figure(figsize=(14, 4 * rows), dpi=120)

    for j in range(n_dof):
        ax = fig.add_subplot(rows, cols, j + 1)
        ax.plot(r_gt[:n, j], label="GT $i_{motor}$ residual", linewidth=1.0)
        ax.plot(r_pred[:n, j], label="LSTM $i_{motor}$ residual", linewidth=1.0, alpha=0.85)
        ax.set_title(f"Residual joint {j}")
        ax.grid(True, alpha=0.2)
        if j == 0:
            ax.legend()

    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    out = os.path.join(out_dir, out_name)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def save_torque_overlay_grid(
    tau_gt: np.ndarray,
    tau_delan: np.ndarray,
    tau_combined: np.ndarray,
    out_dir: str,
    *,
    max_samples: int = 800,
    title: str = "$i_{motor}$: GT vs DeLaN vs Combined (valid region)",
    out_name: str = "torque_gt_vs_delan_vs_combined.png",
) -> str:
    os.makedirs(out_dir, exist_ok=True)

    tau_gt = np.asarray(tau_gt)
    tau_delan = np.asarray(tau_delan)
    tau_combined = np.asarray(tau_combined)
    n = min(int(max_samples), int(tau_gt.shape[0]))
    n_dof = int(tau_gt.shape[1])

    cols = 2
    rows = int(np.ceil(n_dof / cols))
    fig = plt.figure(figsize=(14, 4 * rows), dpi=120)

    for j in range(n_dof):
        ax = fig.add_subplot(rows, cols, j + 1)
        ax.plot(tau_gt[:n, j], label="GT $i_{motor}$", linewidth=1.0)
        ax.plot(tau_delan[:n, j], label="DeLaN $\\hat{i}_{motor}$", linewidth=1.0, alpha=0.85)
        ax.plot(tau_combined[:n, j], label="Combined $i_{motor}$", linewidth=1.0, alpha=0.85)
        ax.set_title(f"$i_{{motor}}$ joint {j} [A]")
        ax.grid(True, alpha=0.2)
        if j == 0:
            ax.legend()

    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    out = os.path.join(out_dir, out_name)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def save_torque_rmse_per_joint_grouped_bar(
    tau_gt: np.ndarray,
    tau_delan: np.ndarray,
    tau_combined: np.ndarray,
    out_dir: str,
    *,
    max_samples: int | None = None,
    max_joints: int = 6,
    title: str = "$i_{motor}$ RMSE per joint (GT baseline)",
    out_name: str = "torque_rmse_per_joint_grouped.png",
) -> str:
    os.makedirs(out_dir, exist_ok=True)

    tau_gt = np.asarray(tau_gt)
    tau_delan = np.asarray(tau_delan)
    tau_combined = np.asarray(tau_combined)
    n_dof = int(tau_gt.shape[1])

    K = int(tau_gt.shape[0])
    if max_samples is not None:
        K = min(int(max_samples), K)

    J = min(int(max_joints), n_dof)
    tau_gt = tau_gt[:K, :J]
    tau_delan = tau_delan[:K, :J]
    tau_combined = tau_combined[:K, :J]

    rmse_gt = np.zeros((J,), dtype=np.float32)
    rmse_delan = np.sqrt(np.mean((tau_delan - tau_gt) ** 2, axis=0))
    rmse_combined = np.sqrt(np.mean((tau_combined - tau_gt) ** 2, axis=0))

    x = np.arange(J)
    width = 0.25

    fig = plt.figure(figsize=(10, 4.5), dpi=140)
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(x - width, rmse_gt, width=width, label="GT", color="C2", alpha=0.75)
    ax.bar(x, rmse_delan, width=width, label="DeLaN", color="C1", alpha=0.9)
    ax.bar(x + width, rmse_combined, width=width, label="Combined", color="C0", alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("Joint")
    ax.set_ylabel("$i_{motor}$ RMSE [A]")
    ax.set_xticks(x)
    ax.set_xticklabels([f"joint{j}" for j in x])
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    out = os.path.join(out_dir, out_name)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def save_torque_rmse_time_curve(
    tau_gt: np.ndarray,
    tau_delan: np.ndarray,
    tau_combined: np.ndarray,
    out_dir: str,
    *,
    dt: float | None = None,
    max_samples: int | None = None,
    max_joints: int = 6,
    title: str = "$i_{motor}$ RMSE over time (joint-avg)",
    out_name: str = "torque_rmse_time.png",
) -> str:
    os.makedirs(out_dir, exist_ok=True)

    tau_gt = np.asarray(tau_gt)
    tau_delan = np.asarray(tau_delan)
    tau_combined = np.asarray(tau_combined)
    n_dof = int(tau_gt.shape[1])

    K = int(tau_gt.shape[0])
    if max_samples is not None:
        K = min(int(max_samples), K)

    J = min(int(max_joints), n_dof)
    tau_gt = tau_gt[:K, :J]
    tau_delan = tau_delan[:K, :J]
    tau_combined = tau_combined[:K, :J]

    rmse_t_delan = np.sqrt(np.mean((tau_delan - tau_gt) ** 2, axis=1))
    rmse_t_combined = np.sqrt(np.mean((tau_combined - tau_gt) ** 2, axis=1))

    x, x_label = _time_axis(rmse_t_delan.shape[0], dt)

    fig = plt.figure(figsize=(10.5, 4.2), dpi=140)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, rmse_t_delan, linewidth=1.4, label="DeLaN")
    ax.plot(x, rmse_t_combined, linewidth=1.4, label="Combined")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("$i_{motor}$ RMSE [A]")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    out = os.path.join(out_dir, out_name)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out
