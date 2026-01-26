# payload_estimation/delan/src/delan_plots.py
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Optional, Sequence, Dict, Any

import numpy as np

@dataclass
class DelanPlotter:
    model_dir: str
    run_name: str
    plt: Any = None  # pass matplotlib.pyplot from caller (already backend-configured)

    def _enabled(self) -> bool:
        return self.plt is not None

    def _save_fig(self, fig, filename: str, dpi: int = 150) -> Optional[str]:
        if not self._enabled():
            return None
        os.makedirs(self.model_dir, exist_ok=True)
        path = os.path.join(self.model_dir, filename)
        fig.savefig(path, dpi=dpi)
        self.plt.close(fig)
        return path

    # --- keep names that your train script already calls ---

    def save_loss_curve(
        self,
        hist_epoch: Sequence[float],
        hist_loss: Sequence[float],
        *,
        prefix: Optional[str] = None,
    ) -> Optional[str]:
        if not self._enabled() or len(hist_epoch) == 0:
            return None
        pfx = prefix or self.run_name
        zoom_end = min(60, max(hist_epoch))

        fig = self.plt.figure(figsize=(12, 4), dpi=120)
        ax_full = fig.add_subplot(1, 2, 1)
        ax_zoom = fig.add_subplot(1, 2, 2)

        ax_full.plot(hist_epoch, hist_loss)
        ax_full.set_title(f"{self.run_name} | Training Loss (full)")
        ax_full.set_xlabel("Epoch")
        ax_full.set_ylabel("Loss")
        ax_full.set_yscale("log")
        ax_full.grid(True, alpha=0.25)

        ax_zoom.plot(hist_epoch, hist_loss)
        ax_zoom.set_title(f"{self.run_name} | Training Loss (0-{int(zoom_end)})")
        ax_zoom.set_xlabel("Epoch")
        ax_zoom.set_ylabel("Loss")
        ax_zoom.set_xlim(0, zoom_end)
        ax_zoom.grid(True, alpha=0.25)

        self.plt.tight_layout()
        return self._save_fig(fig, f"{pfx}__loss_curve.png")

    def save_loss_components(
        self,
        hist_epoch: Sequence[float],
        hist_inv: Sequence[float],
        hist_for: Sequence[float],
        hist_energy: Sequence[float],
        *,
        prefix: Optional[str] = None,
    ) -> Optional[str]:
        if not self._enabled() or len(hist_epoch) == 0:
            return None
        pfx = prefix or self.run_name

        fig = self.plt.figure(figsize=(8, 4), dpi=120)
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(hist_epoch, hist_inv, label="inverse_mean")
        ax.plot(hist_epoch, hist_for, label="forward_mean")
        ax.plot(hist_epoch, hist_energy, label="energy_mean")
        ax.set_title(f"{self.run_name} | Loss Components")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.set_ylim(0, 1000)
        ax.grid(True, alpha=0.25)
        ax.legend()
        self.plt.tight_layout()
        return self._save_fig(fig, f"{pfx}__loss_components.png")

    def save_elbow(
        self,
        hist_test_epoch: Sequence[float],
        hist_test_mse: Sequence[float],
        hist_epoch: Sequence[float],
        hist_loss: Sequence[float],
        *,
        prefix: Optional[str] = None,
    ) -> Optional[str]:
        if (not self._enabled()) or len(hist_test_epoch) == 0 or len(hist_epoch) == 0:
            return None
        pfx = prefix or self.run_name
        zoom_end = min(60, max(hist_epoch))

        fig = self.plt.figure(figsize=(12, 4), dpi=120)
        ax_full = fig.add_subplot(1, 2, 1)
        ax_zoom = fig.add_subplot(1, 2, 2)

        # Full range (log scale)
        ax_full.plot(hist_epoch, hist_loss, label="train_loss")
        ax_full.set_xlabel("Epoch")
        ax_full.set_ylabel("Train loss")
        ax_full.set_yscale("log")
        ax_full.tick_params(axis="x", colors="black", labelcolor="black")
        ax_full.xaxis.label.set_color("black")
        ax_full.spines["bottom"].set_color("black")
        ax_full.grid(True, alpha=0.25)

        ax_full_t = ax_full.twinx()
        ax_full_t.plot(hist_test_epoch, hist_test_mse, label="val_mse", color="C1")
        ax_full_t.set_ylabel("Val torque MSE (mean over joints)", color="black")
        ax_full_t.set_yscale("log")
        ax_full_t.tick_params(axis="y", labelcolor="black", color="black")
        ax_full_t.tick_params(axis="x", bottom=False, labelbottom=False)
        ax_full_t.xaxis.set_visible(False)
        ax_full_t.spines["bottom"].set_visible(False)
        ax_full_t.spines["right"].set_color("black")

        l1, lab1 = ax_full.get_legend_handles_labels()
        l2, lab2 = ax_full_t.get_legend_handles_labels()
        ax_full.legend(l1 + l2, lab1 + lab2, loc="best")
        ax_full.set_title(f"{self.run_name} | Elbow (full)")

        # Zoomed range (linear scale)
        ax_zoom.plot(hist_epoch, hist_loss, label="train_loss")
        ax_zoom.set_xlabel("Epoch")
        ax_zoom.set_ylabel("Train loss")
        ax_zoom.set_xlim(0, zoom_end)
        ax_zoom.tick_params(axis="x", colors="black", labelcolor="black")
        ax_zoom.xaxis.label.set_color("black")
        ax_zoom.spines["bottom"].set_color("black")
        ax_zoom.grid(True, alpha=0.25)

        ax_zoom_t = ax_zoom.twinx()
        ax_zoom_t.plot(hist_test_epoch, hist_test_mse, label="val_mse", color="C1")
        ax_zoom_t.set_ylabel("Val torque MSE (mean over joints)", color="black")
        ax_zoom_t.tick_params(axis="y", labelcolor="black", color="black")
        ax_zoom_t.tick_params(axis="x", bottom=False, labelbottom=False)
        ax_zoom_t.xaxis.set_visible(False)
        ax_zoom_t.spines["bottom"].set_visible(False)
        ax_zoom_t.spines["right"].set_color("black")

        l1, lab1 = ax_zoom.get_legend_handles_labels()
        l2, lab2 = ax_zoom_t.get_legend_handles_labels()
        ax_zoom.legend(l1 + l2, lab1 + lab2, loc="best")
        ax_zoom.set_title(f"{self.run_name} | Elbow (0-{int(zoom_end)})")

        self.plt.tight_layout()
        return self._save_fig(fig, f"{pfx}__elbow_train_vs_test.png")

    def save_torque_overlay(
        self,
        tau_gt: np.ndarray,
        tau_pred: np.ndarray,
        *,
        prefix: Optional[str] = None,
        max_samples: int = 3000,
        max_joints: int = 6,
    ) -> Optional[str]:
        if not self._enabled():
            return None

        pfx = prefix or self.run_name
        tau_gt = np.asarray(tau_gt)
        tau_pred = np.asarray(tau_pred)
        n_dof = tau_gt.shape[1]

        K = min(int(max_samples), tau_gt.shape[0])
        tau_gt = tau_gt[:K]
        tau_pred = tau_pred[:K]

        fig = self.plt.figure(figsize=(14, 8), dpi=100)
        for j in range(min(n_dof, max_joints)):
            ax = fig.add_subplot(3, 2, j + 1)
            ax.set_title(f"Joint {j}")
            ax.plot(tau_gt[:, j], label="GT", linewidth=1.0)
            ax.plot(tau_pred[:, j], label="DeLaN", linewidth=1.0, alpha=0.85)
            ax.grid(True, alpha=0.2)
            if j == 0:
                ax.legend()

        self.plt.tight_layout()

        # IMPORTANT: do NOT append model_choice/seed if already encoded in run_name
        return self._save_fig(fig, f"{pfx}__DeLaN_Torque.png")
    
    def save_torque_plot(
        self,
        tau_gt: np.ndarray,
        tau_pred: np.ndarray,
        *,
        model_choice: str | None = None,
        seed: int | None = None,
        prefix: Optional[str] = None,
        max_samples: int = 3000,
        max_joints: int = 6,
    ) -> Optional[str]:
        # model_choice/seed are accepted for compatibility,
        # but we don't append them since run_name already encodes it.
        return self.save_torque_overlay(
            tau_gt=tau_gt,
            tau_pred=tau_pred,
            prefix=prefix,
            max_samples=max_samples,
            max_joints=max_joints,
        )

    def save_torque_rmse_time_curve(
        self,
        tau_gt: np.ndarray,
        tau_pred: np.ndarray,
        *,
        dt: float | None = None,
        prefix: Optional[str] = None,
        max_samples: int | None = None,
        max_joints: int = 6,
    ) -> Optional[str]:
        if not self._enabled():
            return None

        pfx = prefix or self.run_name
        tau_gt = np.asarray(tau_gt)
        tau_pred = np.asarray(tau_pred)
        n_dof = tau_gt.shape[1]

        K = tau_gt.shape[0]
        if max_samples is not None:
            K = min(int(max_samples), K)
        tau_gt = tau_gt[:K, : min(n_dof, max_joints)]
        tau_pred = tau_pred[:K, : min(n_dof, max_joints)]

        err = tau_pred - tau_gt
        rmse_t = np.sqrt(np.mean(err ** 2, axis=1))

        if dt is not None and dt > 0:
            x = np.arange(rmse_t.shape[0]) * float(dt)
            x_label = "Time [s]"
        else:
            x = np.arange(rmse_t.shape[0])
            x_label = "Sample"

        fig = self.plt.figure(figsize=(10, 4), dpi=120)
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(x, rmse_t, linewidth=1.2)
        ax.set_title(f"{self.run_name} | Torque RMSE (joints 0-{min(n_dof, max_joints)-1})")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Torque RMSE")
        ax.grid(True, alpha=0.25)
        self.plt.tight_layout()
        return self._save_fig(fig, f"{pfx}__DeLaN_Torque_RMSE_Time.png")

    def save_torque_rmse_per_joint_bar(
        self,
        tau_gt: np.ndarray,
        tau_pred: np.ndarray,
        *,
        prefix: Optional[str] = None,
        max_samples: int | None = None,
        max_joints: int = 6,
    ) -> Optional[str]:
        if not self._enabled():
            return None

        pfx = prefix or self.run_name
        tau_gt = np.asarray(tau_gt)
        tau_pred = np.asarray(tau_pred)
        n_dof = tau_gt.shape[1]

        K = tau_gt.shape[0]
        if max_samples is not None:
            K = min(int(max_samples), K)
        tau_gt = tau_gt[:K, : min(n_dof, max_joints)]
        tau_pred = tau_pred[:K, : min(n_dof, max_joints)]

        err = tau_pred - tau_gt
        rmse_per_joint = np.sqrt(np.mean(err ** 2, axis=0))

        fig = self.plt.figure(figsize=(8, 4), dpi=120)
        ax = fig.add_subplot(1, 1, 1)
        joints = np.arange(rmse_per_joint.shape[0])
        ax.bar(joints, rmse_per_joint)
        ax.set_title(f"{self.run_name} | Torque RMSE per joint")
        ax.set_xlabel("Joint")
        ax.set_ylabel("Torque RMSE")
        ax.set_xticks(joints)
        ax.set_xticklabels([f"joint{j}" for j in joints])
        ax.grid(True, axis="y", alpha=0.25)
        self.plt.tight_layout()
        return self._save_fig(fig, f"{pfx}__DeLaN_Torque_RMSE_per_joint.png")

    def save_metrics_txt(self, lines: Sequence[str], filename: str = "metrics_test.txt") -> str:
        os.makedirs(self.model_dir, exist_ok=True)
        path = os.path.join(self.model_dir, filename)
        with open(path, "w") as f:
            for ln in lines:
                f.write(ln.rstrip("\n") + "\n")
        return path

    def save_metrics_json(self, metrics: Dict[str, Any], filename: str = "metrics.json") -> str:
        os.makedirs(self.model_dir, exist_ok=True)
        path = os.path.join(self.model_dir, filename)
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)
        return path
