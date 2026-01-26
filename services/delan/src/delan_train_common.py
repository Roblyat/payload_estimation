from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from delan_plots import DelanPlotter


@dataclass
class DelanRunPaths:
    run_name: str
    model_dir: str
    ckpt_path: str

    @staticmethod
    def from_save_path(save_path: str) -> "DelanRunPaths":
        if not save_path:
            raise ValueError("save_path must be a non-empty string.")

        model_stem = os.path.splitext(os.path.basename(save_path))[0]
        base_save_dir = os.path.dirname(save_path) or "."

        # Avoid run_name/run_name nesting if save_path already points inside it.
        if os.path.basename(base_save_dir) == model_stem:
            model_dir = base_save_dir
            ckpt_path = save_path
        else:
            model_dir = os.path.join(base_save_dir, model_stem)
            ckpt_path = os.path.join(model_dir, os.path.basename(save_path))

        os.makedirs(model_dir, exist_ok=True)
        return DelanRunPaths(run_name=model_stem, model_dir=model_dir, ckpt_path=ckpt_path)


@dataclass
class DelanTrainHistory:
    epoch: List[int] = field(default_factory=list)
    time_s: List[float] = field(default_factory=list)
    loss: List[float] = field(default_factory=list)
    inverse: List[float] = field(default_factory=list)
    forward: List[float] = field(default_factory=list)
    energy: List[float] = field(default_factory=list)
    test_epoch: List[int] = field(default_factory=list)
    test_mse: List[float] = field(default_factory=list)

    def record_train(
        self,
        epoch: int,
        time_s: float,
        loss: float,
        inverse_mean: float,
        forward_mean: float,
        energy_mean: float,
    ) -> None:
        self.epoch.append(int(epoch))
        self.time_s.append(float(time_s))
        self.loss.append(float(loss))
        self.inverse.append(float(inverse_mean))
        self.forward.append(float(forward_mean))
        self.energy.append(float(energy_mean))

    def record_test(self, epoch: int, mse: float) -> None:
        self.test_epoch.append(int(epoch))
        self.test_mse.append(float(mse))


@dataclass(frozen=True)
class EarlyStopConfig:
    enabled: bool = False
    patience: int = 10
    min_delta: float = 0.0
    warmup_evals: int = 0
    mode: str = "min"


class EarlyStopper:
    """
    Simple framework-agnostic early stopping helper.

    Notes:
      - patience is counted in "evaluation events" (i.e., how often you call update),
        not in epochs.
      - mode='min' means lower metric is better.
    """

    def __init__(self, cfg: EarlyStopConfig) -> None:
        if cfg.mode not in ("min",):
            raise ValueError(f"Unsupported early stop mode: {cfg.mode}")
        if cfg.patience < 0:
            raise ValueError("patience must be >= 0")
        if cfg.warmup_evals < 0:
            raise ValueError("warmup_evals must be >= 0")

        self.cfg = cfg
        self.num_evals: int = 0
        self.bad_evals: int = 0
        self.best_metric: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self.stopped: bool = False
        self.stop_epoch: Optional[int] = None

    def update(self, *, metric: float, epoch: int) -> Tuple[bool, bool]:
        """
        Returns: (should_stop, improved)
        """
        self.num_evals += 1

        m = float(metric)
        e = int(epoch)

        improved = False
        if self.best_metric is None:
            improved = True
        else:
            improved = m < (float(self.best_metric) - float(self.cfg.min_delta))

        if improved:
            self.best_metric = m
            self.best_epoch = e
            self.bad_evals = 0
            return False, True

        if self.num_evals <= int(self.cfg.warmup_evals):
            return False, False

        self.bad_evals += 1
        if self.cfg.enabled and self.bad_evals >= int(self.cfg.patience):
            self.stopped = True
            self.stop_epoch = e
            return True, False

        return False, False

    def to_dict(
        self,
        *,
        monitor: str,
        restored_best: bool,
        monitor_split: str,
    ) -> Dict[str, Any]:
        return {
            "enabled": bool(self.cfg.enabled),
            "monitor": str(monitor),
            "monitor_split": str(monitor_split),
            "mode": str(self.cfg.mode),
            "patience": int(self.cfg.patience),
            "min_delta": float(self.cfg.min_delta),
            "warmup_evals": int(self.cfg.warmup_evals),
            "num_evals": int(self.num_evals),
            "bad_evals": int(self.bad_evals),
            "best_metric": None if self.best_metric is None else float(self.best_metric),
            "best_epoch": self.best_epoch,
            "stopped": bool(self.stopped),
            "stop_epoch": self.stop_epoch,
            "restored_best": bool(restored_best),
        }


def compute_tau_metrics(
    tau_true: np.ndarray,
    tau_pred: np.ndarray,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    err = np.asarray(tau_pred) - np.asarray(tau_true)
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mse_per_joint = np.mean(err ** 2, axis=0)
    rmse_per_joint = np.sqrt(mse_per_joint)
    return mse, rmse, mse_per_joint, rmse_per_joint


class DelanTrainRun:
    def __init__(
        self,
        run_paths: DelanRunPaths,
        *,
        model_choice: str,
        hp_preset: str,
        npz_path: str,
        hyper_json: Dict[str, Any],
        args_summary: Dict[str, Any],
        seed: int,
        plt: Any = None,
    ) -> None:
        self.run_paths = run_paths
        self.run_name = run_paths.run_name
        self.model_choice = model_choice
        self.plotter = DelanPlotter(model_dir=run_paths.model_dir, run_name=run_paths.run_name, plt=plt)
        self.history = DelanTrainHistory()

        self.metrics: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "run_name": self.run_name,
            "model_type": model_choice,
            "hp_preset": hp_preset,
            "npz": npz_path,
            "model_dir": run_paths.model_dir,
            "ckpt_path": run_paths.ckpt_path,
            "seed": int(seed),
            "dt": None,
            "n_dof": None,
            "hyper": hyper_json,
            "args": args_summary,
            "dataset": {},
            "train": {},
            "eval_test": {},
            "artifacts": {},
        }

    def update_dataset_info(
        self,
        *,
        n_dof: int,
        dt: float,
        train_labels: Sequence[Any],
        test_labels: Sequence[Any],
        train_samples: int,
        test_samples: int,
        dataset_name: str,
    ) -> None:
        self.metrics["n_dof"] = int(n_dof)
        self.metrics["dt"] = float(dt)
        self.metrics["dataset"] = {
            "npz": self.metrics["npz"],
            "dataset_name": dataset_name,
            "dt": float(dt),
            "dof": int(n_dof),
            "train_trajectories": int(len(train_labels)),
            "test_trajectories": int(len(test_labels)),
            "train_samples": int(train_samples),
            "test_samples": int(test_samples),
        }

    def record_train_point(
        self,
        *,
        epoch: int,
        time_s: float,
        loss: float,
        inverse_mean: float,
        forward_mean: float,
        energy_mean: float,
    ) -> None:
        self.history.record_train(epoch, time_s, loss, inverse_mean, forward_mean, energy_mean)

    def record_test_point(self, *, epoch: int, mse: float) -> None:
        self.history.record_test(epoch, mse)

    def finalize_train_metrics(self, *, epochs_ran: int, early_stop: Optional[Dict[str, Any]] = None) -> None:
        self.metrics["train"]["epochs_ran"] = int(epochs_ran)
        self.metrics["train"]["history_points"] = int(len(self.history.epoch))
        self.metrics["train"]["elbow_points"] = int(len(self.history.test_epoch))
        if early_stop is not None:
            self.metrics["train"]["early_stop"] = dict(early_stop)

    def save_training_artifacts(self, *, save_model: bool) -> None:
        if not save_model:
            return

        csv_path = os.path.join(self.run_paths.model_dir, f"{self.run_name}__train_history.csv")
        with open(csv_path, "w") as f:
            f.write("epoch,time_s,loss,inverse_mean,forward_mean,energy_mean\n")
            for e, ts, lo, inv, fo, en in zip(
                self.history.epoch,
                self.history.time_s,
                self.history.loss,
                self.history.inverse,
                self.history.forward,
                self.history.energy,
            ):
                f.write(f"{e},{ts},{lo},{inv},{fo},{en}\n")

        elbow_mse = list(self.history.test_mse)
        n_dof = self.metrics.get("n_dof", None)
        if n_dof:
            elbow_mse = [float(x) / float(n_dof) for x in elbow_mse]
            self.metrics["train"]["elbow_mse_scaled_by_n_dof"] = True
            self.metrics["train"]["elbow_mse_scale"] = float(n_dof)

        if len(self.history.test_epoch) > 0 and len(self.history.epoch) > 0:
            test_epoch = np.asarray(self.history.test_epoch, dtype=float)
            train_epoch = np.asarray(self.history.epoch, dtype=float)
            train_loss = np.asarray(self.history.loss, dtype=float)
            val_mse = np.asarray(elbow_mse, dtype=float)
            interp_loss = np.interp(test_epoch, train_epoch, train_loss)
            eps = 1e-12
            rel_delta = np.abs(interp_loss - val_mse) / (np.abs(val_mse) + eps)
            threshold = 0.02
            exceeds = rel_delta > threshold
            self.metrics["train"]["loss_vs_val_mse_delta"] = {
                "rel_delta_threshold": float(threshold),
                "rel_delta_exceeds_count": int(np.sum(exceeds)),
                "rel_delta_max": float(np.max(rel_delta)),
                "rel_delta_mean": float(np.mean(rel_delta)),
                "rel_delta_exceeds_epochs": [int(e) for e in test_epoch[exceeds].tolist()],
            }

        elbow_csv_path = os.path.join(self.run_paths.model_dir, f"{self.run_name}__elbow_history.csv")
        with open(elbow_csv_path, "w") as f:
            f.write("epoch,eval_mse\n")
            for e, mse in zip(self.history.test_epoch, elbow_mse):
                f.write(f"{e},{mse}\n")

        loss_path = self.plotter.save_loss_curve(self.history.epoch, self.history.loss)
        comp_path = self.plotter.save_loss_components(
            self.history.epoch,
            self.history.inverse,
            self.history.forward,
            self.history.energy,
        )
        elbow_path = self.plotter.save_elbow(
            self.history.test_epoch,
            elbow_mse,
            self.history.epoch,
            self.history.loss,
        )

        self.metrics["artifacts"]["train_history_csv"] = csv_path
        self.metrics["artifacts"]["elbow_history_csv"] = elbow_csv_path
        if loss_path is not None:
            self.metrics["artifacts"]["loss_curve_png"] = loss_path
        if comp_path is not None:
            self.metrics["artifacts"]["loss_components_png"] = comp_path
        if elbow_path is not None:
            self.metrics["artifacts"]["elbow_png"] = elbow_path

    def save_eval_metrics(
        self,
        *,
        tau_true: np.ndarray,
        tau_pred: np.ndarray,
        t_eval: float,
        hyper: Dict[str, Any],
        dt: float,
        n_dof: int,
        seed: int,
    ) -> Dict[str, Any]:
        mse, rmse, mse_per_joint, rmse_per_joint = compute_tau_metrics(tau_true, tau_pred)

        self.metrics["eval_test"] = {
            "torque_mse": float(mse),
            "torque_rmse": float(rmse),
            "torque_mse_per_joint": [float(x) for x in mse_per_joint],
            "torque_rmse_per_joint": [float(x) for x in rmse_per_joint],
            "time_per_sample_s": float(t_eval),
            "hz": float(1.0 / t_eval) if t_eval > 0 else None,
        }

        torque_plot_path = self.plotter.save_torque_plot(
            tau_gt=np.array(tau_true),
            tau_pred=np.array(tau_pred),
            model_choice=self.model_choice,
            seed=int(seed),
            max_samples=3000,
        )
        if torque_plot_path is not None:
            self.metrics["artifacts"]["torque_plot_png"] = torque_plot_path

        err = np.asarray(tau_pred) - np.asarray(tau_true)
        rmse_time = np.sqrt(np.mean(err ** 2, axis=1))
        rmse_time_npy = os.path.join(
            self.run_paths.model_dir,
            f"{self.run_name}__torque_rmse_time.npy",
        )
        np.save(rmse_time_npy, rmse_time.astype(np.float32))
        self.metrics["artifacts"]["torque_rmse_time_npy"] = rmse_time_npy

        rmse_time_path = self.plotter.save_torque_rmse_time_curve(
            tau_gt=np.array(tau_true),
            tau_pred=np.array(tau_pred),
            dt=float(dt) if dt is not None else None,
            max_samples=None,
        )
        if rmse_time_path is not None:
            self.metrics["artifacts"]["torque_rmse_time_png"] = rmse_time_path

        rmse_per_joint_path = self.plotter.save_torque_rmse_per_joint_bar(
            tau_gt=np.array(tau_true),
            tau_pred=np.array(tau_pred),
            max_samples=None,
        )
        if rmse_per_joint_path is not None:
            self.metrics["artifacts"]["torque_rmse_per_joint_png"] = rmse_per_joint_path

        metrics_path = os.path.join(self.run_paths.model_dir, "metrics_test.txt")
        with open(metrics_path, "w") as f:
            f.write(f"run_name={self.run_name}\n")
            f.write(f"hp_preset={self.metrics['hp_preset']}\n")
            f.write(f"hyper={hyper}\n")
            f.write(f"npz={self.metrics['npz']}\n")
            f.write(f"ckpt={self.run_paths.ckpt_path}\n")
            f.write(f"seed={seed}\n")
            f.write(f"model_type={self.model_choice}\n")
            f.write(f"dt={dt}\n")
            f.write(f"n_dof={n_dof}\n")
            f.write(f"torque_mse={mse}\n")
            f.write(f"torque_rmse={rmse}\n")
            f.write("torque_mse_per_joint=" + " ".join([str(float(x)) for x in mse_per_joint]) + "\n")
            f.write("torque_rmse_per_joint=" + " ".join([str(float(x)) for x in rmse_per_joint]) + "\n")
            f.write(f"time_per_sample_s={t_eval}\n")
            f.write(f"hz={1.0/t_eval if t_eval > 0 else None}\n")

        metrics_json_path = os.path.join(self.run_paths.model_dir, "metrics.json")
        with open(metrics_json_path, "w") as f:
            json.dump(self.metrics, f, indent=2)

        self.metrics["artifacts"]["metrics_txt"] = metrics_path
        self.metrics["artifacts"]["metrics_json"] = metrics_json_path

        return {
            "torque_mse": float(mse),
            "torque_rmse": float(rmse),
            "torque_mse_per_joint": mse_per_joint,
            "torque_rmse_per_joint": rmse_per_joint,
        }
