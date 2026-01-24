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

    def finalize_train_metrics(self, *, epochs_ran: int) -> None:
        self.metrics["train"]["epochs_ran"] = int(epochs_ran)
        self.metrics["train"]["history_points"] = int(len(self.history.epoch))
        self.metrics["train"]["elbow_points"] = int(len(self.history.test_epoch))

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

        loss_path = self.plotter.save_loss_curve(self.history.epoch, self.history.loss)
        comp_path = self.plotter.save_loss_components(
            self.history.epoch,
            self.history.inverse,
            self.history.forward,
            self.history.energy,
        )
        elbow_path = self.plotter.save_elbow(
            self.history.test_epoch,
            self.history.test_mse,
            self.history.epoch,
            self.history.loss,
        )

        self.metrics["artifacts"]["train_history_csv"] = csv_path
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
