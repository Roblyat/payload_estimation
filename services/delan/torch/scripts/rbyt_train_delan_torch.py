import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Matplotlib must be configured AFTER args are parsed (render vs headless).
plt = None


def _init_matplotlib(render: bool):
    """Return pyplot (plt) or None. Uses Qt5Agg only when render=True, else Agg."""
    try:
        import matplotlib as mp

        backend = "Qt5Agg" if render else "Agg"
        mp.use(backend, force=True)
        import matplotlib.pyplot as _plt

        return _plt
    except Exception as e:
        print(f"[warn] Matplotlib disabled: {e}")
        return None


def _add_path(path: str) -> None:
    if path and path not in sys.path:
        sys.path.insert(0, path)


def _resolve_npz_path(p: str) -> str:
    """
    Accept both:
      - /root/stem/stem.npz (new)
      - /root/stem.npz      (old)
    If the given path does not exist, try the other convention.
    """
    p = os.path.expanduser(str(p))
    if p.endswith(".npz") and (not os.path.exists(p)):
        stem = Path(p).stem
        root = os.path.dirname(p)
        if os.path.basename(root) == stem:
            candidate = os.path.join(os.path.dirname(root), f"{stem}.npz")
            if os.path.exists(candidate):
                return candidate
        else:
            candidate = os.path.join(root, stem, f"{stem}.npz")
            if os.path.exists(candidate):
                return candidate
    return p


DELAN_REPO_DIR = os.environ.get("DELAN_REPO_DIR", "/workspace/delan_repo")
DELAN_COMMON_SRC = "/workspace/delan_src"
DELAN_TORCH_SRC = "/workspace/delan_torch/src"

for p in [DELAN_COMMON_SRC, DELAN_TORCH_SRC, DELAN_REPO_DIR]:
    _add_path(p)

from delan_train_common import DelanRunPaths, DelanTrainRun
from load_npz_dataset import load_npz_trajectory_dataset

from deep_lagrangian_networks.DeLaN_model import DeepLagrangianNetwork
from deep_lagrangian_networks.replay_memory import PyTorchReplayMemory
from deep_lagrangian_networks.utils import init_env


def _map_activation(name: str) -> str:
    if name is None:
        raise ValueError("Activation must be provided for torch training.")
    key = name.lower()
    mapping = {
        "relu": "ReLu",
        "softplus": "SoftPlus",
    }
    if key not in mapping:
        raise ValueError(f"Torch DeLaN_model supports activation=relu|softplus, got '{name}'.")
    return mapping[key]


class TorchDelanTrainer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.render_flag = bool(args.r[0])
        self.plt = _init_matplotlib(self.render_flag)

        self.seed, self.cuda, self.render, self.load_model, self.save_model = init_env(args)
        self.device = torch.device("cuda" if self.cuda else "cpu")

        self.model_choice = str(args.t[0])
        if self.model_choice != "structured":
            raise ValueError("Torch DeLaN_model only supports -t structured.")

        self.run_paths = DelanRunPaths.from_save_path(args.save_path)
        self.run_name = self.run_paths.run_name

        self.hyper = self._build_hyper()
        self.hyper_json = dict(self.hyper)

        self.saved_state = None
        if self.load_model:
            self._load_checkpoint()

        print(f"Final hyper: {self.hyper}")

        args_summary = {
            "render": int(args.r[0]),
            "eval_every": int(args.eval_every),
            "eval_n": int(args.eval_n),
            "loss_norm": args.loss_norm,
        }
        self.train_run = DelanTrainRun(
            run_paths=self.run_paths,
            model_choice=self.model_choice,
            hp_preset=args.hp_preset,
            npz_path=args.npz,
            hyper_json=self.hyper_json,
            args_summary=args_summary,
            seed=self.seed,
            plt=self.plt,
        )

    def _build_hyper(self) -> dict:
        hyper = {
            "dataset": self.run_name,
            "n_width": 64,
            "n_depth": 2,
            "n_minibatch": 512,
            "diagonal_epsilon": 0.1,
            "diagonal_shift": 2.0,
            "activation": "softplus",
            "learning_rate": 1.0e-4,
            "weight_decay": 1.0e-5,
            "max_epoch": int(2.0 * 1e3),
            "lagrangian_type": "structured",
        }

        presets = {
            "default": {},
            "fast_debug": {
                "max_epoch": 300,
                "n_minibatch": 256,
                "n_width": 64,
                "n_depth": 2,
                "learning_rate": 3e-4,
            },
            "long_train": {
                "max_epoch": 8000,
                "n_minibatch": 512,
                "n_width": 128,
                "n_depth": 3,
                "learning_rate": 1e-4,
                "weight_decay": 1e-5,
            },
        }
        hyper.update(presets.get(self.args.hp_preset, {}))

        if self.args.n_width is not None:
            hyper["n_width"] = self.args.n_width
        if self.args.n_depth is not None:
            hyper["n_depth"] = self.args.n_depth
        if self.args.batch is not None:
            hyper["n_minibatch"] = self.args.batch
        if self.args.lr is not None:
            hyper["learning_rate"] = self.args.lr
        if self.args.wd is not None:
            hyper["weight_decay"] = self.args.wd
        if self.args.epochs is not None:
            hyper["max_epoch"] = int(self.args.epochs)
        if self.args.diag_eps is not None:
            hyper["diagonal_epsilon"] = self.args.diag_eps
        if self.args.diag_shift is not None:
            hyper["diagonal_shift"] = self.args.diag_shift
        if self.args.activation is not None:
            hyper["activation"] = self.args.activation

        return hyper

    def _load_checkpoint(self) -> None:
        self.saved_state = torch.load(self.run_paths.ckpt_path, map_location=self.device)
        self.hyper = self.saved_state.get("hyper", self.hyper)
        self.hyper_json = dict(self.hyper)
        print(f"Loaded DeLaN checkpoint: {self.run_paths.ckpt_path}")

    def _build_model(self, n_dof: int) -> DeepLagrangianNetwork:
        hyper_model = dict(self.hyper)
        hyper_model["activation"] = _map_activation(hyper_model.get("activation", "softplus"))
        model = DeepLagrangianNetwork(n_dof, **hyper_model)
        model = model.cuda() if self.cuda else model.cpu()
        if self.saved_state is not None:
            model.load_state_dict(self.saved_state["state_dict"])
        return model

    def run(self) -> None:
        args = self.args
        print("\n\n################################################")
        print(f"DeLaN run: {self.run_name}")
        print(f"  model_dir = {self.run_paths.model_dir}")
        print(f"  ckpt_path = {self.run_paths.ckpt_path}")
        print(f"  type = {self.model_choice}")
        print(f"  hp_preset = {args.hp_preset}")
        print("################################################")

        train_data, test_data, divider, dt = load_npz_trajectory_dataset(args.npz)
        train_labels, train_qp, train_qv, train_qa, train_tau = train_data
        test_labels, test_qp, test_qv, test_qa, test_tau = test_data

        n_dof = train_qp.shape[-1]
        npz_name = os.path.basename(args.npz)

        print("\n\n################################################")
        print(f"Dataset: {npz_name}")
        print(f"  npz = {args.npz}")
        print(f"   dt ≈ {dt}")
        print(f"  dof = {n_dof}")
        print(f"  Train trajectories = {len(train_labels)}")
        print(f"  Test trajectories  = {len(test_labels)}")
        print(f"  Train samples = {train_qp.shape[0]}")
        print(f"  Test samples  = {test_qp.shape[0]}")
        print("################################################\n")

        self.train_run.update_dataset_info(
            n_dof=int(n_dof),
            dt=float(dt),
            train_labels=train_labels,
            test_labels=test_labels,
            train_samples=int(train_qp.shape[0]),
            test_samples=int(test_qp.shape[0]),
            dataset_name=npz_name,
        )

        delan_model = self._build_model(n_dof)
        delan_model.train()

        optimizer = torch.optim.AdamW(
            delan_model.parameters(),
            lr=self.hyper["learning_rate"],
            weight_decay=self.hyper["weight_decay"],
        )

        mem_dim = ((n_dof,), (n_dof,), (n_dof,), (n_dof,))
        mem = PyTorchReplayMemory(train_qp.shape[0], self.hyper["n_minibatch"], mem_dim, self.cuda)
        mem.add_samples([train_qp, train_qv, train_qa, train_tau])

        use_norm = args.loss_norm == "per_joint"
        if use_norm:
            norm_tau = torch.from_numpy(np.var(train_tau, axis=0)).float().to(delan_model.device)
            norm_qdd = torch.from_numpy(np.var(train_qa, axis=0)).float().to(delan_model.device)
        else:
            norm_tau = None
            norm_qdd = None

        print("################################################")
        print(f"Training DeLaN | run={self.run_name} | type={self.model_choice} | dof={n_dof}")
        print(f"Loss normalization = {args.loss_norm}")
        print("################################################")

        t0_start = time.perf_counter()
        epoch_i = 0
        while epoch_i < self.hyper["max_epoch"] and not self.load_model:
            n_batches = 0
            loss_sum = 0.0
            inv_sum = 0.0
            for_sum = 0.0
            energy_sum = 0.0

            for q, qd, qdd, tau in mem:
                optimizer.zero_grad()

                tau_hat, dEdt_hat = delan_model(q, qd, qdd)
                qdd_pred = delan_model.for_dyn(q, qd, tau)

                if use_norm:
                    tau_error = torch.sum((tau - tau_hat) ** 2 / norm_tau, dim=1)
                    qdd_error = torch.sum((qdd - qdd_pred) ** 2 / norm_qdd, dim=1)
                else:
                    tau_error = torch.sum((tau - tau_hat) ** 2, dim=1)
                    qdd_error = torch.sum((qdd - qdd_pred) ** 2, dim=1)
                dEdt_true = torch.sum(qd * tau, dim=1)
                dEdt_error = (dEdt_hat - dEdt_true) ** 2

                inverse_mean = torch.mean(tau_error)
                forward_mean = torch.mean(qdd_error)
                energy_mean = torch.mean(dEdt_error)

                loss = inverse_mean
                loss.backward()
                optimizer.step()

                n_batches += 1
                loss_sum += float(loss.item())
                inv_sum += float(inverse_mean.item())
                for_sum += float(forward_mean.item())
                energy_sum += float(energy_mean.item())

            epoch_i += 1
            loss_mean = loss_sum / n_batches
            inv_mean = inv_sum / n_batches
            for_mean = for_sum / n_batches
            energy_mean = energy_sum / n_batches

            if epoch_i == 1 or np.mod(epoch_i, 50) == 0:
                print(
                    f"Epoch {epoch_i:05d}: "
                    f"Time={time.perf_counter()-t0_start:6.1f}s, "
                    f"Loss={loss_mean:.2e}, "
                    f"Inv={inv_mean:.2e}, "
                    f"For={for_mean:.2e}, "
                    f"Power={energy_mean:.2e}"
                )
                self.train_run.record_train_point(
                    epoch=epoch_i,
                    time_s=time.perf_counter() - t0_start,
                    loss=loss_mean,
                    inverse_mean=inv_mean,
                    forward_mean=for_mean,
                    energy_mean=energy_mean,
                )

            if args.eval_every > 0 and (epoch_i == 1 or (epoch_i % args.eval_every) == 0):
                q_eval = test_qp
                qd_eval = test_qv
                qdd_eval = test_qa
                tau_eval = test_tau

                if args.eval_n and args.eval_n > 0:
                    n = min(int(args.eval_n), q_eval.shape[0])
                    q_eval = q_eval[:n]
                    qd_eval = qd_eval[:n]
                    qdd_eval = qdd_eval[:n]
                    tau_eval = tau_eval[:n]

                with torch.no_grad():
                    qj = torch.from_numpy(q_eval).float().to(delan_model.device)
                    qdj = torch.from_numpy(qd_eval).float().to(delan_model.device)
                    qddj = torch.from_numpy(qdd_eval).float().to(delan_model.device)
                    tauj = torch.from_numpy(tau_eval).float().to(delan_model.device)
                    pred_tau_eval = delan_model.inv_dyn(qj, qdj, qddj)
                    test_mse = float(torch.sum((pred_tau_eval - tauj) ** 2) / qj.shape[0])

                self.train_run.record_test_point(epoch=epoch_i, mse=test_mse)
                print(f"  [eval] test_mse={test_mse:.3e}  (n={qj.shape[0]})")

        if self.save_model:
            torch.save(
                {
                    "epoch": epoch_i,
                    "hyper": self.hyper,
                    "state_dict": delan_model.state_dict(),
                    "seed": self.seed,
                },
                self.run_paths.ckpt_path,
            )
            print(f"Saved DeLaN checkpoint: {self.run_paths.ckpt_path}")

        self.train_run.finalize_train_metrics(epochs_ran=epoch_i)
        self.train_run.save_training_artifacts(save_model=self.save_model)

        train_csv = self.train_run.metrics["artifacts"].get("train_history_csv")
        if train_csv:
            print(f"Saved training history: {train_csv}")

        print("\n################################################")
        print(f"Evaluating DeLaN | run={self.run_name}")

        delan_model.eval()
        with torch.no_grad():
            q = torch.from_numpy(test_qp).float().to(delan_model.device)
            qd = torch.from_numpy(test_qv).float().to(delan_model.device)
            qdd = torch.from_numpy(test_qa).float().to(delan_model.device)

            if self.cuda:
                torch.cuda.synchronize()
            t0_eval = time.perf_counter()
            pred_tau = delan_model.inv_dyn(q, qd, qdd)
            if self.cuda:
                torch.cuda.synchronize()
            t_eval = (time.perf_counter() - t0_eval) / float(q.shape[0])

        tau_pred_np = pred_tau.cpu().numpy()
        tau_true_np = np.asarray(test_tau)
        err = tau_pred_np - tau_true_np
        err_tau = float(np.mean(err ** 2))
        err_tau_rmse = float(np.sqrt(err_tau))
        err_tau_j = np.mean(err ** 2, axis=0)
        err_tau_rmse_j = np.sqrt(err_tau_j)

        print(f"Torque MSE  = {err_tau:.3e}")
        print(f"Torque RMSE = {err_tau_rmse:.3e}")
        print("Per-joint MSE :", " ".join([f"{x:.3e}" for x in err_tau_j]))
        print("Per-joint RMSE:", " ".join([f"{x:.3e}" for x in err_tau_rmse_j]))
        print(f"Comp Time per Sample = {t_eval:.3e}s / {1./t_eval:.1f}Hz")

        self.train_run.save_eval_metrics(
            tau_true=tau_true_np,
            tau_pred=tau_pred_np,
            t_eval=float(t_eval),
            hyper=self.hyper,
            dt=float(dt),
            n_dof=int(n_dof),
            seed=int(self.seed),
        )

        metrics_txt = self.train_run.metrics["artifacts"].get("metrics_txt")
        metrics_json = self.train_run.metrics["artifacts"].get("metrics_json")
        if metrics_txt:
            print(f"Saved metrics TXT: {metrics_txt}")
        if metrics_json:
            print(f"Saved metrics JSON: {metrics_json}")

    def finalize_plot(self) -> None:
        if self.plt is None:
            return
        if self.render:
            self.plt.show()
        else:
            self.plt.close("all")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", nargs=1, type=int, required=False, default=[1], help="Use CUDA (via torch availability check).")
    parser.add_argument("-i", nargs=1, type=int, required=False, default=[0], help="CUDA id.")
    parser.add_argument("-s", nargs=1, type=int, required=False, default=[4], help="Random seed.")
    parser.add_argument("-r", nargs=1, type=int, required=False, default=[0], help="Render plots.")
    parser.add_argument("-l", nargs=1, type=int, required=False, default=[0], help="Load model.")
    parser.add_argument("-m", nargs=1, type=int, required=False, default=[1], help="Save model.")
    parser.add_argument(
        "--npz",
        type=str,
        required=False,
        default="/workspace/shared/data/preprocessed/delan_ur5_dataset/delan_ur5_dataset.npz",
        help="Path to delan_ur5_dataset.npz",
    )
    parser.add_argument(
        "-t",
        nargs=1,
        type=str,
        required=False,
        default=["structured"],
        help="Lagrangian Type: structured",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="/workspace/shared/models/delan/delan_ur5.torch",
    )
    parser.add_argument(
        "--hp_preset",
        type=str,
        default="default",
        choices=["default", "fast_debug", "long_train"],
        help="Hyperparameter preset (UI dropdown).",
    )
    parser.add_argument("--n_width", type=int, default=None)
    parser.add_argument("--n_depth", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--wd", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--diag_eps", type=float, default=None)
    parser.add_argument("--diag_shift", type=float, default=None)
    parser.add_argument(
        "--activation",
        type=str,
        default=None,
        choices=["tanh", "relu", "softplus", "gelu", "swish"],
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=200,
        help="Evaluate on test split every N epochs (for elbow plot). 0 disables periodic eval.",
    )
    parser.add_argument(
        "--eval_n",
        type=int,
        default=0,
        help="If >0, evaluate only on first eval_n test samples for speed. 0 = full test set.",
    )
    parser.add_argument(
        "--loss_norm",
        type=str,
        default="per_joint",
        choices=["per_joint", "torch_example"],
        help="Loss normalization: per_joint uses variance normalization; torch_example uses raw squared error.",
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    args.npz = _resolve_npz_path(args.npz)
    trainer = TorchDelanTrainer(args)
    trainer.run()
    trainer.finalize_plot()
