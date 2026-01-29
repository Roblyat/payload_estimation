import argparse
import functools
import os
import sys
import time
from pathlib import Path

import dill as pickle
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

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
DELAN_JAX_SRC = "/workspace/delan_jax/src"

for p in [DELAN_COMMON_SRC, DELAN_JAX_SRC, DELAN_REPO_DIR]:
    _add_path(p)

from delan_train_common import DelanRunPaths, DelanTrainRun, compute_tau_metrics
from delan_train_common import EarlyStopConfig, EarlyStopper
from load_npz_dataset import load_npz_trajectory_dataset

import deep_lagrangian_networks.jax_DeLaN_model as delan
from deep_lagrangian_networks.replay_memory import ReplayMemory
from deep_lagrangian_networks.utils import init_env, activations


class JaxDelanTrainer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.render_flag = bool(args.r[0])
        self.plt = _init_matplotlib(self.render_flag)

        self.seed, self.cuda, self.render, self.load_model, self.save_model = init_env(args)
        self.rng_key = jax.random.PRNGKey(self.seed)

        self.model_choice = str(args.t[0])
        self.lagrangian_type = self._resolve_lagrangian_type(self.model_choice)

        self.run_paths = DelanRunPaths.from_save_path(args.save_path)
        self.run_name = self.run_paths.run_name

        self.hyper = self._build_hyper()
        self.hyper_json = self._build_hyper_json(self.hyper)

        self.params = None
        if self.load_model:
            self._load_checkpoint()

        print(f"Final hyper: {self.hyper}")

        args_summary = {
            "render": int(args.r[0]),
            "eval_every": int(args.eval_every),
            "eval_n": int(args.eval_n),
            "log_every": int(args.log_every),
            "early_stop": bool(args.early_stop),
            "early_stop_patience": int(args.early_stop_patience),
            "early_stop_min_delta": float(args.early_stop_min_delta),
            "early_stop_warmup_evals": int(args.early_stop_warmup_evals),
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

    def _resolve_lagrangian_type(self, model_choice: str):
        if model_choice == "structured":
            return delan.structured_lagrangian_fn
        if model_choice == "black_box":
            return delan.blackbox_lagrangian_fn
        raise ValueError("Unknown -t. Use structured or black_box.")

    def _build_hyper(self) -> dict:
        hyper = {
            "dataset": self.run_name,
            "n_width": 64,
            "n_depth": 2,
            "n_minibatch": 512,
            "diagonal_epsilon": 0.1,
            "diagonal_shift": 2.0,
            "activation": "tanh",
            "learning_rate": 1.0e-4,
            "weight_decay": 1.0e-5,
            "max_epoch": int(2.0 * 1e3),
            "lagrangian_type": self.lagrangian_type,
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
            "lutter_like": {
                "activation": "softplus",
                "n_minibatch": 1024,
                "n_width": 128,
                "n_depth": 2,
                "learning_rate": 1e-4,
                "weight_decay": 1e-5,
            },
            "lutter_like_128": {
                "activation": "softplus",
                "n_minibatch": 1024,
                "n_width": 128,
                "n_depth": 2,
                "learning_rate": 1e-4,
                "weight_decay": 1e-5,
            },
            "lutter_like_256": {
                "activation": "softplus",
                "n_minibatch": 1024,
                "n_width": 256,
                "n_depth": 2,
                "learning_rate": 1e-4,
                "weight_decay": 1e-5,
            },
            "lutter_like_256_d3": {
                "activation": "softplus",
                "n_minibatch": 1024,
                "n_width": 256,
                "n_depth": 3,
                "learning_rate": 1e-4,
                "weight_decay": 1e-5,
            },
            "lutter_like_256_wd1e4": {
                "activation": "softplus",
                "n_minibatch": 1024,
                "n_width": 256,
                "n_depth": 2,
                "learning_rate": 1e-4,
                "weight_decay": 1e-4,
            },
            "lutter_like_256_lr5e5": {
                "activation": "softplus",
                "n_minibatch": 1024,
                "n_width": 256,
                "n_depth": 2,
                "learning_rate": 5e-5,
                "weight_decay": 1e-5,
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

    @staticmethod
    def _build_hyper_json(hyper: dict) -> dict:
        hyper_json = dict(hyper)
        lt = hyper_json.get("lagrangian_type")
        if callable(lt) and hasattr(lt, "__name__"):
            hyper_json["lagrangian_type"] = lt.__name__
        else:
            hyper_json["lagrangian_type"] = str(lt)
        return hyper_json

    def _load_checkpoint(self) -> None:
        load_path = self.run_paths.ckpt_path
        with open(load_path, "rb") as f:
            saved = pickle.load(f)
        self.hyper = saved.get("hyper", self.hyper)
        self.hyper_json = self._build_hyper_json(self.hyper)
        self.params = saved["params"]
        print(f"Loaded DeLaN checkpoint: {load_path}")

    def run(self) -> None:
        args = self.args
        print("\n\n################################################")
        print(f"DeLaN run: {self.run_name}")
        print(f"  model_dir = {self.run_paths.model_dir}")
        print(f"  ckpt_path = {self.run_paths.ckpt_path}")
        print(f"  type = {self.model_choice}")
        print(f"  hp_preset = {args.hp_preset}")
        print("################################################")

        try:
            train_data, val_data, test_data, divider, dt = load_npz_trajectory_dataset(args.npz)
        except ValueError as e:
            print(f"[error] {e}")
            raise
        train_labels, train_qp, train_qv, train_qa, train_tau = train_data
        val_labels, val_qp, val_qv, val_qa, val_tau = val_data
        test_labels, test_qp, test_qv, test_qa, test_tau = test_data

        n_dof = train_qp.shape[-1]
        npz_name = os.path.basename(args.npz)

        print("\n\n################################################")
        print(f"Dataset: {npz_name}")
        print(f"  npz = {args.npz}")
        print(f"   dt ≈ {dt}")
        print(f"  dof = {n_dof}")
        print(f"  Train trajectories = {len(train_labels)}")
        print(f"  Val trajectories   = {len(val_labels)}")
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

        mem_dim = ((n_dof,), (n_dof,), (n_dof,), (n_dof,))
        mem = ReplayMemory(train_qp.shape[0], self.hyper["n_minibatch"], mem_dim)
        mem.add_samples([train_qp, train_qv, train_qa, train_tau])

        lagrangian_fn = hk.transform(
            functools.partial(
                self.hyper["lagrangian_type"],
                n_dof=n_dof,
                shape=(self.hyper["n_width"],) * self.hyper["n_depth"],
                activation=activations[self.hyper["activation"]],
                epsilon=self.hyper["diagonal_epsilon"],
                shift=self.hyper["diagonal_shift"],
            )
        )

        q, qd, qdd, tau = [jnp.array(x) for x in next(iter(mem))]
        self.rng_key, init_key = jax.random.split(self.rng_key)

        if self.params is None:
            self.params = lagrangian_fn.init(init_key, q[0], qd[0])

        lagrangian = lagrangian_fn.apply
        delan_model = jax.jit(
            functools.partial(delan.dynamics_model, lagrangian=lagrangian, n_dof=n_dof)
        )
        _ = delan_model(self.params, None, q[:1], qd[:1], qdd[:1], tau[:1])

        optimizer = optax.adamw(
            learning_rate=self.hyper["learning_rate"], weight_decay=self.hyper["weight_decay"]
        )
        opt_state = optimizer.init(self.params)

        loss_fn = functools.partial(
            delan.inverse_loss_fn,
            lagrangian=lagrangian,
            n_dof=n_dof,
            norm_tau=jnp.var(train_tau, axis=0),
            norm_qdd=jnp.var(train_qa, axis=0),
        )

        def update_fn(params, opt_state, q, qd, qdd, tau):
            (_, logs), grads = jax.value_and_grad(loss_fn, 0, has_aux=True)(params, q, qd, qdd, tau)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, logs

        update_fn = jax.jit(update_fn)
        _, _, logs0 = update_fn(self.params, opt_state, q[:1], qd[:1], qdd[:1], tau[:1])

        print("################################################")
        print(f"Training DeLaN | run={self.run_name} | type={self.model_choice} | dof={n_dof}")
        print("################################################")

        early_cfg = EarlyStopConfig(
            enabled=bool(args.early_stop),
            patience=int(args.early_stop_patience),
            min_delta=float(args.early_stop_min_delta),
            warmup_evals=int(args.early_stop_warmup_evals),
            mode="min",
        )
        early_stopper = EarlyStopper(early_cfg)
        best_params_snapshot = None
        restored_best = False
        monitor_split = "val"

        t0_start = time.perf_counter()
        epoch_i = 0
        while epoch_i < self.hyper["max_epoch"] and not self.load_model:
            n_batches = 0
            logs = jax.tree.map(lambda x: x * 0.0, logs0)

            for data_batch in mem:
                q, qd, qdd, tau = [jnp.array(x) for x in data_batch]
                self.params, opt_state, batch_logs = update_fn(
                    self.params, opt_state, q, qd, qdd, tau
                )
                n_batches += 1
                logs = jax.tree.map(lambda x, y: x + y, logs, batch_logs)

            epoch_i += 1
            logs = jax.tree.map(lambda x: x / n_batches, logs)

            if epoch_i == 1 or (args.log_every > 0 and (epoch_i % args.log_every) == 0):
                print(
                    f"Epoch {epoch_i:05d}: "
                    f"Time={time.perf_counter()-t0_start:6.1f}s, "
                    f"Loss={float(logs['loss']):.2e}, "
                    f"Inv={float(logs['inverse_mean']):.2e}, "
                    f"For={float(logs['forward_mean']):.2e}, "
                    f"Power={float(logs['energy_mean']):.2e}"
                )
                self.train_run.record_train_point(
                    epoch=epoch_i,
                    time_s=time.perf_counter() - t0_start,
                    loss=float(logs["loss"]),
                    inverse_mean=float(logs["inverse_mean"]),
                    forward_mean=float(logs["forward_mean"]),
                    energy_mean=float(logs["energy_mean"]),
                )

            if args.eval_every > 0 and (epoch_i == 1 or (epoch_i % args.eval_every) == 0):
                use_val = val_qp.shape[0] > 0
                q_eval = val_qp if use_val else test_qp
                qd_eval = val_qv if use_val else test_qv
                qdd_eval = val_qa if use_val else test_qa
                tau_eval = val_tau if use_val else test_tau
                monitor_split = "val" if use_val else "test"

                if args.eval_n and args.eval_n > 0:
                    n = min(int(args.eval_n), q_eval.shape[0])
                    q_eval = q_eval[:n]
                    qd_eval = qd_eval[:n]
                    qdd_eval = qdd_eval[:n]
                    tau_eval = tau_eval[:n]

                qj = jnp.array(q_eval)
                qdj = jnp.array(qd_eval)
                qddj = jnp.array(qdd_eval)
                tauj = jnp.array(tau_eval)

                pred_tau_eval = delan_model(self.params, None, qj, qdj, qddj, 0.0 * qj)[1]
                eval_mse = float((1.0 / qj.shape[0]) * jnp.sum((pred_tau_eval - tauj) ** 2))

                self.train_run.record_test_point(epoch=epoch_i, mse=eval_mse)
                split_name = "val" if use_val else "test"
                print(f"  [eval] {split_name}_mse={eval_mse:.3e}  (n={qj.shape[0]})")

                should_stop, improved = early_stopper.update(metric=eval_mse, epoch=epoch_i)
                if improved:
                    best_params_snapshot = jax.device_get(self.params)
                if should_stop:
                    print(
                        f"  [early_stop] stop at epoch={epoch_i} "
                        f"(best_epoch={early_stopper.best_epoch}, best_{split_name}_mse={early_stopper.best_metric:.3e})"
                    )
                    break

        if best_params_snapshot is not None and early_cfg.enabled:
            self.params = jax.device_put(best_params_snapshot)
            restored_best = True

        early_stop_metrics = early_stopper.to_dict(
            monitor="val_mse",
            restored_best=restored_best,
            monitor_split=monitor_split,
        )

        if self.save_model:
            epoch_to_save = early_stopper.best_epoch if restored_best and early_stopper.best_epoch is not None else epoch_i
            with open(self.run_paths.ckpt_path, "wb") as f:
                pickle.dump(
                    {
                        "epoch": int(epoch_to_save),
                        "hyper": self.hyper,
                        "params": self.params,
                        "seed": self.seed,
                        "early_stop": early_stop_metrics,
                    },
                    f,
                )
            print(f"Saved DeLaN checkpoint: {self.run_paths.ckpt_path}")

        self.train_run.finalize_train_metrics(epochs_ran=epoch_i, early_stop=early_stop_metrics)
        self.train_run.save_training_artifacts(save_model=self.save_model)

        train_csv = self.train_run.metrics["artifacts"].get("train_history_csv")
        if train_csv:
            print(f"Saved training history: {train_csv}")

        print("\n################################################")
        print(f"Evaluating DeLaN | run={self.run_name}")

        if val_qp.shape[0] > 0:
            qv = jnp.array(val_qp)
            qdv = jnp.array(val_qv)
            qddv = jnp.array(val_qa)

            t0_val = time.perf_counter()
            pred_tau_val = delan_model(self.params, None, qv, qdv, qddv, 0.0 * qv)[1]
            t_eval_val = (time.perf_counter() - t0_val) / float(qv.shape[0])

            val_mse, val_rmse, val_mse_j, val_rmse_j = compute_tau_metrics(
                tau_true=np.array(val_tau),
                tau_pred=np.array(pred_tau_val),
            )

            # store separately for selection
            self.train_run.metrics["eval_val"] = {
                "torque_mse": float(val_mse),
                "torque_rmse": float(val_rmse),
                "torque_mse_per_joint": [float(x) for x in val_mse_j],
                "torque_rmse_per_joint": [float(x) for x in val_rmse_j],
                "time_per_sample_s": float(t_eval_val),
                "hz": float(1.0 / t_eval_val) if t_eval_val > 0 else None,
            }

            print("  [val] Torque RMSE =", f"{val_rmse:.3e}")

        q = jnp.array(test_qp)
        qd = jnp.array(test_qv)
        qdd = jnp.array(test_qa)

        t0_eval = time.perf_counter()
        pred_tau = delan_model(self.params, None, q, qd, qdd, 0.0 * q)[1]
        t_eval = (time.perf_counter() - t0_eval) / float(q.shape[0])

        tau_true = jnp.array(test_tau)
        err = pred_tau - tau_true

        err_tau = float(jnp.mean(err ** 2))
        err_tau_rmse = float(np.sqrt(err_tau))
        err_tau_j = np.array(jnp.mean(err ** 2, axis=0))
        err_tau_rmse_j = np.sqrt(err_tau_j)

        print(f"Torque MSE  = {err_tau:.3e}")
        print(f"Torque RMSE = {err_tau_rmse:.3e}")
        print("Per-joint MSE :", " ".join([f'{x:.3e}' for x in err_tau_j]))
        print("Per-joint RMSE:", " ".join([f'{x:.3e}' for x in err_tau_rmse_j]))
        print(f"Comp Time per Sample = {t_eval:.3e}s / {1./t_eval:.1f}Hz")

        self.train_run.save_eval_metrics(
            tau_true=np.array(test_tau),
            tau_pred=np.array(pred_tau),
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
    parser.add_argument("-i", nargs=1, type=int, required=False, default=[0], help="CUDA id (torch side).")
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
        help="Lagrangian Type: structured|black_box",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="/workspace/shared/models/delan/delan_ur5.jax",
    )
    parser.add_argument(
        "--hp_preset",
        type=str,
        default="default",
        choices=[
            "default",
            "fast_debug",
            "lutter_like",
            "lutter_like_128",
            "lutter_like_256",
            "lutter_like_256_d3",
            "lutter_like_256_wd1e4",
            "lutter_like_256_lr5e5",
            "long_train",
        ],
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
        default=5,
        help="Evaluate on test split every N epochs (for elbow plot). 0 disables periodic eval.",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=5,
        help="Print/record training curves every N epochs. 0 disables periodic logging (epoch 1 still logs).",
    )
    parser.add_argument(
        "--eval_n",
        type=int,
        default=0,
        help="If >0, evaluate only on first eval_n test samples for speed. 0 = full test set.",
    )
    parser.add_argument(
        "--early_stop",
        type = bool,
        default=True, 
        help="Enable early stopping monitored on val_mse (evaluated every --eval_every epochs).",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=10,
        help="Early stopping patience in evaluation events (not epochs).",
    )
    parser.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=0.0,
        help="Minimum improvement required to reset patience.",
    )
    parser.add_argument(
        "--early_stop_warmup_evals",
        type=int,
        default=0,
        help="Ignore non-improving evals for the first N evaluation events.",
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    args.npz = _resolve_npz_path(args.npz)
    trainer = JaxDelanTrainer(args)
    trainer.run()
    trainer.finalize_plot()
