import argparse
import contextlib
import io
import json
import os
import sys
import time
import traceback

import numpy as np
import torch


def _add_path(path: str) -> None:
    if path and path not in sys.path:
        sys.path.insert(0, path)


DELAN_REPO_DIR = os.environ.get("DELAN_REPO_DIR", "/workspace/delan_repo")
DELAN_COMMON_SRC = "/workspace/delan_src"
DELAN_TORCH_SRC = "/workspace/delan_torch/src"

for p in [DELAN_COMMON_SRC, DELAN_TORCH_SRC, DELAN_REPO_DIR]:
    _add_path(p)

from delan_residuals import DelanResidualComputer, build_residual_output_paths, load_traj_npz
from deep_lagrangian_networks.DeLaN_model import DeepLagrangianNetwork


class _Tee(io.TextIOBase):
    """Write to two streams (console + buffer)."""

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def write(self, s):
        self.a.write(s)
        self.a.flush()
        self.b.write(s)
        return len(s)

    def flush(self):
        self.a.flush()
        self.b.flush()


def _map_activation(name: str) -> str:
    if name is None:
        raise ValueError("Activation must be provided for torch residual export.")
    key = name.lower()
    mapping = {
        "relu": "ReLu",
        "softplus": "SoftPlus",
    }
    if key not in mapping:
        raise ValueError(f"Torch DeLaN_model supports activation=relu|softplus, got '{name}'.")
    return mapping[key]


def _predict_tau_traj(model: DeepLagrangianNetwork, device: torch.device, q_np, qd_np, qdd_np):
    with torch.no_grad():
        q = torch.from_numpy(q_np).float().to(device)
        qd = torch.from_numpy(qd_np).float().to(device)
        qdd = torch.from_numpy(qdd_np).float().to(device)
        tau_hat = model.inv_dyn(q, qd, qdd)
    return tau_hat.cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--npz_in",
        type=str,
        required=True,
        help="Trajectory-wise DeLaN dataset (delan_ur5_dataset.npz)",
    )
    ap.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Saved DeLaN checkpoint (.torch) from rbyt_train_delan_torch.py",
    )
    ap.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output NPZ path (trajectory-wise residual dataset)",
    )
    args = ap.parse_args()

    requested_out = args.out
    paths = build_residual_output_paths(requested_out)
    out_dir = paths.out_dir
    out_npz = paths.out_npz
    out_json = paths.out_json
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.time()
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    tee_out = _Tee(sys.stdout, stdout_buf)
    tee_err = _Tee(sys.stderr, stderr_buf)

    meta = {
        "status": "running",
        "timestamp_unix": t0,
        "npz_in": args.npz_in,
        "ckpt": args.ckpt,
        "requested_out": requested_out,
        "out_dir": out_dir,
        "out_npz": out_npz,
        "out_json": out_json,
        "device": str(device),
    }

    try:
        with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
            print("Loading dataset:", args.npz_in)
            data = load_traj_npz(args.npz_in)

            print("Loading checkpoint:", args.ckpt)
            state = torch.load(args.ckpt, map_location=device)
            hyper = state.get("hyper", {})
            seed = state.get("seed", None)

            n_dof = int(np.asarray(data["train_q"][0]).shape[1])
            print(f"Detected n_dof={n_dof} (seed={seed})")

            hyper_model = dict(hyper)
            hyper_model["activation"] = _map_activation(hyper_model.get("activation", "softplus"))

            model = DeepLagrangianNetwork(n_dof, **hyper_model)
            model = model.cuda() if device.type == "cuda" else model.cpu()
            model.load_state_dict(state["state_dict"])
            model.eval()

            # warmup on 1 sample
            q0 = np.asarray(data["train_q"][0], dtype=np.float32)[:1]
            qd0 = np.asarray(data["train_qd"][0], dtype=np.float32)[:1]
            qdd0 = np.asarray(data["train_qdd"][0], dtype=np.float32)[:1]
            _ = _predict_tau_traj(model, device, q0, qd0, qdd0)

            computer = DelanResidualComputer(
                lambda q_i, qd_i, qdd_i: _predict_tau_traj(model, device, q_i, qd_i, qdd_i)
            )
            out, n_dof = computer.compute(data)

            np.savez(out_npz, **out)
            print("\nSaved residual trajectory dataset:", out_npz)
            print("Keys:", sorted(out.keys()))

            hyper_json = dict(hyper) if isinstance(hyper, dict) else {}
            lt = hyper_json.get("lagrangian_type", None)
            if callable(lt) and hasattr(lt, "__name__"):
                hyper_json["lagrangian_type"] = lt.__name__
            else:
                hyper_json["lagrangian_type"] = str(lt)

            meta.update(
                {
                    "status": "ok",
                    "dataset_npz_basename": os.path.basename(args.npz_in),
                    "checkpoint_basename": os.path.basename(args.ckpt),
                    "seed": seed,
                    "n_dof": n_dof,
                    "hyper": hyper_json,
                    "train_n_traj": len(data["train_q"]),
                    "test_n_traj": len(data["test_q"]),
                    "keys": sorted(list(out.keys())),
                    "elapsed_s": float(time.time() - t0),
                }
            )

        meta["stdout"] = stdout_buf.getvalue()
        meta["stderr"] = stderr_buf.getvalue()
        with open(out_json, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved export JSON: {out_json}")

    except Exception:
        meta["status"] = "error"
        meta["elapsed_s"] = float(time.time() - t0)
        meta["stdout"] = stdout_buf.getvalue()
        meta["stderr"] = stderr_buf.getvalue()
        meta["traceback"] = traceback.format_exc()

        try:
            os.makedirs(out_dir, exist_ok=True)
            with open(out_json, "w") as f:
                json.dump(meta, f, indent=2)
            print(f"[error] Saved export JSON (error state): {out_json}")
        except Exception:
            pass

        raise


if __name__ == "__main__":
    main()
