import argparse
import os
import json
import io
import time
import traceback
import contextlib

import dill as pickle
import numpy as np

import jax
import jax.numpy as jnp
import haiku as hk
import functools

import sys

# --- NEW: allow importing deep_lagrangian_networks from the submodule mount ---
DELAN_REPO_DIR = os.environ.get("DELAN_REPO_DIR", "/workspace/delan_repo")
if DELAN_REPO_DIR not in sys.path:
    sys.path.insert(0, DELAN_REPO_DIR)

import deep_lagrangian_networks.jax_DeLaN_model as delan
from deep_lagrangian_networks.utils import activations

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

def load_traj_npz(npz_path: str):
    """
    Load trajectory-wise DeLaN dataset written by your preprocess pipeline.

    Expected keys:
      train_labels, train_t, train_q, train_qd, train_qdd, train_tau
      test_labels,  test_t,  test_q,  test_qd,  test_qdd,  test_tau

    Returns dict with lists of trajectories.
    """
    d = np.load(npz_path, allow_pickle=True)

    out = {}
    out["train_labels"] = list(d["train_labels"])
    out["test_labels"] = list(d["test_labels"])

    for split in ["train", "test"]:
        out[f"{split}_t"]   = list(d[f"{split}_t"])
        out[f"{split}_q"]   = list(d[f"{split}_q"])
        out[f"{split}_qd"]  = list(d[f"{split}_qd"])
        out[f"{split}_qdd"] = list(d[f"{split}_qdd"])
        out[f"{split}_tau"] = list(d[f"{split}_tau"])

    return out


def build_delan_model(hyper: dict, n_dof: int):
    """
    Rebuild the Haiku-transformed Lagrangian network + jit dynamics model,
    matching train_ur5_jax.py.
    """
    lagrangian_type = hyper["lagrangian_type"]
    act = activations[hyper["activation"]]

    lagrangian_fn = hk.transform(functools.partial(
        lagrangian_type,
        n_dof=n_dof,
        shape=(hyper["n_width"],) * hyper["n_depth"],
        activation=act,
        epsilon=hyper["diagonal_epsilon"],
        shift=hyper["diagonal_shift"],
    ))

    lagrangian_apply = lagrangian_fn.apply
    dynamics = jax.jit(functools.partial(delan.dynamics_model, lagrangian=lagrangian_apply, n_dof=n_dof))
    return lagrangian_fn, dynamics


def predict_tau_traj(dynamics_model, params, q_np, qd_np, qdd_np):
    """
    Run DeLaN torque prediction on one trajectory.
    Inputs are numpy arrays (T, n_dof).
    Returns numpy array (T, n_dof).
    """
    q   = jnp.asarray(q_np, dtype=jnp.float32)
    qd  = jnp.asarray(qd_np, dtype=jnp.float32)
    qdd = jnp.asarray(qdd_np, dtype=jnp.float32)

    # tau input is unused for pure forward prediction in this call pattern,
    # but dynamics_model signature expects it
    dummy_tau = jnp.zeros_like(q)

    # dynamics_model returns (qdd_hat?, tau_hat?) -> in train_ur5_jax.py it uses [1] as tau
    tau_hat = dynamics_model(params, None, q, qd, qdd, dummy_tau)[1]
    return np.asarray(tau_hat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_in", type=str, required=True,
                    help="Trajectory-wise DeLaN dataset (delan_ur5_dataset.npz)")
    ap.add_argument("--ckpt", type=str, required=True,
                    help="Saved DeLaN checkpoint (.jax) from train_ur5_jax.py")
    ap.add_argument("--out", type=str, required=True,
                    help="Output NPZ path (trajectory-wise residual dataset)")
    ap.add_argument("--mem_frac", type=float, default=0.4,
                    help="XLA memory fraction (default 0.4)")
    args = ap.parse_args()

    # --- NEW: structured output folder next to the requested NPZ ---
    requested_out = args.out  # keep old CLI behavior as "canonical" path
    if requested_out.endswith(".npz"):
        base = os.path.basename(requested_out)[:-4]  # strip .npz
        parent = os.path.dirname(requested_out)
        out_dir = os.path.join(parent, base)
        os.makedirs(out_dir, exist_ok=True)

        out_npz = os.path.join(out_dir, f"{base}.npz")
        out_json = os.path.join(out_dir, f"{base}.json")
    else:
        # If user passed a directory, we still create files inside it.
        out_dir = requested_out
        os.makedirs(out_dir, exist_ok=True)
        base = "residual_export"
        out_npz = os.path.join(out_dir, f"{base}.npz")
        out_json = os.path.join(out_dir, f"{base}.json")

    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(args.mem_frac)

    # --- NEW: capture console output for JSON audit ---
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
        "mem_frac": args.mem_frac,
    }

    try:
        with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
            print("Loading dataset:", args.npz_in)
            data = load_traj_npz(args.npz_in)

            print("Loading checkpoint:", args.ckpt)
            with open(args.ckpt, "rb") as f:
                saved = pickle.load(f)

            hyper = saved.get("hyper", {})
            params = saved.get("params", None)
            seed = saved.get("seed", None)

            # infer dof from first train trajectory
            n_dof = int(np.asarray(data["train_q"][0]).shape[1])
            print(f"Detected n_dof={n_dof} (seed={seed})")

            # build model
            lagrangian_fn, dynamics_model = build_delan_model(hyper, n_dof=n_dof)

            # warmup compile with 1 sample
            q0   = np.asarray(data["train_q"][0], dtype=np.float32)[:1]
            qd0  = np.asarray(data["train_qd"][0], dtype=np.float32)[:1]
            qdd0 = np.asarray(data["train_qdd"][0], dtype=np.float32)[:1]
            _ = predict_tau_traj(dynamics_model, params, q0, qd0, qdd0)

            out = {
                "train_labels": np.asarray(data["train_labels"], dtype=object),
                "test_labels":  np.asarray(data["test_labels"], dtype=object),
            }

            for split in ["train", "test"]:
                tau_hat_list = []
                r_tau_list = []

                q_list   = data[f"{split}_q"]
                qd_list  = data[f"{split}_qd"]
                qdd_list = data[f"{split}_qdd"]
                tau_list = data[f"{split}_tau"]

                print(f"\nExporting split={split}: {len(q_list)} trajectories")
                for i in range(len(q_list)):
                    q_i   = np.asarray(q_list[i], dtype=np.float32)
                    qd_i  = np.asarray(qd_list[i], dtype=np.float32)
                    qdd_i = np.asarray(qdd_list[i], dtype=np.float32)
                    tau_i = np.asarray(tau_list[i], dtype=np.float32)

                    tau_hat_i = predict_tau_traj(dynamics_model, params, q_i, qd_i, qdd_i)
                    r_tau_i = tau_i - tau_hat_i

                    tau_hat_list.append(np.asarray(tau_hat_i, dtype=np.float32))
                    r_tau_list.append(np.asarray(r_tau_i, dtype=np.float32))

                    if (i + 1) % 25 == 0 or (i + 1) == len(q_list):
                        print(f"  done {i+1}/{len(q_list)}", flush=True)

                # store original trajectory-wise signals too (for later windowing)
                out[f"{split}_t"]   = np.asarray(data[f"{split}_t"], dtype=object)
                out[f"{split}_q"]   = np.asarray(data[f"{split}_q"], dtype=object)
                out[f"{split}_qd"]  = np.asarray(data[f"{split}_qd"], dtype=object)
                out[f"{split}_qdd"] = np.asarray(data[f"{split}_qdd"], dtype=object)
                out[f"{split}_tau"] = np.asarray(data[f"{split}_tau"], dtype=object)

                out[f"{split}_tau_hat"] = np.asarray(tau_hat_list, dtype=object)
                out[f"{split}_r_tau"]   = np.asarray(r_tau_list, dtype=object)

            # write main NPZ into folder
            np.savez(out_npz, **out)

            print("\nSaved residual trajectory dataset:", out_npz)
            print("Keys:", sorted(out.keys()))

            print("\nSaved residual trajectory dataset:", out_npz)
            print("Keys:", sorted(out.keys()))

            # store netword type
            hyper_json = dict(hyper) if isinstance(hyper, dict) else {}
            lt = hyper_json.get("lagrangian_type", None)
            if callable(lt) and hasattr(lt, "__name__"):
                hyper_json["lagrangian_type"] = lt.__name__
            else:
                hyper_json["lagrangian_type"] = str(lt)

            # meta for JSON
            meta.update({
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
            })

        # write JSON (outside redirect so file write errors show normally)
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

        # best-effort JSON write
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