from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


def _resolve_npz_path(npz_path: str) -> str:
    """
    Accepts:
      - /root/stem/stem.npz   (new)
      - /root/stem           (dir -> /root/stem/stem.npz)
      - /root/stem.npz       (old flat; if not found, try /root/stem/stem.npz)
    """
    p = os.path.expanduser(str(npz_path))

    if os.path.isdir(p):
        stem = os.path.basename(os.path.normpath(p))
        return os.path.join(p, f"{stem}.npz")

    if p.endswith(".npz") and (not os.path.exists(p)):
        stem = Path(p).stem
        root = os.path.dirname(p)
        if os.path.basename(root) == stem:
            # new path was provided but doesn't exist -> try old flat
            candidate = os.path.join(os.path.dirname(root), f"{stem}.npz")
            if os.path.exists(candidate):
                return candidate
        else:
            # old flat was provided but doesn't exist -> try new folder layout
            candidate = os.path.join(root, stem, f"{stem}.npz")
            if os.path.exists(candidate):
                return candidate

    return p


def load_traj_npz(npz_path: str) -> Dict[str, Any]:
    """
    Load trajectory-wise DeLaN dataset written by the preprocess pipeline.

    Expected keys:
      train_labels, train_t, train_q, train_qd, train_qdd, train_tau
      test_labels,  test_t,  test_q,  test_qd,  test_qdd,  test_tau

    Returns a dict with lists of trajectories.
    """
    npz_path = _resolve_npz_path(npz_path)
    d = np.load(npz_path, allow_pickle=True)

    out: Dict[str, Any] = {}
    out["train_labels"] = list(d["train_labels"])
    out["test_labels"] = list(d["test_labels"])

    for split in ["train", "test"]:
        out[f"{split}_t"] = list(d[f"{split}_t"])
        out[f"{split}_q"] = list(d[f"{split}_q"])
        out[f"{split}_qd"] = list(d[f"{split}_qd"])
        out[f"{split}_qdd"] = list(d[f"{split}_qdd"])
        out[f"{split}_tau"] = list(d[f"{split}_tau"])

    return out


@dataclass
class ResidualExportPaths:
    out_dir: str
    out_npz: str
    out_json: str
    base_name: str


def build_residual_output_paths(requested_out: str) -> ResidualExportPaths:
    if requested_out.endswith(".npz"):
        base = os.path.basename(requested_out)[:-4]  # strip .npz
        parent = os.path.dirname(requested_out)
        out_dir = os.path.join(parent, base)
        out_npz = os.path.join(out_dir, f"{base}.npz")
        out_json = os.path.join(out_dir, f"{base}.json")
    else:
        out_dir = requested_out
        base = "residual_export"
        out_npz = os.path.join(out_dir, f"{base}.npz")
        out_json = os.path.join(out_dir, f"{base}.json")

    return ResidualExportPaths(out_dir=out_dir, out_npz=out_npz, out_json=out_json, base_name=base)


class DelanResidualComputer:
    def __init__(self, predict_tau_traj, *, progress_every: int = 25) -> None:
        self._predict_tau_traj = predict_tau_traj
        self._progress_every = int(progress_every)

    def compute(self, data: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        n_dof = int(np.asarray(data["train_q"][0]).shape[1])

        out: Dict[str, Any] = {
            "train_labels": np.asarray(data["train_labels"], dtype=object),
            "test_labels": np.asarray(data["test_labels"], dtype=object),
        }

        for split in ["train", "test"]:
            tau_hat_list = []
            r_tau_list = []

            q_list = data[f"{split}_q"]
            qd_list = data[f"{split}_qd"]
            qdd_list = data[f"{split}_qdd"]
            tau_list = data[f"{split}_tau"]

            print(f"\nExporting split={split}: {len(q_list)} trajectories")
            for i in range(len(q_list)):
                q_i = np.asarray(q_list[i], dtype=np.float32)
                qd_i = np.asarray(qd_list[i], dtype=np.float32)
                qdd_i = np.asarray(qdd_list[i], dtype=np.float32)
                tau_i = np.asarray(tau_list[i], dtype=np.float32)

                tau_hat_i = self._predict_tau_traj(q_i, qd_i, qdd_i)
                r_tau_i = tau_i - tau_hat_i

                tau_hat_list.append(np.asarray(tau_hat_i, dtype=np.float32))
                r_tau_list.append(np.asarray(r_tau_i, dtype=np.float32))

                if self._progress_every and ((i + 1) % self._progress_every == 0 or (i + 1) == len(q_list)):
                    print(f"  done {i+1}/{len(q_list)}", flush=True)

            out[f"{split}_t"] = np.asarray(data[f"{split}_t"], dtype=object)
            out[f"{split}_q"] = np.asarray(data[f"{split}_q"], dtype=object)
            out[f"{split}_qd"] = np.asarray(data[f"{split}_qd"], dtype=object)
            out[f"{split}_qdd"] = np.asarray(data[f"{split}_qdd"], dtype=object)
            out[f"{split}_tau"] = np.asarray(data[f"{split}_tau"], dtype=object)
            out[f"{split}_tau_hat"] = np.asarray(tau_hat_list, dtype=object)
            out[f"{split}_r_tau"] = np.asarray(r_tau_list, dtype=object)

        return out, n_dof
