#!/usr/bin/env python3
"""
Compute and append per-joint RMSE percentage (fraction, 0.01 = 1%) into
metrics_<split>_H*.json for each benchmark model.

Why this recomputes from GT:
  The percentage needs the ground-truth RMS per joint, so we must reload the
  residual NPZ (which contains GT tau and residuals) and re-run the residual
  LSTM to reconstruct:
    - DeLaN torque error (tau_hat vs tau)
    - Residual LSTM error (r_hat vs r_gt)
    - Combined torque error (tau_hat + r_hat vs tau)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent

def _add_if_exists(path: Path) -> None:
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

_add_if_exists(_SCRIPT_DIR.parent / "src")  # services/evaluation/src
for anc in _SCRIPT_DIR.parents:
    _add_if_exists(anc / "shared" / "src")
_add_if_exists(Path("/workspace/shared/src"))
_add_if_exists(Path("/workspace/payload_estimation/shared/src"))

from feature_builders import FEATURE_MODES, build_features  # type: ignore


def apply_x_scaler_feat(feat: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray) -> np.ndarray:
    return ((feat - x_mean[None, :]) / x_std[None, :]).astype(np.float32)


def invert_y_scaler(y_scaled: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray) -> np.ndarray:
    return (y_scaled * y_std[None, :] + y_mean[None, :]).astype(np.float32)


def build_windows(feat: np.ndarray, H: int) -> np.ndarray:
    T, D = feat.shape
    if T < H:
        return np.zeros((0, H, D), dtype=np.float32)
    X = np.zeros((T - H + 1, H, D), dtype=np.float32)
    for i, k in enumerate(range(H - 1, T)):
        X[i] = feat[k - H + 1 : k + 1]
    return X


def per_joint_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))


def per_joint_rmse_pct(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Fraction of RMS magnitude; 0.01 = 1%."""
    rmse_abs = per_joint_rmse(y_true, y_pred)
    denom = np.sqrt(np.mean(y_true ** 2, axis=0))
    denom = np.maximum(denom, 1e-8)
    return rmse_abs / denom


def concat_valid(chunks: List[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack(chunks).astype(np.float32)


def resolve_local_path(p: str, project_root: Optional[Path]) -> Path:
    p_expanded = Path(os.path.expanduser(p))
    if p_expanded.exists():
        return p_expanded

    if p.startswith("/workspace/shared"):
        rel = Path(p).relative_to("/workspace/shared")
        if project_root:
            candidate = project_root / "shared" / rel
            if candidate.exists():
                return candidate
        candidate = Path("/workspace/shared") / rel
        if candidate.exists():
            return candidate

    if p.startswith("/workspace"):
        rel = Path(p).relative_to("/workspace")
        if project_root:
            candidate = project_root / rel
            if candidate.exists():
                return candidate
        candidate = Path("/workspace") / rel
        if candidate.exists():
            return candidate

    return p_expanded


def maybe_swap_to_unseen(residual_npz: Path, unseen_tag: Optional[str]) -> Path:
    """
    If an unseen residual NPZ exists for a 5x10^4 path, prefer it.
    """
    if not unseen_tag:
        return residual_npz
    path_str = str(residual_npz)
    if "5x10^4_under" in path_str:
        candidate = Path(path_str.replace("5x10^4_under", unseen_tag))
        if candidate.exists():
            return candidate
    return residual_npz


def compute_pct_metrics(
    metrics_json: Path,
    batch: int,
    unseen_tag: Optional[str],
    force_unseen: bool,
) -> Dict[str, List[float]]:
    with open(metrics_json, "r") as f:
        meta = json.load(f)

    H = int(meta["H"])
    feature_mode = meta.get("feature_mode", "full")
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"Unsupported feature mode '{feature_mode}' in {metrics_json}")

    paths_cfg = meta["paths"]
    # Infer a project root from the metrics file path
    project_root = metrics_json.parents[4] if len(metrics_json.parents) >= 5 else None
    residual_npz = resolve_local_path(paths_cfg["residual_npz"], project_root)
    residual_npz_unseen = maybe_swap_to_unseen(residual_npz, unseen_tag)
    if force_unseen and ("5x10^4_under" in str(residual_npz)) and (residual_npz_unseen == residual_npz):
        raise FileNotFoundError(
            f"Unseen residual NPZ not found for {residual_npz}. Expected tag '{unseen_tag}'."
        )
    residual_npz = residual_npz_unseen
    lstm_model = resolve_local_path(paths_cfg["lstm_model"], project_root)
    lstm_scalers = resolve_local_path(paths_cfg["lstm_scalers"], project_root)

    data = np.load(residual_npz, allow_pickle=True)
    split = meta.get("split", "test")
    q_list = list(data[f"{split}_q"])
    qd_list = list(data[f"{split}_qd"])
    qdd_list = list(data[f"{split}_qdd"])
    tau_list = list(data[f"{split}_tau"])
    tau_hat_list = list(data[f"{split}_tau_hat"])
    r_tau_list = list(data[f"{split}_r_tau"])

    # Load LSTM to reconstruct residual predictions aligned to the original time axis.
    model = tf.keras.models.load_model(lstm_model)
    sc = np.load(lstm_scalers)
    x_mean = sc["x_mean"].astype(np.float32)
    x_std = sc["x_std"].astype(np.float32)
    y_mean = sc["y_mean"].astype(np.float32)
    y_std = sc["y_std"].astype(np.float32)

    tau_gt_valid_all: List[np.ndarray] = []
    tau_delan_valid_all: List[np.ndarray] = []
    tau_combined_valid_all: List[np.ndarray] = []
    r_gt_valid_all: List[np.ndarray] = []
    r_hat_valid_all: List[np.ndarray] = []

    for q, qd, qdd, tau, tau_hat, r_gt in zip(q_list, qd_list, qdd_list, tau_list, tau_hat_list, r_tau_list):
        q = np.asarray(q, dtype=np.float32)
        qd = np.asarray(qd, dtype=np.float32)
        qdd = np.asarray(qdd, dtype=np.float32)
        tau = np.asarray(tau, dtype=np.float32)
        tau_hat = np.asarray(tau_hat, dtype=np.float32)
        r_gt = np.asarray(r_gt, dtype=np.float32)

        T = q.shape[0]
        if T < H:
            continue

        feat = build_features(q, qd, qdd, tau_hat, mode=feature_mode)
        feat_n = apply_x_scaler_feat(feat, x_mean, x_std)
        X = build_windows(feat_n, H)

        r_hat_valid_scaled = model.predict(X, batch_size=batch, verbose=0).astype(np.float32)
        r_hat_valid = invert_y_scaler(r_hat_valid_scaled, y_mean, y_std)

        tau_combined_valid = tau_hat[H - 1 :] + r_hat_valid

        tau_gt_valid_all.append(tau[H - 1 :])
        tau_delan_valid_all.append(tau_hat[H - 1 :])
        tau_combined_valid_all.append(tau_combined_valid)
        r_gt_valid_all.append(r_gt[H - 1 :])
        r_hat_valid_all.append(r_hat_valid)

    tau_gt_valid = concat_valid(tau_gt_valid_all)
    tau_delan_valid = concat_valid(tau_delan_valid_all)
    tau_combined_valid = concat_valid(tau_combined_valid_all)
    r_gt_valid = concat_valid(r_gt_valid_all)
    r_hat_valid = concat_valid(r_hat_valid_all)

    return {
        # Percent as fraction (0.01 = 1%)
        "delan_joint_rmse_pct": per_joint_rmse_pct(tau_gt_valid, tau_delan_valid).tolist(),
        "res_joint_rmse_pct": per_joint_rmse_pct(r_gt_valid, r_hat_valid).tolist(),
        "rg_joint_rmse_pct": per_joint_rmse_pct(tau_gt_valid, tau_combined_valid).tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-root", type=str, default=None, help="evaluation root")
    ap.add_argument("--use-eval-root", action=argparse.BooleanOptionalAction, default=True,
                    help="Default to /shared/evaluation instead of /shared/evaluation/benchmark_models.")
    ap.add_argument("--split", type=str, default="test", choices=["test", "train"])
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--include", action="append", default=[],
                    help="Only process metrics files whose path contains this substring (repeatable).")
    ap.add_argument("--unseen-tag", type=str, default="5x10^3_under",
                    help="Dataset tag used for unseen residuals (e.g., 5x10^3_under).")
    ap.add_argument("--force-unseen", action="store_true",
                    help="Fail if unseen residuals are not found for a 5x10^4 path.")
    args = ap.parse_args()

    bench_root = Path(args.bench_root) if args.bench_root else None
    if bench_root is None:
        # Try to find evaluation root from script location or container default.
        if args.use_eval_root:
            for anc in _SCRIPT_DIR.parents:
                candidate = anc / "shared" / "evaluation"
                if candidate.exists():
                    bench_root = candidate
                    break
            if bench_root is None:
                candidate = Path("/workspace/shared/evaluation")
                if candidate.exists():
                    bench_root = candidate
        else:
            for anc in _SCRIPT_DIR.parents:
                candidate = anc / "shared" / "evaluation" / "benchmark_models"
                if candidate.exists():
                    bench_root = candidate
                    break
            if bench_root is None:
                candidate = Path("/workspace/shared/evaluation/benchmark_models")
                if candidate.exists():
                    bench_root = candidate
    if bench_root is None:
        raise FileNotFoundError("Could not locate benchmark_models; pass --bench-root explicitly.")

    metrics_files = list(bench_root.rglob(f"metrics_{args.split}_H*.json"))
    if not metrics_files:
        raise RuntimeError(f"No metrics_{args.split}_H*.json found under {bench_root}")

    for mf in metrics_files:
        if args.include:
            s = str(mf)
            if not any(tok in s for tok in args.include):
                continue
        print(f"[update] {mf}")
        pct = compute_pct_metrics(
            mf,
            batch=args.batch,
            unseen_tag=args.unseen_tag,
            force_unseen=args.force_unseen,
        )
        with open(mf, "r") as f:
            meta = json.load(f)
        metrics = meta.get("metrics", {})
        metrics.update(pct)
        metrics["pct_scale"] = "fraction"  # 0.01 = 1%
        meta["metrics"] = metrics
        with open(mf, "w") as f:
            json.dump(meta, f, indent=2)

    print(f"Updated {len(metrics_files)} metrics files.")


if __name__ == "__main__":
    main()
