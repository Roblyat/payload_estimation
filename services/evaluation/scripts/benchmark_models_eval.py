#!/usr/bin/env python3
"""
Re-evaluate packaged benchmark DeLaN+LSTM models and regenerate metrics/plots.

The script scans payload_estimation/shared/evaluation/benchmark_models, loads the
saved residual datasets, LSTM residual model, and DeLaN torque estimates, then
recomputes:
  - RMSE over time ("progress") for DeLaN, residual LSTM, and combined torque.
  - Per-joint RMSE for the same.
  - Overlay and RMSE plots.
It also emits comparison plots for UR3_Load0 vs UR10_Load0.

Example:
  python3 benchmark_models_eval.py --split test --batch 256
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
import os as _os

# Avoid permission issues inside containers where /workspace/shared/.mplconfig is read-only
_os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/.mplconfig")))
Path(_os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

# Use headless backend for CLI environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent

def _add_if_exists(path: Path) -> None:
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Likely locations for source modules (handle both host and container layouts)
_add_if_exists(_SCRIPT_DIR.parent / "src")  # services/evaluation/src
for anc in _SCRIPT_DIR.parents:
    _add_if_exists(anc / "shared" / "src")  # walk up to find shared/src
_add_if_exists(Path("/workspace/shared/src"))                  # container mount
_add_if_exists(Path("/workspace/payload_estimation/shared/src"))

from eval_plots import (  # type: ignore  # local import
    save_residual_overlay_grid,
    save_torque_overlay_grid,
    save_torque_rmse_per_joint_grouped_bar,
    save_torque_rmse_time_curve,
)
from feature_builders import FEATURE_MODES, build_features  # type: ignore  # local import

# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------
def apply_x_scaler_feat(feat: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray) -> np.ndarray:
    return ((feat - x_mean[None, :]) / x_std[None, :]).astype(np.float32)


def invert_y_scaler(y_scaled: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray) -> np.ndarray:
    return (y_scaled * y_std[None, :] + y_mean[None, :]).astype(np.float32)


def build_windows(feat: np.ndarray, H: int) -> np.ndarray:
    """feat: (T, D) -> (T-H+1, H, D) sliding windows that end at k>=H-1."""
    T, D = feat.shape
    if T < H:
        return np.zeros((0, H, D), dtype=np.float32)
    X = np.zeros((T - H + 1, H, D), dtype=np.float32)
    for i, k in enumerate(range(H - 1, T)):
        X[i] = feat[k - H + 1 : k + 1]
    return X


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def per_joint_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))


def rmse_over_time(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-sample RMSE (avg over joints)."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=1))


def concat_valid(chunks: List[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack(chunks).astype(np.float32)


def infer_dt(t_list: Optional[Iterable[np.ndarray]]) -> Optional[float]:
    """Median delta from time traces if available."""
    if not t_list:
        return None
    diffs: List[np.ndarray] = []
    for t in t_list:
        t = np.asarray(t, dtype=np.float32).reshape(-1)
        if t.size >= 2:
            d = np.diff(t)
            d = d[np.isfinite(d)]
            if d.size:
                diffs.append(d)
    if not diffs:
        return None
    all_d = np.concatenate(diffs)
    if not np.isfinite(all_d).any():
        return None
    return float(np.median(all_d))


def _time_axis(n: int, dt: Optional[float]) -> Tuple[np.ndarray, str]:
    if dt is not None and dt > 0:
        return np.arange(n) * float(dt), "Time [s]"
    return np.arange(n), "Sample"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class BenchmarkPaths:
    residual_npz: Path
    lstm_model: Path
    lstm_scalers: Path


@dataclass
class BenchmarkResult:
    dataset: str
    split: str
    H: int
    feature_mode: str
    n_dof: int
    n_traj: int
    dt: Optional[float]
    metrics: Dict[str, float]
    per_joint: Dict[str, List[float]]
    rmse_time: Dict[str, np.ndarray]
    tau_gt_valid: np.ndarray
    tau_delan_valid: np.ndarray
    tau_combined_valid: np.ndarray
    r_gt_valid: np.ndarray
    r_hat_valid: np.ndarray
    out_dir: Path


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def resolve_local_path(p: str, project_root: Optional[Path]) -> Path:
    """
    Map legacy /workspace/... paths into the local repository layout.
    """
    p_expanded = Path(os.path.expanduser(p))
    if p_expanded.exists():
        return p_expanded

    if p.startswith("/workspace/shared"):
        rel = Path(p).relative_to("/workspace/shared")
        # Try project_root/shared when provided
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

    # Fallback: return the original (will raise later if missing)
    return p_expanded


def find_metrics_json(model_dir: Path, split: str) -> Optional[Path]:
    """Choose metrics JSON matching split if present."""
    split_matches = sorted(model_dir.glob(f"metrics_{split}_*.json"))
    if split_matches:
        return split_matches[0]
    any_matches = sorted(model_dir.glob("metrics_*.json"))
    return any_matches[0] if any_matches else None


# ---------------------------------------------------------------------------
# Plot helpers (small wrappers so we do not overwrite existing files)
# ---------------------------------------------------------------------------
def save_rmse_time_plot(series: List[Tuple[str, np.ndarray]], out_dir: Path, out_name: str, dt: Optional[float], title: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(9.5, 4.0), dpi=150)
    ax = fig.add_subplot(1, 1, 1)
    max_len = max(s[1].shape[0] for s in series)
    x, x_label = _time_axis(max_len, dt)

    for label, values in series:
        n = values.shape[0]
        ax.plot(x[:n], values, linewidth=1.2, label=label)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("RMSE [A]")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    out_path = out_dir / out_name
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def save_pair_overlay(results: Dict[str, BenchmarkResult], keys: List[str], out_dir: Path, out_name: str, max_samples: int = 800) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(keys), figsize=(12, 4.5), dpi=150)
    if len(keys) == 1:
        axes = [axes]
    for ax, k in zip(axes, keys):
        res = results[k]
        gt_mean = np.mean(res.tau_gt_valid, axis=1)
        delan_mean = np.mean(res.tau_delan_valid, axis=1)
        comb_mean = np.mean(res.tau_combined_valid, axis=1)
        n = min(max_samples, gt_mean.shape[0])
        ax.plot(gt_mean[:n], label="GT mean")
        ax.plot(delan_mean[:n], label="DeLaN mean")
        ax.plot(comb_mean[:n], label="Combined mean")
        ax.set_title(res.dataset)
        ax.set_xlabel("Sample")
        ax.set_ylabel("$i_{motor}$ [A]")
        ax.grid(True, alpha=0.25)
        if ax is axes[0]:
            ax.legend()
    fig.tight_layout()
    out_path = out_dir / out_name
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def save_pair_rmse_time(results: Dict[str, BenchmarkResult], keys: List[str], out_dir: Path, out_name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10.5, 4.2), dpi=150)
    ax = fig.add_subplot(1, 1, 1)

    colors = {
        "UR3": ("C0", "C1"),
        "UR10": ("C3", "C4"),
    }

    for k in keys:
        res = results[k]
        prefix = "UR3" if "UR3" in k else "UR10"
        c_delan, c_comb = colors.get(prefix, ("C0", "C1"))
        x_delan = np.arange(res.rmse_time["delan"].shape[0])
        x_comb = np.arange(res.rmse_time["combined"].shape[0])
        ax.plot(x_delan, res.rmse_time["delan"], color=c_delan, linewidth=1.2, label=f"{res.dataset} DeLaN")
        ax.plot(x_comb, res.rmse_time["combined"], color=c_comb, linewidth=1.2, label=f"{res.dataset} Combined", linestyle="--")

    ax.set_title("RMSE over progress (UR3_Load0 vs UR10_Load0)")
    ax.set_xlabel("Sample")
    ax.set_ylabel("$i_{motor}$ RMSE [A]")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    out_path = out_dir / out_name
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def save_pair_rmse_per_joint(results: Dict[str, BenchmarkResult], keys: List[str], out_dir: Path, out_name: str, max_joints: int = 6) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10.5, 4.2), dpi=170)
    ax = fig.add_subplot(1, 1, 1)

    J = min(max_joints, results[keys[0]].n_dof)
    x = np.arange(J)
    width = 0.18
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]

    for key, off, color in zip(keys * 2, offsets, ["C0", "C1", "C3", "C4"]):
        res = results[key]
        mode = "delan" if off in offsets[:2] else "combined"
        vals = np.asarray(res.per_joint[mode])[:J]
        ax.bar(x + off, vals, width=width, label=f"{res.dataset} {mode}", color=color, alpha=0.85)

    ax.set_title("Per-joint RMSE comparison (UR3_Load0 vs UR10_Load0)")
    ax.set_xlabel("Joint")
    ax.set_ylabel("$i_{motor}$ RMSE [A]")
    ax.set_xticks(x)
    ax.set_xticklabels([f"j{j}" for j in range(J)])
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    out_path = out_dir / out_name
    fig.savefig(out_path, dpi=190)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------
def evaluate_benchmark_model(
    model_dir: Path,
    metrics_path: Path,
    paths: BenchmarkPaths,
    *,
    dataset_name: str,
    split: str,
    batch: int,
    max_plot_samples: int,
) -> BenchmarkResult:
    with open(metrics_path, "r") as f:
        meta = json.load(f)

    H = int(meta["H"])
    feature_mode = meta.get("feature_mode", "full")
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"Unsupported feature mode '{feature_mode}' in {metrics_path}")

    data = np.load(paths.residual_npz, allow_pickle=True)
    q_list = list(data[f"{split}_q"])
    qd_list = list(data[f"{split}_qd"])
    qdd_list = list(data[f"{split}_qdd"])
    tau_list = list(data[f"{split}_tau"])
    tau_hat_list = list(data[f"{split}_tau_hat"])
    r_tau_list = list(data[f"{split}_r_tau"])
    t_list = list(data[f"{split}_t"]) if f"{split}_t" in data else None

    n_traj = len(q_list)
    n_dof = int(np.asarray(q_list[0]).shape[1])

    model = tf.keras.models.load_model(paths.lstm_model)
    sc = np.load(paths.lstm_scalers)
    x_mean = sc["x_mean"].astype(np.float32)
    x_std = sc["x_std"].astype(np.float32)
    y_mean = sc["y_mean"].astype(np.float32)
    y_std = sc["y_std"].astype(np.float32)

    tau_gt_valid_all: List[np.ndarray] = []
    tau_delan_valid_all: List[np.ndarray] = []
    tau_combined_valid_all: List[np.ndarray] = []
    r_gt_valid_all: List[np.ndarray] = []
    r_hat_valid_all: List[np.ndarray] = []

    for i in range(n_traj):
        q = np.asarray(q_list[i], dtype=np.float32)
        qd = np.asarray(qd_list[i], dtype=np.float32)
        qdd = np.asarray(qdd_list[i], dtype=np.float32)
        tau = np.asarray(tau_list[i], dtype=np.float32)
        tau_hat = np.asarray(tau_hat_list[i], dtype=np.float32)
        r_gt = np.asarray(r_tau_list[i], dtype=np.float32)

        T = q.shape[0]
        if T < H:
            # Not enough samples to form a window
            continue

        feat = build_features(q, qd, qdd, tau_hat, mode=feature_mode)
        feat_n = apply_x_scaler_feat(feat, x_mean, x_std)
        X = build_windows(feat_n, H)

        r_hat_valid_scaled = model.predict(X, batch_size=batch, verbose=0).astype(np.float32)
        r_hat_valid = invert_y_scaler(r_hat_valid_scaled, y_mean, y_std)

        r_hat_full = np.full((T, n_dof), np.nan, dtype=np.float32)
        tau_combined_full = np.full((T, n_dof), np.nan, dtype=np.float32)
        r_hat_full[H - 1 :] = r_hat_valid
        tau_combined_full[H - 1 :] = tau_hat[H - 1 :] + r_hat_valid

        tau_gt_valid_all.append(tau[H - 1 :])
        tau_delan_valid_all.append(tau_hat[H - 1 :])
        tau_combined_valid_all.append(tau_combined_full[H - 1 :])
        r_gt_valid_all.append(r_gt[H - 1 :])
        r_hat_valid_all.append(r_hat_valid)

        if (i + 1) % 10 == 0 or (i + 1) == n_traj:
            print(f"[{dataset_name}] processed {i+1}/{n_traj} trajectories", flush=True)

    tau_gt_valid = concat_valid(tau_gt_valid_all)
    tau_delan_valid = concat_valid(tau_delan_valid_all)
    tau_combined_valid = concat_valid(tau_combined_valid_all)
    r_gt_valid = concat_valid(r_gt_valid_all)
    r_hat_valid = concat_valid(r_hat_valid_all)

    if tau_gt_valid.size == 0:
        raise RuntimeError(f"No valid samples found for {dataset_name} (H={H}, split={split})")

    delan_rmse = rmse(tau_gt_valid, tau_delan_valid)
    combined_rmse = rmse(tau_gt_valid, tau_combined_valid)
    residual_rmse = rmse(r_gt_valid, r_hat_valid)

    metrics = {
        "delan_rmse": delan_rmse,
        "combined_rmse": combined_rmse,
        "residual_rmse": residual_rmse,
        "delan_mse": mse(tau_gt_valid, tau_delan_valid),
        "combined_mse": mse(tau_gt_valid, tau_combined_valid),
        "residual_mse": mse(r_gt_valid, r_hat_valid),
    }

    per_joint_metrics = {
        "delan": per_joint_rmse(tau_gt_valid, tau_delan_valid).tolist(),
        "combined": per_joint_rmse(tau_gt_valid, tau_combined_valid).tolist(),
        "residual": per_joint_rmse(r_gt_valid, r_hat_valid).tolist(),
    }

    rmse_time_metrics = {
        "delan": rmse_over_time(tau_gt_valid, tau_delan_valid),
        "combined": rmse_over_time(tau_gt_valid, tau_combined_valid),
        "residual": rmse_over_time(r_gt_valid, r_hat_valid),
    }

    dt_val = infer_dt(t_list)

    # --- Plots inside model dir ---
    out_dir = model_dir
    save_residual_overlay_grid(
        r_gt_valid,
        r_hat_valid,
        out_dir=str(out_dir),
        max_samples=max_plot_samples,
        title=f"$i_{{motor}}$ residual: GT vs LSTM ({split}, valid k>=H-1)",
        out_name=f"residual_gt_vs_pred_{split}_H{H}_bench.png",
    )

    save_torque_overlay_grid(
        tau_gt_valid,
        tau_delan_valid,
        tau_combined_valid,
        out_dir=str(out_dir),
        max_samples=max_plot_samples,
        title=f"$i_{{motor}}$: GT vs DeLaN vs Combined ({split}, valid k>=H-1)",
        out_name=f"torque_gt_vs_delan_vs_combined_{split}_H{H}_bench.png",
    )

    save_torque_rmse_per_joint_grouped_bar(
        tau_gt_valid,
        tau_delan_valid,
        tau_combined_valid,
        out_dir=str(out_dir),
        max_samples=None,
        max_joints=n_dof,
        title=f"$i_{{motor}}$ RMSE per joint ({split}, valid k>=H-1)",
        out_name=f"torque_rmse_per_joint_grouped_{split}_H{H}_bench.png",
    )

    save_torque_rmse_time_curve(
        tau_gt_valid,
        tau_delan_valid,
        tau_combined_valid,
        out_dir=str(out_dir),
        dt=dt_val,
        max_samples=max_plot_samples,
        max_joints=n_dof,
        title=f"$i_{{motor}}$ RMSE over time (joint-avg) ({split}, valid k>=H-1)",
        out_name=f"torque_rmse_time_{split}_H{H}_bench.png",
    )

    save_rmse_time_plot(
        [
            ("DeLaN", rmse_time_metrics["delan"]),
            ("Combined", rmse_time_metrics["combined"]),
            ("Residual LSTM", rmse_time_metrics["residual"]),
        ],
        out_dir,
        out_name=f"rmse_time_{split}_H{H}_bench.png",
        dt=dt_val,
        title=f"RMSE over progress ({dataset_name}, H={H}, split={split})",
    )

    # --- Persist metrics ---
    summary = {
        "dataset": dataset_name,
        "split": split,
        "H": H,
        "feature_mode": feature_mode,
        "n_traj": n_traj,
        "n_dof": n_dof,
        "dt": dt_val,
        "paths": {
            "metrics_json": str(metrics_path),
            "residual_npz": str(paths.residual_npz),
            "lstm_model": str(paths.lstm_model),
            "lstm_scalers": str(paths.lstm_scalers),
        },
        "metrics": metrics,
        "per_joint_rmse": per_joint_metrics,
        "rmse_time_lengths": {k: int(v.shape[0]) for k, v in rmse_time_metrics.items()},
    }

    out_json = model_dir / f"benchmark_eval_{split}_H{H}.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    out_npz = model_dir / f"benchmark_eval_rmse_time_{split}_H{H}.npz"
    np.savez(
        out_npz,
        delan=rmse_time_metrics["delan"],
        combined=rmse_time_metrics["combined"],
        residual=rmse_time_metrics["residual"],
    )

    print(f"[{dataset_name}] wrote {out_json.name} and {out_npz.name}")

    return BenchmarkResult(
        dataset=dataset_name,
        split=split,
        H=H,
        feature_mode=feature_mode,
        n_dof=n_dof,
        n_traj=n_traj,
        dt=dt_val,
        metrics=metrics,
        per_joint=per_joint_metrics,
        rmse_time=rmse_time_metrics,
        tau_gt_valid=tau_gt_valid,
        tau_delan_valid=tau_delan_valid,
        tau_combined_valid=tau_combined_valid,
        r_gt_valid=r_gt_valid,
        r_hat_valid=r_hat_valid,
        out_dir=model_dir,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-root", type=str, default=None, help="Root folder containing benchmark model bundles.")
    ap.add_argument("--raw-root", type=str, default=None, help="Optional raw CSV root (unused but logged).")
    ap.add_argument("--split", type=str, default="test", choices=["test", "train"], help="Split to evaluate.")
    ap.add_argument("--batch", type=int, default=256, help="Batch size for LSTM inference.")
    ap.add_argument("--max-plot-samples", type=int, default=800, help="Max samples to show in overlay plots.")
    args = ap.parse_args()

    def first_existing(paths):
        for p in paths:
            if p and Path(p).exists():
                return Path(p)
        return None

    # Determine bench/data roots, try host and container layouts.
    if args.bench_root:
        bench_root = Path(args.bench_root)
    else:
        bench_root = first_existing(
            [
                _SCRIPT_DIR.parents[i] / "shared" / "evaluation" / "benchmark_models"
                for i in range(len(_SCRIPT_DIR.parents))
            ]
            + ["/workspace/shared/evaluation/benchmark_models"]
        )
    if bench_root is None:
        raise FileNotFoundError("Could not locate benchmark_models directory; pass --bench-root explicitly.")

    if args.raw_root:
        raw_root = Path(args.raw_root)
    else:
        raw_root = first_existing(
            [
                _SCRIPT_DIR.parents[i] / "shared" / "data" / "raw"
                for i in range(len(_SCRIPT_DIR.parents))
            ]
            + ["/workspace/shared/data/raw"]
        )
    if raw_root is None:
        raw_root = Path("shared/data/raw")  # only for logging; may not be used

    if not bench_root.exists():
        raise FileNotFoundError(f"Benchmark root not found: {bench_root}")

    model_dirs = sorted([p for p in bench_root.iterdir() if p.is_dir()])
    if not model_dirs:
        raise RuntimeError(f"No benchmark model directories found in {bench_root}")

    print(f"Benchmark root: {bench_root}")
    print(f"Raw data root : {raw_root}")

    results: Dict[str, BenchmarkResult] = {}

    for model_dir in model_dirs:
        dataset_name = model_dir.name.split("__", 1)[0]
        metrics_json = find_metrics_json(model_dir, args.split)
        if not metrics_json:
            print(f"[skip] no metrics JSON in {model_dir}")
            continue

        with open(metrics_json, "r") as f:
            meta = json.load(f)

        paths_cfg = meta.get("paths", {})
        parent_root = bench_root.parents[2] if len(bench_root.parents) >= 3 else None
        bp = BenchmarkPaths(
            residual_npz=resolve_local_path(paths_cfg["residual_npz"], parent_root),
            lstm_model=resolve_local_path(paths_cfg["lstm_model"], parent_root),
            lstm_scalers=resolve_local_path(paths_cfg["lstm_scalers"], parent_root),
        )

        # Sanity check that referenced files exist
        for label, p in [("residual", bp.residual_npz), ("lstm_model", bp.lstm_model), ("lstm_scalers", bp.lstm_scalers)]:
            if not p.exists():
                raise FileNotFoundError(f"{label} path missing for {dataset_name}: {p}")

        result = evaluate_benchmark_model(
            model_dir,
            metrics_json,
            bp,
            dataset_name=dataset_name,
            split=args.split,
            batch=args.batch,
            max_plot_samples=args.max_plot_samples,
        )
        results[dataset_name] = result

    # ------------------------------------------------------------------
    # UR3 vs UR10 comparison plots (only if both present)
    # ------------------------------------------------------------------
    comp_dir = bench_root / "_comparison"
    ur3_key = next((k for k in results if k.startswith("UR3_Load0")), None)
    ur10_key = next((k for k in results if k.startswith("UR10_Load0")), None)
    comp_keys = [k for k in (ur3_key, ur10_key) if k]

    if len(comp_keys) == 2:
        save_pair_overlay(results, comp_keys, comp_dir, "compare_ur3_vs_ur10_overlay.png")
        save_pair_rmse_time(results, comp_keys, comp_dir, "compare_ur3_vs_ur10_rmse_time.png")
        save_pair_rmse_per_joint(results, comp_keys, comp_dir, "compare_ur3_vs_ur10_rmse_per_joint.png")
        print(f"Wrote UR3 vs UR10 comparison plots to {comp_dir}")
    else:
        print("UR3_Load0 and UR10_Load0 benchmarks not both found; skipping comparison plots.")


if __name__ == "__main__":
    main()
