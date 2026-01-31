#!/usr/bin/env python3
"""
Visual comparisons between the published baseline controller and the DeLaN+LSTM
pipeline for UR3/UR10 in loaded and unloaded settings.

Inputs:
  - Baseline metrics (Table II) JSON: shared/evaluation/benchmark_models/_comparison/baseline_metrics.json
  - Our pipeline eval JSONs: benchmark_eval_<split>_H*.json inside benchmark_models/*/

Outputs (PNG) are written to the comparison folder (default:
shared/evaluation/benchmark_models/_comparison).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def load_baseline(path: Path) -> Dict[Tuple[str, str], Dict[str, np.ndarray]]:
    """Load baseline per-joint RMSE and RMSE% (converted to fraction)."""
    with open(path, "r") as f:
        data = json.load(f)
    out: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
    for entry in data["datasets"]:
        key = (entry["dataset"], entry["condition"])
        out[key] = {
            "per_joint_rmse": np.asarray(entry["per_joint_rmse"], dtype=np.float32),
            # baseline file stores percentages; convert to fraction (0.01 = 1%)
            "per_joint_rmse_frac": np.asarray(entry["per_joint_rmse_pct"], dtype=np.float32) / 100.0,
        }
        out[key]["overall_rmse"] = float(np.mean(out[key]["per_joint_rmse"]))
        out[key]["overall_rmse_frac"] = float(np.mean(out[key]["per_joint_rmse_frac"]))
    return out


def map_dataset(dataset_name: str) -> Tuple[str, str] | None:
    """Map our dataset naming to (robot, condition)."""
    if dataset_name.startswith("UR3_Load0"):
        return ("UR3e", "without_load")
    if dataset_name.startswith("UR3_Load2"):
        return ("UR3e", "with_load")
    if dataset_name.startswith("UR10_Load0"):
        return ("UR10e", "without_load")
    return None  # unsupported (e.g., UR10 with load not in our pipeline)


def load_pipeline_metrics(bench_root: Path, split: str) -> Dict[Tuple[str, str], Dict[str, np.ndarray]]:
    """Scan benchmark_eval_<split> files (any depth) and collect combined metrics."""
    out: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
    for jf in bench_root.rglob(f"benchmark_eval_{split}_*.json"):
        if "_comparison" in jf.parts:
            continue
        with open(jf, "r") as f:
            meta = json.load(f)
        key = map_dataset(meta["dataset"])
        if key is None:
            continue
        per_joint = np.asarray(meta["per_joint_rmse"]["combined"], dtype=np.float32)
        per_joint_frac = np.asarray(meta["per_joint_rmse_pct"]["combined"], dtype=np.float32)
        out[key] = {
            "per_joint_rmse": per_joint,
            "per_joint_rmse_frac": per_joint_frac,
            "overall_rmse": float(np.mean(per_joint)),
            "overall_rmse_frac": float(np.mean(per_joint_frac)),
            "H": meta.get("H", None),
            "feature_mode": meta.get("feature_mode", None),
            "n_traj": meta.get("n_traj", None),
            "source": str(jf),
        }
    return out


def ensure_out_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def label_key(key: Tuple[str, str]) -> str:
    robot, cond = key
    if robot == "UR3e":
        base = "UR3"
    elif robot == "UR10e":
        base = "UR10"
    else:
        base = robot
    suffix = "no load" if cond == "without_load" else "with load"
    return f"{base} ({suffix})"


def plot_overall_rmse(baseline, ours, out_dir: Path):
    # Shows how much the end-to-end RMSE drops when swapping the baseline for DeLaN+LSTM on each available robot/load combo.
    ensure_out_dir(out_dir)
    keys = [k for k in baseline.keys() if k in ours]
    labels = [label_key(k) for k in keys]
    x = np.arange(len(keys))
    width = 0.35

    base_vals = [baseline[k]["overall_rmse"] for k in keys]
    ours_vals = [ours[k]["overall_rmse"] for k in keys]

    fig, ax = plt.subplots(figsize=(9, 4), dpi=150)
    ax.bar(x - width / 2, base_vals, width, label="Baseline")
    ax.bar(x + width / 2, ours_vals, width, label="DeLaN+LSTM")
    ax.set_ylabel("RMSE [A]")
    ax.set_title("Overall RMSE by robot & load")
    ax.set_xticks(x, labels, rotation=12)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "overall_rmse_baseline_vs_pipeline.png", dpi=200)
    plt.close(fig)


def plot_per_joint_bars(baseline, ours, out_dir: Path):
    # For each robot/load, compares per-joint RMSE to reveal which joints benefit most from the pipeline.
    ensure_out_dir(out_dir)
    keys = [k for k in baseline.keys() if k in ours]
    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.2), dpi=150, sharey=True)
    if n == 1:
        axes = [axes]
    width = 0.35
    for ax, k in zip(axes, keys):
        J = len(baseline[k]["per_joint_rmse"])
        x = np.arange(J)
        ax.bar(x - width / 2, baseline[k]["per_joint_rmse"], width, label="Baseline")
        ax.bar(x + width / 2, ours[k]["per_joint_rmse"], width, label="DeLaN+LSTM")
        ax.set_title(label_key(k))
        ax.set_xlabel("Joint")
        ax.set_xticks(x, [f"j{j}" for j in x])
        ax.grid(True, axis="y", alpha=0.25)
        if ax is axes[0]:
            ax.set_ylabel("RMSE [A]")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "per_joint_rmse_baseline_vs_pipeline.png", dpi=200)
    plt.close(fig)


def plot_gain_per_joint(baseline, ours, out_dir: Path):
    # Shows fractional change per joint: negative bars mean the pipeline reduces RMSE (green), positive means worse (red).
    ensure_out_dir(out_dir)
    keys = [k for k in baseline.keys() if k in ours]
    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.2), dpi=150, sharey=True)
    if n == 1:
        axes = [axes]
    for ax, k in zip(axes, keys):
        base = baseline[k]["per_joint_rmse"]
        ours_vals = ours[k]["per_joint_rmse"]
        # Negative => improvement, Positive => degradation
        change = (ours_vals - base) / base
        x = np.arange(len(change))
        colors = ["C2" if v < 0 else "C3" for v in change]  # green for improvement, red for worse
        ax.bar(x, change, color=colors, alpha=0.85)
        ax.axhline(0, color="k", linewidth=0.8)
        ax.set_title(f"Δ RMSE vs baseline\n{label_key(k)}")
        ax.set_xlabel("Joint")
        ax.set_xticks(x, [f"j{j}" for j in x])
        if ax is axes[0]:
            ax.set_ylabel("Relative change (ours-baseline)/baseline\n(0.01 = 1%)")
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "per_joint_relative_gain.png", dpi=200)
    plt.close(fig)


def plot_load_effect_ur3(baseline, ours, out_dir: Path):
    # Compares how adding a load changes performance on UR3 for both baseline and pipeline, highlighting robustness to payload changes.
    ensure_out_dir(out_dir)
    keys = [("UR3e", "without_load"), ("UR3e", "with_load")]
    if not all(k in baseline and k in ours for k in keys):
        return
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(6.5, 4), dpi=150)
    ax.plot(x, [baseline[k]["overall_rmse"] for k in keys], marker="o", label="Baseline", color="C1")
    ax.plot(x, [ours[k]["overall_rmse"] for k in keys], marker="o", label="DeLaN+LSTM", color="C0")
    ax.set_xticks(x, ["UR3 no load", "UR3 with load"])
    ax.set_ylabel("Overall RMSE [A]")
    ax.set_title("Load sensitivity on UR3")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "ur3_load_effect.png", dpi=200)
    plt.close(fig)


def plot_kinematics_diff(baseline, ours, out_dir: Path):
    # Contrasts UR3 vs UR10 (both unloaded) to see how kinematic differences affect errors for each model.
    ensure_out_dir(out_dir)
    k3 = ("UR3e", "without_load")
    k10 = ("UR10e", "without_load")
    if not (k3 in baseline and k3 in ours and k10 in baseline and k10 in ours):
        return
    labels = ["UR3", "UR10"]
    x = np.arange(2)
    width = 0.35
    fig, ax = plt.subplots(figsize=(6.5, 4), dpi=150)
    ax.bar(x - width / 2, [baseline[k3]["overall_rmse"], baseline[k10]["overall_rmse"]], width, label="Baseline")
    ax.bar(x + width / 2, [ours[k3]["overall_rmse"], ours[k10]["overall_rmse"]], width, label="DeLaN+LSTM")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Overall RMSE [A]")
    ax.set_title("Kinematic difference (UR3 vs UR10, no load)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "ur3_vs_ur10_no_load.png", dpi=200)
    plt.close(fig)


def main():
    split = os.environ.get("SPLIT", "test")

    # Locate benchmark root robustly: env override, repo-relative, or container mount.
    def first_existing(paths):
        for p in paths:
            if p and Path(p).exists():
                return Path(p).resolve()
        return None

    env_bench = os.environ.get("BENCH_ROOT")
    script_dir = Path(__file__).resolve().parent
    candidates = [
        env_bench,
        script_dir.parents[2] / "shared" / "evaluation" / "benchmark_models",  # payload_estimation/shared/...
        script_dir.parents[3] / "shared" / "evaluation" / "benchmark_models" if len(script_dir.parents) >= 4 else None,
        Path("payload_estimation/shared/evaluation/benchmark_models"),
        Path("/workspace/shared/evaluation/benchmark_models"),
    ]
    bench_root = first_existing(candidates)
    if bench_root is None:
        raise FileNotFoundError("Could not locate benchmark_models; set BENCH_ROOT env var.")

    comparison_dir = bench_root / "_comparison"
    baseline_json = comparison_dir / "baseline_metrics.json"

    if not bench_root.exists():
        raise FileNotFoundError(f"benchmark root not found: {bench_root}")
    if not baseline_json.exists():
        raise FileNotFoundError(f"baseline metrics not found: {baseline_json}")

    baseline = load_baseline(baseline_json)
    ours = load_pipeline_metrics(bench_root, split)

    if not ours:
        raise RuntimeError(f"No benchmark_eval_{split}_*.json files found under {bench_root}")

    out_dir = ensure_out_dir(comparison_dir)

    plot_overall_rmse(baseline, ours, out_dir)
    plot_per_joint_bars(baseline, ours, out_dir)
    plot_gain_per_joint(baseline, ours, out_dir)
    plot_load_effect_ur3(baseline, ours, out_dir)
    plot_kinematics_diff(baseline, ours, out_dir)

    print(f"Wrote comparison plots to {out_dir}")


if __name__ == "__main__":
    main()
