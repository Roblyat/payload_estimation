#!/usr/bin/env python3
"""
Plot baseline vs DeLaN+LSTM comparisons from final_metrics JSONs.

Inputs:
  - baseline_metrics.json
  - delan_lstm_metrics.json
Output:
  - PNGs in /shared/data/benchmark (or chosen output dir)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import os as _os

# Avoid permission issues inside containers where /workspace/shared/.mplconfig is read-only
_os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/.mplconfig")))
Path(_os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def load_metrics(path: Path):
    with open(path, "r") as f:
        data = json.load(f)
    out = {}
    for entry in data["datasets"]:
        key = (entry["dataset"], entry["condition"])
        out[key] = {
            "per_joint_rmse": np.asarray(entry["per_joint_rmse"], dtype=np.float32),
            "per_joint_rmse_pct": np.asarray(entry["per_joint_rmse_pct"], dtype=np.float32),
        }
        out[key]["overall_rmse"] = float(np.mean(out[key]["per_joint_rmse"]))
        out[key]["overall_rmse_pct"] = float(np.mean(out[key]["per_joint_rmse_pct"]))
    return out


def label_key(key):
    dataset, cond = key
    base = "UR3" if "UR3" in dataset else ("UR10" if "UR10" in dataset else dataset)
    suffix = "no load" if cond == "without_load" else "with load"
    return f"{base} ({suffix})"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_overall_rmse(baseline, ours, out_dir: Path):
    # Overall RMSE per robot/load: baseline vs DeLaN+LSTM
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
    fig.savefig(out_dir / "final_overall_rmse_baseline_vs_pipeline.png", dpi=200)
    plt.close(fig)


def plot_per_joint_bars(baseline, ours, out_dir: Path):
    # Per-joint RMSE comparison for each robot/load
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
    fig.savefig(out_dir / "final_per_joint_rmse_baseline_vs_pipeline.png", dpi=200)
    plt.close(fig)


def plot_relative_change(baseline, ours, out_dir: Path):
    # Relative change per joint: negative = improvement
    keys = [k for k in baseline.keys() if k in ours]
    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.2), dpi=150, sharey=True)
    if n == 1:
        axes = [axes]
    for ax, k in zip(axes, keys):
        base = baseline[k]["per_joint_rmse"]
        ours_vals = ours[k]["per_joint_rmse"]
        change = (ours_vals - base) / base
        x = np.arange(len(change))
        colors = ["C2" if v < 0 else "C3" for v in change]
        ax.bar(x, change, color=colors, alpha=0.85)
        ax.axhline(0, color="k", linewidth=0.8)
        ax.set_title(f"Δ RMSE vs baseline\n{label_key(k)}")
        ax.set_xlabel("Joint")
        ax.set_xticks(x, [f"j{j}" for j in x])
        if ax is axes[0]:
            ax.set_ylabel("Relative change (ours-baseline)/baseline\n(0.01 = 1%)")
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "final_per_joint_relative_change.png", dpi=200)
    plt.close(fig)


def plot_load_effect_ur3(baseline, ours, out_dir: Path):
    # Load sensitivity on UR3
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
    fig.savefig(out_dir / "final_ur3_load_effect.png", dpi=200)
    plt.close(fig)


def plot_kinematics_diff(baseline, ours, out_dir: Path):
    # UR3 vs UR10 (no load)
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
    fig.savefig(out_dir / "final_ur3_vs_ur10_no_load.png", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    def first_existing(paths):
        for p in paths:
            if p and Path(p).exists():
                return Path(p).resolve()
        return None

    def find_metrics_dir(script_dir: Path) -> Path | None:
        candidates: list[Path] = [
            Path("/workspace/shared/data/benchmark"),
            Path("/workspace/shared/data/benchmark/final_metrics"),
            Path("/workspace/shared/evaluation/final_metrics"),
        ]
        for parent in script_dir.parents:
            candidates.extend(
                [
                    parent / "shared" / "data" / "benchmark",
                    parent / "shared" / "data" / "benchmark" / "final_metrics",
                    parent / "shared" / "evaluation" / "final_metrics",
                ]
            )
        for c in candidates:
            if (c / "baseline_metrics.json").exists() and (c / "delan_lstm_metrics.json").exists():
                return c.resolve()
        return None

    script_dir = Path(__file__).resolve().parent
    default_base = find_metrics_dir(script_dir)
    if default_base is None:
        default_base = first_existing(
            [
                script_dir.parents[i] / "shared" / "data" / "benchmark"
                for i in range(len(script_dir.parents))
            ]
            + [Path("/workspace/shared/data/benchmark")]
        )
    if default_base is None:
        default_base = Path("payload_estimation/shared/data/benchmark").resolve()
    ap.add_argument("--baseline", type=str, default=str(default_base / "baseline_metrics.json"),
                    help="Path to baseline_metrics.json")
    ap.add_argument("--pipeline", type=str, default=str(default_base / "delan_lstm_metrics.json"),
                    help="Path to delan_lstm_metrics.json")
    ap.add_argument("--out-dir", type=str, default=None, help="Output directory for plots")
    args = ap.parse_args()

    baseline = load_metrics(Path(args.baseline))
    pipeline = load_metrics(Path(args.pipeline))

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.baseline).parent
    ensure_dir(out_dir)

    plot_overall_rmse(baseline, pipeline, out_dir)
    plot_per_joint_bars(baseline, pipeline, out_dir)
    plot_relative_change(baseline, pipeline, out_dir)
    plot_load_effect_ur3(baseline, pipeline, out_dir)
    plot_kinematics_diff(baseline, pipeline, out_dir)

    print(f"Wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
