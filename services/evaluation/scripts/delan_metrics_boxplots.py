#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _safe_mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _parse_delan_tag(tag: str) -> Dict[str, Any]:
    # tag like: delan_struct_s4_ep200 or delan_black_s4_ep500
    out: Dict[str, Any] = {"delan_tag": tag}
    m = re.search(r"delan_(struct|black)_s(\d+)_ep(\d+)", tag)
    if m:
        out["model_short"] = m.group(1)
        out["seed"] = int(m.group(2))
        out["epochs"] = int(m.group(3))
    else:
        out["model_short"] = "unknown"
    return out


def _boxplot(groups: Dict[str, List[float]], title: str, ylabel: str, out_png: str) -> None:
    labels = sorted(groups.keys())
    data = [groups[k] for k in labels]
    plt.figure(figsize=(max(10, 1.2 * len(labels)), 5), dpi=140)
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def _per_joint_boxplot(
    groups_per_joint: Dict[str, List[List[float]]],  # group -> [joint0_list, ..., joint5_list]
    title: str,
    ylabel: str,
    out_png: str,
    n_dof: int,
) -> None:
    labels = sorted(groups_per_joint.keys())
    fig = plt.figure(figsize=(14, 8), dpi=140)
    for j in range(n_dof):
        ax = fig.add_subplot(3, 2, j + 1)
        data = [groups_per_joint[g][j] for g in labels]
        ax.boxplot(data, labels=labels, showmeans=True)
        ax.set_title(f"Joint {j}")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close(fig)


def _scatter(xs, ys, xlabel: str, ylabel: str, title: str, out_png: str) -> None:
    plt.figure(figsize=(7, 5), dpi=140)
    plt.scatter(xs, ys)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delan_root", default="/workspace/shared/models/delan", help="Root folder containing delan/*/metrics.json")
    ap.add_argument("--out_dir", default="/workspace/shared/models/delan/_plots", help="Where to save plots")
    args = ap.parse_args()

    _safe_mkdir(args.out_dir)

    metrics_paths = sorted(glob.glob(os.path.join(args.delan_root, "*", "metrics.json")))
    if not metrics_paths:
        print(f"[delan_metrics_boxplots] No metrics.json found under: {args.delan_root}")
        return

    rows: List[Dict[str, Any]] = []
    for mp in metrics_paths:
        run_dir = os.path.dirname(mp)
        run_name = os.path.basename(run_dir)  # delan_ur5_B__delan_struct_s4_ep200
        try:
            with open(mp, "r") as f:
                d = json.load(f)
        except Exception as e:
            print(f"Skipping {mp}: {e}")
            continue

        # delan_tag is suffix after "__" if present
        parts = run_name.split("__")
        delan_tag = parts[-1] if len(parts) >= 2 else "unknown"
        tag_info = _parse_delan_tag(delan_tag)

        eval_test = d.get("eval_test", {})
        torque_rmse = float(eval_test.get("torque_rmse", np.nan))
        joint_rmse = eval_test.get("torque_rmse_per_joint", None)

        # speed
        tps = d.get("time_per_sample_s", None)
        hz = d.get("hz", None)

        row = {
            "run_name": run_name,
            "run_dir": run_dir,
            "delan_tag": tag_info.get("delan_tag", delan_tag),
            "model_short": tag_info.get("model_short", "unknown"),
            "seed": tag_info.get("seed", None),
            "epochs": tag_info.get("epochs", None),
            "torque_rmse": torque_rmse,
            "time_per_sample_s": float(tps) if tps is not None else np.nan,
            "hz": float(hz) if hz is not None else np.nan,
        }

        if isinstance(joint_rmse, list):
            row["n_dof"] = len(joint_rmse)
            row["torque_rmse_per_joint"] = [float(x) for x in joint_rmse]
        else:
            row["n_dof"] = None
            row["torque_rmse_per_joint"] = None

        # keep hyper + dataset info for later (optional)
        row["hyper"] = d.get("hyper", {})
        row["dataset"] = d.get("dataset", {})
        rows.append(row)

    # infer dof
    n_dof = None
    for r in rows:
        if isinstance(r.get("torque_rmse_per_joint"), list):
            n_dof = len(r["torque_rmse_per_joint"])
            break
    if n_dof is None:
        n_dof = 6  # fallback

    # 1) Torque RMSE boxplot grouped by model_short
    groups: Dict[str, List[float]] = {}
    for r in rows:
        g = r["model_short"]
        groups.setdefault(g, []).append(r["torque_rmse"])
    _boxplot(
        groups,
        title="DeLaN torque RMSE (test) grouped by model type",
        ylabel="torque_rmse",
        out_png=os.path.join(args.out_dir, "delan_torque_rmse_by_model_type.png"),
    )

    # 2) Torque RMSE boxplot grouped by delan_tag (seed/epochs)
    groups = {}
    for r in rows:
        g = r["delan_tag"]
        groups.setdefault(g, []).append(r["torque_rmse"])
    _boxplot(
        groups,
        title="DeLaN torque RMSE (test) grouped by delan_tag",
        ylabel="torque_rmse",
        out_png=os.path.join(args.out_dir, "delan_torque_rmse_by_delan_tag.png"),
    )

    # 3) Per-joint RMSE grid grouped by delan_tag
    groups_per_joint: Dict[str, List[List[float]]] = {}
    for r in rows:
        pj = r.get("torque_rmse_per_joint")
        if not isinstance(pj, list):
            continue
        g = r["delan_tag"]
        if g not in groups_per_joint:
            groups_per_joint[g] = [[] for _ in range(n_dof)]
        for j in range(n_dof):
            groups_per_joint[g][j].append(float(pj[j]))
    if groups_per_joint:
        _per_joint_boxplot(
            groups_per_joint,
            title="DeLaN per-joint RMSE (test) grouped by delan_tag",
            ylabel="joint torque_rmse",
            out_png=os.path.join(args.out_dir, "delan_joint_rmse_grid_by_delan_tag.png"),
            n_dof=n_dof,
        )

    # 4) Speed vs accuracy scatter (time_per_sample_s vs torque_rmse)
    xs = [r["time_per_sample_s"] for r in rows if np.isfinite(r["time_per_sample_s"]) and np.isfinite(r["torque_rmse"])]
    ys = [r["torque_rmse"] for r in rows if np.isfinite(r["time_per_sample_s"]) and np.isfinite(r["torque_rmse"])]
    if xs and ys:
        _scatter(
            xs, ys,
            xlabel="time_per_sample_s",
            ylabel="torque_rmse",
            title="DeLaN speed vs accuracy (lower is better)",
            out_png=os.path.join(args.out_dir, "delan_speed_vs_rmse.png"),
        )

    # 5) Seed stability (if seeds present)
    seed_groups: Dict[str, List[float]] = {}
    for r in rows:
        if r.get("seed") is None:
            continue
        seed_groups.setdefault(f"s{r['seed']}", []).append(r["torque_rmse"])
    if len(seed_groups) >= 2:
        _boxplot(
            seed_groups,
            title="DeLaN torque RMSE (test) grouped by seed",
            ylabel="torque_rmse",
            out_png=os.path.join(args.out_dir, "delan_torque_rmse_by_seed.png"),
        )

    print(f"[delan_metrics_boxplots] Saved plots to: {args.out_dir}")


if __name__ == "__main__":
    main()