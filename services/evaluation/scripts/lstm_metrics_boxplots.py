#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _safe_mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _parse_lstm_folder(name: str) -> Dict[str, Any]:
    # Example:
    # ur5__B__delan_struct_s4_ep500__feat_full__lstm_s4_H60_ep60_b64_u128_do0p2
    parts = name.split("__")
    out: Dict[str, Any] = {"folder": name}
    if len(parts) >= 5:
        out["dataset"] = parts[0]
        out["run_tag"] = parts[1]
        out["delan_tag"] = parts[2]
        out["feature_mode"] = parts[3].replace("feat_", "")
        out["lstm_tag"] = parts[4].replace("lstm_", "")
    else:
        out["dataset"] = None
        out["run_tag"] = None
        out["delan_tag"] = None
        out["feature_mode"] = None
        out["lstm_tag"] = None

    # Extract H from lstm_tag
    mH = re.search(r"_H(\d+)", out.get("lstm_tag") or "")
    out["H"] = int(mH.group(1)) if mH else None

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
    groups_per_joint: Dict[str, List[List[float]]],
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
    ap.add_argument("--lstm_root", default="/workspace/shared/models/lstm", help="Root folder containing lstm runs")
    ap.add_argument("--out_dir", default="/workspace/shared/models/lstm/_plots", help="Where to save plots")
    args = ap.parse_args()

    _safe_mkdir(args.out_dir)

    metric_paths = sorted(glob.glob(os.path.join(args.lstm_root, "*", "metrics_train_test_H*.json")))
    if not metric_paths:
        print(f"[lstm_metrics_boxplots] No metrics_train_test_H*.json found under: {args.lstm_root}")
        return

    rows: List[Dict[str, Any]] = []
    for mp in metric_paths:
        run_dir = os.path.dirname(mp)
        folder = os.path.basename(run_dir)
        meta = _parse_lstm_folder(folder)

        try:
            with open(mp, "r") as f:
                d = json.load(f)
        except Exception as e:
            print(f"Skipping {mp}: {e}")
            continue

        eval_test = d.get("eval_test", {})
        rmse_total = float(eval_test.get("rmse_total", np.nan))
        rmse_per_joint = eval_test.get("rmse_per_joint", None)
        args_d = d.get("args", {})
        train_d = d.get("train", {})

        row = {
            "folder": folder,
            "run_dir": run_dir,
            "metrics_path": mp,
            "feature_mode": meta.get("feature_mode"),
            "H": int(d.get("H", meta.get("H") or 0)),
            "rmse_total": rmse_total,
            "best_val_loss": float(train_d.get("best_val_loss", np.nan)),
            "final_train_loss": float(train_d.get("final_train_loss", np.nan)),
            "final_val_loss": float(train_d.get("final_val_loss", np.nan)),
            "best_epoch": int(train_d.get("best_epoch", -1)) if train_d.get("best_epoch") is not None else -1,
            "seed": int(args_d.get("seed", -1)) if args_d.get("seed") is not None else -1,
            "units": int(args_d.get("units", -1)) if args_d.get("units") is not None else -1,
            "dropout": float(args_d.get("dropout", np.nan)) if args_d.get("dropout") is not None else np.nan,
        }

        if isinstance(rmse_per_joint, list):
            row["n_dof"] = len(rmse_per_joint)
            row["rmse_per_joint"] = [float(x) for x in rmse_per_joint]
        else:
            row["n_dof"] = None
            row["rmse_per_joint"] = None

        rows.append(row)

    # infer dof
    n_dof = None
    for r in rows:
        if isinstance(r.get("rmse_per_joint"), list):
            n_dof = len(r["rmse_per_joint"])
            break
    if n_dof is None:
        n_dof = 6

    # 1) Residual RMSE boxplot grouped by feature_mode
    g1: Dict[str, List[float]] = {}
    for r in rows:
        g = r.get("feature_mode") or "unknown"
        g1.setdefault(g, []).append(r["rmse_total"])
    _boxplot(
        g1,
        title="LSTM residual RMSE (test) grouped by feature_mode",
        ylabel="rmse_total (residual torque units)",
        out_png=os.path.join(args.out_dir, "lstm_residual_rmse_by_feature_mode.png"),
    )

    # 1b) Residual RMSE grouped by H (optional helpful)
    gH: Dict[str, List[float]] = {}
    for r in rows:
        gH.setdefault(f"H{r['H']}", []).append(r["rmse_total"])
    if len(gH) >= 2:
        _boxplot(
            gH,
            title="LSTM residual RMSE (test) grouped by H",
            ylabel="rmse_total",
            out_png=os.path.join(args.out_dir, "lstm_residual_rmse_by_H.png"),
        )

    # 2) Per-joint residual RMSE grid grouped by feature_mode
    groups_per_joint: Dict[str, List[List[float]]] = {}
    for r in rows:
        pj = r.get("rmse_per_joint")
        if not isinstance(pj, list):
            continue
        g = r.get("feature_mode") or "unknown"
        if g not in groups_per_joint:
            groups_per_joint[g] = [[] for _ in range(n_dof)]
        for j in range(n_dof):
            groups_per_joint[g][j].append(float(pj[j]))
    if groups_per_joint:
        _per_joint_boxplot(
            groups_per_joint,
            title="LSTM per-joint residual RMSE (test) grouped by feature_mode",
            ylabel="joint residual rmse",
            out_png=os.path.join(args.out_dir, "lstm_joint_rmse_grid_by_feature_mode.png"),
            n_dof=n_dof,
        )

    # 3) Generalization scatter: best_val_loss vs rmse_total
    xs = [r["best_val_loss"] for r in rows if np.isfinite(r["best_val_loss"]) and np.isfinite(r["rmse_total"])]
    ys = [r["rmse_total"] for r in rows if np.isfinite(r["best_val_loss"]) and np.isfinite(r["rmse_total"])]
    if xs and ys:
        _scatter(
            xs, ys,
            xlabel="best_val_loss (scaled MSE)",
            ylabel="rmse_total (real units)",
            title="Does scaled validation loss track real-unit residual RMSE?",
            out_png=os.path.join(args.out_dir, "lstm_best_val_loss_vs_rmse_total.png"),
        )

    # 4) Overfit indicator grouped by feature_mode: final_val_loss / final_train_loss
    overfit: Dict[str, List[float]] = {}
    for r in rows:
        tr = r["final_train_loss"]
        va = r["final_val_loss"]
        if not (np.isfinite(tr) and np.isfinite(va)) or tr <= 0:
            continue
        ratio = va / tr
        g = r.get("feature_mode") or "unknown"
        overfit.setdefault(g, []).append(float(ratio))
    if overfit:
        _boxplot(
            overfit,
            title="LSTM overfit indicator grouped by feature_mode (final_val_loss / final_train_loss)",
            ylabel="val/train loss ratio",
            out_png=os.path.join(args.out_dir, "lstm_overfit_ratio_by_feature_mode.png"),
        )

    print(f"[lstm_metrics_boxplots] Saved plots to: {args.out_dir}")


if __name__ == "__main__":
    main()