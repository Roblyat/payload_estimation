#!/usr/bin/env python3
"""
Scan /workspace/shared/evaluation/*/metrics_*_H*.txt and create:
- boxplots across feature_mode
- boxplots across delan_id
- correlation scatters

Assumes your evaluation run dir names contain:
  ...__delan_<DELANTAG>__feat_<FEATUREMODE>__lstm_<LSTMTAG>
Example:
  ur5__B__delan_struct_s4_ep500__feat_full__lstm_s4_H60_ep60_b64_u128_do0p2
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


FEATURE_RE = re.compile(r"__feat_([^_]+)__")
DELAN_RE = re.compile(r"__delan_([^_]+(?:_[^_]+)*)__feat_")  # greedy-ish between markers
LSTM_RE = re.compile(r"__lstm_(.+)$")
H_FROM_FILENAME_RE = re.compile(r"_H(\d+)\.txt$")


def _parse_float_list(s: str) -> List[float]:
    s = s.strip()
    if not s:
        return []
    return [float(x) for x in s.split()]


def parse_metrics_txt(path: str) -> Dict:
    """
    Parse your metrics_test_H*.txt format:
      key=value
      key=v1 v2 v3 ...
    """
    out: Dict[str, object] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()

            # known vector keys
            if k in ("delan_joint_rmse", "res_joint_rmse", "rg_joint_rmse"):
                out[k] = _parse_float_list(v)
                continue

            # scalars
            if k in ("H",):
                out[k] = int(v)
                continue

            # try float
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def parse_run_dir_name(run_dir: str) -> Dict[str, Optional[str]]:
    """
    Pull ids from run folder name.
    """
    name = os.path.basename(run_dir.rstrip("/"))

    feat = None
    m = FEATURE_RE.search(name)
    if m:
        feat = m.group(1)

    delan_id = None
    m = DELAN_RE.search(name)
    if m:
        delan_id = m.group(1)

    lstm_id = None
    m = LSTM_RE.search(name)
    if m:
        lstm_id = m.group(1)

    # dataset + run_tag = first two tokens separated by "__" (your convention)
    toks = name.split("__")
    dataset = toks[0] if len(toks) >= 1 else None
    run_tag = toks[1] if len(toks) >= 2 else None

    return {
        "dataset": dataset,
        "run_tag": run_tag,
        "feature_mode": feat,
        "delan_id": delan_id,
        "lstm_id": lstm_id,
        "run_dir": name,
    }


def save_boxplot(df: pd.DataFrame, group_col: str, value_col: str, out_path: str, title: str):
    g = df[[group_col, value_col]].dropna()
    if g.empty:
        print(f"[warn] Empty data for boxplot {value_col} by {group_col}")
        return

    order = sorted(g[group_col].unique().tolist())
    data = [g.loc[g[group_col] == k, value_col].values for k in order]

    plt.figure(figsize=(10, 4), dpi=140)
    plt.boxplot(data, labels=order, showfliers=True)
    plt.title(title)
    plt.ylabel(value_col)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"Saved: {out_path}")


def save_scatter(df: pd.DataFrame, x: str, y: str, out_path: str, title: str):
    g = df[[x, y]].dropna()
    if g.empty:
        print(f"[warn] Empty data for scatter {y} vs {x}")
        return

    plt.figure(figsize=(5.5, 5.0), dpi=140)
    plt.scatter(g[x].values, g[y].values, s=18)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_root", default="/workspace/shared/evaluation",
                    help="Root folder that contains per-run evaluation subfolders.")
    ap.add_argument("--out_dir", default="/workspace/shared/evaluation/_plots",
                    help="Where to save plots + summary CSV.")
    ap.add_argument("--pattern", default="*/metrics_*_H*.txt",
                    help="Glob pattern under eval_root.")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.eval_root, args.pattern)))
    if not paths:
        raise SystemExit(f"No metrics files found with: {os.path.join(args.eval_root, args.pattern)}")

    rows = []
    for p in paths:
        run_dir = os.path.dirname(p)
        name_info = parse_run_dir_name(run_dir)
        met = parse_metrics_txt(p)

        # H from filename if missing
        if "H" not in met:
            m = H_FROM_FILENAME_RE.search(os.path.basename(p))
            if m:
                met["H"] = int(m.group(1))

        row = {**name_info}
        # expected keys from your txt file
        for k in [
            "split", "H",
            "delan_mse", "delan_rmse",
            "res_mse", "res_rmse",
            "rg_mse", "rg_rmse",
        ]:
            row[k] = met.get(k, None)

        # store joints as fixed columns
        dj = met.get("delan_joint_rmse", [])
        rj = met.get("res_joint_rmse", [])
        gj = met.get("rg_joint_rmse", [])
        for j in range(6):  # UR5 dof=6 in your pipeline
            row[f"delan_joint_rmse_{j}"] = dj[j] if j < len(dj) else np.nan
            row[f"res_joint_rmse_{j}"] = rj[j] if j < len(rj) else np.nan
            row[f"rg_joint_rmse_{j}"] = gj[j] if j < len(gj) else np.nan

        row["metrics_path"] = p
        rows.append(row)

    df = pd.DataFrame(rows)

    # ----------------------------
    # Derived interpretability metrics
    # ----------------------------
    # absolute improvement: how much LSTM+DeLaN improves over DeLaN alone
    df["gain"] = df["delan_rmse"] - df["rg_rmse"]

    # relative improvement: <1 means improved (since rg_rmse < delan_rmse)
    df["gain_ratio"] = df["rg_rmse"] / df["delan_rmse"]

    # per-joint absolute improvement
    for j in range(6):
        df[f"joint_gain_{j}"] = df[f"delan_joint_rmse_{j}"] - df[f"rg_joint_rmse_{j}"]
        df[f"joint_gain_ratio_{j}"] = df[f"rg_joint_rmse_{j}"] / df[f"delan_joint_rmse_{j}"]

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "summary_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    print(df[["run_dir", "feature_mode", "delan_id", "rg_rmse", "delan_rmse", "res_rmse"]].tail(12).to_string(index=False))

    # ----------------------------
    # 1) rg_rmse by feature_mode
    # ----------------------------
    save_boxplot(
        df, "feature_mode", "rg_rmse",
        os.path.join(args.out_dir, "box_rg_rmse_by_feature_mode.png"),
        "Final torque RMSE (rg_rmse) grouped by feature_mode"
    )

    # ----------------------------
    # 2) res_rmse by feature_mode
    # ----------------------------
    save_boxplot(
        df, "feature_mode", "res_rmse",
        os.path.join(args.out_dir, "box_res_rmse_by_feature_mode.png"),
        "Residual RMSE (res_rmse) grouped by feature_mode"
    )

    # ----------------------------
    # 3) delan_rmse by delan_id
    # ----------------------------
    save_boxplot(
        df, "delan_id", "delan_rmse",
        os.path.join(args.out_dir, "box_delan_rmse_by_delan_id.png"),
        "DeLaN torque RMSE (delan_rmse) grouped by delan_id"
    )

    # ----------------------------
    # 4) per-joint rg boxplots by feature_mode
    # ----------------------------
    for j in range(6):
        save_boxplot(
            df, "feature_mode", f"rg_joint_rmse_{j}",
            os.path.join(args.out_dir, f"box_rg_joint_rmse_{j}_by_feature_mode.png"),
            f"Final torque RMSE joint {j} grouped by feature_mode"
        )

    # ----------------------------
    # 5) correlation scatters
    # ----------------------------
    save_scatter(
        df, "delan_rmse", "rg_rmse",
        os.path.join(args.out_dir, "scatter_delan_rmse_vs_rg_rmse.png"),
        "Does better DeLaN (lower rmse) correlate with better final torque?"
    )
    save_scatter(
        df, "res_rmse", "rg_rmse",
        os.path.join(args.out_dir, "scatter_res_rmse_vs_rg_rmse.png"),
        "Does better residual fit translate to better final torque?"
    )
    for j in range(6):
        save_scatter(
            df, f"delan_joint_rmse_{j}", f"rg_joint_rmse_{j}",
            os.path.join(args.out_dir, f"scatter_delan_vs_rg_joint_{j}.png"),
            f"Joint {j}: DeLaN RMSE vs final torque RMSE"
        )

    # ----------------------------
    # More informative correlations
    # ----------------------------
    save_scatter(
        df, "delan_rmse", "gain",
        os.path.join(args.out_dir, "scatter_delan_rmse_vs_gain.png"),
        "Does worse DeLaN create more room for improvement? (delan_rmse vs gain)"
    )

    save_scatter(
        df, "delan_rmse", "gain_ratio",
        os.path.join(args.out_dir, "scatter_delan_rmse_vs_gain_ratio.png"),
        "Does DeLaN quality correlate with relative improvement? (delan_rmse vs gain_ratio)"
    )

    for j in range(6):
        save_scatter(
            df, f"delan_joint_rmse_{j}", f"joint_gain_{j}",
            os.path.join(args.out_dir, f"scatter_delan_joint_rmse_{j}_vs_joint_gain_{j}.png"),
            f"Joint {j}: DeLaN RMSE vs absolute improvement (gain)"
        )

    # ----------------------------
    # Improvement over DeLaN
    # ----------------------------
    save_boxplot(
        df, "feature_mode", "gain",
        os.path.join(args.out_dir, "box_gain_by_feature_mode.png"),
        "Improvement gain = delan_rmse - rg_rmse (higher is better) grouped by feature_mode"
    )

    save_boxplot(
        df, "feature_mode", "gain_ratio",
        os.path.join(args.out_dir, "box_gain_ratio_by_feature_mode.png"),
        "Relative gain_ratio = rg_rmse / delan_rmse (lower is better) grouped by feature_mode"
    )

    # ----------------------------
    # Per-joint improvement / if needed single 3x2 grid image by adding save_joint_grid_boxplots(...)-helper
    # ----------------------------
    for j in range(6):
        save_boxplot(
            df, "feature_mode", f"joint_gain_{j}",
            os.path.join(args.out_dir, f"box_joint_gain_{j}_by_feature_mode.png"),
            f"Joint {j} gain = delan_joint_rmse - rg_joint_rmse (higher is better) by feature_mode"
    )

if __name__ == "__main__":
    main()