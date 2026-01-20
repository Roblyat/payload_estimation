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

# ----------------------------
# Plot helpers
# ----------------------------
def save_boxplot(
    df: pd.DataFrame,
    group_col: str | None = None,
    value_col: str | None = None,
    out_path: str = "",
    title: str = "",
    # aliases (so older/newer call styles both work)
    x_col: str | None = None,
    y_col: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
):
    # allow either (group_col/value_col) or (x_col/y_col)
    group_col = group_col or x_col
    value_col = value_col or y_col
    if group_col is None or value_col is None:
        raise ValueError("save_boxplot needs (group_col,value_col) or (x_col,y_col)")

    plt.figure(figsize=(10, 4), dpi=140)
    df.boxplot(column=value_col, by=group_col, grid=False, rot=25)
    plt.title(title)
    plt.suptitle("")  # remove pandas' automatic title
    plt.xlabel(xlabel or group_col)
    plt.ylabel(ylabel or value_col)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def save_scatter(
    df: pd.DataFrame,
    x: str | None = None,
    y: str | None = None,
    out_path: str = "",
    title: str = "",
    # aliases
    x_col: str | None = None,
    y_col: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
):
    x = x or x_col
    y = y or y_col
    if x is None or y is None:
        raise ValueError("save_scatter needs (x,y) or (x_col,y_col)")

    plt.figure(figsize=(6, 4), dpi=140)
    plt.scatter(df[x], df[y], s=18, alpha=0.8)
    plt.title(title)
    plt.xlabel(xlabel or x)
    plt.ylabel(ylabel or y)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
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
    ap.add_argument("--backend", choices=["all", "jax", "torch"], default="all")
    
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

    os.makedirs(args.out_dir, exist_ok=True)

    # backend from delan_id OR run_dir (robust: handles "jax_*", "delan_jax_*", or strings containing it)
    def _infer_backend(delan_id: object, run_dir: object) -> str:
        s = f"{delan_id or ''} {run_dir or ''}"
        if ("delan_jax_" in s) or ("jax_" in s):
            return "jax"
        if ("delan_torch_" in s) or ("torch_" in s):
            return "torch"
        return "unknown"

    df["backend"] = df.apply(lambda r: _infer_backend(r.get("delan_id"), r.get("run_dir")), axis=1)

    if args.backend != "all":
        df = df[df["backend"] == args.backend].copy()

    df["gain"] = df["delan_rmse"] - df["rg_rmse"]
    df["gain_ratio"] = df["rg_rmse"] / df["delan_rmse"].replace(0, np.nan)
    df["feature_backend"] = df["feature_mode"].astype(str) + "|" + df["backend"].astype(str)

    save_boxplot(
        df, x_col="feature_backend", y_col="gain",
        out_path=os.path.join(args.out_dir, "gain_by_feature_backend.png"),
        title="Gain (delan_rmse - rg_rmse) by feature_mode|backend",
        xlabel="feature_mode|backend", ylabel="gain"
    )

    save_boxplot(
        df, x_col="feature_backend", y_col="gain_ratio",
        out_path=os.path.join(args.out_dir, "gain_ratio_by_feature_backend.png"),
        title="Gain ratio (rg_rmse / delan_rmse) by feature_mode|backend",
        xlabel="feature_mode|backend", ylabel="gain_ratio"
    )

    # scatter per backend (keeps your existing scatter helper)
    for b in sorted(df["backend"].unique()):
        sub = df[df["backend"] == b]
        if len(sub) < 2:
            continue
        save_scatter(
            sub, x_col="delan_rmse", y_col="gain_ratio",
            out_path=os.path.join(args.out_dir, f"scatter_delan_rmse_vs_gain_ratio__{b}.png"),
            title=f"delan_rmse vs gain_ratio ({b})",
            xlabel="delan_rmse", ylabel="gain_ratio"
        )


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