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
from typing import Dict, List, Optional, Tuple, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


FEATURE_RE = re.compile(r"__feat_([^_]+(?:_[^_]+)*)__")
DELAN_RE = re.compile(r"__delan_([^_]+(?:_[^_]+)*)__feat_")  # greedy-ish between markers
LSTM_RE = re.compile(r"__lstm_(.+)$")
H_FROM_FILENAME_RE = re.compile(r"_H(\d+)\.txt$")

def _common_prefix_token(labels: list[str]) -> str:
    if not labels:
        return ""
    pref = os.path.commonprefix(labels)
    if "_" in pref:
        pref = pref[: pref.rfind("_") + 1]
    return pref if len(pref) >= 6 else ""

def _strip_prefix(labels: list[str], pref: str) -> list[str]:
    if not pref:
        return labels
    alts = _prefix_variants(pref)
    out: list[str] = []
    for s in labels:
        stripped = s
        for a in alts:
            if stripped.startswith(a):
                stripped = stripped[len(a):]
                break
        out.append(stripped)
    return out

def _prefix_variants(pref: str) -> list[str]:
    variants = [pref]
    if pref.startswith("delan_"):
        variants.append(pref[len("delan_"):])
    else:
        variants.append(f"delan_{pref}")
    return variants

def _label_has_prefix(label: str, pref: str) -> bool:
    if label.startswith(pref):
        return True
    if pref.startswith("delan_"):
        return label.startswith(pref[len("delan_"):])
    return label.startswith(f"delan_{pref}")

def _force_prefix(labels: list[str], prefixes: list[str]) -> str:
    for p in prefixes:
        if labels and all(_label_has_prefix(s, p) for s in labels):
            return p
    return ""

def _strip_any_prefixes(labels: list[str], prefixes: list[str]) -> tuple[list[str], list[str]]:
    if not prefixes:
        return labels, []
    removed: list[str] = []
    out: list[str] = []
    for s in labels:
        stripped = s
        used = None
        for p in prefixes:
            for v in _prefix_variants(p):
                if stripped.startswith(v):
                    stripped = stripped[len(v):]
                    used = p
                    break
            if used:
                break
        if used:
            removed.append(used)
        out.append(stripped)
    seen = set()
    dedup = []
    for p in removed:
        if p not in seen:
            seen.add(p)
            dedup.append(p)
    return out, dedup

def _title_suffix_from_prefixes(prefixes: list[str]) -> str:
    cleaned: list[str] = []
    for p in prefixes:
        if not p or p == "delan_":
            continue
        if p.startswith("delan_"):
            p = p[len("delan_"):]
        if p not in cleaned:
            cleaned.append(p)
    return ", ".join(cleaned)

def _boxplot_groups(
    groups: dict[str, list[float]],
    *,
    title: str,
    ylabel: str,
    out_png: str,
    strip_common_prefix: bool = False,
    split_two_rows: bool = False,
    strip_prefixes: list[str] | None = None,
):
    labels = sorted(groups.keys())
    vals = [groups[k] for k in labels]

    pref = _common_prefix_token(labels) if strip_common_prefix else ""
    if strip_common_prefix:
        forced = _force_prefix(labels, ["delan_jax_struct_", "jax_struct_"])
        if forced:
            pref = forced
    labels_disp = _strip_prefix(labels, pref)
    labels_disp, removed = _strip_any_prefixes(labels_disp, strip_prefixes or [])
    title_disp = title
    title_suffix = _title_suffix_from_prefixes([pref] + removed)
    if title_suffix:
        title_disp = f"{title} | {title_suffix}"

    if split_two_rows and len(labels_disp) > 1:
        n = len(labels_disp)
        k = (n + 1) // 2
        top_l, top_v = labels_disp[:k], vals[:k]
        bot_l, bot_v = labels_disp[k:], vals[k:]
        fig_w = max(12, 0.75 * max(len(top_l), len(bot_l)))
        fig = plt.figure(figsize=(fig_w, 7.5), dpi=140)
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2, sharey=ax1)
        ax1.boxplot(top_v, labels=top_l, showmeans=True)
        ax2.boxplot(bot_v, labels=bot_l, showmeans=True)
        for ax in (ax1, ax2):
            ax.grid(True, axis="y", alpha=0.2)
            for t in ax.get_xticklabels():
                t.set_rotation(28); t.set_ha("right")
        ax1.set_title(title_disp)
        ax2.set_ylabel(ylabel)
        fig.subplots_adjust(top=0.90, bottom=0.22, hspace=0.35)
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        print(f"Saved: {out_png}")
        return

    fig_w = max(12, 0.75 * len(labels_disp))
    fig = plt.figure(figsize=(fig_w, 4.8), dpi=140)
    ax = fig.add_subplot(1, 1, 1)
    ax.boxplot(vals, labels=labels_disp, showmeans=True)
    ax.set_title(title_disp)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.2)
    for t in ax.get_xticklabels():
        t.set_rotation(28); t.set_ha("right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_png}")


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
    # Match style with custom matplotlib boxplots (black boxes, orange median, green mean)
    df.boxplot(
        column=value_col,
        by=group_col,
        grid=False,
        rot=25,
        showmeans=True,
        boxprops={"color": "black"},
        whiskerprops={"color": "black"},
        capprops={"color": "black"},
        medianprops={"color": "C1"},
        meanprops={"marker": "^", "markerfacecolor": "C2", "markeredgecolor": "C2"},
    )
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
        "Final $i_{motor}$ RMSE (rg_rmse) grouped by feature_mode"
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

    # --- shorten delan_id labels (move common prefix into title) ---
    delan_ids = df["delan_id"].dropna().astype(str).tolist()
    pref = ""
    if delan_ids:
        import os as _os
        pref = _os.path.commonprefix(delan_ids)
        if "_" in pref:
            pref = pref[: pref.rfind("_") + 1]
        if len(pref) < 4:
            pref = ""

    if pref:
        df["delan_id_short"] = df["delan_id"].astype(str).str.replace("^" + re.escape(pref), "", regex=True)
    else:
        df["delan_id_short"] = df["delan_id"].astype(str)

    # Custom boxplot so we can strip shared prefix and optionally split into two rows
    groups: dict[str, list[float]] = {}
    for _, r in df.iterrows():
        groups.setdefault(str(r["delan_id"]), []).append(float(r["delan_rmse"]))
    _boxplot_groups(
        groups,
        title="DeLaN $i_{motor}$ RMSE (delan_rmse) grouped by delan_id",
        ylabel="$i_{motor}$ RMSE [A]",
        out_png=os.path.join(args.out_dir, "box_delan_rmse_by_delan_id.png"),
        strip_common_prefix=True,
        split_two_rows=True,
        strip_prefixes=["jax_struct_", "torch_struct_"],
    )

    # ----------------------------
    # 4) per-joint rg boxplots by feature_mode
    # ----------------------------
    for j in range(6):
        save_boxplot(
            df, "feature_mode", f"rg_joint_rmse_{j}",
            os.path.join(args.out_dir, f"box_rg_joint_rmse_{j}_by_feature_mode.png"),
            f"Final $i_{{motor}}$ RMSE joint {j} grouped by feature_mode"
        )

    # ----------------------------
    # 5) correlation scatters
    # ----------------------------
    save_scatter(
        df, "delan_rmse", "rg_rmse",
        os.path.join(args.out_dir, "scatter_delan_rmse_vs_rg_rmse.png"),
        "Does better DeLaN (lower rmse) correlate with better final $i_{motor}$?"
    )
    save_scatter(
        df, "res_rmse", "rg_rmse",
        os.path.join(args.out_dir, "scatter_res_rmse_vs_rg_rmse.png"),
        "Does better residual fit translate to better final $i_{motor}$?"
    )
    for j in range(6):
        save_scatter(
            df, f"delan_joint_rmse_{j}", f"rg_joint_rmse_{j}",
            os.path.join(args.out_dir, f"scatter_delan_vs_rg_joint_{j}.png"),
            f"Joint {j}: DeLaN RMSE vs final $i_{{motor}}$ RMSE"
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
