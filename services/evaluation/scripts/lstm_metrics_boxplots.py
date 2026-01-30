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


_PALETTE = [
    "#e41a1c",  # red
    "#ff7f00",  # orange
    "#377eb8",  # blue
    "#4daf4a",  # green
    "#984ea3",  # purple
    "#a65628",
    "#f781bf",
    "#999999",
    "#1b9e77",
    "#d95f02",
    "#7570b3",
]

_H_EDGE_PALETTE = [
    "#4daf4a",  # green
    "#377eb8",  # blue
    "#984ea3",  # purple
]


def _color_map(keys: List[str | int], palette: List[str] | None = None) -> Dict[str | int, str]:
    pal = palette or _PALETTE
    return {k: pal[i % len(pal)] for i, k in enumerate(keys)}


def _safe_mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _parse_lstm_folder(name: str) -> Dict[str, Any]:
    # Examples:
    # ur5__B__delan_struct_s4_ep500__feat_full__lstm_s4_H60_ep60_b64_u128_do0p2
    # delan_UR3_Load0_combined_26__A__K100_tf0p2__delan_jax_struct_s0_ep300__feat_full__lstm_s0_H25_ep80_b256_u128_do0p1
    parts = name.split("__")
    out: Dict[str, Any] = {
        "folder": name,
        "dataset": parts[0] if parts else None,
        "run_tag": parts[1] if len(parts) >= 2 else None,
        "delan_tag": None,
        "feature_mode": None,
        "lstm_tag": None,
    }

    for p in parts:
        if p.startswith("feat_"):
            out["feature_mode"] = p.replace("feat_", "")
            continue
        if p.startswith("lstm_"):
            out["lstm_tag"] = p.replace("lstm_", "")
            continue
        if re.match(r"^delan_(jax|torch)_(struct|black)_", p) or re.match(r"^delan_(struct|black)_", p):
            out["delan_tag"] = p

    # Fallback for older naming conventions
    if out["delan_tag"] is None and len(parts) >= 3:
        out["delan_tag"] = parts[2]
    if out["feature_mode"] is None:
        for p in parts:
            if p.startswith("feat_"):
                out["feature_mode"] = p.replace("feat_", "")
                break
    if out["lstm_tag"] is None:
        for p in parts:
            if p.startswith("lstm_"):
                out["lstm_tag"] = p.replace("lstm_", "")
                break

    # Extract H from lstm_tag
    mH = re.search(r"_H(\d+)", out.get("lstm_tag") or "")
    out["H"] = int(mH.group(1)) if mH else None

    return out

def _infer_backend_from_name(name: str) -> str:
    if "delan_jax_" in name:
        return "jax"
    if "delan_torch_" in name:
        return "torch"
    return "unknown"

def _common_prefix_token(labels: List[str]) -> str:
    """Longest common prefix, cut back to last '_' to avoid cutting tokens."""
    if not labels:
        return ""
    import os as _os
    pref = _os.path.commonprefix(labels)
    if "_" in pref:
        pref = pref[: pref.rfind("_") + 1]
    # only keep if it actually helps
    return pref if len(pref) >= 4 else ""

def _label_has_prefix(label: str, pref: str) -> bool:
    if label.startswith(pref):
        return True
    if pref.startswith("delan_"):
        return label.startswith(pref[len("delan_"):])
    return label.startswith(f"delan_{pref}")

def _force_prefix(labels: List[str], prefixes: List[str]) -> str:
    for p in prefixes:
        if labels and all(_label_has_prefix(l, p) for l in labels):
            return p
    return ""

def _strip_prefix_variants(labels: List[str], pref: str) -> List[str]:
    if not pref:
        return labels
    alts = [pref]
    if pref.startswith("delan_"):
        alts.append(pref[len("delan_"):])
    else:
        alts.append(f"delan_{pref}")
    out: List[str] = []
    for s in labels:
        stripped = s
        for a in alts:
            if stripped.startswith(a):
                stripped = stripped[len(a):]
                break
        out.append(stripped)
    return out

def _split_labels(labels: List[str]) -> Tuple[List[str], List[str]]:
    mid = (len(labels) + 1) // 2
    return labels[:mid], labels[mid:]


def _common_delan_prefix_from_rows(rows: List[Dict[str, Any]]) -> str:
    tags = [str(r.get("delan_tag") or "") for r in rows if r.get("delan_tag")]
    if not tags:
        return ""
    import os as _os
    pref = _os.path.commonprefix(tags)
    if "_" in pref:
        pref = pref[: pref.rfind("_") + 1]
    return pref if len(pref) >= 4 else ""


def _boxplot(
    groups: Dict[str, List[float]],
    title: str,
    ylabel: str,
    out_png: str,
    *,
    strip_common_prefix: bool = False,
    split_two_rows: bool = False,
    prefix_hint: str | None = None,
    force_prefixes: List[str] | None = None,
    title_prefix_override: str | None = None,
    title_fontsize: int = 11,
) -> None:
    labels = sorted(groups.keys())

    pref = _common_prefix_token(labels) if strip_common_prefix else ""
    if prefix_hint and all(l.startswith(prefix_hint) for l in labels):
        pref = prefix_hint
    forced = _force_prefix(labels, force_prefixes or [])
    if forced:
        pref = forced
    disp = _strip_prefix_variants(labels, pref)
    title_pref = title_prefix_override if (pref and title_prefix_override) else pref
    full_title = title if (not title_pref or title_pref in title) else f"{title} | {title_pref}"

    if split_two_rows and len(labels) > 1:
        top_lbl, bot_lbl = _split_labels(labels)
        top_disp, bot_disp = _split_labels(disp)

        fig = plt.figure(figsize=(max(10, 1.2 * max(len(top_lbl), len(bot_lbl))), 8), dpi=140)

        ax1 = fig.add_subplot(2, 1, 1)
        ax1.boxplot([groups[k] for k in top_lbl], labels=top_disp, showmeans=True)
        ax1.set_title(full_title, fontsize=title_fontsize)
        ax1.set_ylabel(ylabel)
        ax1.grid(True, alpha=0.25)
        ax1.tick_params(axis="x", rotation=30)

        ax2 = fig.add_subplot(2, 1, 2)
        ax2.boxplot([groups[k] for k in bot_lbl], labels=bot_disp, showmeans=True)
        ax2.set_ylabel(ylabel)
        ax2.grid(True, alpha=0.25)
        ax2.tick_params(axis="x", rotation=30)

        plt.tight_layout()
        plt.savefig(out_png, dpi=160)
        plt.close(fig)
        return

    data = [groups[k] for k in labels]
    plt.figure(figsize=(max(10, 1.2 * len(labels)), 5), dpi=140)
    plt.boxplot(data, labels=disp, showmeans=True)
    plt.title(full_title, fontsize=title_fontsize)
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
    *,
    strip_common_prefix: bool = False,
    prefix_hint: str | None = None,
    force_prefixes: List[str] | None = None,
    title_prefix_override: str | None = None,
    suptitle_fontsize: int = 11,
) -> None:
    labels = sorted(groups_per_joint.keys())

    pref = _common_prefix_token(labels) if strip_common_prefix else ""
    if prefix_hint and all(l.startswith(prefix_hint) for l in labels):
        pref = prefix_hint
    forced = _force_prefix(labels, force_prefixes or [])
    if forced:
        pref = forced
    disp = _strip_prefix_variants(labels, pref)
    title_pref = title_prefix_override if (pref and title_prefix_override) else pref
    full_title = title if (not title_pref or title_pref in title) else f"{title} | {title_pref}"

    # 6x1 vertical layout
    fig = plt.figure(figsize=(max(12, 1.2 * len(labels)), 2.3 * n_dof), dpi=140)

    for j in range(n_dof):
        ax = fig.add_subplot(n_dof, 1, j + 1)
        data = [groups_per_joint[g][j] for g in labels]
        ax.boxplot(data, labels=disp, showmeans=True)
        ax.set_title(f"Joint {j}")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

        # reduce clutter: only show x tick labels on the last subplot
        if j != n_dof - 1:
            ax.set_xticklabels([])
        else:
            ax.tick_params(axis="x", rotation=30)

    fig.suptitle(full_title, fontsize=suptitle_fontsize)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close(fig)

def _scatter(
    rows: List[Dict[str, Any]],
    xlabel: str,
    ylabel: str,
    title: str,
    out_png: str,
    *,
    show_legend: bool = True,
) -> None:
    pts = []
    for r in rows:
        x = r.get("best_val_loss")
        y = r.get("rmse_total")
        feat = r.get("feature_mode") or "unknown"
        H = r.get("H")
        if np.isfinite(x) and np.isfinite(y):
            pts.append((float(x), float(y), str(feat), H))

    if not pts:
        return

    feats = sorted({p[2] for p in pts})
    hs = sorted({p[3] for p in pts if p[3] is not None})
    feat_colors = _color_map(feats)
    h_colors = _color_map(hs, _H_EDGE_PALETTE)

    fig = plt.figure(figsize=(7.2, 5), dpi=140)
    ax = fig.add_subplot(1, 1, 1)

    for x, y, feat, H in pts:
        edge = h_colors.get(H, "#000000")
        ax.scatter(
            x,
            y,
            s=55,
            facecolor=feat_colors.get(feat, "#777777"),
            edgecolor=edge,
            linewidth=1.0,
            alpha=0.9,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.25)

    if show_legend:
        from matplotlib.lines import Line2D

        feat_handles = [
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=feat_colors[f], markeredgecolor="black",
                   markersize=7, label=f)
            for f in feats
        ]
        h_handles = [
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor="none", markeredgecolor=h_colors[h],
                   markersize=7, markeredgewidth=1.2, label=f"H={h}")
            for h in hs
        ]

        leg_h = None
        if h_handles:
            leg_h = ax.legend(handles=h_handles, title="H", loc="upper right",
                              bbox_to_anchor=(1.0, 1.0), fontsize=8, title_fontsize=9, framealpha=0.9)
            ax.add_artist(leg_h)
        ax.legend(handles=feat_handles, title="Feature", loc="upper right",
                  bbox_to_anchor=(0.78, 1.0), fontsize=8, title_fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lstm_root", default="/workspace/shared/models/lstm", help="Root folder containing lstm runs")
    ap.add_argument("--out_dir", default="/workspace/shared/models/lstm/_metrics_plots", help="Where to save plots")
    ap.add_argument("--backend", choices=["all", "jax", "torch"], default="all")
    ap.add_argument("--no_scatter_legend", action="store_true", help="Disable legend on scatter plot")

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

        backend = _infer_backend_from_name(folder)

        row = {
            "folder": folder,
            "run_dir": run_dir,
            "backend": backend,
            "metrics_path": mp,
            "delan_tag": meta.get("delan_tag"),
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

    if args.backend != "all":
        rows = [r for r in rows if r.get("backend") == args.backend]

    delan_pref = _common_delan_prefix_from_rows(rows)

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
        title="LSTM $i_{motor}$ Residual RMSE (test) by Feature Mode",
        ylabel="$i_{motor}$ RMSE Total [A]",
        out_png=os.path.join(args.out_dir, "lstm_residual_rmse_by_feature_mode.png"),
        split_two_rows=True,
        strip_common_prefix=True,
        prefix_hint=delan_pref,
        force_prefixes=["jax_struct_"],
        title_prefix_override="delan_jax_struct_",
        title_fontsize=11,
    )


    # 1a) Residual RMSE grouped by feature_mode|backend (jax vs torch side-by-side)
    g1b: Dict[str, List[float]] = {}
    for r in rows:
        fm = r.get("feature_mode") or "unknown"
        key = f"{fm}|{r.get('backend','unknown')}"
        g1b.setdefault(key, []).append(r["rmse_total"])
    _boxplot(
        g1b,
        title="LSTM $i_{motor}$ Residual RMSE (test) by Feature Mode / Backend",
        ylabel="$i_{motor}$ RMSE Total [A]",
        out_png=os.path.join(args.out_dir, "lstm_residual_rmse_by_feature_mode_backend.png"),
        split_two_rows=True,
        strip_common_prefix=True,
        prefix_hint=delan_pref,
        force_prefixes=["jax_struct_"],
        title_prefix_override="delan_jax_struct_",
        title_fontsize=11,
    )

    # 1b) Residual RMSE grouped by H (optional helpful)
    gH: Dict[str, List[float]] = {}
    for r in rows:
        gH.setdefault(f"H{r['H']}", []).append(r["rmse_total"])
    if len(gH) >= 2:
        _boxplot(
            gH,
            title="LSTM $i_{motor}$ Residual RMSE (test) by H",
            ylabel="$i_{motor}$ RMSE Total [A]",
            out_png=os.path.join(args.out_dir, "lstm_residual_rmse_by_H.png"),
            title_fontsize=11,
        )

    # 1b2) Residual RMSE grouped by backend only
    gb: Dict[str, List[float]] = {}
    for r in rows:
        gb.setdefault(r.get("backend","unknown"), []).append(r["rmse_total"])
    if len(gb) >= 2:
        _boxplot(
            gb,
            title="LSTM $i_{motor}$ Residual RMSE (test) by Backend",
            ylabel="$i_{motor}$ RMSE Total [A]",
            out_png=os.path.join(args.out_dir, "lstm_residual_rmse_by_backend.png"),
            title_fontsize=11,
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
            title="Per-joint $i_{motor}$ Residual RMSE (test) by Feature Mode",
            ylabel="Joint $i_{motor}$ RMSE [A]",
            out_png=os.path.join(args.out_dir, "lstm_joint_rmse_grid_by_feature_mode.png"),
            n_dof=n_dof,
            strip_common_prefix=True,
            prefix_hint=delan_pref,
            force_prefixes=["jax_struct_"],
            title_prefix_override="delan_jax_struct_",
            suptitle_fontsize=11,
        )


    # 3) Generalization scatter: best_val_loss vs rmse_total
    if rows:
        _scatter(
            rows,
            xlabel="Best Val Loss (scaled MSE)",
            ylabel="$i_{motor}$ RMSE Total [A]",
            title="Best Val Loss vs $i_{motor}$ RMSE Total",
            out_png=os.path.join(args.out_dir, "lstm_best_val_loss_vs_rmse_total.png"),
            show_legend=not args.no_scatter_legend,
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
            title="Overfit Indicator (final val / train) by Feature Mode",
            ylabel="Val/Train Loss Ratio",
            out_png=os.path.join(args.out_dir, "lstm_overfit_ratio_by_feature_mode.png"),
            split_two_rows=True,
            strip_common_prefix=True,
            prefix_hint=delan_pref,
            force_prefixes=["jax_struct_"],
            title_prefix_override="delan_jax_struct_",
            title_fontsize=11,
        )


    print(f"[lstm_metrics_boxplots] Saved plots to: {args.out_dir}")


if __name__ == "__main__":
    main()
