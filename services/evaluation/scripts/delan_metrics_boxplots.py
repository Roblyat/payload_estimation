#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from typing import Any, Dict, List, Optional, Tuple

def _common_prefix_token(labels: List[str]) -> str:
    """Longest common prefix, cut back to last '_' to avoid cutting tokens."""
    if not labels:
        return ""
    import os as _os
    pref = _os.path.commonprefix(labels)
    if "_" in pref:
        pref = pref[: pref.rfind("_") + 1]
    # only keep if it actually helps
    return pref if len(pref) >= 6 else ""

def _strip_prefix(labels: List[str], pref: str) -> List[str]:
    if not pref:
        return labels
    alts = _prefix_variants(pref)
    out: List[str] = []
    for s in labels:
        stripped = s
        for a in alts:
            if stripped.startswith(a):
                stripped = stripped[len(a):]
                break
        out.append(stripped)
    return out

def _prefix_variants(pref: str) -> List[str]:
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

def _force_prefix(labels: List[str], prefixes: List[str]) -> str:
    for p in prefixes:
        if labels and all(_label_has_prefix(l, p) for l in labels):
            return p
    return ""

def _strip_any_prefixes(labels: List[str], prefixes: List[str]) -> Tuple[List[str], List[str]]:
    if not prefixes:
        return labels, []
    removed: List[str] = []
    out: List[str] = []
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
    # de-dup but keep order
    seen = set()
    dedup = []
    for p in removed:
        if p not in seen:
            seen.add(p)
            dedup.append(p)
    return out, dedup

def _title_suffix_from_prefixes(prefixes: List[str]) -> str:
    cleaned: List[str] = []
    for p in prefixes:
        if not p or p == "delan_":
            continue
        if p.startswith("delan_"):
            p = p[len("delan_"):]
        if p not in cleaned:
            cleaned.append(p)
    return ", ".join(cleaned)

def _split_labels_vals(labels: List[str], vals: List[List[float]]) -> Tuple[List[str], List[List[float]], List[str], List[List[float]]]:
    n = len(labels)
    k = (n + 1) // 2
    return labels[:k], vals[:k], labels[k:], vals[k:]


def _safe_mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _parse_delan_tag(tag: str) -> Dict[str, Any]:
    # tag like: delan_jax_struct_s4_ep200 or delan_torch_black_s4_ep500
    out: Dict[str, Any] = {"delan_tag": tag}

    m = re.search(r"delan_(jax|torch)_(struct|black)_s(\d+)_ep(\d+)", tag)
    if m:
        out["backend"] = m.group(1)
        out["model_short"] = m.group(2)
        out["seed"] = int(m.group(3))
        out["epochs"] = int(m.group(4))
        return out

    # fallback: old tags without backend
    m = re.search(r"delan_(struct|black)_s(\d+)_ep(\d+)", tag)
    if m:
        out["backend"] = "unknown"
        out["model_short"] = m.group(1)
        out["seed"] = int(m.group(2))
        out["epochs"] = int(m.group(3))
    else:
        out["backend"] = "unknown"
        out["model_short"] = "unknown"
    return out

def _boxplot(
    groups: Dict[str, List[float]],
    *,
    title: str,
    ylabel: str,
    out_png: str,
    strip_common_prefix: bool = False,
    split_two_rows: bool = False,
    strip_prefixes: List[str] | None = None,
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
        top_l, top_v, bot_l, bot_v = _split_labels_vals(labels_disp, vals)
        fig_w = max(12, 0.75 * max(len(top_l), len(bot_l)))
        fig = plt.figure(figsize=(fig_w, 7.5), dpi=140)
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2, sharey=ax1)

        ax1.boxplot(top_v, tick_labels=top_l, showmeans=True)
        ax2.boxplot(bot_v, tick_labels=bot_l, showmeans=True)

        for ax in (ax1, ax2):
            ax.tick_params(axis="x", labelsize=9)
            ax.tick_params(axis="y", labelsize=9)
            for t in ax.get_xticklabels():
                t.set_rotation(28)
                t.set_ha("right")
            ax.grid(True, axis="y", alpha=0.2)

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
    ax.boxplot(vals, tick_labels=labels_disp, showmeans=True)
    ax.set_title(title_disp)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.2)
    ax.tick_params(axis="x", labelsize=9)
    for t in ax.get_xticklabels():
        t.set_rotation(28)
        t.set_ha("right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_png}")

def _per_joint_boxplot(
    groups_per_joint: Dict[str, List[List[float]]],
    *,
    title: str,
    ylabel: str,
    out_png: str,
    n_dof: int,
    strip_common_prefix: bool = False,
    strip_prefixes: List[str] | None = None,
):
    group_names = sorted(groups_per_joint.keys())
    pref = _common_prefix_token(group_names) if strip_common_prefix else ""
    if strip_common_prefix:
        forced = _force_prefix(group_names, ["delan_jax_struct_", "jax_struct_"])
        if forced:
            pref = forced
    group_disp = _strip_prefix(group_names, pref)
    group_disp, removed = _strip_any_prefixes(group_disp, strip_prefixes or [])
    title_disp = title
    title_suffix = _title_suffix_from_prefixes([pref] + removed)
    if title_suffix:
        title_disp = f"{title} | {title_suffix}"

    fig_w = max(12, 0.75 * len(group_disp))
    fig_h = max(10, 2.1 * n_dof)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=140)

    for j in range(n_dof):
        ax = fig.add_subplot(n_dof, 1, j + 1)
        vals = [groups_per_joint[g][j] for g in group_names]
        ax.boxplot(vals, tick_labels=group_disp, showmeans=True)
        ax.set_title(f"Joint {j}", fontsize=10, pad=6)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, axis="y", alpha=0.2)
        ax.tick_params(axis="y", labelsize=9)

        if j < n_dof - 1:
            ax.set_xticklabels([])
        else:
            ax.tick_params(axis="x", labelsize=9)
            for t in ax.get_xticklabels():
                t.set_rotation(28)
                t.set_ha("right")

    fig.suptitle(title_disp, y=0.995)
    fig.tight_layout(rect=[0, 0.0, 1, 0.97])
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_png}")

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
    ap.add_argument("--out_dir", default="/workspace/shared/models/delan/_metrics_plots", help="Where to save plots")
    ap.add_argument("--backend", choices=["all", "jax", "torch"], default="all")

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
        backend = tag_info.get("backend", "unknown")

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
            "backend": backend,
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

    
    if args.backend != "all":
        rows = [r for r in rows if r.get("backend") == args.backend]

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
        title="DeLaN $i_{motor}$ RMSE (test) grouped by model type",
        ylabel="$i_{motor}$ RMSE [A]",
        out_png=os.path.join(args.out_dir, "delan_torque_rmse_by_model_type.png"),
    )

    # 1b) Torque RMSE grouped by model_short|backend  (jax vs torch side-by-side)
    groups_mb: Dict[str, List[float]] = {}
    for r in rows:
        key = f"{r['model_short']}|{r['backend']}"
        groups_mb.setdefault(key, []).append(r["torque_rmse"])
    _boxplot(
        groups_mb,
        title="DeLaN $i_{motor}$ RMSE (test) grouped by model_short|backend",
        ylabel="$i_{motor}$ RMSE [A]",
        out_png=os.path.join(args.out_dir, "delan_torque_rmse_by_model_short_backend.png"),
    )

    # 1c) Torque RMSE grouped by backend only
    groups_b: Dict[str, List[float]] = {}
    for r in rows:
        groups_b.setdefault(r["backend"], []).append(r["torque_rmse"])
    if len(groups_b) >= 2:
        _boxplot(
            groups_b,
            title="DeLaN $i_{motor}$ RMSE (test) grouped by backend",
            ylabel="$i_{motor}$ RMSE [A]",
            out_png=os.path.join(args.out_dir, "delan_torque_rmse_by_backend.png"),
        )

    # 2) Torque RMSE boxplot grouped by delan_tag (seed/epochs)
    groups = {}
    for r in rows:
        g = r["delan_tag"]
        groups.setdefault(g, []).append(r["torque_rmse"])
    _boxplot(
         groups,
         title="DeLaN $i_{motor}$ RMSE (test) grouped by delan_tag",
         ylabel="$i_{motor}$ RMSE [A]",
         out_png=os.path.join(args.out_dir, "delan_torque_rmse_by_delan_tag.png"),
        split_two_rows=True,
        strip_common_prefix=True,
        strip_prefixes=["jax_struct_", "torch_struct_"],
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
             title="DeLaN per-joint $i_{motor}$ RMSE (test) grouped by delan_tag",
             ylabel="Joint $i_{motor}$ RMSE [A]",
             out_png=os.path.join(args.out_dir, "delan_joint_rmse_grid_by_delan_tag.png"),
             n_dof=n_dof,
            strip_common_prefix=True,
            strip_prefixes=["jax_struct_", "torch_struct_"],
         )


    # 4) Speed vs accuracy scatter (time_per_sample_s vs torque_rmse)
    xs = [r["time_per_sample_s"] for r in rows if np.isfinite(r["time_per_sample_s"]) and np.isfinite(r["torque_rmse"])]
    ys = [r["torque_rmse"] for r in rows if np.isfinite(r["time_per_sample_s"]) and np.isfinite(r["torque_rmse"])]
    if xs and ys:
        _scatter(
            xs, ys,
            xlabel="time_per_sample_s",
            ylabel="$i_{motor}$ RMSE [A]",
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
            title="DeLaN $i_{motor}$ RMSE (test) grouped by seed",
            ylabel="$i_{motor}$ RMSE [A]",
            out_png=os.path.join(args.out_dir, "delan_torque_rmse_by_seed.png"),
        )

    print(f"[delan_metrics_boxplots] Saved plots to: {args.out_dir}")


if __name__ == "__main__":
    main()
