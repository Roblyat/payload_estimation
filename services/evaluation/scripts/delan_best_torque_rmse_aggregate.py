#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

_mpl_cfg = os.environ.get("MPLCONFIGDIR")
if not _mpl_cfg:
    _mpl_cfg = "/workspace/shared/.mplconfig"
    os.environ["MPLCONFIGDIR"] = _mpl_cfg
os.makedirs(_mpl_cfg, exist_ok=True)

import numpy as np

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

_TAB10 = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def _resolve_shared_path(path: str) -> str:
    if not path:
        return path
    if os.path.exists(path):
        return path
    marker = "/shared/"
    if marker in path:
        return os.path.join("/workspace/shared", path.split(marker, 1)[1])
    return path


def _resample_progress(curve: np.ndarray, n_bins: int) -> np.ndarray:
    if curve is None:
        return np.full((n_bins,), np.nan, dtype=np.float32)
    curve = np.asarray(curve).reshape(-1)
    if curve.size == 0:
        return np.full((n_bins,), np.nan, dtype=np.float32)
    if curve.size == 1:
        return np.full((n_bins,), float(curve[0]), dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, num=curve.size)
    x_new = np.linspace(0.0, 1.0, num=int(n_bins))
    return np.interp(x_new, x_old, curve).astype(np.float32)


def _median_iqr(curves: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    stack = np.stack(curves, axis=0)
    median = np.nanmedian(stack, axis=0)
    q25 = np.nanpercentile(stack, 25, axis=0)
    q75 = np.nanpercentile(stack, 75, axis=0)
    return median, q25, q75


def _parse_hp_presets(s: str | None) -> List[str]:
    if not s:
        return []
    out: List[str] = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok:
            out.append(tok)
    return out


def _hp_color_map(hp_list: List[str]) -> Dict[str, str]:
    return {hp: _TAB10[i % len(_TAB10)] for i, hp in enumerate(hp_list)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--bins", type=int, default=200)
    ap.add_argument("--split", type=str, default="test", choices=["test", "val"])
    ap.add_argument("--hp_presets", type=str, default="")
    ap.add_argument("--only_selected", action="store_true")
    ap.add_argument("--max_joints", type=int, default=6)
    args = ap.parse_args()

    summary_path = _resolve_shared_path(args.summary_jsonl)
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"summary_jsonl not found: {summary_path}")

    with open(summary_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    hp_filter = _parse_hp_presets(args.hp_presets)

    rmse_time_groups: Dict[Tuple[str, int], List[np.ndarray]] = defaultdict(list)
    per_joint_groups: Dict[Tuple[str, int], List[np.ndarray]] = defaultdict(list)

    for r in rows:
        if args.only_selected and not r.get("selected", False):
            continue
        hp = r.get("hp_preset")
        ds = r.get("dataset_seed")
        if hp is None or ds is None:
            continue
        if hp_filter and hp not in hp_filter:
            continue
        try:
            ds_i = int(ds)
        except Exception:
            continue

        metrics_json = _resolve_shared_path(
            r.get("metrics_json_container") or r.get("metrics_json", "")
        )
        if not metrics_json or not os.path.exists(metrics_json):
            continue
        with open(metrics_json, "r", encoding="utf-8") as mf:
            d = json.load(mf)

        artifacts = d.get("artifacts", {})
        rmse_path = _resolve_shared_path(artifacts.get("torque_rmse_time_npy", ""))
        if rmse_path and os.path.exists(rmse_path):
            try:
                rmse_t = np.load(rmse_path)
                rmse_t = _resample_progress(rmse_t, args.bins)
                rmse_time_groups[(str(hp), ds_i)].append(rmse_t)
            except Exception:
                pass

        eval_key = "eval_test" if args.split == "test" else "eval_val"
        per_joint = d.get(eval_key, {}).get("torque_rmse_per_joint", None)
        if isinstance(per_joint, list) and per_joint:
            vec = np.asarray(per_joint, dtype=np.float32)
            if args.max_joints is not None:
                vec = vec[: int(args.max_joints)]
            per_joint_groups[(str(hp), ds_i)].append(vec)

    os.makedirs(args.out_dir, exist_ok=True)

    # A3-style: RMSE vs normalized progress by hp_preset
    per_hp_seed_time: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = defaultdict(dict)
    for (hp, ds), curves in rmse_time_groups.items():
        if not curves:
            continue
        per_hp_seed_time[hp][ds] = _median_iqr(curves)

    per_hp_time: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for hp, seed_map in per_hp_seed_time.items():
        seed_curves = [seed_map[s][0] for s in sorted(seed_map.keys()) if seed_map[s] is not None]
        if seed_curves:
            per_hp_time[hp] = _median_iqr(seed_curves)

    plot_order = hp_filter or sorted(per_hp_time.keys())
    color_map = _hp_color_map(plot_order)

    if per_hp_time and plot_order:
        plt.figure(figsize=(9, 5), dpi=160)
        xs = np.linspace(0.0, 1.0, int(args.bins))
        for hp in plot_order:
            if hp not in per_hp_time:
                continue
            med, q25, q75 = per_hp_time[hp]
            color = color_map.get(hp, "#1f77b4")
            plt.plot(xs, med, label=str(hp), color=color, linewidth=1.4)
            plt.fill_between(xs, q25, q75, color=color, alpha=0.18)
        max_median = None
        for med, _, _ in per_hp_time.values():
            try:
                m = float(np.nanmax(med))
            except Exception:
                continue
            if np.isfinite(m):
                max_median = m if max_median is None else max(max_median, m)
        if max_median is not None:
            plt.ylim(0, max_median + 0.25)
        plt.title("DeLaN torque RMSE over progress (median ± IQR) by hp_preset")
        plt.xlabel("Progress (0 → 1)")
        plt.ylabel("Torque RMSE")
        plt.grid(True, alpha=0.25)
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "A3_torque_rmse_progress_by_hp.png"), dpi=180)
        plt.close()

        np.savez(
            os.path.join(args.out_dir, "A3_torque_rmse_progress_by_hp.npz"),
            bins=int(args.bins),
            hp_presets=np.array([hp for hp in plot_order if hp in per_hp_time], dtype=object),
            curves=np.stack([per_hp_time[hp][0] for hp in plot_order if hp in per_hp_time], axis=0),
            q25=np.stack([per_hp_time[hp][1] for hp in plot_order if hp in per_hp_time], axis=0),
            q75=np.stack([per_hp_time[hp][2] for hp in plot_order if hp in per_hp_time], axis=0),
        )

    # A4-style: RMSE per joint grouped by hp_preset
    per_hp_seed_joint: Dict[str, Dict[int, np.ndarray]] = defaultdict(dict)
    for (hp, ds), vecs in per_joint_groups.items():
        if not vecs:
            continue
        stack = np.stack(vecs, axis=0)
        per_hp_seed_joint[hp][ds] = np.nanmedian(stack, axis=0)

    per_hp_joint: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for hp, seed_map in per_hp_seed_joint.items():
        seed_vecs = [seed_map[s] for s in sorted(seed_map.keys()) if seed_map[s] is not None]
        if seed_vecs:
            med, q25, q75 = _median_iqr(seed_vecs)
            per_hp_joint[hp] = (med, q25, q75)

    available_hp = sorted(per_hp_joint.keys())
    selected_hp = hp_filter or available_hp
    color_map = _hp_color_map(selected_hp)

    if selected_hp:
        n_dof = int(per_hp_joint[selected_hp[0]][0].shape[0])
        x = np.arange(n_dof)
        n_groups = len(selected_hp)
        width = 0.8 / max(1, n_groups)

        plt.figure(figsize=(10, 4.8), dpi=160)
        for i, hp in enumerate(selected_hp):
            med, q25, q75 = per_hp_joint[hp]
            offsets = x - 0.4 + (i + 0.5) * width
            yerr = np.vstack([med - q25, q75 - med])
            color = color_map.get(hp, "#1f77b4")
            plt.bar(offsets, med, width=width, label=str(hp), alpha=0.9, color=color)
            plt.errorbar(offsets, med, yerr=yerr, fmt="none", ecolor="k", elinewidth=0.9, capsize=2, alpha=0.8)
        max_median = None
        for med, _, _ in per_hp_joint.values():
            try:
                m = float(np.nanmax(med))
            except Exception:
                continue
            if np.isfinite(m):
                max_median = m if max_median is None else max(max_median, m)
        if max_median is not None:
            plt.ylim(0, max_median + 0.25)

        plt.title("DeLaN torque RMSE per joint (median ± IQR) by hp_preset")
        plt.xlabel("Joint")
        plt.ylabel("Torque RMSE")
        plt.xticks(x, [f"joint{j}" for j in x])
        plt.grid(True, axis="y", alpha=0.25)
        plt.legend(ncol=min(3, len(selected_hp)), fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "A4_torque_rmse_per_joint_grouped.png"), dpi=200)
        plt.close()

        np.savez(
            os.path.join(args.out_dir, "A4_torque_rmse_per_joint_grouped.npz"),
            hp_presets=np.array(selected_hp, dtype=object),
            joint_median=np.stack([per_hp_joint[hp][0] for hp in selected_hp], axis=0),
            joint_q25=np.stack([per_hp_joint[hp][1] for hp in selected_hp], axis=0),
            joint_q75=np.stack([per_hp_joint[hp][2] for hp in selected_hp], axis=0),
        )


if __name__ == "__main__":
    main()
