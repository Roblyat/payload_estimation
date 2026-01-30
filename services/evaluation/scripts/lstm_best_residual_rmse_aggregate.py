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


def _color_map(keys: List[int], palette: List[str] | None = None) -> Dict[int, str]:
    pal = palette or _PALETTE
    return {k: pal[i % len(pal)] for i, k in enumerate(keys)}


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


def _parse_h_values(s: str | None) -> List[int]:
    if not s:
        return []
    out: List[int] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except Exception:
            continue
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--bins", type=int, default=200)
    ap.add_argument("--feature", type=str, default="")
    ap.add_argument("--h_values", type=str, default="")
    args = ap.parse_args()

    summary_path = _resolve_shared_path(args.summary_jsonl)
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"summary_jsonl not found: {summary_path}")

    with open(summary_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    if args.feature:
        rows = [r for r in rows if str(r.get("feature_mode", "")) == args.feature]

    time_groups: Dict[Tuple[int, int, int], List[np.ndarray]] = defaultdict(list)
    joint_groups: Dict[Tuple[int, int, int], List[np.ndarray]] = defaultdict(list)

    for r in rows:
        try:
            H = int(r["H"])
            ds = int(r["dataset_seed"])
            seed = int(r["lstm_seed"])
        except Exception:
            continue

        metrics_json = _resolve_shared_path(r.get("metrics_json", ""))
        if not metrics_json or not os.path.exists(metrics_json):
            continue
        with open(metrics_json, "r", encoding="utf-8") as mf:
            d = json.load(mf)

        preds_path = _resolve_shared_path(d.get("eval_test", {}).get("predictions_npz", ""))
        if not preds_path or not os.path.exists(preds_path):
            continue
        try:
            p = np.load(preds_path)
            Y_test = p["Y_test"]
            Y_pred = p["Y_pred"]
        except Exception:
            continue

        err = np.asarray(Y_pred) - np.asarray(Y_test)
        rmse_t = np.sqrt(np.mean(err ** 2, axis=1))
        rmse_t = _resample_progress(rmse_t, int(args.bins))
        rmse_joint = np.sqrt(np.mean(err ** 2, axis=0))

        time_groups[(H, ds, seed)].append(rmse_t)
        joint_groups[(H, ds, seed)].append(rmse_joint)

    os.makedirs(args.out_dir, exist_ok=True)

    per_h_seed_time: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = defaultdict(dict)
    per_h_seed_joint: Dict[int, Dict[int, np.ndarray]] = defaultdict(dict)

    for (H, ds, seed), curves in time_groups.items():
        if curves:
            per_h_seed_time[(H, ds)][seed] = _median_iqr(curves)

    for (H, ds, seed), vecs in joint_groups.items():
        if vecs:
            stack = np.stack(vecs, axis=0)
            per_h_seed_joint[(H, ds)][seed] = np.nanmedian(stack, axis=0)

    per_h_time: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    per_h_joint: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for (H, ds), seed_map in per_h_seed_time.items():
        seed_curves = [seed_map[s][0] for s in sorted(seed_map.keys()) if seed_map[s] is not None]
        if seed_curves:
            per_h_time[(H, ds)] = _median_iqr(seed_curves)

    for (H, ds), seed_map in per_h_seed_joint.items():
        seed_vecs = [seed_map[s] for s in sorted(seed_map.keys()) if seed_map[s] is not None]
        if seed_vecs:
            per_h_joint[(H, ds)] = _median_iqr(seed_vecs)

    # Aggregate across dataset seeds
    agg_time: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    agg_joint: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    hs = sorted({h for (h, _) in per_h_time.keys()} | {h for (h, _) in per_h_joint.keys()})
    for H in hs:
        ds_curves = [per_h_time[(H, ds)][0] for (h, ds) in per_h_time.keys() if h == H]
        if ds_curves:
            agg_time[H] = _median_iqr(ds_curves)
        ds_vecs = [per_h_joint[(H, ds)][0] for (h, ds) in per_h_joint.keys() if h == H]
        if ds_vecs:
            agg_joint[H] = _median_iqr(ds_vecs)

    xs = np.linspace(0.0, 1.0, int(args.bins))
    if agg_time:
        plt.figure(figsize=(9, 5), dpi=160)
        hs_order = sorted(agg_time.keys())
        color_map = _color_map(hs_order)
        for H in hs_order:
            med, q25, q75 = agg_time[H]
            color = color_map.get(H, "#1f77b4")
            plt.plot(xs, med, label=f"H={H}", color=color, linewidth=1.4)
            plt.fill_between(xs, q25, q75, color=color, alpha=0.18)
        max_median = None
        for med, _, _ in agg_time.values():
            try:
                m = float(np.nanmax(med))
            except Exception:
                continue
            if np.isfinite(m):
                max_median = m if max_median is None else max(max_median, m)
        if max_median is not None:
            plt.ylim(0, max_median + 0.25)
        title_feat = f" (feat={args.feature})" if args.feature else ""
        plt.title(f"LSTM $i_{motor}$ Residual RMSE over Progress by H{title_feat}")
        plt.xlabel("Progress (0 → 1)")
        plt.ylabel("$i_{motor}$ Residual RMSE [A]")
        plt.grid(True, alpha=0.25)
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "B3_residual_rmse_progress_by_H.png"), dpi=180)
        plt.close()

        np.savez(
            os.path.join(args.out_dir, "B3_residual_rmse_progress_by_H.npz"),
            bins=int(args.bins),
            Hs=np.array(sorted(agg_time.keys()), dtype=np.int32),
            curves=np.stack([agg_time[H][0] for H in sorted(agg_time.keys())], axis=0),
            q25=np.stack([agg_time[H][1] for H in sorted(agg_time.keys())], axis=0),
            q75=np.stack([agg_time[H][2] for H in sorted(agg_time.keys())], axis=0),
        )

    available_H = sorted(agg_joint.keys())
    selected_H = _parse_h_values(args.h_values)
    if selected_H:
        selected_H = [h for h in selected_H if h in agg_joint]
    if not selected_H:
        if len(available_H) <= 3:
            selected_H = available_H
        elif available_H:
            mid = available_H[len(available_H) // 2]
            selected_H = [available_H[0], mid, available_H[-1]]
            selected_H = list(dict.fromkeys(selected_H))

    if selected_H:
        n_dof = int(agg_joint[selected_H[0]][0].shape[0])
        x = np.arange(n_dof)
        n_groups = len(selected_H)
        width = 0.8 / max(1, n_groups)

        plt.figure(figsize=(10, 4.8), dpi=160)
        color_map = _color_map(selected_H)
        for i, H in enumerate(selected_H):
            med, q25, q75 = agg_joint[H]
            offsets = x - 0.4 + (i + 0.5) * width
            yerr = np.vstack([med - q25, q75 - med])
            color = color_map.get(H, "#1f77b4")
            plt.bar(offsets, med, width=width, label=f"H={H}", alpha=0.9, color=color)
            plt.errorbar(offsets, med, yerr=yerr, fmt="none", ecolor="k", elinewidth=0.9, capsize=2, alpha=0.8)
        max_median = None
        for med, _, _ in agg_joint.values():
            try:
                m = float(np.nanmax(med))
            except Exception:
                continue
            if np.isfinite(m):
                max_median = m if max_median is None else max(max_median, m)
        if max_median is not None:
            plt.ylim(0, max_median + 0.25)

        title_feat = f" (feat={args.feature})" if args.feature else ""
        plt.title(f"LSTM $i_{motor}$ Residual RMSE per Joint (median ± IQR) by H{title_feat}")
        plt.xlabel("Joint")
        plt.ylabel("Joint $i_{motor}$ Residual RMSE [A]")
        plt.xticks(x, [f"joint{j}" for j in x])
        plt.grid(True, axis="y", alpha=0.25)
        plt.legend(ncol=min(3, len(selected_H)), fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "B4_residual_rmse_per_joint_grouped.png"), dpi=200)
        plt.close()

        np.savez(
            os.path.join(args.out_dir, "B4_residual_rmse_per_joint_grouped.npz"),
            Hs=np.array(selected_H, dtype=np.int32),
            joint_median=np.stack([agg_joint[H][0] for H in selected_H], axis=0),
            joint_q25=np.stack([agg_joint[H][1] for H in selected_H], axis=0),
            joint_q75=np.stack([agg_joint[H][2] for H in selected_H], axis=0),
        )


if __name__ == "__main__":
    main()
