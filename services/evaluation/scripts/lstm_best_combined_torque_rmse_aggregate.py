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


def _concat_valid(
    tau_list: List[np.ndarray],
    tau_hat_list: List[np.ndarray],
    tau_rg_list: List[np.ndarray],
    H: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tau_gt_valid_all: List[np.ndarray] = []
    tau_delan_valid_all: List[np.ndarray] = []
    tau_rg_valid_all: List[np.ndarray] = []

    for tau, tau_hat, tau_rg in zip(tau_list, tau_hat_list, tau_rg_list):
        tau = np.asarray(tau)
        tau_hat = np.asarray(tau_hat)
        tau_rg = np.asarray(tau_rg)
        if tau.shape[0] < H:
            continue
        sl = slice(H - 1, None)
        tau_valid = tau[sl]
        tau_hat_valid = tau_hat[sl]
        tau_rg_valid = tau_rg[sl]

        if tau_valid.size == 0:
            continue

        tau_gt_valid_all.append(tau_valid)
        tau_delan_valid_all.append(tau_hat_valid)
        tau_rg_valid_all.append(tau_rg_valid)

    if not tau_gt_valid_all:
        return np.empty((0, 0)), np.empty((0, 0)), np.empty((0, 0))

    return (
        np.vstack(tau_gt_valid_all),
        np.vstack(tau_delan_valid_all),
        np.vstack(tau_rg_valid_all),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--bins", type=int, default=200)
    ap.add_argument("--split", type=str, default="test", choices=["test", "val"])
    ap.add_argument("--feature", type=str, default="")
    ap.add_argument("--h_values", type=str, default="")
    args = ap.parse_args()

    summary_path = _resolve_shared_path(args.summary_jsonl)
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"summary_jsonl not found: {summary_path}")

    with open(summary_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    rows = [r for r in rows if str(r.get("split", "")) == args.split]
    if args.feature:
        rows = [r for r in rows if str(r.get("feature_mode", "")) == args.feature]

    time_groups_delan: Dict[Tuple[int, int, int], List[np.ndarray]] = defaultdict(list)
    time_groups_rg: Dict[Tuple[int, int, int], List[np.ndarray]] = defaultdict(list)

    for r in rows:
        try:
            H = int(r["H"])
            ds = int(r.get("dataset_seed", -1))
            seed = int(r["seed"])
        except Exception:
            continue

        residual_npz = _resolve_shared_path(r.get("residual_npz", ""))
        out_dir = _resolve_shared_path(r.get("out_dir", ""))
        if not residual_npz or not os.path.exists(residual_npz) or not out_dir:
            continue

        pred_npz = os.path.join(out_dir, f"combined_predictions_{args.split}_H{H}.npz")
        pred_npz = _resolve_shared_path(pred_npz)
        if not os.path.exists(pred_npz):
            continue

        try:
            d = np.load(residual_npz, allow_pickle=True)
            p = np.load(pred_npz, allow_pickle=True)
        except Exception:
            continue

        split_prefix = "test" if args.split == "test" else "val"
        tau_key = f"{split_prefix}_tau"
        tau_hat_key = f"{split_prefix}_tau_hat"
        if tau_key not in d or tau_hat_key not in d:
            continue

        tau_list = list(d[tau_key])
        tau_hat_list = list(d[tau_hat_key])
        tau_rg_list = list(p["tau_rg"]) if "tau_rg" in p else []
        if not tau_rg_list:
            continue

        tau_gt_valid_all, tau_delan_valid_all, tau_rg_valid_all = _concat_valid(
            tau_list, tau_hat_list, tau_rg_list, H
        )
        if tau_gt_valid_all.size == 0:
            continue

        err_delan = tau_delan_valid_all - tau_gt_valid_all
        err_rg = tau_rg_valid_all - tau_gt_valid_all

        rmse_time_delan = np.sqrt(np.mean(err_delan ** 2, axis=1))
        rmse_time_rg = np.sqrt(np.mean(err_rg ** 2, axis=1))

        rmse_time_delan = _resample_progress(rmse_time_delan, int(args.bins))
        rmse_time_rg = _resample_progress(rmse_time_rg, int(args.bins))

        time_groups_delan[(H, ds, seed)].append(rmse_time_delan)
        time_groups_rg[(H, ds, seed)].append(rmse_time_rg)

    os.makedirs(args.out_dir, exist_ok=True)

    per_h_seed_time_delan: Dict[Tuple[int, int], Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = defaultdict(dict)
    per_h_seed_time_rg: Dict[Tuple[int, int], Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = defaultdict(dict)

    for (H, ds, seed), curves in time_groups_delan.items():
        if curves:
            per_h_seed_time_delan[(H, ds)][seed] = _median_iqr(curves)

    for (H, ds, seed), curves in time_groups_rg.items():
        if curves:
            per_h_seed_time_rg[(H, ds)][seed] = _median_iqr(curves)

    per_h_time_delan: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    per_h_time_rg: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    hs = sorted({h for (h, _) in per_h_seed_time_delan.keys()} | {h for (h, _) in per_h_seed_time_rg.keys()})
    for H in hs:
        ds_curves_d = [per_h_seed_time_delan[(H, ds)][s][0]
                       for (h, ds), seed_map in per_h_seed_time_delan.items() if h == H
                       for s in seed_map.keys()]
        if ds_curves_d:
            per_h_time_delan[H] = _median_iqr(ds_curves_d)

        ds_curves_r = [per_h_seed_time_rg[(H, ds)][s][0]
                       for (h, ds), seed_map in per_h_seed_time_rg.items() if h == H
                       for s in seed_map.keys()]
        if ds_curves_r:
            per_h_time_rg[H] = _median_iqr(ds_curves_r)

    xs = np.linspace(0.0, 1.0, int(args.bins))
    if per_h_time_delan:
        plt.figure(figsize=(9, 5), dpi=160)
        colors = plt.cm.tab10(np.linspace(0, 1, len(per_h_time_delan)))
        for (H, (med, q25, q75)), color in zip(sorted(per_h_time_delan.items(), key=lambda kv: kv[0]), colors):
            plt.plot(xs, med, label=f"H={H}", color=color, linewidth=1.4)
            plt.fill_between(xs, q25, q75, color=color, alpha=0.18)
        max_median = None
        for med, _, _ in per_h_time_delan.values():
            try:
                m = float(np.nanmax(med))
            except Exception:
                continue
            if np.isfinite(m):
                max_median = m if max_median is None else max(max_median, m)
        if max_median is not None:
            plt.ylim(0, max_median + 0.25)
        title_feat = f" (feat={args.feature})" if args.feature else ""
        plt.title(f"DeLaN torque RMSE over progress by H{title_feat}")
        plt.xlabel("Progress (0 → 1)")
        plt.ylabel("Torque RMSE")
        plt.grid(True, alpha=0.25)
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "C1_delan_rmse_progress_by_H.png"), dpi=180)
        plt.close()

        np.savez(
            os.path.join(args.out_dir, "C1_delan_rmse_progress_by_H.npz"),
            bins=int(args.bins),
            Hs=np.array(sorted(per_h_time_delan.keys()), dtype=np.int32),
            curves=np.stack([per_h_time_delan[H][0] for H in sorted(per_h_time_delan.keys())], axis=0),
            q25=np.stack([per_h_time_delan[H][1] for H in sorted(per_h_time_delan.keys())], axis=0),
            q75=np.stack([per_h_time_delan[H][2] for H in sorted(per_h_time_delan.keys())], axis=0),
        )

    if per_h_time_rg:
        plt.figure(figsize=(9, 5), dpi=160)
        colors = plt.cm.tab10(np.linspace(0, 1, len(per_h_time_rg)))
        for (H, (med, q25, q75)), color in zip(sorted(per_h_time_rg.items(), key=lambda kv: kv[0]), colors):
            plt.plot(xs, med, label=f"H={H}", color=color, linewidth=1.4)
            plt.fill_between(xs, q25, q75, color=color, alpha=0.18)
        max_median = None
        for med, _, _ in per_h_time_rg.values():
            try:
                m = float(np.nanmax(med))
            except Exception:
                continue
            if np.isfinite(m):
                max_median = m if max_median is None else max(max_median, m)
        if max_median is not None:
            plt.ylim(0, max_median + 0.25)
        title_feat = f" (feat={args.feature})" if args.feature else ""
        plt.title(f"Combined torque RMSE over progress by H{title_feat}")
        plt.xlabel("Progress (0 → 1)")
        plt.ylabel("Torque RMSE")
        plt.grid(True, alpha=0.25)
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "C1_combined_rmse_progress_by_H.png"), dpi=180)
        plt.close()

        np.savez(
            os.path.join(args.out_dir, "C1_combined_rmse_progress_by_H.npz"),
            bins=int(args.bins),
            Hs=np.array(sorted(per_h_time_rg.keys()), dtype=np.int32),
            curves=np.stack([per_h_time_rg[H][0] for H in sorted(per_h_time_rg.keys())], axis=0),
            q25=np.stack([per_h_time_rg[H][1] for H in sorted(per_h_time_rg.keys())], axis=0),
            q75=np.stack([per_h_time_rg[H][2] for H in sorted(per_h_time_rg.keys())], axis=0),
        )


if __name__ == "__main__":
    main()
