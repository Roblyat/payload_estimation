#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

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


def _parse_k_values(s: str | None) -> List[int]:
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
    ap.add_argument("--k_values", type=str, default="")
    ap.add_argument("--feature", type=str, default="")
    args = ap.parse_args()

    summary_path = _resolve_shared_path(args.summary_jsonl)
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"summary_jsonl not found: {summary_path}")

    with open(summary_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    if args.feature:
        rows = [r for r in rows if str(r.get("features", "")) == args.feature]

    time_groups: Dict[Tuple[int, int, int], List[np.ndarray]] = defaultdict(list)
    joint_groups: Dict[Tuple[int, int, int], List[np.ndarray]] = defaultdict(list)

    for r in rows:
        try:
            K = int(r["K"])
            H = int(r["H"])
            seed = int(r["seed"])
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

        time_groups[(K, H, seed)].append(rmse_t)
        joint_groups[(K, H, seed)].append(rmse_joint)

    os.makedirs(args.out_dir, exist_ok=True)

    per_kh_seed_time: Dict[Tuple[int, int], Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = defaultdict(dict)
    per_kh_seed_joint: Dict[Tuple[int, int], Dict[int, np.ndarray]] = defaultdict(dict)

    for (K, H, seed), curves in time_groups.items():
        if curves:
            per_kh_seed_time[(K, H)][seed] = _median_iqr(curves)

    for (K, H, seed), vecs in joint_groups.items():
        if vecs:
            stack = np.stack(vecs, axis=0)
            per_kh_seed_joint[(K, H)][seed] = np.nanmedian(stack, axis=0)

    per_kh_time: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    per_kh_joint: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for key, seed_map in per_kh_seed_time.items():
        seed_curves = [seed_map[s][0] for s in sorted(seed_map.keys()) if seed_map[s] is not None]
        if seed_curves:
            per_kh_time[key] = _median_iqr(seed_curves)

    for key, seed_map in per_kh_seed_joint.items():
        seed_vecs = [seed_map[s] for s in sorted(seed_map.keys()) if seed_map[s] is not None]
        if seed_vecs:
            per_kh_joint[key] = _median_iqr(seed_vecs)

    hs = sorted({h for (_, h) in per_kh_time.keys()} | {h for (_, h) in per_kh_joint.keys()})
    if not hs:
        return

    multi_h = len(hs) > 1
    xs = np.linspace(0.0, 1.0, int(args.bins))

    for H in hs:
        time_by_k = {K: v for (K, h), v in per_kh_time.items() if h == H}
        joint_by_k = {K: v for (K, h), v in per_kh_joint.items() if h == H}
        suffix = f"_H{H}" if multi_h else ""
        title_h = f" (H={H})" if multi_h else ""

        if time_by_k:
            plt.figure(figsize=(9, 5), dpi=160)
            colors = plt.cm.tab10(np.linspace(0, 1, len(time_by_k)))
            for (K, (med, q25, q75)), color in zip(sorted(time_by_k.items(), key=lambda kv: kv[0]), colors):
                plt.plot(xs, med, label=f"K={K}", color=color, linewidth=1.4)
                plt.fill_between(xs, q25, q75, color=color, alpha=0.18)
            plt.title(f"LSTM residual RMSE over progress by K{title_h}")
            plt.xlabel("Progress (0 → 1)")
            plt.ylabel("Residual RMSE")
            plt.grid(True, alpha=0.25)
            plt.legend(ncol=2, fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(args.out_dir, f"B3_residual_rmse_progress_by_K{suffix}.png"), dpi=180)
            plt.close()

            np.savez(
                os.path.join(args.out_dir, f"B3_residual_rmse_progress_by_K{suffix}.npz"),
                bins=int(args.bins),
                Ks=np.array(sorted(time_by_k.keys()), dtype=np.int32),
                curves=np.stack([time_by_k[K][0] for K in sorted(time_by_k.keys())], axis=0),
                q25=np.stack([time_by_k[K][1] for K in sorted(time_by_k.keys())], axis=0),
                q75=np.stack([time_by_k[K][2] for K in sorted(time_by_k.keys())], axis=0),
            )

        available_K = sorted(joint_by_k.keys())
        selected_K = _parse_k_values(args.k_values)
        if selected_K:
            selected_K = [k for k in selected_K if k in joint_by_k]
        if not selected_K:
            if len(available_K) <= 3:
                selected_K = available_K
            elif available_K:
                mid = available_K[len(available_K) // 2]
                selected_K = [available_K[0], mid, available_K[-1]]
                selected_K = list(dict.fromkeys(selected_K))

        if selected_K:
            n_dof = int(joint_by_k[selected_K[0]][0].shape[0])
            x = np.arange(n_dof)
            n_groups = len(selected_K)
            width = 0.8 / max(1, n_groups)

            plt.figure(figsize=(10, 4.8), dpi=160)
            for i, K in enumerate(selected_K):
                med, q25, q75 = joint_by_k[K]
                offsets = x - 0.4 + (i + 0.5) * width
                yerr = np.vstack([med - q25, q75 - med])
                plt.bar(offsets, med, width=width, label=f"K={K}", alpha=0.9)
                plt.errorbar(offsets, med, yerr=yerr, fmt="none", ecolor="k", elinewidth=0.9, capsize=2, alpha=0.8)

            plt.title(f"LSTM residual RMSE per joint (median ± IQR){title_h}")
            plt.xlabel("Joint")
            plt.ylabel("Residual RMSE")
            plt.xticks(x, [f"joint{j}" for j in x])
            plt.grid(True, axis="y", alpha=0.25)
            plt.legend(ncol=min(3, len(selected_K)), fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(args.out_dir, f"B4_residual_rmse_per_joint_grouped{suffix}.png"), dpi=200)
            plt.close()

            np.savez(
                os.path.join(args.out_dir, f"B4_residual_rmse_per_joint_grouped{suffix}.npz"),
                Ks=np.array(selected_K, dtype=np.int32),
                joint_median=np.stack([joint_by_k[K][0] for K in selected_K], axis=0),
                joint_q25=np.stack([joint_by_k[K][1] for K in selected_K], axis=0),
                joint_q75=np.stack([joint_by_k[K][2] for K in selected_K], axis=0),
            )


if __name__ == "__main__":
    main()
