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
    ap.add_argument("--split", type=str, default="test", choices=["test", "val"])
    ap.add_argument("--k_values", type=str, default="")
    ap.add_argument("--only_selected", action="store_true")
    ap.add_argument("--max_joints", type=int, default=6)
    args = ap.parse_args()

    summary_path = _resolve_shared_path(args.summary_jsonl)
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"summary_jsonl not found: {summary_path}")

    with open(summary_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    rmse_time_groups: Dict[Tuple[int, int], List[np.ndarray]] = defaultdict(list)
    per_joint_groups: Dict[Tuple[int, int], List[np.ndarray]] = defaultdict(list)

    for r in rows:
        if args.only_selected and not r.get("selected", False):
            continue
        try:
            K = int(r["K"])
            seed = int(r["seed"])
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
                rmse_time_groups[(K, seed)].append(rmse_t)
            except Exception:
                pass

        eval_key = "eval_test" if args.split == "test" else "eval_val"
        per_joint = d.get(eval_key, {}).get("torque_rmse_per_joint", None)
        if isinstance(per_joint, list) and per_joint:
            vec = np.asarray(per_joint, dtype=np.float32)
            if args.max_joints is not None:
                vec = vec[: int(args.max_joints)]
            per_joint_groups[(K, seed)].append(vec)

    os.makedirs(args.out_dir, exist_ok=True)

    # A3: RMSE vs normalized progress
    per_k_seed_time: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = defaultdict(dict)
    for (K, seed), curves in rmse_time_groups.items():
        if not curves:
            continue
        per_k_seed_time[K][seed] = _median_iqr(curves)

    per_k_time: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for K, seed_map in per_k_seed_time.items():
        seed_curves = [seed_map[s][0] for s in sorted(seed_map.keys()) if seed_map[s] is not None]
        if seed_curves:
            per_k_time[K] = _median_iqr(seed_curves)

    if per_k_time:
        plt.figure(figsize=(9, 5), dpi=160)
        colors = plt.cm.tab10(np.linspace(0, 1, len(per_k_time)))
        xs = np.linspace(0.0, 1.0, int(args.bins))
        for (K, (med, q25, q75)), color in zip(sorted(per_k_time.items(), key=lambda kv: kv[0]), colors):
            plt.plot(xs, med, label=f"K={K}", color=color, linewidth=1.4)
            plt.fill_between(xs, q25, q75, color=color, alpha=0.18)
        max_median = None
        for med, _, _ in per_k_time.values():
            try:
                m = float(np.nanmax(med))
            except Exception:
                continue
            if np.isfinite(m):
                max_median = m if max_median is None else max(max_median, m)
        if max_median is not None:
            plt.ylim(0, max_median + 0.25)
        plt.title("DeLaN $i_{motor}$ RMSE over progress (median ± IQR) by K")
        plt.xlabel("Progress (0 → 1)")
        plt.ylabel("$i_{motor}$ RMSE [A]")
        plt.grid(True, alpha=0.25)
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "A3_torque_rmse_progress_by_K.png"), dpi=180)
        plt.close()

        np.savez(
            os.path.join(args.out_dir, "A3_torque_rmse_progress_by_K.npz"),
            bins=int(args.bins),
            Ks=np.array(sorted(per_k_time.keys()), dtype=np.int32),
            curves=np.stack([per_k_time[K][0] for K in sorted(per_k_time.keys())], axis=0),
            q25=np.stack([per_k_time[K][1] for K in sorted(per_k_time.keys())], axis=0),
            q75=np.stack([per_k_time[K][2] for K in sorted(per_k_time.keys())], axis=0),
        )

    # A4: RMSE per joint (grouped bars for selected K)
    per_k_seed_joint: Dict[int, Dict[int, np.ndarray]] = defaultdict(dict)
    for (K, seed), vecs in per_joint_groups.items():
        if not vecs:
            continue
        stack = np.stack(vecs, axis=0)
        per_k_seed_joint[K][seed] = np.nanmedian(stack, axis=0)

    per_k_joint: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for K, seed_map in per_k_seed_joint.items():
        seed_vecs = [seed_map[s] for s in sorted(seed_map.keys()) if seed_map[s] is not None]
        if seed_vecs:
            med, q25, q75 = _median_iqr(seed_vecs)
            per_k_joint[K] = (med, q25, q75)

    available_K = sorted(per_k_joint.keys())
    selected_K = _parse_k_values(args.k_values)
    if selected_K:
        selected_K = [k for k in selected_K if k in per_k_joint]
    if not selected_K:
        selected_K = available_K

    if selected_K:
        n_dof = int(per_k_joint[selected_K[0]][0].shape[0])
        x = np.arange(n_dof)
        n_groups = len(selected_K)
        width = 0.8 / max(1, n_groups)

        plt.figure(figsize=(10, 4.8), dpi=160)
        for i, K in enumerate(selected_K):
            med, q25, q75 = per_k_joint[K]
            offsets = x - 0.4 + (i + 0.5) * width
            yerr = np.vstack([med - q25, q75 - med])
            plt.bar(offsets, med, width=width, label=f"K={K}", alpha=0.9)
            plt.errorbar(offsets, med, yerr=yerr, fmt="none", ecolor="k", elinewidth=0.9, capsize=2, alpha=0.8)
        max_median = None
        for med, _, _ in per_k_joint.values():
            try:
                m = float(np.nanmax(med))
            except Exception:
                continue
            if np.isfinite(m):
                max_median = m if max_median is None else max(max_median, m)
        if max_median is not None:
            plt.ylim(0, max_median + 0.25)

        plt.title("DeLaN $i_{motor}$ RMSE per joint (median ± IQR)")
        plt.xlabel("Joint")
        plt.ylabel("Joint $i_{motor}$ RMSE [A]")
        plt.xticks(x, [f"joint{j}" for j in x])
        plt.grid(True, axis="y", alpha=0.25)
        plt.legend(ncol=min(3, len(selected_K)), fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "A4_torque_rmse_per_joint_grouped.png"), dpi=200)
        plt.close()

        np.savez(
            os.path.join(args.out_dir, "A4_torque_rmse_per_joint_grouped.npz"),
            Ks=np.array(selected_K, dtype=np.int32),
            joint_median=np.stack([per_k_joint[K][0] for K in selected_K], axis=0),
            joint_q25=np.stack([per_k_joint[K][1] for K in selected_K], axis=0),
            joint_q75=np.stack([per_k_joint[K][2] for K in selected_K], axis=0),
        )


if __name__ == "__main__":
    main()
