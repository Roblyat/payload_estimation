#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

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


def _resolve_npz_path(npz_path: str) -> str:
    p = os.path.expanduser(str(npz_path))
    if os.path.isdir(p):
        stem = os.path.basename(os.path.normpath(p))
        return os.path.join(p, f"{stem}.npz")
    if p.endswith(".npz") and (not os.path.exists(p)):
        stem = os.path.splitext(os.path.basename(p))[0]
        root = os.path.dirname(p)
        if os.path.basename(root) == stem:
            candidate = os.path.join(os.path.dirname(root), f"{stem}.npz")
            if os.path.exists(candidate):
                return candidate
        else:
            candidate = os.path.join(root, stem, f"{stem}.npz")
            if os.path.exists(candidate):
                return candidate
    return p


def _parse_delan_tag(run_name: str) -> str:
    if "__" in run_name:
        return run_name.split("__")[-1]
    return run_name


def _parse_delan_seed(delan_tag: str) -> Optional[int]:
    m = re.search(r"_s(\d+)_", delan_tag)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _pick_exemplar(rows: List[dict], K: int) -> Tuple[Optional[dict], Optional[dict]]:
    candidates = [r for r in rows if int(r.get("K", -1)) == int(K)]
    selected = [r for r in candidates if r.get("selected", False)]
    use = selected or candidates
    if not use:
        return None, None

    scored: List[Tuple[float, dict, dict]] = []
    for r in use:
        metrics_json = _resolve_shared_path(
            r.get("metrics_json_container") or r.get("metrics_json", "")
        )
        if not metrics_json or not os.path.exists(metrics_json):
            continue
        with open(metrics_json, "r", encoding="utf-8") as mf:
            d = json.load(mf)
        rmse = float(d.get("eval_test", {}).get("torque_rmse", float("inf")))
        scored.append((rmse, r, d))
    if not scored:
        return None, None

    rmses = [s[0] for s in scored]
    med = float(np.median(rmses))
    scored.sort(key=lambda t: abs(t[0] - med))
    _, row, metrics = scored[0]
    return row, metrics


def _load_residual_npz(row: dict, metrics: dict) -> Optional[Dict[str, Any]]:
    dataset = row.get("dataset")
    run_tag = row.get("run_tag")
    K = int(row.get("K"))
    if dataset is None or run_tag is None:
        return None

    run_name = metrics.get("run_name", "")
    delan_tag = _parse_delan_tag(run_name)
    base_id = f"{dataset}__{run_tag}"
    residual_name = f"{base_id}__K{K}__residual__{delan_tag}.npz"
    residual_path = _resolve_npz_path(f"/workspace/shared/data/processed/{residual_name}")
    if not os.path.exists(residual_path):
        return None

    return dict(np.load(residual_path, allow_pickle=True))


def _pick_longest_traj(tau_list: np.ndarray) -> int:
    lengths = [int(np.asarray(t).shape[0]) for t in tau_list]
    return int(np.argmax(lengths))


def _pick_joints(tau_gt: np.ndarray, tau_hat: np.ndarray) -> List[int]:
    err = tau_hat - tau_gt
    rmse_j = np.sqrt(np.mean(err ** 2, axis=0))
    j_max = int(np.argmax(rmse_j))
    if rmse_j.shape[0] <= 1:
        return [j_max]
    j_alt = 0 if j_max != 0 else 1
    return [j_max, j_alt]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--k_high", type=int, default=None)
    ap.add_argument("--k_low", type=int, default=None)
    ap.add_argument("--max_samples", type=int, default=2000)
    args = ap.parse_args()

    summary_path = _resolve_shared_path(args.summary_jsonl)
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"summary_jsonl not found: {summary_path}")

    with open(summary_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    if not rows:
        raise RuntimeError("No rows found in summary_jsonl")

    available_K = sorted({int(r.get("K")) for r in rows if r.get("K") is not None})
    if not available_K:
        raise RuntimeError("No valid K values in summary_jsonl")

    k_high = args.k_high if args.k_high is not None else available_K[-1]
    row_high, metrics_high = _pick_exemplar(rows, k_high)
    if row_high is None or metrics_high is None:
        raise RuntimeError(f"No exemplar found for K={k_high}")

    data_high = _load_residual_npz(row_high, metrics_high)
    if data_high is None:
        raise RuntimeError(f"Residual npz not found for K={k_high}")

    tau_list = data_high.get("test_tau")
    tau_hat_list = data_high.get("test_tau_hat")
    if tau_list is None or tau_hat_list is None:
        raise RuntimeError("Residual npz missing test_tau/test_tau_hat")

    idx_high = _pick_longest_traj(tau_list)
    tau_gt_h = np.asarray(tau_list[idx_high])
    tau_hat_h = np.asarray(tau_hat_list[idx_high])

    os.makedirs(args.out_dir, exist_ok=True)

    if args.k_low is not None:
        row_low, metrics_low = _pick_exemplar(rows, int(args.k_low))
        if row_low is None or metrics_low is None:
            raise RuntimeError(f"No exemplar found for K={args.k_low}")
        data_low = _load_residual_npz(row_low, metrics_low)
        if data_low is None:
            raise RuntimeError(f"Residual npz not found for K={args.k_low}")
        tau_list_l = data_low.get("test_tau")
        tau_hat_list_l = data_low.get("test_tau_hat")
        idx_low = _pick_longest_traj(tau_list_l)
        tau_gt_l = np.asarray(tau_list_l[idx_low])
        tau_hat_l = np.asarray(tau_hat_list_l[idx_low])

        joints = _pick_joints(tau_gt_h, tau_hat_h)
        j = joints[0]
        fig = plt.figure(figsize=(12, 4), dpi=160)
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        n = min(int(args.max_samples), tau_gt_h.shape[0])
        ax1.plot(tau_gt_h[:n, j], label="GT", linewidth=1.0)
        ax1.plot(tau_hat_h[:n, j], label="DeLaN", linewidth=1.0, alpha=0.85)
        ax1.set_title(f"K={k_high} | joint {j}")
        ax1.grid(True, alpha=0.2)
        ax1.legend()

        n2 = min(int(args.max_samples), tau_gt_l.shape[0])
        ax2.plot(tau_gt_l[:n2, j], label="GT", linewidth=1.0)
        ax2.plot(tau_hat_l[:n2, j], label="DeLaN", linewidth=1.0, alpha=0.85)
        ax2.set_title(f"K={args.k_low} | joint {j}")
        ax2.grid(True, alpha=0.2)

        fig.tight_layout()
        out_path = os.path.join(args.out_dir, f"DeLaN_Torque_vs_GT_K{k_high}_vs_K{args.k_low}.png")
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return

    joints = _pick_joints(tau_gt_h, tau_hat_h)
    fig = plt.figure(figsize=(10, 4 * len(joints)), dpi=160)
    for i, j in enumerate(joints):
        ax = fig.add_subplot(len(joints), 1, i + 1)
        n = min(int(args.max_samples), tau_gt_h.shape[0])
        ax.plot(tau_gt_h[:n, j], label="GT", linewidth=1.0)
        ax.plot(tau_hat_h[:n, j], label="DeLaN", linewidth=1.0, alpha=0.85)
        ax.set_title(f"K={k_high} | joint {j}")
        ax.grid(True, alpha=0.2)
        if i == 0:
            ax.legend()

    fig.tight_layout()
    out_path = os.path.join(args.out_dir, f"DeLaN_Torque_vs_GT_K{k_high}.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
