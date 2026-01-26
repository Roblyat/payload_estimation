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
    ap.add_argument("--k_values", type=str, default="")
    ap.add_argument("--split", type=str, default="test", choices=["test", "val"])
    ap.add_argument("--feature", type=str, default="")
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
    joint_groups_delan: Dict[Tuple[int, int, int], List[np.ndarray]] = defaultdict(list)
    joint_groups_rg: Dict[Tuple[int, int, int], List[np.ndarray]] = defaultdict(list)

    for r in rows:
        try:
            K = int(r["K"])
            H = int(r["H"])
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

        rmse_joint_delan = np.sqrt(np.mean(err_delan ** 2, axis=0))
        rmse_joint_rg = np.sqrt(np.mean(err_rg ** 2, axis=0))

        time_groups_delan[(K, H, seed)].append(rmse_time_delan)
        time_groups_rg[(K, H, seed)].append(rmse_time_rg)
        joint_groups_delan[(K, H, seed)].append(rmse_joint_delan)
        joint_groups_rg[(K, H, seed)].append(rmse_joint_rg)

    os.makedirs(args.out_dir, exist_ok=True)

    per_kh_seed_time_delan: Dict[Tuple[int, int], Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = defaultdict(dict)
    per_kh_seed_time_rg: Dict[Tuple[int, int], Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = defaultdict(dict)
    per_kh_seed_joint_delan: Dict[Tuple[int, int], Dict[int, np.ndarray]] = defaultdict(dict)
    per_kh_seed_joint_rg: Dict[Tuple[int, int], Dict[int, np.ndarray]] = defaultdict(dict)

    for (K, H, seed), curves in time_groups_delan.items():
        if curves:
            per_kh_seed_time_delan[(K, H)][seed] = _median_iqr(curves)

    for (K, H, seed), curves in time_groups_rg.items():
        if curves:
            per_kh_seed_time_rg[(K, H)][seed] = _median_iqr(curves)

    for (K, H, seed), vecs in joint_groups_delan.items():
        if vecs:
            stack = np.stack(vecs, axis=0)
            per_kh_seed_joint_delan[(K, H)][seed] = np.nanmedian(stack, axis=0)

    for (K, H, seed), vecs in joint_groups_rg.items():
        if vecs:
            stack = np.stack(vecs, axis=0)
            per_kh_seed_joint_rg[(K, H)][seed] = np.nanmedian(stack, axis=0)

    per_kh_time_delan: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    per_kh_time_rg: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    per_kh_joint_delan: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    per_kh_joint_rg: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for key, seed_map in per_kh_seed_time_delan.items():
        seed_curves = [seed_map[s][0] for s in sorted(seed_map.keys()) if seed_map[s] is not None]
        if seed_curves:
            per_kh_time_delan[key] = _median_iqr(seed_curves)

    for key, seed_map in per_kh_seed_time_rg.items():
        seed_curves = [seed_map[s][0] for s in sorted(seed_map.keys()) if seed_map[s] is not None]
        if seed_curves:
            per_kh_time_rg[key] = _median_iqr(seed_curves)

    for key, seed_map in per_kh_seed_joint_delan.items():
        seed_vecs = [seed_map[s] for s in sorted(seed_map.keys()) if seed_map[s] is not None]
        if seed_vecs:
            per_kh_joint_delan[key] = _median_iqr(seed_vecs)

    for key, seed_map in per_kh_seed_joint_rg.items():
        seed_vecs = [seed_map[s] for s in sorted(seed_map.keys()) if seed_map[s] is not None]
        if seed_vecs:
            per_kh_joint_rg[key] = _median_iqr(seed_vecs)

    hs = sorted({h for (_, h) in per_kh_time_delan.keys()} | {h for (_, h) in per_kh_time_rg.keys()})
    if not hs:
        return

    multi_h = len(hs) > 1
    xs = np.linspace(0.0, 1.0, int(args.bins))

    for H in hs:
        time_by_k_delan = {K: v for (K, h), v in per_kh_time_delan.items() if h == H}
        time_by_k_rg = {K: v for (K, h), v in per_kh_time_rg.items() if h == H}
        joint_by_k_delan = {K: v for (K, h), v in per_kh_joint_delan.items() if h == H}
        joint_by_k_rg = {K: v for (K, h), v in per_kh_joint_rg.items() if h == H}

        suffix = f"_H{H}" if multi_h else ""
        title_h = f" (H={H})" if multi_h else ""

        if time_by_k_delan and time_by_k_rg:
            fig = plt.figure(figsize=(12, 4.6), dpi=160)
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)

            colors = plt.cm.tab10(np.linspace(0, 1, len(time_by_k_delan)))
            for (K, (med, q25, q75)), color in zip(sorted(time_by_k_delan.items(), key=lambda kv: kv[0]), colors):
                ax1.plot(xs, med, label=f"K={K}", color=color, linewidth=1.3)
                ax1.fill_between(xs, q25, q75, color=color, alpha=0.18)
            for (K, (med, q25, q75)), color in zip(sorted(time_by_k_rg.items(), key=lambda kv: kv[0]), colors):
                ax2.plot(xs, med, label=f"K={K}", color=color, linewidth=1.3)
                ax2.fill_between(xs, q25, q75, color=color, alpha=0.18)

            ax1.set_title(f"DeLaN torque RMSE over progress{title_h}")
            ax2.set_title(f"Combined torque RMSE over progress{title_h}")
            for ax in (ax1, ax2):
                ax.set_xlabel("Progress (0 → 1)")
                ax.set_ylabel("Torque RMSE")
                ax.grid(True, alpha=0.25)
                ax.legend(ncol=2, fontsize=8)

            fig.tight_layout()
            fig.savefig(os.path.join(args.out_dir, f"C1_torque_rmse_progress_by_K{suffix}.png"), dpi=180)
            plt.close(fig)

            np.savez(
                os.path.join(args.out_dir, f"C1_torque_rmse_progress_by_K{suffix}.npz"),
                bins=int(args.bins),
                Ks=np.array(sorted(time_by_k_delan.keys()), dtype=np.int32),
                delan_curves=np.stack([time_by_k_delan[K][0] for K in sorted(time_by_k_delan.keys())], axis=0),
                delan_q25=np.stack([time_by_k_delan[K][1] for K in sorted(time_by_k_delan.keys())], axis=0),
                delan_q75=np.stack([time_by_k_delan[K][2] for K in sorted(time_by_k_delan.keys())], axis=0),
                rg_curves=np.stack([time_by_k_rg[K][0] for K in sorted(time_by_k_rg.keys())], axis=0),
                rg_q25=np.stack([time_by_k_rg[K][1] for K in sorted(time_by_k_rg.keys())], axis=0),
                rg_q75=np.stack([time_by_k_rg[K][2] for K in sorted(time_by_k_rg.keys())], axis=0),
            )

        available_K = sorted(joint_by_k_delan.keys())
        selected_K = _parse_k_values(args.k_values)
        if selected_K:
            selected_K = [k for k in selected_K if k in joint_by_k_delan and k in joint_by_k_rg]
        if not selected_K:
            if len(available_K) <= 3:
                selected_K = available_K
            elif available_K:
                mid = available_K[len(available_K) // 2]
                selected_K = [available_K[0], mid, available_K[-1]]
                selected_K = list(dict.fromkeys(selected_K))

        if selected_K and joint_by_k_delan and joint_by_k_rg:
            n_dof = int(joint_by_k_delan[selected_K[0]][0].shape[0])
            fig_h = max(4.5, 3.0 * len(selected_K))
            fig = plt.figure(figsize=(10, fig_h), dpi=160)
            for idx, K in enumerate(selected_K):
                ax = fig.add_subplot(len(selected_K), 1, idx + 1)
                x = np.arange(n_dof)
                width = 0.35

                med_d, q25_d, q75_d = joint_by_k_delan[K]
                med_r, q25_r, q75_r = joint_by_k_rg[K]

                ax.bar(x - width / 2, med_d, width=width, label="DeLaN", alpha=0.9)
                ax.bar(x + width / 2, med_r, width=width, label="Combined", alpha=0.9)

                yerr_d = np.vstack([med_d - q25_d, q75_d - med_d])
                yerr_r = np.vstack([med_r - q25_r, q75_r - med_r])
                ax.errorbar(x - width / 2, med_d, yerr=yerr_d, fmt="none", ecolor="k", elinewidth=0.9, capsize=2, alpha=0.8)
                ax.errorbar(x + width / 2, med_r, yerr=yerr_r, fmt="none", ecolor="k", elinewidth=0.9, capsize=2, alpha=0.8)

                ax.set_title(f"K={K}{title_h}")
                ax.set_ylabel("Torque RMSE")
                ax.set_xticks(x)
                if idx == len(selected_K) - 1:
                    ax.set_xticklabels([f"joint{j}" for j in x])
                else:
                    ax.set_xticklabels([])
                ax.grid(True, axis="y", alpha=0.25)
                if idx == 0:
                    ax.legend()

            fig.tight_layout()
            fig.savefig(os.path.join(args.out_dir, f"C2_torque_rmse_per_joint_grouped{suffix}.png"), dpi=200)
            plt.close(fig)

            np.savez(
                os.path.join(args.out_dir, f"C2_torque_rmse_per_joint_grouped{suffix}.npz"),
                Ks=np.array(selected_K, dtype=np.int32),
                delan_joint_median=np.stack([joint_by_k_delan[K][0] for K in selected_K], axis=0),
                delan_joint_q25=np.stack([joint_by_k_delan[K][1] for K in selected_K], axis=0),
                delan_joint_q75=np.stack([joint_by_k_delan[K][2] for K in selected_K], axis=0),
                rg_joint_median=np.stack([joint_by_k_rg[K][0] for K in selected_K], axis=0),
                rg_joint_q25=np.stack([joint_by_k_rg[K][1] for K in selected_K], axis=0),
                rg_joint_q75=np.stack([joint_by_k_rg[K][2] for K in selected_K], axis=0),
            )


if __name__ == "__main__":
    main()
