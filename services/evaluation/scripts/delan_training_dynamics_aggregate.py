#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass
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


def _read_csv_curve(path: str, x_col: str, y_col: str) -> Tuple[List[int], List[float]]:
    xs: List[int] = []
    ys: List[float] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                xs.append(int(float(row[x_col])))
                ys.append(float(row[y_col]))
            except Exception:
                continue
    return xs, ys


def _pad_to_epochs(xs: List[int], ys: List[float], emax: int) -> np.ndarray:
    if not xs or not ys or emax <= 0:
        return np.full((emax,), np.nan, dtype=np.float32)
    arr = np.full((emax,), np.nan, dtype=np.float32)
    for x, y in zip(xs, ys):
        if 1 <= x <= emax:
            arr[x - 1] = float(y)
    # forward fill
    last = np.nan
    for i in range(emax):
        if np.isfinite(arr[i]):
            last = arr[i]
        elif np.isfinite(last):
            arr[i] = last
    # backfill leading NaNs
    if np.isnan(arr[0]):
        finite_idx = np.where(np.isfinite(arr))[0]
        if finite_idx.size > 0:
            arr[: finite_idx[0]] = arr[finite_idx[0]]
    return arr


def _median_iqr(curves: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    stack = np.stack(curves, axis=0)
    median = np.nanmedian(stack, axis=0)
    q25 = np.nanpercentile(stack, 25, axis=0)
    q75 = np.nanpercentile(stack, 75, axis=0)
    return median, q25, q75


@dataclass
class CurveSet:
    epochs: List[int]
    values: List[float]
    delan_seed: int


def _plot_overlay(curves: List[CurveSet], title: str, ylabel: str, out_path: str) -> None:
    if not curves:
        return
    plt.figure(figsize=(8, 4), dpi=140)
    for c in sorted(curves, key=lambda x: x.delan_seed):
        plt.plot(c.epochs, c.values, label=f"s{c.delan_seed}", linewidth=1.2, alpha=0.9)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--pad_to_epochs", type=int, default=None)
    ap.add_argument("--write_best_overlays", action="store_true")
    args = ap.parse_args()

    summary_path = _resolve_shared_path(args.summary_jsonl)
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"summary_jsonl not found: {summary_path}")

    with open(summary_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    groups: Dict[Tuple[int, int], List[dict]] = defaultdict(list)
    for r in rows:
        try:
            key = (int(r["K"]), int(r["seed"]))
        except Exception:
            continue
        groups[key].append(r)

    os.makedirs(args.out_dir, exist_ok=True)

    # Storage for aggregated curves
    per_k_seed_train: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = defaultdict(dict)
    per_k_seed_val: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = defaultdict(dict)

    for (K, seed), items in groups.items():
        # determine max epochs
        if args.pad_to_epochs is not None:
            emax = int(args.pad_to_epochs)
        else:
            emax = max(int(r.get("delan_epochs", 0)) for r in items if r.get("delan_epochs")) or None
        if not emax:
            continue

        train_curves: List[np.ndarray] = []
        val_curves: List[np.ndarray] = []
        overlay_train: List[CurveSet] = []
        overlay_val: List[CurveSet] = []

        best_model_dir = None
        best_run_name = None

        for r in items:
            metrics_json = _resolve_shared_path(
                r.get("metrics_json_container") or r.get("metrics_json", "")
            )
            if not os.path.exists(metrics_json):
                continue
            with open(metrics_json, "r", encoding="utf-8") as mf:
                d = json.load(mf)

            artifacts = d.get("artifacts", {})
            train_csv = _resolve_shared_path(artifacts.get("train_history_csv", ""))
            elbow_csv = _resolve_shared_path(artifacts.get("elbow_history_csv", ""))

            if train_csv and os.path.exists(train_csv):
                epochs, loss = _read_csv_curve(train_csv, "epoch", "loss")
                train_curves.append(_pad_to_epochs(epochs, loss, emax))
                if r.get("delan_seed") is not None:
                    overlay_train.append(CurveSet(epochs, loss, int(r["delan_seed"])))

            if elbow_csv and os.path.exists(elbow_csv):
                e_epochs, e_mse = _read_csv_curve(elbow_csv, "epoch", "eval_mse")
                val_curves.append(_pad_to_epochs(e_epochs, e_mse, emax))
                if r.get("delan_seed") is not None:
                    overlay_val.append(CurveSet(e_epochs, e_mse, int(r["delan_seed"])))

            if r.get("selected"):
                best_model_dir = d.get("model_dir", None)
                best_run_name = d.get("run_name", None)

        if train_curves:
            per_k_seed_train[K][seed] = _median_iqr(train_curves)
        if val_curves:
            per_k_seed_val[K][seed] = _median_iqr(val_curves)

        if args.write_best_overlays and best_model_dir and best_run_name:
            os.makedirs(best_model_dir, exist_ok=True)
            if overlay_train:
                out_path = os.path.join(best_model_dir, f"{best_run_name}__train_loss_all_delan_seeds.png")
                _plot_overlay(
                    overlay_train,
                    title=f"{best_run_name} | Train loss by delan_seed",
                    ylabel="Train loss",
                    out_path=out_path,
                )
            if overlay_val:
                out_path = os.path.join(best_model_dir, f"{best_run_name}__val_mse_all_delan_seeds.png")
                _plot_overlay(
                    overlay_val,
                    title=f"{best_run_name} | Val MSE by delan_seed",
                    ylabel="Val MSE",
                    out_path=out_path,
                )

    # Aggregate across dataset seeds to final per-K curves
    per_k_train: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    per_k_val: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for K, seed_map in per_k_seed_train.items():
        seed_curves = [seed_map[s][0] for s in sorted(seed_map.keys()) if seed_map[s] is not None]
        if seed_curves:
            per_k_train[K] = _median_iqr(seed_curves)

    for K, seed_map in per_k_seed_val.items():
        seed_curves = [seed_map[s][0] for s in sorted(seed_map.keys()) if seed_map[s] is not None]
        if seed_curves:
            per_k_val[K] = _median_iqr(seed_curves)

    # Plot A1 / A2
    def _plot_by_k(data: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]], title: str, ylabel: str, out_name: str) -> None:
        if not data:
            return
        plt.figure(figsize=(9, 5), dpi=160)
        colors = plt.cm.tab10(np.linspace(0, 1, len(data)))
        for (K, (med, q25, q75)), color in zip(sorted(data.items(), key=lambda kv: kv[0]), colors):
            xs = np.arange(1, len(med) + 1)
            plt.plot(xs, med, label=f"K={K}", color=color, linewidth=1.4)
            plt.fill_between(xs, q25, q75, color=color, alpha=0.18)
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.25)
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        out_path = os.path.join(args.out_dir, out_name)
        plt.savefig(out_path, dpi=180)
        plt.close()

    _plot_by_k(
        per_k_train,
        title="DeLaN training loss (median ± IQR) by K",
        ylabel="Train loss",
        out_name="A1_train_loss_by_K.png",
    )

    _plot_by_k(
        per_k_val,
        title="DeLaN validation MSE (median ± IQR) by K",
        ylabel="Val MSE",
        out_name="A2_val_mse_by_K.png",
    )


if __name__ == "__main__":
    main()
