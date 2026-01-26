#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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


def _plot_by_k(
    data: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    title: str,
    ylabel: str,
    out_path: str,
) -> None:
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
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--pad_to_epochs", type=int, default=None)
    ap.add_argument("--feature", type=str, default="")
    args = ap.parse_args()

    summary_path = _resolve_shared_path(args.summary_jsonl)
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"summary_jsonl not found: {summary_path}")

    with open(summary_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    if args.feature:
        rows = [r for r in rows if str(r.get("features", "")) == args.feature]

    groups: Dict[Tuple[int, int, int], List[dict]] = defaultdict(list)
    for r in rows:
        try:
            key = (int(r["K"]), int(r["H"]), int(r["seed"]))
        except Exception:
            continue
        groups[key].append(r)

    os.makedirs(args.out_dir, exist_ok=True)

    per_kh_seed_train: Dict[Tuple[int, int], Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = defaultdict(dict)
    per_kh_seed_val: Dict[Tuple[int, int], Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = defaultdict(dict)

    for (K, H, seed), items in groups.items():
        emax = args.pad_to_epochs or None
        train_curves: List[np.ndarray] = []
        val_curves: List[np.ndarray] = []

        for r in items:
            metrics_json = _resolve_shared_path(r.get("metrics_json", ""))
            if not metrics_json or not os.path.exists(metrics_json):
                continue
            with open(metrics_json, "r", encoding="utf-8") as mf:
                d = json.load(mf)
            history_csv = _resolve_shared_path(d.get("train", {}).get("history_csv", ""))
            if not history_csv or not os.path.exists(history_csv):
                continue
            epochs_ran = d.get("train", {}).get("epochs_ran", None)
            if emax is None:
                emax = int(epochs_ran) if epochs_ran else None
            if emax is None:
                # fallback from CSV length
                emax = sum(1 for _ in open(history_csv, "r", encoding="utf-8")) - 1
            epochs, loss = _read_csv_curve(history_csv, "epoch", "loss")
            _, val_loss = _read_csv_curve(history_csv, "epoch", "val_loss")
            if emax:
                train_curves.append(_pad_to_epochs(epochs, loss, int(emax)))
                val_curves.append(_pad_to_epochs(epochs, val_loss, int(emax)))

        if train_curves:
            per_kh_seed_train[(K, H)][seed] = _median_iqr(train_curves)
        if val_curves:
            per_kh_seed_val[(K, H)][seed] = _median_iqr(val_curves)

    per_kh_train: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    per_kh_val: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for key, seed_map in per_kh_seed_train.items():
        seed_curves = [seed_map[s][0] for s in sorted(seed_map.keys()) if seed_map[s] is not None]
        if seed_curves:
            per_kh_train[key] = _median_iqr(seed_curves)

    for key, seed_map in per_kh_seed_val.items():
        seed_curves = [seed_map[s][0] for s in sorted(seed_map.keys()) if seed_map[s] is not None]
        if seed_curves:
            per_kh_val[key] = _median_iqr(seed_curves)

    hs = sorted({h for (_, h) in per_kh_train.keys()} | {h for (_, h) in per_kh_val.keys()})
    if not hs:
        return

    multi_h = len(hs) > 1
    for H in hs:
        train_by_k = {K: v for (K, h), v in per_kh_train.items() if h == H}
        val_by_k = {K: v for (K, h), v in per_kh_val.items() if h == H}
        suffix = f"_H{H}" if multi_h else ""
        title_h = f" (H={H})" if multi_h else ""

        if train_by_k:
            _plot_by_k(
                train_by_k,
                title=f"LSTM training loss (median ± IQR) by K{title_h}",
                ylabel="Train loss",
                out_path=os.path.join(args.out_dir, f"B1_train_loss_by_K{suffix}.png"),
            )
            np.savez(
                os.path.join(args.out_dir, f"B1_train_loss_by_K{suffix}.npz"),
                Ks=np.array(sorted(train_by_k.keys()), dtype=np.int32),
                curves=np.stack([train_by_k[K][0] for K in sorted(train_by_k.keys())], axis=0),
                q25=np.stack([train_by_k[K][1] for K in sorted(train_by_k.keys())], axis=0),
                q75=np.stack([train_by_k[K][2] for K in sorted(train_by_k.keys())], axis=0),
            )

        if val_by_k:
            _plot_by_k(
                val_by_k,
                title=f"LSTM validation loss (median ± IQR) by K{title_h}",
                ylabel="Val loss",
                out_path=os.path.join(args.out_dir, f"B2_val_loss_by_K{suffix}.png"),
            )
            np.savez(
                os.path.join(args.out_dir, f"B2_val_loss_by_K{suffix}.npz"),
                Ks=np.array(sorted(val_by_k.keys()), dtype=np.int32),
                curves=np.stack([val_by_k[K][0] for K in sorted(val_by_k.keys())], axis=0),
                q25=np.stack([val_by_k[K][1] for K in sorted(val_by_k.keys())], axis=0),
                q75=np.stack([val_by_k[K][2] for K in sorted(val_by_k.keys())], axis=0),
            )


if __name__ == "__main__":
    main()
