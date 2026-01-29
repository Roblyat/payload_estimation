#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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
        if 1 <= int(x) <= emax:
            arr[int(x) - 1] = float(y)
    last = np.nan
    for i in range(emax):
        if np.isfinite(arr[i]):
            last = arr[i]
        elif np.isfinite(last):
            arr[i] = last
    if emax > 0 and np.isnan(arr[0]):
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


def _safe_tag(s: str) -> str:
    return str(s).replace(".", "p").replace("/", "_")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    summary_path = _resolve_shared_path(args.summary_jsonl)
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"summary_jsonl not found: {summary_path}")

    with open(summary_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    print(f"[info] summary_jsonl={summary_path} rows={len(rows)}")
    print(f"[info] out_dir={args.out_dir} MPLCONFIGDIR={os.environ.get('MPLCONFIGDIR')}")

    groups: Dict[Tuple[str, int], List[dict]] = defaultdict(list)
    for r in rows:
        if r.get("diverged"):
            continue
        hp = r.get("hp_preset")
        ds = r.get("dataset_seed")
        if hp is None or ds is None:
            continue
        try:
            ds_i = int(ds)
        except Exception:
            continue
        groups[(str(hp), ds_i)].append(r)

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[info] groups={len(groups)}")

    for (hp_preset, dataset_seed), items in groups.items():
        train_raw: List[Tuple[List[int], List[float]]] = []
        val_raw: List[Tuple[List[int], List[float]]] = []
        emax = 0
        K = None

        for r in items:
            if K is None and r.get("K") is not None:
                try:
                    K = int(r["K"])
                except Exception:
                    K = None
            metrics_json = _resolve_shared_path(
                r.get("metrics_json_container") or r.get("metrics_json", "")
            )
            if not metrics_json or not os.path.exists(metrics_json):
                continue
            with open(metrics_json, "r", encoding="utf-8") as mf:
                d = json.load(mf)
            artifacts = d.get("artifacts", {}) or {}
            train_csv = _resolve_shared_path(artifacts.get("train_history_csv", ""))
            elbow_csv = _resolve_shared_path(artifacts.get("elbow_history_csv", ""))

            if train_csv and os.path.exists(train_csv):
                epochs, loss = _read_csv_curve(train_csv, "epoch", "loss")
                if epochs:
                    emax = max(emax, max(epochs))
                train_raw.append((epochs, loss))

            if elbow_csv and os.path.exists(elbow_csv):
                e_epochs, e_mse = _read_csv_curve(elbow_csv, "epoch", "eval_mse")
                if e_epochs:
                    emax = max(emax, max(e_epochs))
                val_raw.append((e_epochs, e_mse))

        if emax <= 0:
            print(f"[warn] no curve data for hp={hp_preset} d={dataset_seed}")
            continue

        train_curves = [_pad_to_epochs(xs, ys, emax) for xs, ys in train_raw if xs and ys]
        val_curves = [_pad_to_epochs(xs, ys, emax) for xs, ys in val_raw if xs and ys]

        fig = plt.figure(figsize=(10, 4.5), dpi=160)
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        epochs = np.arange(1, emax + 1)

        if train_curves:
            med, q25, q75 = _median_iqr(train_curves)
            ax1.plot(epochs, med, color="#1f77b4", linewidth=1.6, label="median")
            ax1.fill_between(epochs, q25, q75, color="#1f77b4", alpha=0.18, label="IQR")
            if np.nanmin(med) > 0:
                ax1.set_yscale("log")
        else:
            ax1.text(0.5, 0.5, "no train curves", ha="center", va="center")

        if val_curves:
            med, q25, q75 = _median_iqr(val_curves)
            ax2.plot(epochs, med, color="#ff7f0e", linewidth=1.6, label="median")
            ax2.fill_between(epochs, q25, q75, color="#ff7f0e", alpha=0.18, label="IQR")
            if np.nanmin(med) > 0:
                ax2.set_yscale("log")
        else:
            ax2.text(0.5, 0.5, "no val curves", ha="center", va="center")

        k_label = f"K={K} " if K is not None else ""
        fig.suptitle(f"DeLaN best fold curves | {k_label}hp={hp_preset} d={dataset_seed}")
        ax1.set_title("Train loss (median ± IQR)")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.grid(True, alpha=0.25)
        ax1.legend(fontsize=8)

        ax2.set_title("Val MSE (median ± IQR)")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Val MSE")
        ax2.grid(True, alpha=0.25)
        ax2.legend(fontsize=8)

        fig.tight_layout()

        k_part = f"K{K}__" if K is not None else ""
        tag = f"{k_part}hp_{_safe_tag(hp_preset)}__d{dataset_seed}"
        out_png = os.path.join(args.out_dir, f"delan_best_fold_{tag}.png")
        fig.savefig(out_png, dpi=200)
        print(f"[info] saved {out_png}")
        plt.close(fig)


if __name__ == "__main__":
    main()
