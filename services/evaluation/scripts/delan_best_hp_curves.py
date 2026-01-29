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


def _pad_curve(arr: np.ndarray, length: int) -> np.ndarray:
    if arr is None or length <= 0:
        return np.full((length,), np.nan, dtype=np.float32)
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return np.full((length,), np.nan, dtype=np.float32)
    if arr.size == length:
        return arr
    out = np.full((length,), np.nan, dtype=np.float32)
    n = min(length, arr.size)
    out[:n] = arr[:n]
    last = np.nan
    for i in range(length):
        if np.isfinite(out[i]):
            last = out[i]
        elif np.isfinite(last):
            out[i] = last
    if length > 0 and np.isnan(out[0]):
        finite_idx = np.where(np.isfinite(out))[0]
        if finite_idx.size > 0:
            out[: finite_idx[0]] = out[finite_idx[0]]
    return out


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
    ap.add_argument("--hp_presets", type=str, default="")
    args = ap.parse_args()

    summary_path = _resolve_shared_path(args.summary_jsonl)
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"summary_jsonl not found: {summary_path}")

    with open(summary_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    print(f"[info] summary_jsonl={summary_path} rows={len(rows)}")
    print(f"[info] out_dir={args.out_dir} MPLCONFIGDIR={os.environ.get('MPLCONFIGDIR')}")

    hp_filter = _parse_hp_presets(args.hp_presets)

    groups: Dict[Tuple[str, int], List[dict]] = defaultdict(list)
    for r in rows:
        if r.get("diverged"):
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
        groups[(str(hp), ds_i)].append(r)

    os.makedirs(args.out_dir, exist_ok=True)

    per_hp_seed_train: Dict[str, List[np.ndarray]] = defaultdict(list)
    per_hp_seed_val: Dict[str, List[np.ndarray]] = defaultdict(list)

    for (hp_preset, dataset_seed), items in groups.items():
        train_raw: List[Tuple[List[int], List[float]]] = []
        val_raw: List[Tuple[List[int], List[float]]] = []
        emax = 0

        for r in items:
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

        if train_curves:
            med, _, _ = _median_iqr(train_curves)
            per_hp_seed_train[hp_preset].append(med)
        if val_curves:
            med, _, _ = _median_iqr(val_curves)
            per_hp_seed_val[hp_preset].append(med)

    hp_list = hp_filter or sorted(set(list(per_hp_seed_train.keys()) + list(per_hp_seed_val.keys())))
    if not hp_list:
        print("[warn] no curves to plot")
        return

    color_map = _hp_color_map(hp_list)

    train_by_hp: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    val_by_hp: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for hp in hp_list:
        train_curves = per_hp_seed_train.get(hp, [])
        if train_curves:
            emax_hp = max(c.shape[0] for c in train_curves)
            padded = [_pad_curve(c, emax_hp) for c in train_curves]
            train_by_hp[hp] = _median_iqr(padded)
        val_curves = per_hp_seed_val.get(hp, [])
        if val_curves:
            emax_hp = max(c.shape[0] for c in val_curves)
            padded = [_pad_curve(c, emax_hp) for c in val_curves]
            val_by_hp[hp] = _median_iqr(padded)

    fig = plt.figure(figsize=(10, 4.6), dpi=160)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    for hp in hp_list:
        color = color_map.get(hp, "#1f77b4")
        if hp in train_by_hp:
            med, q25, q75 = train_by_hp[hp]
            xs = np.arange(1, med.shape[0] + 1)
            ax1.plot(xs, med, label=str(hp), color=color, linewidth=1.5)
            ax1.fill_between(xs, q25, q75, color=color, alpha=0.18)
        if hp in val_by_hp:
            med, q25, q75 = val_by_hp[hp]
            xs = np.arange(1, med.shape[0] + 1)
            ax2.plot(xs, med, label=str(hp), color=color, linewidth=1.5)
            ax2.fill_between(xs, q25, q75, color=color, alpha=0.18)

    if train_by_hp:
        min_train = min(float(np.nanmin(v[0])) for v in train_by_hp.values())
        if min_train > 0:
            ax1.set_yscale("log")
    if val_by_hp:
        min_val = min(float(np.nanmin(v[0])) for v in val_by_hp.values())
        if min_val > 0:
            ax2.set_yscale("log")

    fig.suptitle("DeLaN best train/val curves (median ± IQR) by hp_preset")
    ax1.set_title("Train loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.25)

    ax2.set_title("Val MSE")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val MSE")
    ax2.grid(True, alpha=0.25)

    ax1.legend(fontsize=8, ncol=2)
    ax2.legend(fontsize=8, ncol=2)
    fig.tight_layout()

    out_png = os.path.join(args.out_dir, "delan_best_train_val_curves_by_hp.png")
    fig.savefig(out_png, dpi=200)
    print(f"[info] saved {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
