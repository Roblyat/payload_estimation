#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import List, Tuple

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


def _annotate(ax, xs: List[float], ys: List[float], labels: List[str]) -> None:
    for x, y, label in zip(xs, ys, labels):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)


def _filter_finite(
    labels: List[str],
    xs: List[float],
    ys: List[float],
) -> Tuple[List[str], List[float], List[float]]:
    out_labels: List[str] = []
    out_xs: List[float] = []
    out_ys: List[float] = []
    for l, x, y in zip(labels, xs, ys):
        if np.isfinite(x) and np.isfinite(y):
            out_labels.append(l)
            out_xs.append(float(x))
            out_ys.append(float(y))
    return out_labels, out_xs, out_ys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--run_tag", type=str, default="", help="optional run tag for logging only")
    args = ap.parse_args()

    summary_path = _resolve_shared_path(args.summary_jsonl)
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"summary_jsonl not found: {summary_path}")

    with open(summary_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    print(f"[info] summary_jsonl={summary_path} rows={len(rows)} run_tag={args.run_tag or 'n/a'}")
    print(f"[info] out_dir={args.out_dir} MPLCONFIGDIR={os.environ.get('MPLCONFIGDIR')}")

    labels = [str(r.get("hp_preset")) for r in rows]
    val_rmse_median = [r.get("val_rmse_median") for r in rows]
    val_rmse_iqr_median = [r.get("val_rmse_iqr_median") for r in rows]
    test_rmse_median = [r.get("test_rmse_median") for r in rows]

    labels_a, xs_a, ys_a = _filter_finite(labels, val_rmse_median, val_rmse_iqr_median)
    labels_b, xs_b, ys_b = _filter_finite(labels, val_rmse_median, test_rmse_median)

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[info] points_a={len(xs_a)} points_b={len(xs_b)}")

    if xs_a and ys_a:
        fig = plt.figure(figsize=(7.2, 5.2), dpi=160)
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(xs_a, ys_a, s=55, alpha=0.9)
        _annotate(ax, xs_a, ys_a, labels_a)
        ax.set_title("Hyper-set comparison: accuracy vs seed-stability")
        ax.set_xlabel("Val RMSE median")
        ax.set_ylabel("Val RMSE IQR median (within-fold)")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        out_png = os.path.join(args.out_dir, "scatter_accuracy_vs_stability.png")
        fig.savefig(out_png, dpi=200)
        print(f"[info] saved {out_png}")
        plt.close(fig)

        np.savez(
            os.path.join(args.out_dir, "scatter_accuracy_vs_stability.npz"),
            labels=np.array(labels_a, dtype=object),
            x=np.array(xs_a, dtype=np.float32),
            y=np.array(ys_a, dtype=np.float32),
        )

    if xs_b and ys_b:
        fig = plt.figure(figsize=(7.2, 5.2), dpi=160)
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(xs_b, ys_b, s=55, alpha=0.9)
        _annotate(ax, xs_b, ys_b, labels_b)
        ax.set_title("Hyper-set comparison: validation vs test")
        ax.set_xlabel("Val RMSE median")
        ax.set_ylabel("Test RMSE median")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        out_png = os.path.join(args.out_dir, "scatter_val_vs_test.png")
        fig.savefig(out_png, dpi=200)
        print(f"[info] saved {out_png}")
        plt.close(fig)

        np.savez(
            os.path.join(args.out_dir, "scatter_val_vs_test.npz"),
            labels=np.array(labels_b, dtype=object),
            x=np.array(xs_b, dtype=np.float32),
            y=np.array(ys_b, dtype=np.float32),
        )


if __name__ == "__main__":
    main()
