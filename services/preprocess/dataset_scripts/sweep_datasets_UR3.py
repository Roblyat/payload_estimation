#!/usr/bin/env python3
"""
Build balanced subsets from UR3_Load0_cc.csv (or user-provided CSV).

Outputs (saved to chosen out dir):
  - <basename>_5x10^4_under.csv
  - <basename>_5x10^4_over.csv
  - <basename>_5x10^3_under.csv
  - <basename>_5x10^3_over.csv
  - (optional, when --include_k_sets) <basename>_K86_dist.csv
  - (optional, when --include_k_sets) <basename>_K86_uniform.csv
  - (optional, when --include_k_sets) <basename>_K36_dist.csv
  - (optional, when --include_k_sets) <basename>_K36_uniform.csv

Also writes a JSON log with per-trajectory stats and selection details.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def resolve_raw_dir() -> Path:
    env = os.environ.get("PAYLOAD_ESTIMATION_RAW_DIR")
    if env:
        return Path(env)
    if os.path.isdir("/workspace/payload_estimation/shared/data/raw"):
        return Path("/workspace/payload_estimation/shared/data/raw")
    if os.path.isdir("/workspace/shared/data/raw"):
        return Path("/workspace/shared/data/raw")
    if os.path.isdir("/payload_estimation/shared/data/raw"):
        return Path("/payload_estimation/shared/data/raw")
    # repo-local fallback
    return Path(__file__).resolve().parents[3] / "shared" / "data" / "raw"


def resolve_input_csv(input_csv: str, raw_dir: Path) -> Path:
    path = Path(input_csv)
    if path.is_absolute():
        return path
    return raw_dir / input_csv


def compute_trajectory_stats(
    df: pd.DataFrame,
    *,
    time_col: str,
    id_col: str,
) -> tuple[list[dict[str, Any]], dict[Any, int], dict[Any, int]]:
    stats: list[dict[str, Any]] = []
    length_by_id: dict[Any, int] = {}
    order_by_id: dict[Any, int] = {}

    next_order = 0
    for traj_id, group in df.groupby(id_col, sort=False):
        times = pd.to_numeric(group[time_col], errors="coerce").to_numpy()
        finite_mask = np.isfinite(times)
        finite_times = times[finite_mask]

        if finite_times.size > 0:
            duration_s = float(finite_times[-1] - finite_times[0])
        else:
            duration_s = float("nan")

        dts = np.diff(times)
        dts_finite = dts[np.isfinite(dts)]
        dts_pos = dts_finite[dts_finite > 0]

        dt_mean = float(np.mean(dts_pos)) if dts_pos.size > 0 else None
        dt_median = float(np.median(dts_pos)) if dts_pos.size > 0 else None
        dt_min = float(np.min(dts_pos)) if dts_pos.size > 0 else None
        dt_max = float(np.max(dts_pos)) if dts_pos.size > 0 else None
        dt_std = float(np.std(dts_pos)) if dts_pos.size > 0 else None

        dt_nonpositive = int(np.sum((np.isfinite(dts)) & (dts <= 0)))
        dt_nonfinite = int(np.sum(~np.isfinite(dts)))

        samples = int(len(group))

        stats.append(
            {
                "trajectory_id": traj_id,
                "samples": samples,
                "duration_s": duration_s,
                "dt_mean_s": dt_mean,
                "dt_median_s": dt_median,
                "dt_min_s": dt_min,
                "dt_max_s": dt_max,
                "dt_std_s": dt_std,
                "dt_nonpositive": dt_nonpositive,
                "dt_nonfinite": dt_nonfinite,
            }
        )

        length_by_id[traj_id] = samples
        order_by_id[traj_id] = next_order
        next_order += 1

    return stats, length_by_id, order_by_id


def compute_overall_dt_stats(df: pd.DataFrame, *, time_col: str) -> dict[str, Any]:
    times = pd.to_numeric(df[time_col], errors="coerce").to_numpy()
    dts = np.diff(times)
    dts_finite = dts[np.isfinite(dts)]
    dts_pos = dts_finite[dts_finite > 0]

    return {
        "dt_mean_s": float(np.mean(dts_pos)) if dts_pos.size > 0 else None,
        "dt_median_s": float(np.median(dts_pos)) if dts_pos.size > 0 else None,
        "dt_min_s": float(np.min(dts_pos)) if dts_pos.size > 0 else None,
        "dt_max_s": float(np.max(dts_pos)) if dts_pos.size > 0 else None,
        "dt_std_s": float(np.std(dts_pos)) if dts_pos.size > 0 else None,
        "dt_nonpositive": int(np.sum((np.isfinite(dts)) & (dts <= 0))),
        "dt_nonfinite": int(np.sum(~np.isfinite(dts))),
        "dt_count_positive": int(dts_pos.size),
        "dt_count_total": int(dts.size),
    }


def build_length_bins(
    length_by_id: dict[Any, int],
    *,
    bins: int,
) -> tuple[list[list[Any]], list[float], dict[Any, int]]:
    lengths = np.array(list(length_by_id.values()), dtype=float)
    unique_lengths = np.unique(lengths)
    if unique_lengths.size < bins:
        bins = int(unique_lengths.size)

    if bins <= 1:
        edges = [float(np.min(lengths)), float(np.max(lengths))]
    else:
        quantiles = np.linspace(0.0, 1.0, bins + 1)
        edges = [float(x) for x in np.quantile(lengths, quantiles)]
        # Ensure strictly increasing edges for digitize.
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1.0

    bins_list: list[list[Any]] = [[] for _ in range(bins)]
    bin_index_by_id: dict[Any, int] = {}
    for traj_id, length in length_by_id.items():
        idx = int(np.digitize(length, edges[1:-1], right=True))
        idx = max(0, min(idx, bins - 1))
        bins_list[idx].append(traj_id)
        bin_index_by_id[traj_id] = idx

    return bins_list, edges, bin_index_by_id


def _best_candidate(
    ids: Iterable[Any],
    length_by_id: dict[Any, int],
    *,
    total: int,
    target: int,
    max_total: int,
) -> Any | None:
    best_id = None
    best_score = math.inf
    for traj_id in ids:
        length = length_by_id[traj_id]
        candidate_total = total + length
        if candidate_total > max_total:
            continue
        score = abs(candidate_total - target)
        if score < best_score:
            best_score = score
            best_id = traj_id
    return best_id


def _global_best_candidate(
    remaining_ids: Iterable[Any],
    length_by_id: dict[Any, int],
    *,
    total: int,
    target: int,
    max_total: int,
) -> Any | None:
    return _best_candidate(remaining_ids, length_by_id, total=total, target=target, max_total=max_total)


def select_balanced_sample_target(
    bins_list: list[list[Any]],
    length_by_id: dict[Any, int],
    *,
    target: int,
    min_total: int,
    max_total: int,
) -> tuple[list[Any], int, dict[int, int]]:
    remaining_by_bin = [list(ids) for ids in bins_list]
    selected: list[Any] = []
    selected_counts = {idx: 0 for idx in range(len(bins_list))}
    total = 0

    while True:
        made_pick = False
        bins_order = sorted(selected_counts, key=lambda k: (selected_counts[k], k))
        for bin_idx in bins_order:
            ids = remaining_by_bin[bin_idx]
            if not ids:
                continue
            best_id = _best_candidate(
                ids, length_by_id, total=total, target=target, max_total=max_total
            )
            if best_id is None:
                continue
            remaining_by_bin[bin_idx].remove(best_id)
            selected.append(best_id)
            selected_counts[bin_idx] += 1
            total += int(length_by_id[best_id])
            made_pick = True
        if not made_pick:
            break

    # If we are below minimum required, add best-fit globally.
    remaining_ids = [traj_id for ids in remaining_by_bin for traj_id in ids]
    while total < min_total:
        best_id = _global_best_candidate(
            remaining_ids, length_by_id, total=total, target=target, max_total=max_total
        )
        if best_id is None:
            break
        remaining_ids.remove(best_id)
        bin_idx = None
        for idx, ids in enumerate(remaining_by_bin):
            if best_id in ids:
                ids.remove(best_id)
                bin_idx = idx
                break
        selected.append(best_id)
        total += int(length_by_id[best_id])
        if bin_idx is not None:
            selected_counts[bin_idx] += 1

    return selected, total, selected_counts


def _compute_desired_counts_dist(counts_available: list[int], total: int) -> list[int]:
    total_available = sum(counts_available)
    if total_available == 0:
        return [0] * len(counts_available)

    raw = [total * (count / total_available) for count in counts_available]
    desired = [int(math.floor(x)) for x in raw]
    remainder = total - sum(desired)

    if remainder > 0:
        frac = [x - math.floor(x) for x in raw]
        order = sorted(range(len(frac)), key=lambda i: frac[i], reverse=True)
        for idx in order:
            if remainder <= 0:
                break
            if desired[idx] < counts_available[idx]:
                desired[idx] += 1
                remainder -= 1

    return desired


def _compute_desired_counts_uniform(counts_available: list[int], total: int) -> list[int]:
    bins = len(counts_available)
    desired = [0] * bins
    if bins == 0:
        return desired

    base = total // bins
    remaining = total
    for i in range(bins):
        take = min(base, counts_available[i])
        desired[i] = take
        remaining -= take

    while remaining > 0:
        capacities = [counts_available[i] - desired[i] for i in range(bins)]
        if max(capacities) <= 0:
            break
        idx = int(np.argmax(capacities))
        desired[idx] += 1
        remaining -= 1

    return desired


def _select_evenly(ids: list[Any], length_by_id: dict[Any, int], count: int) -> list[Any]:
    if count <= 0:
        return []
    if count >= len(ids):
        return list(ids)

    ids_sorted = sorted(ids, key=lambda x: length_by_id[x])
    step = len(ids_sorted) / count
    indices = []
    used = set()
    for i in range(count):
        idx = int(i * step + step / 2)
        idx = max(0, min(idx, len(ids_sorted) - 1))
        while idx in used and idx + 1 < len(ids_sorted):
            idx += 1
        while idx in used and idx - 1 >= 0:
            idx -= 1
        used.add(idx)
        indices.append(idx)

    indices = sorted(set(indices))
    selected = [ids_sorted[i] for i in indices]
    if len(selected) < count:
        remaining = [x for x in ids_sorted if x not in selected]
        selected.extend(remaining[: count - len(selected)])
    return selected[:count]


def select_trajectory_count(
    bins_list: list[list[Any]],
    length_by_id: dict[Any, int],
    *,
    count: int,
    mode: str,
) -> tuple[list[Any], dict[int, int]]:
    counts_available = [len(ids) for ids in bins_list]
    if mode == "dist":
        desired = _compute_desired_counts_dist(counts_available, count)
    elif mode == "uniform":
        desired = _compute_desired_counts_uniform(counts_available, count)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    selected: list[Any] = []
    selected_counts: dict[int, int] = {idx: 0 for idx in range(len(bins_list))}

    remaining_by_bin = [list(ids) for ids in bins_list]
    for bin_idx, want in enumerate(desired):
        ids = remaining_by_bin[bin_idx]
        take = min(want, len(ids))
        chosen = _select_evenly(ids, length_by_id, take)
        for traj_id in chosen:
            ids.remove(traj_id)
        selected.extend(chosen)
        selected_counts[bin_idx] = len(chosen)

    # Fill leftovers if not enough due to availability.
    if len(selected) < count:
        remaining = [traj_id for ids in remaining_by_bin for traj_id in ids]
        remaining_sorted = sorted(remaining, key=lambda x: length_by_id[x], reverse=True)
        needed = count - len(selected)
        selected.extend(remaining_sorted[:needed])
        for traj_id in remaining_sorted[:needed]:
            bin_idx = None
            for idx, ids in enumerate(remaining_by_bin):
                if traj_id in ids:
                    ids.remove(traj_id)
                    bin_idx = idx
                    break
            if bin_idx is not None:
                selected_counts[bin_idx] += 1

    return selected, selected_counts


def save_dataset(
    df: pd.DataFrame,
    *,
    id_col: str,
    selected_ids: list[Any],
    out_path: Path,
) -> dict[str, Any]:
    selected_set = set(selected_ids)
    df_out = df[df[id_col].isin(selected_set)]
    df_out.to_csv(out_path, index=False)
    return {
        "path": str(out_path),
        "rows": int(len(df_out)),
        "trajectories": int(len(selected_set)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_csv",
        type=str,
        default="UR3_Load0_cc.csv",
        help="Input CSV filename or absolute path.",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Directory for outputs/log (default: input_csv parent).",
    )
    ap.add_argument(
        "--raw_dir",
        type=str,
        default=None,
        help="Override auto-detected raw directory (used to resolve relative input_csv/out_dir).",
    )
    ap.add_argument(
        "--basename",
        type=str,
        default="UR3_Load0",
        help="Base name for output datasets.",
    )
    ap.add_argument("--time_col", type=str, default="t1")
    ap.add_argument("--id_col", type=str, default="ID")
    ap.add_argument("--bins", type=int, default=5, help="Number of length bins.")
    ap.add_argument(
        "--tolerance_frac",
        type=float,
        default=0.05,
        help="Allowed deviation (fraction) for sample-target datasets.",
    )
    ap.add_argument(
        "--include_k_sets",
        action="store_true",
        help="Also generate K86/K36 trajectory-count datasets.",
    )
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir) if args.raw_dir else resolve_raw_dir()
    input_csv = resolve_input_csv(args.input_csv, raw_dir)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    out_dir = Path(args.out_dir) if args.out_dir else input_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    if args.time_col not in df.columns or args.id_col not in df.columns:
        raise ValueError(
            f"Missing required columns. Need '{args.time_col}' and '{args.id_col}'."
        )

    traj_stats, length_by_id, order_by_id = compute_trajectory_stats(
        df, time_col=args.time_col, id_col=args.id_col
    )
    overall_dt_stats = compute_overall_dt_stats(df, time_col=args.time_col)

    bins_list, bin_edges, bin_index_by_id = build_length_bins(
        length_by_id, bins=args.bins
    )

    # Sample-target datasets (under/over)
    sample_targets = [50_000, 5_000]
    sample_outputs: list[dict[str, Any]] = []
    for target in sample_targets:
        min_total = int(math.floor(target * (1.0 - args.tolerance_frac)))
        max_total = int(math.ceil(target * (1.0 + args.tolerance_frac)))

        under_ids, under_total, under_bins = select_balanced_sample_target(
            bins_list,
            length_by_id,
            target=target,
            min_total=min_total,
            max_total=target,
        )
        over_ids, over_total, over_bins = select_balanced_sample_target(
            bins_list,
            length_by_id,
            target=target,
            min_total=target,
            max_total=max_total,
        )

        base_tag = "5x10^4" if target == 50_000 else "5x10^3"
        under_path = out_dir / f"{args.basename}_{base_tag}_under.csv"
        over_path = out_dir / f"{args.basename}_{base_tag}_over.csv"

        under_ids = sorted(under_ids, key=lambda tid: order_by_id[tid])
        over_ids = sorted(over_ids, key=lambda tid: order_by_id[tid])

        under_info = save_dataset(
            df, id_col=args.id_col, selected_ids=under_ids, out_path=under_path
        )
        over_info = save_dataset(
            df, id_col=args.id_col, selected_ids=over_ids, out_path=over_path
        )

        sample_outputs.append(
            {
                "name": f"{args.basename}_{base_tag}_under",
                "target_samples": target,
                "mode": "under",
                "tolerance_frac": args.tolerance_frac,
                "selected_bins": under_bins,
                "selected_ids": [str(x) for x in under_ids],
                **under_info,
            }
        )
        sample_outputs.append(
            {
                "name": f"{args.basename}_{base_tag}_over",
                "target_samples": target,
                "mode": "over",
                "tolerance_frac": args.tolerance_frac,
                "selected_bins": over_bins,
                "selected_ids": [str(x) for x in over_ids],
                **over_info,
            }
        )

        print(
            f"[{args.basename}_{base_tag}] under={under_total:,} over={over_total:,} "
            f"(range {min_total:,}..{max_total:,})"
        )

    k_outputs: list[dict[str, Any]] = []
    if args.include_k_sets:
        # Trajectory-count datasets (dist/uniform)
        k_targets = [86, 36]
        for k in k_targets:
            for mode in ("dist", "uniform"):
                ids, bins_sel = select_trajectory_count(
                    bins_list, length_by_id, count=k, mode=mode
                )
                ids = sorted(ids, key=lambda tid: order_by_id[tid])
                out_path = out_dir / f"{args.basename}_K{k}_{mode}.csv"
                info = save_dataset(df, id_col=args.id_col, selected_ids=ids, out_path=out_path)
                k_outputs.append(
                    {
                        "name": f"{args.basename}_K{k}_{mode}",
                        "target_trajectories": k,
                        "mode": mode,
                        "selected_bins": bins_sel,
                        "selected_ids": [str(x) for x in ids],
                        **info,
                    }
                )
                print(
                    f"[{args.basename}_K{k}_{mode}] trajectories={len(ids):,} rows={info['rows']:,}"
                )

    # Log JSON
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = out_dir / f"sweep_{args.basename}_{stamp}.json"

    log_payload = {
        "input_csv": str(input_csv),
        "basename": args.basename,
        "time_col": args.time_col,
        "id_col": args.id_col,
        "total_rows": int(len(df)),
        "total_trajectories": int(df[args.id_col].nunique()),
        "bins": {
            "count": len(bins_list),
            "edges": bin_edges,
            "counts": [len(ids) for ids in bins_list],
        },
        "overall_dt_stats": overall_dt_stats,
        "trajectory_stats": [
            {
                **item,
                "trajectory_id": str(item["trajectory_id"]),
                "length_bin": int(bin_index_by_id[item["trajectory_id"]]),
            }
            for item in traj_stats
        ],
        "datasets": sample_outputs + k_outputs,
    }

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_payload, f, indent=2)

    print("\n=== Summary ===")
    print(f"Input: {input_csv}")
    print(f"Rows: {len(df):,}  Trajectories: {df[args.id_col].nunique():,}")
    print(
        "Overall dt stats: "
        f"median={overall_dt_stats['dt_median_s']} "
        f"mean={overall_dt_stats['dt_mean_s']} "
        f"min={overall_dt_stats['dt_min_s']} "
        f"max={overall_dt_stats['dt_max_s']}"
    )
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
