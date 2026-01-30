#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Optional


def _to_float(value: str, *, field_name: str, line_no: int) -> float:
    try:
        return float(value)
    except ValueError as exc:
        msg = f"Invalid float in field '{field_name}' on line {line_no}: {value!r}"
        raise ValueError(msg) from exc


def _find_col_index(header: list[str], col: str, *, csv_path: Path) -> int:
    try:
        return header.index(col)
    except ValueError as exc:
        raise ValueError(
            f"Missing column '{col}' in {csv_path}. Available: {', '.join(header)}"
        ) from exc


def _trajectory_report(
    *,
    trajectory_id: str,
    start_row_index: int,
    end_row_index: int,
    start_time_s: float,
    end_time_s: float,
    dts_pos: list[float],
    dt_sum: float,
    dt_min: float,
    dt_max: float,
    slow_dt_count: int,
    fast_dt_count: int,
    nonpositive_dt_count: int,
    nonfinite_dt_count: int,
    slow_examples: list[dict[str, Any]],
    fast_examples: list[dict[str, Any]],
    min_allowed_dt_s: float,
    max_allowed_dt_s: float,
    slow_fraction_threshold: float,
    gap_s: float,
) -> tuple[dict[str, Any], list[str]]:
    n_samples = (end_row_index - start_row_index) + 1
    n_dts = len(dts_pos)

    mean_dt_s: Optional[float] = None
    median_dt_s: Optional[float] = None
    if n_dts > 0:
        mean_dt_s = dt_sum / n_dts
        median_dt_s = float(statistics.median(dts_pos))

    report: dict[str, Any] = {
        "trajectory_id": trajectory_id,
        "start_row_index": int(start_row_index),
        "end_row_index": int(end_row_index),
        "start_line_no": int(start_row_index + 2),
        "end_line_no": int(end_row_index + 2),
        "samples": int(n_samples),
        "duration_s": float(end_time_s - start_time_s),
        "dts_positive": int(n_dts),
        "mean_dt_s": mean_dt_s,
        "median_dt_s": median_dt_s,
        "min_dt_s": float(dt_min) if n_dts > 0 else None,
        "max_dt_s": float(dt_max) if n_dts > 0 else None,
        "slow_dt_count": int(slow_dt_count),
        "fast_dt_count": int(fast_dt_count),
        "slow_fraction": (slow_dt_count / n_dts) if n_dts > 0 else None,
        "fast_fraction": (fast_dt_count / n_dts) if n_dts > 0 else None,
        "nonpositive_dt_count": int(nonpositive_dt_count),
        "nonfinite_dt_count": int(nonfinite_dt_count),
        "slow_examples": slow_examples,
        "fast_examples": fast_examples,
    }

    reasons: list[str] = []
    if n_dts == 0 or median_dt_s is None:
        reasons.append("insufficient_positive_dts")
        return report, reasons

    eps = 1e-9  # avoid threshold false-positives due to float rounding
    if median_dt_s > (max_allowed_dt_s + eps):
        reasons.append("median_dt_too_slow")
    if median_dt_s < (min_allowed_dt_s - eps):
        reasons.append("median_dt_too_fast")

    slow_fraction = slow_dt_count / n_dts
    if slow_fraction > slow_fraction_threshold:
        reasons.append("slow_fraction_exceeds_threshold")

    if nonpositive_dt_count > 0:
        reasons.append("nonpositive_dt_present")
    if nonfinite_dt_count > 0:
        reasons.append("nonfinite_dt_present")

    # A large internal gap while ID stays constant is usually suspicious.
    if dt_max >= gap_s:
        reasons.append("internal_gap_ge_gap_s")

    return report, reasons


def check_timespace(
    csv_path: Path,
    *,
    time_col: str,
    traj_col: str,
    expected_rate_hz: float,
    rel_tolerance: float,
    abs_tolerance_s: float,
    gap_s: float,
    slow_fraction_threshold: float,
    max_examples: int,
) -> dict[str, Any]:
    dt_expected = 1.0 / expected_rate_hz
    min_dt = dt_expected * (1.0 - rel_tolerance) - abs_tolerance_s
    max_dt = dt_expected * (1.0 + rel_tolerance) + abs_tolerance_s
    eps = 1e-9

    total_rows = 0
    dt_all_pos: list[float] = []
    dt_all_sum = 0.0
    dt_all_min = math.inf
    dt_all_max = -math.inf

    id_change_gaps: list[dict[str, Any]] = []
    bad_id_change_gaps: list[dict[str, Any]] = []

    trajectories: list[dict[str, Any]] = []
    flagged_trajectories: list[dict[str, Any]] = []

    # Current trajectory accumulators
    cur_id: Optional[str] = None
    cur_start_row: Optional[int] = None
    cur_end_row: Optional[int] = None
    cur_start_time: Optional[float] = None
    cur_end_time: Optional[float] = None

    cur_prev_time: Optional[float] = None
    cur_prev_row: Optional[int] = None

    cur_dts_pos: list[float] = []
    cur_dt_sum = 0.0
    cur_dt_min = math.inf
    cur_dt_max = -math.inf
    cur_slow = 0
    cur_fast = 0
    cur_nonpositive = 0
    cur_nonfinite = 0
    cur_slow_examples: list[dict[str, Any]] = []
    cur_fast_examples: list[dict[str, Any]] = []

    def finalize_current() -> None:
        nonlocal cur_id
        nonlocal cur_start_row, cur_end_row, cur_start_time, cur_end_time
        nonlocal cur_dts_pos, cur_dt_sum, cur_dt_min, cur_dt_max
        nonlocal cur_slow, cur_fast, cur_nonpositive, cur_nonfinite
        nonlocal cur_slow_examples, cur_fast_examples
        nonlocal cur_prev_time, cur_prev_row

        if cur_id is None:
            return
        if (
            cur_start_row is None
            or cur_end_row is None
            or cur_start_time is None
            or cur_end_time is None
        ):
            raise RuntimeError("Internal error: trajectory accumulator incomplete")

        report, reasons = _trajectory_report(
            trajectory_id=cur_id,
            start_row_index=cur_start_row,
            end_row_index=cur_end_row,
            start_time_s=cur_start_time,
            end_time_s=cur_end_time,
            dts_pos=cur_dts_pos,
            dt_sum=cur_dt_sum,
            dt_min=cur_dt_min,
            dt_max=cur_dt_max,
            slow_dt_count=cur_slow,
            fast_dt_count=cur_fast,
            nonpositive_dt_count=cur_nonpositive,
            nonfinite_dt_count=cur_nonfinite,
            slow_examples=cur_slow_examples,
            fast_examples=cur_fast_examples,
            min_allowed_dt_s=min_dt,
            max_allowed_dt_s=max_dt,
            slow_fraction_threshold=slow_fraction_threshold,
            gap_s=gap_s,
        )

        trajectories.append(report)
        if reasons:
            flagged_trajectories.append({"trajectory_id": cur_id, "reasons": reasons, **report})

        # Reset
        cur_id = None
        cur_start_row = None
        cur_end_row = None
        cur_start_time = None
        cur_end_time = None
        cur_prev_time = None
        cur_prev_row = None

        cur_dts_pos = []
        cur_dt_sum = 0.0
        cur_dt_min = math.inf
        cur_dt_max = -math.inf
        cur_slow = 0
        cur_fast = 0
        cur_nonpositive = 0
        cur_nonfinite = 0
        cur_slow_examples = []
        cur_fast_examples = []

    with csv_path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            raise ValueError(f"CSV has no header: {csv_path}")
        time_idx = _find_col_index(header, time_col, csv_path=csv_path)
        traj_idx = _find_col_index(header, traj_col, csv_path=csv_path)

        for row_idx, row in enumerate(reader):
            # line_no: 1 header + 1-based data line index
            line_no = 2 + row_idx
            if time_idx >= len(row) or traj_idx >= len(row):
                raise ValueError(
                    f"Missing field on line {line_no}: row has {len(row)} columns"
                )
            t = _to_float(row[time_idx], field_name=time_col, line_no=line_no)
            traj_raw = row[traj_idx].strip()
            if traj_raw:
                try:
                    traj_id = str(int(float(traj_raw)))
                except ValueError:
                    traj_id = traj_raw
            else:
                traj_id = ""

            if cur_id is None:
                cur_id = traj_id
                cur_start_row = row_idx
                cur_end_row = row_idx
                cur_start_time = t
                cur_end_time = t
                cur_prev_time = t
                cur_prev_row = row_idx
                total_rows += 1
                continue

            # New trajectory boundary (ID changes)
            if traj_id != cur_id:
                if cur_prev_time is not None and cur_prev_row is not None:
                    gap_dt = t - cur_prev_time
                    gap_rec = {
                        "prev_trajectory_id": cur_id,
                        "next_trajectory_id": traj_id,
                        "prev_row_index": int(cur_prev_row),
                        "row_index": int(row_idx),
                        "prev_time_s": float(cur_prev_time),
                        "time_s": float(t),
                        "gap_dt_s": float(gap_dt),
                    }
                    id_change_gaps.append(gap_rec)
                    if gap_dt < (gap_s - eps):
                        bad_id_change_gaps.append(gap_rec)

                finalize_current()

                cur_id = traj_id
                cur_start_row = row_idx
                cur_end_row = row_idx
                cur_start_time = t
                cur_end_time = t
                cur_prev_time = t
                cur_prev_row = row_idx
                total_rows += 1
                continue

            # Same trajectory: compute dt
            if cur_prev_time is not None and cur_prev_row is not None:
                dt = t - cur_prev_time
                if not math.isfinite(dt):
                    cur_nonfinite += 1
                elif dt <= 0:
                    cur_nonpositive += 1
                else:
                    cur_dts_pos.append(dt)
                    cur_dt_sum += dt
                    cur_dt_min = min(cur_dt_min, dt)
                    cur_dt_max = max(cur_dt_max, dt)

                    dt_all_pos.append(dt)
                    dt_all_sum += dt
                    dt_all_min = min(dt_all_min, dt)
                    dt_all_max = max(dt_all_max, dt)

                    too_fast = dt < (min_dt - eps)
                    too_slow = dt > (max_dt + eps)
                    if too_fast:
                        cur_fast += 1
                        if len(cur_fast_examples) < max_examples:
                            cur_fast_examples.append(
                                {
                                    "prev_row_index": int(cur_prev_row),
                                    "row_index": int(row_idx),
                                    "prev_time_s": float(cur_prev_time),
                                    "time_s": float(t),
                                    "dt_s": float(dt),
                                }
                            )
                    if too_slow:
                        cur_slow += 1
                        if len(cur_slow_examples) < max_examples:
                            cur_slow_examples.append(
                                {
                                    "prev_row_index": int(cur_prev_row),
                                    "row_index": int(row_idx),
                                    "prev_time_s": float(cur_prev_time),
                                    "time_s": float(t),
                                    "dt_s": float(dt),
                                }
                            )

            cur_end_row = row_idx
            cur_end_time = t
            cur_prev_time = t
            cur_prev_row = row_idx
            total_rows += 1

    finalize_current()

    dataset_stats: dict[str, Any] = {
        "rows": int(total_rows),
        "trajectories": int(len(trajectories)),
        "expected_rate_hz": float(expected_rate_hz),
        "expected_dt_s": float(dt_expected),
        "rel_tolerance": float(rel_tolerance),
        "abs_tolerance_s": float(abs_tolerance_s),
        "min_allowed_dt_s": float(min_dt),
        "max_allowed_dt_s": float(max_dt),
        "gap_s": float(gap_s),
        "slow_fraction_threshold": float(slow_fraction_threshold),
        "id_change_gaps": int(len(id_change_gaps)),
        "bad_id_change_gaps": int(len(bad_id_change_gaps)),
        "flagged_trajectories": int(len(flagged_trajectories)),
    }

    if dt_all_pos:
        dataset_stats |= {
            "observed_positive_dt_count": int(len(dt_all_pos)),
            "observed_mean_dt_s": float(dt_all_sum / len(dt_all_pos)),
            "observed_mean_rate_hz": float(len(dt_all_pos) / dt_all_sum) if dt_all_sum > 0 else math.inf,
            "observed_median_dt_s": float(statistics.median(dt_all_pos)),
            "observed_min_dt_s": float(dt_all_min),
            "observed_max_dt_s": float(dt_all_max),
        }

    return {
        "csv": str(csv_path),
        "time_col": time_col,
        "traj_col": traj_col,
        "stats": dataset_stats,
        "id_change_gaps": id_change_gaps,
        "bad_id_change_gaps": bad_id_change_gaps,
        "trajectories": trajectories,
        "flagged_trajectories": flagged_trajectories,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze time-step (dt) per trajectory (ID changes define trajectory boundaries) "
            "and write datasetname_timespace.json next to the CSV (by default)."
        )
    )
    parser.add_argument(
        "--csv",
        dest="csv_path",
        type=Path,
        default=Path("/workspace/shared/data/raw/UR3_Load0_cc.csv"),
        help="Path to the raw CSV dataset",
    )
    parser.add_argument(
        "--time-col",
        type=str,
        default="t1",
        help="Name of the time column in the CSV",
    )
    parser.add_argument(
        "--traj-col",
        type=str,
        default="ID",
        help="Name of the trajectory ID column in the CSV (default: ID)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to write the JSON into (default: CSV parent directory)",
    )
    parser.add_argument(
        "--expected-rate-hz",
        type=float,
        default=100.0,
        help="Expected sample rate in Hz (default: 100.0)",
    )
    parser.add_argument(
        "--rel-tolerance",
        type=float,
        default=0.20,
        help="Relative tolerance on dt (default: 0.20 => +/- 20%%)",
    )
    parser.add_argument(
        "--abs-tolerance-s",
        type=float,
        default=0.0,
        help="Absolute tolerance on dt in seconds (default: 0.0)",
    )
    parser.add_argument(
        "--gap-s",
        type=float,
        default=5.0,
        help="Expected minimum gap at trajectory boundaries (default: 5.0s)",
    )
    parser.add_argument(
        "--slow-fraction-threshold",
        type=float,
        default=0.05,
        help="Flag trajectory if slow-dt fraction exceeds this (default: 0.05)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=20,
        help="Max dt example rows stored per trajectory (default: 20)",
    )
    args = parser.parse_args()

    csv_path: Path = args.csv_path
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    report = check_timespace(
        csv_path,
        time_col=args.time_col,
        traj_col=args.traj_col,
        expected_rate_hz=args.expected_rate_hz,
        rel_tolerance=args.rel_tolerance,
        abs_tolerance_s=args.abs_tolerance_s,
        gap_s=args.gap_s,
        slow_fraction_threshold=args.slow_fraction_threshold,
        max_examples=args.max_examples,
    )

    out_dir = args.out_dir if args.out_dir is not None else csv_path.resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{csv_path.stem}_timespace.json"

    with out_path.open("w", encoding="utf-8") as out:
        json.dump(report, out, indent=2)
        out.write("\n")

    flagged = int(report["stats"]["flagged_trajectories"])
    print(f"Wrote report with {flagged} flagged trajectories to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
