#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from sweep_base import (
    EVAL_DIR,
    EVAL_DIR_HOST,
    LOGS_DIR_HOST,
    DATASET_NAME,
    RUN_TAG,
    FEATURE_MODES,
    LSTM_AGGREGATE_OUT_DIR,
    LSTM_AGGREGATE_BINS,
    LSTM_AGGREGATE_FEATURE,
    LSTM_AGGREGATE_K_VALUES,
    DELAN_TORQUE_OUT_DIR,
    DELAN_TORQUE_BINS,
    DELAN_TORQUE_K_VALUES,
    SCRIPT_DELAN_TORQUE_AGG,
    COMBINED_TORQUE_OUT_DIR,
    COMBINED_TORQUE_BINS,
    COMBINED_TORQUE_FEATURE,
    SCRIPT_LSTM_RESIDUAL_AGG,
    SCRIPT_COMBINED_TORQUE_AGG,
    SVC_EVAL,
    COMPOSE,
    REPO_ROOT,
)
from sweep_helper import banner, compose_exec, run_cmd


def _latest_ts(pattern: str) -> str:
    eval_dir = Path(EVAL_DIR_HOST)
    runs = sorted(eval_dir.glob(pattern))
    if not runs:
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    last = runs[-1].name
    return last.split("_")[-1].replace(".jsonl", "")


def _summary_path(sweep_id: int, ts: str) -> str:
    return f"{EVAL_DIR}/summary_lstm_sweep_{sweep_id}_{ts}.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", type=str, default="", help="timestamp suffix, default: latest summary_lstm_sweep_*")
    ap.add_argument("--sweep_id", type=int, default=1, help="sweep index used in summary filename (default: 1)")
    ap.add_argument("--feature", type=str, default="", help="limit to one feature_mode; default: config/all")
    ap.add_argument("--k_values", type=str, default="", help="optional K list for plots, comma-separated")
    ap.add_argument(
        "--delan_summary",
        type=str,
        default="",
        help="override summary_metrics_sweep_*.jsonl for DeLaN torque aggregate",
    )
    ap.add_argument("--only_combined", action="store_true", help="skip residual agg; run combined + delan torque only")
    ap.add_argument("--skip_residual", action="store_true", help="skip LSTM residual agg")
    ap.add_argument("--skip_combined", action="store_true", help="skip combined torque agg")
    args = ap.parse_args()

    ts = args.ts.strip() or _latest_ts(f"summary_lstm_sweep_{args.sweep_id}_*.jsonl")
    summary = _summary_path(args.sweep_id, ts)
    delan_summary = args.delan_summary.strip() or f"{EVAL_DIR}/summary_delan_sweep_{args.sweep_id}_{ts}.jsonl"

    logs_dir = Path(LOGS_DIR_HOST)
    if not logs_dir.is_absolute():
        logs_dir = Path(REPO_ROOT) / logs_dir
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"kStory_plots_{DATASET_NAME}_{RUN_TAG}_{ts}.log"

    # decide features
    features = [args.feature] if args.feature else []
    if not features:
        if LSTM_AGGREGATE_FEATURE:
            features = [LSTM_AGGREGATE_FEATURE]
        else:
            features = list(FEATURE_MODES)

    # choose K values string
    k_values = args.k_values.strip()
    if not k_values and LSTM_AGGREGATE_K_VALUES:
        k_values = ",".join(str(k) for k in LSTM_AGGREGATE_K_VALUES)

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(banner(["kStory plots", f"ts={ts}", f"sweep_id={args.sweep_id}", f"summary={summary}"]) + "\n")

        do_residual = not args.only_combined and not args.skip_residual
        do_combined = not args.skip_combined

        for feat in features:
            feat_out = f"{LSTM_AGGREGATE_OUT_DIR}/{DATASET_NAME}__{RUN_TAG}__{ts}__feat_{feat}"

            if do_residual:
                cmd = compose_exec(
                    SVC_EVAL,
                    f"python3 {SCRIPT_LSTM_RESIDUAL_AGG} "
                    f"--summary_jsonl {summary} "
                    f"--out_dir {feat_out}/residual "
                    f"--bins {LSTM_AGGREGATE_BINS} "
                    f"--feature {feat} "
                    + (f"--k_values {k_values} " if k_values else "")
                )
                run_cmd(cmd, log_file)

            if do_combined:
                combined_feat = COMBINED_TORQUE_FEATURE or feat
                comb_out = f"{COMBINED_TORQUE_OUT_DIR}/{DATASET_NAME}__{RUN_TAG}__{ts}__feat_{combined_feat}"
                cmd = compose_exec(
                    SVC_EVAL,
                    f"python3 {SCRIPT_COMBINED_TORQUE_AGG} "
                    f"--summary_jsonl {summary} "
                    f"--out_dir {comb_out} "
                    f"--bins {COMBINED_TORQUE_BINS} "
                    f"--feature {combined_feat} "
                    + (f"--k_values {k_values} " if k_values else "")
                )
                run_cmd(cmd, log_file)

        # DeLaN torque aggregate (all Ks by default)
        k_values_d = args.k_values.strip()
        if not k_values_d and DELAN_TORQUE_K_VALUES:
            k_values_d = ",".join(str(k) for k in DELAN_TORQUE_K_VALUES)
        delan_out = f"{DELAN_TORQUE_OUT_DIR}/{DATASET_NAME}__{RUN_TAG}__{ts}"
        cmd = compose_exec(
            SVC_EVAL,
            f"python3 {SCRIPT_DELAN_TORQUE_AGG} "
            f"--summary_jsonl {delan_summary} "
            f"--out_dir {delan_out} "
            f"--bins {DELAN_TORQUE_BINS} "
            + (f"--k_values {k_values_d} " if k_values_d else "")
        )
        run_cmd(cmd, log_file)

        log_file.write(banner(["kStory plots finished", f"ts={ts}", f"log={log_path}"]) + "\n")
        log_file.flush()

    print(banner(["kStory plots finished", f"ts={ts}", f"log={log_path}"]))


if __name__ == "__main__":
    main()
