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
    DELAN_BEST_TORQUE_BINS,
    DELAN_BEST_TORQUE_SPLIT,
    DELAN_BEST_TORQUE_HP_PRESETS,
    DELAN_BEST_PLOTS_OUT_DIR,
    SCRIPT_DELAN_BEST_FOLD_PLOTS,
    SCRIPT_DELAN_BEST_HYPER_SCATTER,
    SCRIPT_DELAN_BEST_TORQUE_AGG,
    SVC_EVAL,
    REPO_ROOT,
)
from sweep_helper import banner, run_cmd, compose_exec


def _default_ts() -> str:
    eval_dir = Path(EVAL_DIR_HOST)
    runs = sorted(eval_dir.glob("summary_delan_best_runs_*.jsonl"))
    if not runs:
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    last = runs[-1].name
    return last.replace("summary_delan_best_runs_", "").replace(".jsonl", "")


def _summary_paths(ts: str, overrides: dict[str, str]) -> dict[str, str]:
    paths = {
        "runs": f"{EVAL_DIR}/summary_delan_best_runs_{ts}.jsonl",
        "folds": f"{EVAL_DIR}/summary_delan_best_folds_{ts}.jsonl",
        "hypers": f"{EVAL_DIR}/summary_delan_best_hypers_{ts}.jsonl",
    }
    paths.update({k: v for k, v in overrides.items() if v})
    return paths


def _plot_base_dir(ts: str) -> str:
    return f"{DELAN_BEST_PLOTS_OUT_DIR}/{DATASET_NAME}__{RUN_TAG}__{ts}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", type=str, default="", help="timestamp; default: latest summary_delan_best_runs_*")
    ap.add_argument("--summary_runs", type=str, default="", help="override runs summary path")
    ap.add_argument("--summary_folds", type=str, default="", help="override folds summary path")
    ap.add_argument("--summary_hypers", type=str, default="", help="override hypers summary path")
    ap.add_argument("--run_tag", type=str, default="", help="override RUN_TAG for outputs/logging")
    ap.add_argument("--skip_folds", action="store_true")
    ap.add_argument("--skip_scatter", action="store_true")
    ap.add_argument("--skip_torque", action="store_true")
    args = ap.parse_args()

    ts = args.ts.strip() or _default_ts()
    paths = _summary_paths(
        ts,
        {
            "runs": args.summary_runs,
            "folds": args.summary_folds,
            "hypers": args.summary_hypers,
        },
    )

    run_tag = args.run_tag.strip() or RUN_TAG
    out_dir = f"{DELAN_BEST_PLOTS_OUT_DIR}/{DATASET_NAME}__{run_tag}__{ts}"
    logs_dir = Path(LOGS_DIR_HOST)
    if not logs_dir.is_absolute():
        logs_dir = Path(REPO_ROOT) / logs_dir
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"delan_best_plots_{DATASET_NAME}_{run_tag}_{ts}.log"

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(banner(["DeLaN best plots", f"ts={ts}", f"run_tag={run_tag}", f"out_dir={out_dir}"]) + "\n")

        if not args.skip_folds:
            cmd = compose_exec(
                SVC_EVAL,
                f"python3 {SCRIPT_DELAN_BEST_FOLD_PLOTS} "
                f"--summary_jsonl {paths['runs']} "
                f"--out_dir {out_dir}/folds"
            )
            run_cmd(cmd, log_file)

        if not args.skip_scatter:
            cmd = compose_exec(
                SVC_EVAL,
                f"python3 {SCRIPT_DELAN_BEST_HYPER_SCATTER} "
                f"--summary_jsonl {paths['hypers']} "
                f"--out_dir {out_dir}/hypers "
                f"--run_tag {run_tag}"
            )
            run_cmd(cmd, log_file)

        if not args.skip_torque:
            hp_presets = ",".join(DELAN_BEST_TORQUE_HP_PRESETS) if DELAN_BEST_TORQUE_HP_PRESETS else ""
            cmd = compose_exec(
                SVC_EVAL,
                f"python3 {SCRIPT_DELAN_BEST_TORQUE_AGG} "
                f"--summary_jsonl {paths['folds']} "
                f"--out_dir {out_dir}/torque "
                f"--bins {DELAN_BEST_TORQUE_BINS} "
                f"--split {DELAN_BEST_TORQUE_SPLIT} "
                + (f"--hp_presets {hp_presets} " if hp_presets else "")
            )
            run_cmd(cmd, log_file)

        log_file.write(banner(["DeLaN best plots finished", f"ts={ts}", f"log={log_path}"]) + "\n")
        log_file.flush()

    print(banner(["DeLaN best plots finished", f"ts={ts}", f"log={log_path}"]))


if __name__ == "__main__":
    main()
