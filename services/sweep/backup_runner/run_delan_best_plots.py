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


def _summary_paths(ts: str, run_tag: str, overrides: dict[str, str]) -> dict[str, str]:
    """
    Build default summary paths. If run_tag is explicitly provided, prefer the
    tag-only filenames (no timestamp) like summary_delan_best_runs_<run_tag>.jsonl.
    """

    def pick(defaults: list[str], override: str | None) -> str:
        if override:
            return override
        return defaults[0]

    def run_path(prefix: str) -> list[str]:
        if run_tag and not ts:
            # explicit run_tag override: trust tag-only
            return [f"{EVAL_DIR}/{prefix}_{run_tag}.jsonl"]
        if run_tag and ts:
            return [
                f"{EVAL_DIR}/{prefix}_{run_tag}_{ts}.jsonl",
                f"{EVAL_DIR}/{prefix}_{run_tag}.jsonl",
            ]
        if ts:
            return [f"{EVAL_DIR}/{prefix}_{ts}.jsonl"]
        # fallback: return empty path, caller should override
        return [f"{EVAL_DIR}/{prefix}.jsonl"]

    runs_defaults = run_path("summary_delan_best_runs")
    folds_defaults = run_path("summary_delan_best_folds")
    hypers_defaults = run_path("summary_delan_best_hypers")

    return {
        "runs": pick(runs_defaults, overrides.get("runs")),
        "folds": pick(folds_defaults, overrides.get("folds")),
        "hypers": pick(hypers_defaults, overrides.get("hypers")),
    }


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

    ts = args.ts.strip()
    run_tag = args.run_tag.strip()

    # If user provided run_tag, trust it and ignore timestamp (tag-only filenames)
    if run_tag:
        ts = ""
    else:
        ts = ts or _default_ts()
        run_tag = RUN_TAG
    paths = _summary_paths(
        ts,
        run_tag,
        {
            "runs": args.summary_runs,
            "folds": args.summary_folds,
            "hypers": args.summary_hypers,
        },
    )

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
