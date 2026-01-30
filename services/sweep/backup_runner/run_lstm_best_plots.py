#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

from sweep_base import (
    EVAL_DIR,
    EVAL_DIR_HOST,
    LOGS_DIR_HOST,
    DATASET_NAME,
    RUN_TAG,
    LSTM_BEST_FEATURE_MODES,
    LSTM_BEST_H_LIST,
    LSTM_BEST_BINS,
    LSTM_BEST_EVAL_SPLIT,
    LSTM_BEST_PLOTS_OUT_DIR,
    LSTM_BEST_MODELS_DIR,
    LSTM_BEST_SCATTER_LEGEND,
    PROCESSED_DIR,
    TEST_FRACTIONS,
    SCRIPT_LSTM_BEST_RESIDUAL_AGG,
    SCRIPT_LSTM_BEST_COMBINED_AGG,
    SCRIPT_LSTM_METRICS_BOXPLOTS,
    SCRIPT_EVAL,
    SVC_EVAL,
    COMPOSE,
    REPO_ROOT,
)
from sweep_helper import banner, run_cmd, compose_exec


def _default_ts() -> str:
    eval_dir = Path(EVAL_DIR_HOST)
    runs = sorted(eval_dir.glob("summary_lstm_best_runs_*.jsonl"))
    if not runs:
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    last = runs[-1].name
    return last.replace("summary_lstm_best_runs_", "").replace(".jsonl", "")


def _summary_paths(ts: str) -> dict:
    return {
        "runs": f"{EVAL_DIR}/summary_lstm_best_runs_{ts}.jsonl",
        "combined": f"{EVAL_DIR}/summary_lstm_best_combined_{ts}.jsonl",
        "configs": f"{EVAL_DIR}/summary_lstm_best_configs_{ts}.jsonl",
        "fh": f"{EVAL_DIR}/summary_lstm_best_fh_{ts}.jsonl",
    }


def _plot_base_dir(ts: str) -> str:
    return f"{LSTM_BEST_PLOTS_OUT_DIR}/{DATASET_NAME}__{RUN_TAG}__{ts}"


def _parse_model_meta(model_path: str) -> tuple[int | None, str]:
    if not model_path:
        return None, ""
    folder = Path(model_path).parent.name
    k_val = None
    m = re.search(r"__K(\\d+)__", folder)
    if m:
        try:
            k_val = int(m.group(1))
        except Exception:
            k_val = None
    delan_tag = ""
    for part in folder.split("__"):
        if part.startswith("delan_"):
            delan_tag = part
            break
    return k_val, delan_tag


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", type=str, default="")
    ap.add_argument("--feature", type=str, default="")
    ap.add_argument("--h_values", type=str, default="")
    ap.add_argument("--skip_boxplots", action="store_true")
    ap.add_argument("--skip_final_eval", action="store_true")
    args = ap.parse_args()

    ts = args.ts.strip() or _default_ts()
    paths = _summary_paths(ts)

    out_dir = _plot_base_dir(ts)
    logs_dir = Path(LOGS_DIR_HOST)
    if not logs_dir.is_absolute():
        logs_dir = Path(REPO_ROOT) / logs_dir
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"lstm_best_plots_{DATASET_NAME}_{RUN_TAG}_{ts}.log"

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(banner(["LSTM best plots", f"ts={ts}", f"out_dir={out_dir}"]) + "\n")

        # Residual + combined plots per feature
        features = [args.feature] if args.feature else list(LSTM_BEST_FEATURE_MODES)
        h_values = args.h_values
        if not h_values and LSTM_BEST_H_LIST:
            h_values = ",".join(str(h) for h in LSTM_BEST_H_LIST)

        for feat in features:
            feat_tag = feat
            feat_out = f"{out_dir}/feat_{feat_tag}"
            cmd = compose_exec(
                SVC_EVAL,
                f"python3 {SCRIPT_LSTM_BEST_RESIDUAL_AGG} "
                f"--summary_jsonl {paths['runs']} "
                f"--out_dir {feat_out}/residual "
                f"--bins {LSTM_BEST_BINS} "
                f"--feature {feat} "
                f"--h_values {h_values}"
            )
            run_cmd(cmd, log_file)

            cmd = compose_exec(
                SVC_EVAL,
                f"python3 {SCRIPT_LSTM_BEST_COMBINED_AGG} "
                f"--summary_jsonl {paths['combined']} "
                f"--out_dir {feat_out}/combined "
                f"--bins {LSTM_BEST_BINS} "
                f"--split {LSTM_BEST_EVAL_SPLIT} "
                f"--feature {feat} "
                f"--h_values {h_values}"
            )
            run_cmd(cmd, log_file)

        # Boxplots (from configs jsonl)
        if not args.skip_boxplots:
            cmd = compose_exec(
                SVC_EVAL,
                f"python3 {SCRIPT_LSTM_METRICS_BOXPLOTS} "
                f"--lstm_root {LSTM_BEST_MODELS_DIR} "
                f"--out_dir {out_dir}/boxplots"
                + ("" if LSTM_BEST_SCATTER_LEGEND else " --no_scatter_legend")
            )
            run_cmd(cmd, log_file)

        # Final best combined eval (optional)
        if not args.skip_final_eval:
            best_json = Path(EVAL_DIR_HOST) / f"lstm_best_model_{ts}.json"
            if best_json.exists():
                try:
                    best = json.loads(best_json.read_text(encoding="utf-8"))
                except Exception:
                    best = {}

                best_feat = best.get("feature_mode")
                best_H = best.get("H")
                best_seed = best.get("lstm_seed")
                best_model = best.get("model_path")
                best_ds = best.get("dataset_seed")

                k_val, delan_tag = _parse_model_meta(str(best_model))
                if best_feat and best_H and best_seed is not None and best_model and k_val and delan_tag:
                    residual_name = f"{DATASET_NAME}__{RUN_TAG}__K{k_val}__residual__{delan_tag}.npz"
                    res_out = f"{PROCESSED_DIR}/{residual_name}"
                    scalers_path = str(best_model).replace("residual_lstm.keras", f"scalers_H{best_H}.npz")
                    best_out = f"{EVAL_DIR}/lstm_best/{DATASET_NAME}__{RUN_TAG}__{ts}/best"

                    cmd = compose_exec(
                        SVC_EVAL,
                        f"python3 {SCRIPT_EVAL} "
                        f"--residual_npz {res_out} "
                        f"--model {best_model} "
                        f"--scalers {scalers_path} "
                        f"--out_dir {best_out} "
                        f"--H {best_H} "
                        f"--split {LSTM_BEST_EVAL_SPLIT} "
                        f"--features {best_feat} "
                        f"--save_pred_npz "
                        f"--K {k_val} "
                        f"--test_fraction {TEST_FRACTIONS} "
                        f"--seed {best_seed} "
                        f"--dataset_seed {best_ds if best_ds is not None else -1}"
                    )
                    run_cmd(cmd, log_file)
                else:
                    log_file.write(banner(["Skip final eval: best model info incomplete"], char="!") + "\n")

        log_file.write(banner(["LSTM best plots finished", f"ts={ts}", f"out_dir={out_dir}"]) + "\n")
        log_file.flush()

    print(banner(["LSTM best plots finished", f"ts={ts}", f"out_dir={out_dir}", f"log={log_path}"]))


if __name__ == "__main__":
    main()
