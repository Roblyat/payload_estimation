from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from sweep_base import (
    RAW_DIR,
    PREPROCESSED_DIR,
    EVAL_DIR,
    EVAL_DIR_HOST,
    MODELS_DELAN_DIR_HOST,
    LOGS_DIR_HOST,
    DATASET_NAME,
    RUN_TAG,
    IN_FORMAT,
    COL_FORMAT,
    DERIVE_QDD,
    LOWPASS_SIGNALS,
    LOWPASS_CUTOFF_HZ,
    LOWPASS_ORDER,
    LOWPASS_QDD_VALUES,
    TEST_FRACTIONS,
    VAL_FRACTION,
    DELAN_MODEL_TYPE,
    DELAN_HP_FLAGS,
    DELAN_EPOCHS,
    DELAN_SEEDS,
    DELAN_BEST_K_MAX,
    DELAN_BEST_DATASET_SEEDS,
    DELAN_BEST_HP_PRESETS,
    DELAN_BEST_SCORE_LAMBDA,
    DELAN_BEST_SCORE_PENALTY,
    DELAN_BEST_FOLD_PLOTS,
    DELAN_BEST_HP_CURVES,
    DELAN_BEST_SCATTER_PLOTS,
    DELAN_BEST_TORQUE_AGGREGATE,
    DELAN_BEST_TORQUE_BINS,
    DELAN_BEST_TORQUE_SPLIT,
    DELAN_BEST_TORQUE_HP_PRESETS,
    DELAN_BEST_PLOTS_OUT_DIR,
    SCRIPT_DELAN_BEST_FOLD_PLOTS,
    SCRIPT_DELAN_BEST_HP_CURVES,
    SCRIPT_DELAN_BEST_HYPER_SCATTER,
    SCRIPT_DELAN_BEST_TORQUE_AGG,
    SVC_EVAL,
)
from sweep_helper import (
    banner,
    run_cmd,
    compose_exec,
    append_csv_row,
    append_jsonl,
    safe_tag,
    median_iqr_scalar,
)
from preprocess.sweep_preprocess_loop import comp_prep
from delan.sweep_best_delan_loop import run_delan_seeds, select_best_candidate
from delan.sweep_delan_helper import copy_candidate_metrics_to_best, cleanup_non_best_plots


def _best_id(ts: str) -> str:
    return f"delan_best_{DATASET_NAME}_{RUN_TAG}_{ts}"


def main() -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_name = _best_id(ts)
    logs_dir = Path(LOGS_DIR_HOST) / sweep_name
    logs_dir.mkdir(parents=True, exist_ok=True)

    raw_csv = f"{RAW_DIR}/{DATASET_NAME}.{IN_FORMAT}"

    master_log_path = logs_dir / f"{sweep_name}.log"

    runs_csv = str(Path(EVAL_DIR_HOST) / f"summary_delan_best_runs_{ts}.csv")
    runs_jsonl = str(Path(EVAL_DIR_HOST) / f"summary_delan_best_runs_{ts}.jsonl")
    folds_csv = str(Path(EVAL_DIR_HOST) / f"summary_delan_best_folds_{ts}.csv")
    folds_jsonl = str(Path(EVAL_DIR_HOST) / f"summary_delan_best_folds_{ts}.jsonl")
    hypers_csv = str(Path(EVAL_DIR_HOST) / f"summary_delan_best_hypers_{ts}.csv")
    hypers_jsonl = str(Path(EVAL_DIR_HOST) / f"summary_delan_best_hypers_{ts}.jsonl")
    best_model_json = Path(EVAL_DIR_HOST) / f"delan_best_model_{ts}.json"
    # run-tagged stable copies
    best_model_json_tag = Path(EVAL_DIR_HOST) / f"delan_best_model_{RUN_TAG}.json"
    hypers_jsonl_tag = Path(EVAL_DIR_HOST) / f"summary_{RUN_TAG}_delan_best_hypers.jsonl"
    folds_jsonl_tag = Path(EVAL_DIR_HOST) / f"summary_{RUN_TAG}_delan_best_folds.jsonl"

    with master_log_path.open("w", encoding="utf-8") as master_log:
        master_log.write(banner([
            f"DeLaN best-model sweep started: {ts}",
            f"sweep_name={sweep_name}",
            f"logs_dir={logs_dir}",
            f"dataset={DATASET_NAME} run_tag={RUN_TAG}",
            f"raw_csv={raw_csv}",
            f"col_format={COL_FORMAT} derive_qdd={DERIVE_QDD}",
            f"K_max={DELAN_BEST_K_MAX} test_fraction={TEST_FRACTIONS} val_fraction={VAL_FRACTION}",
            f"dataset_seeds={DELAN_BEST_DATASET_SEEDS}",
            f"delan_seeds={DELAN_SEEDS}",
            f"hp_presets={DELAN_BEST_HP_PRESETS}",
            f"score_lambda={DELAN_BEST_SCORE_LAMBDA} score_penalty={DELAN_BEST_SCORE_PENALTY}",
            f"DeLaN: backend=jax type={DELAN_MODEL_TYPE} epochs={DELAN_EPOCHS} hp_flags={DELAN_HP_FLAGS}",
            f"runs_csv={runs_csv}",
            f"runs_jsonl={runs_jsonl}",
            f"folds_csv={folds_csv}",
            f"folds_jsonl={folds_jsonl}",
            f"hypers_csv={hypers_csv}",
            f"hypers_jsonl={hypers_jsonl}",
            f"best_model_json={best_model_json}",
        ], char="#") + "\n")

        run_fields = [
            "timestamp",
            "dataset",
            "run_tag",
            "K",
            "test_fraction",
            "val_fraction",
            "dataset_seed",
            "delan_seed",
            "delan_epochs",
            "hp_preset",
            "hp_flags",
            "model_type",
            "ckpt",
            "metrics_json",
            "metrics_json_container",
            "metrics_exists",
            "val_rmse",
            "val_mse",
            "test_rmse",
            "test_mse",
            "best_epoch",
            "diverged",
            "selected",
        ]

        fold_fields = [
            "timestamp",
            "dataset",
            "run_tag",
            "K",
            "test_fraction",
            "val_fraction",
            "dataset_seed",
            "hp_preset",
            "delan_epochs",
            "hp_flags",
            "model_type",
            "val_rmse_median",
            "val_rmse_iqr",
            "test_rmse_median",
            "test_rmse_iqr",
            "divergence_rate",
            "score",
            "best_delan_seed",
            "best_ckpt",
            "best_metrics_json",
        ]

        hyper_fields = [
            "timestamp",
            "dataset",
            "run_tag",
            "K",
            "test_fraction",
            "val_fraction",
            "hp_preset",
            "score_median",
            "score_iqr",
            "val_rmse_median",
            "val_rmse_iqr",
            "val_rmse_iqr_median",
            "test_rmse_median",
            "test_rmse_iqr",
            "divergence_rate_median",
            "best_dataset_seed",
            "best_delan_seed",
            "best_ckpt",
        ]

        all_fold_rows: list[dict] = []
        all_hyper_rows: list[dict] = []

        K = int(DELAN_BEST_K_MAX)

        for hp_preset in DELAN_BEST_HP_PRESETS:
            fold_rows: list[dict] = []

            for dataset_seed in DELAN_BEST_DATASET_SEEDS:
                npz_name = f"delan_{DATASET_NAME}_K{K}_seed{dataset_seed}_dataset.npz"
                npz_stem = Path(npz_name).stem
                npz_in = f"{PREPROCESSED_DIR}/{npz_stem}/{npz_stem}.npz"

                run_log_path = logs_dir / f"run_best_{safe_tag(hp_preset)}_d{dataset_seed}_{ts}.log"
                with run_log_path.open("w", encoding="utf-8") as run_log:
                    run_log.write(banner([
                        f"RUN: hp_preset={hp_preset} K={K} test_fraction={TEST_FRACTIONS} val_fraction={VAL_FRACTION}",
                        f"dataset_seed={dataset_seed}",
                        f"npz_in={npz_in}",
                    ], char="#") + "\n")

                    comp_prep(
                        npz_in=npz_in,
                        K=K,
                        tf=TEST_FRACTIONS,
                        seed=dataset_seed,
                        lowpass_qdd=LOWPASS_QDD_VALUES,
                        raw_csv=raw_csv,
                        log_file=run_log,
                    )

                    candidates = run_delan_seeds(
                        npz_in=npz_in,
                        K=K,
                        dataset_seed=dataset_seed,
                        hp_preset=hp_preset,
                        log_file=run_log,
                    )

                    best = select_best_candidate(candidates, run_log)

                    for cand in candidates:
                        row = {
                            "timestamp": ts,
                            "dataset": DATASET_NAME,
                            "run_tag": RUN_TAG,
                            "K": K,
                            "test_fraction": TEST_FRACTIONS,
                            "val_fraction": VAL_FRACTION,
                            "dataset_seed": dataset_seed,
                            "delan_seed": cand["delan_seed"],
                            "delan_epochs": DELAN_EPOCHS,
                            "hp_preset": hp_preset,
                            "hp_flags": DELAN_HP_FLAGS,
                            "model_type": DELAN_MODEL_TYPE,
                            "ckpt": cand["ckpt"],
                            "metrics_json": cand["metrics_json"],
                            "metrics_json_container": cand["metrics_json_container"],
                            "metrics_exists": bool(cand.get("metrics_exists")),
                            "val_rmse": cand.get("val_rmse"),
                            "val_mse": cand.get("val_mse"),
                            "test_rmse": cand.get("test_rmse"),
                            "test_mse": cand.get("test_mse"),
                            "best_epoch": cand.get("best_epoch"),
                            "diverged": bool(cand.get("diverged")),
                            "selected": bool(best and cand["delan_seed"] == best["delan_seed"]),
                        }
                        append_csv_row(runs_csv, run_fields, row)
                        append_jsonl(runs_jsonl, row)

                    if best is not None:
                        best_dir_host = f"{MODELS_DELAN_DIR_HOST}/{best['delan_id']}"
                        copy_candidate_metrics_to_best(best_dir_host, candidates)
                        cleanup_non_best_plots(best["delan_id"], candidates)

                    non_diverged = [c for c in candidates if not c.get("diverged")]
                    val_rmse_vals = [c.get("val_rmse") for c in non_diverged if c.get("val_rmse") is not None]
                    test_rmse_vals = [c.get("test_rmse") for c in non_diverged if c.get("test_rmse") is not None]

                    val_rmse_median, val_rmse_iqr = median_iqr_scalar(val_rmse_vals)
                    test_rmse_median, test_rmse_iqr = median_iqr_scalar(test_rmse_vals)

                    total_runs = len(candidates)
                    divergence_rate = 1.0
                    if total_runs > 0:
                        divergence_rate = (total_runs - len(non_diverged)) / float(total_runs)

                    score = (
                        float(val_rmse_median)
                        + float(DELAN_BEST_SCORE_LAMBDA) * float(val_rmse_iqr)
                        + float(DELAN_BEST_SCORE_PENALTY) * float(divergence_rate)
                    )

                    fold_row = {
                        "timestamp": ts,
                        "dataset": DATASET_NAME,
                        "run_tag": RUN_TAG,
                        "K": K,
                        "test_fraction": TEST_FRACTIONS,
                        "val_fraction": VAL_FRACTION,
                        "dataset_seed": dataset_seed,
                        "hp_preset": hp_preset,
                        "delan_epochs": DELAN_EPOCHS,
                        "hp_flags": DELAN_HP_FLAGS,
                        "model_type": DELAN_MODEL_TYPE,
                        "val_rmse_median": val_rmse_median,
                        "val_rmse_iqr": val_rmse_iqr,
                        "test_rmse_median": test_rmse_median,
                        "test_rmse_iqr": test_rmse_iqr,
                        "divergence_rate": divergence_rate,
                        "score": score,
                        "best_delan_seed": best["delan_seed"] if best else None,
                        "best_ckpt": best["ckpt"] if best else None,
                        "best_metrics_json": best["metrics_json"] if best else None,
                    }
                    append_csv_row(folds_csv, fold_fields, fold_row)
                    append_jsonl(folds_jsonl, fold_row)
                    fold_rows.append(fold_row)
                    all_fold_rows.append(fold_row)

            score_vals = [r["score"] for r in fold_rows if r.get("score") is not None]
            val_rmse_meds = [r["val_rmse_median"] for r in fold_rows if r.get("val_rmse_median") is not None]
            test_rmse_meds = [r["test_rmse_median"] for r in fold_rows if r.get("test_rmse_median") is not None]
            val_rmse_iqrs = [r["val_rmse_iqr"] for r in fold_rows if r.get("val_rmse_iqr") is not None]
            divergence_rates = [r["divergence_rate"] for r in fold_rows if r.get("divergence_rate") is not None]

            score_median, score_iqr = median_iqr_scalar(score_vals)
            val_rmse_median, val_rmse_iqr = median_iqr_scalar(val_rmse_meds)
            test_rmse_median, test_rmse_iqr = median_iqr_scalar(test_rmse_meds)
            val_rmse_iqr_median, _ = median_iqr_scalar(val_rmse_iqrs)
            divergence_rate_median, _ = median_iqr_scalar(divergence_rates)

            best_fold = None
            if fold_rows:
                best_fold = min(fold_rows, key=lambda r: r["score"])

            hyper_row = {
                "timestamp": ts,
                "dataset": DATASET_NAME,
                "run_tag": RUN_TAG,
                "K": K,
                "test_fraction": TEST_FRACTIONS,
                "val_fraction": VAL_FRACTION,
                "hp_preset": hp_preset,
                "score_median": score_median,
                "score_iqr": score_iqr,
                "val_rmse_median": val_rmse_median,
                "val_rmse_iqr": val_rmse_iqr,
                "val_rmse_iqr_median": val_rmse_iqr_median,
                "test_rmse_median": test_rmse_median,
                "test_rmse_iqr": test_rmse_iqr,
                "divergence_rate_median": divergence_rate_median,
                "best_dataset_seed": best_fold["dataset_seed"] if best_fold else None,
                "best_delan_seed": best_fold["best_delan_seed"] if best_fold else None,
                "best_ckpt": best_fold["best_ckpt"] if best_fold else None,
            }
            append_csv_row(hypers_csv, hyper_fields, hyper_row)
            append_jsonl(hypers_jsonl, hyper_row)
            all_hyper_rows.append(hyper_row)

        best_hyper = None
        if all_hyper_rows:
            best_hyper = min(all_hyper_rows, key=lambda r: r["score_median"])

        if best_hyper:
            best_hp = best_hyper["hp_preset"]
            best_fold = None
            if all_fold_rows:
                candidate_folds = [r for r in all_fold_rows if r["hp_preset"] == best_hp]
                if candidate_folds:
                    best_fold = min(candidate_folds, key=lambda r: r["score"])

            best_manifest = {
                "timestamp": ts,
                "dataset": DATASET_NAME,
                "run_tag": RUN_TAG,
                "K": K,
                "test_fraction": TEST_FRACTIONS,
                "val_fraction": VAL_FRACTION,
                "hp_preset": best_hp,
                "dataset_seed": best_fold["dataset_seed"] if best_fold else None,
                "delan_seed": best_fold["best_delan_seed"] if best_fold else None,
                "ckpt": best_fold["best_ckpt"] if best_fold else None,
                "metrics_json": best_fold["best_metrics_json"] if best_fold else None,
                "score": best_fold["score"] if best_fold else None,
                "val_rmse_median": best_fold["val_rmse_median"] if best_fold else None,
                "test_rmse_median": best_fold["test_rmse_median"] if best_fold else None,
            }
            best_model_json.parent.mkdir(parents=True, exist_ok=True)
            with best_model_json.open("w", encoding="utf-8") as f:
                json.dump(best_manifest, f, indent=2)
            # also write run-tagged stable copies
            with best_model_json_tag.open("w", encoding="utf-8") as f:
                json.dump(best_manifest, f, indent=2)
            # copy hypers/folds to run-tagged stable copies
            try:
                Path(hypers_jsonl).replace(hypers_jsonl_tag)
            except Exception:
                import shutil
                shutil.copyfile(hypers_jsonl, hypers_jsonl_tag)
            try:
                Path(folds_jsonl).replace(folds_jsonl_tag)
            except Exception:
                import shutil
                shutil.copyfile(folds_jsonl, folds_jsonl_tag)

        if DELAN_BEST_FOLD_PLOTS:
            out_dir = f"{DELAN_BEST_PLOTS_OUT_DIR}/{DATASET_NAME}__{RUN_TAG}__{ts}/folds"
            summary_jsonl_container = f"{EVAL_DIR}/summary_delan_best_runs_{ts}.jsonl"
            master_log.write("\n" + banner(["DeLaN best fold plots"], char="#") + "\n")
            cmd = compose_exec(
                SVC_EVAL,
                f"python3 {SCRIPT_DELAN_BEST_FOLD_PLOTS} "
                f"--summary_jsonl {summary_jsonl_container} "
                f"--out_dir {out_dir}"
            )
            run_cmd(cmd, master_log)

        if DELAN_BEST_HP_CURVES:
            out_dir = f"{DELAN_BEST_PLOTS_OUT_DIR}/{DATASET_NAME}__{RUN_TAG}__{ts}/curves"
            summary_jsonl_container = f"{EVAL_DIR}/summary_delan_best_runs_{ts}.jsonl"
            hp_list = ",".join(DELAN_BEST_TORQUE_HP_PRESETS) if DELAN_BEST_TORQUE_HP_PRESETS else ""
            if not hp_list:
                hp_list = ",".join(DELAN_BEST_HP_PRESETS)
            hp_arg = f"--hp_presets {hp_list} " if hp_list else ""
            master_log.write("\n" + banner(["DeLaN best hp-level train/val curves"], char="#") + "\n")
            cmd = compose_exec(
                SVC_EVAL,
                f"python3 {SCRIPT_DELAN_BEST_HP_CURVES} "
                f"--summary_jsonl {summary_jsonl_container} "
                f"--out_dir {out_dir} "
                f"{hp_arg}"
            )
            run_cmd(cmd, master_log)

        if DELAN_BEST_SCATTER_PLOTS:
            out_dir = f"{DELAN_BEST_PLOTS_OUT_DIR}/{DATASET_NAME}__{RUN_TAG}__{ts}/hypers"
            summary_jsonl_container = f"{EVAL_DIR}/summary_delan_best_hypers_{ts}.jsonl"
            master_log.write("\n" + banner(["DeLaN best hyper scatter plots"], char="#") + "\n")
            cmd = compose_exec(
                SVC_EVAL,
                f"python3 {SCRIPT_DELAN_BEST_HYPER_SCATTER} "
                f"--summary_jsonl {summary_jsonl_container} "
                f"--out_dir {out_dir}"
            )
            run_cmd(cmd, master_log)

        if DELAN_BEST_TORQUE_AGGREGATE:
            out_dir = f"{DELAN_BEST_PLOTS_OUT_DIR}/{DATASET_NAME}__{RUN_TAG}__{ts}/torque"
            summary_jsonl_container = f"{EVAL_DIR}/summary_delan_best_runs_{ts}.jsonl"
            hp_list = ",".join(DELAN_BEST_TORQUE_HP_PRESETS) if DELAN_BEST_TORQUE_HP_PRESETS else ""
            hp_arg = f"--hp_presets {hp_list} " if hp_list else ""
            master_log.write("\n" + banner(["DeLaN best torque aggregates"], char="#") + "\n")
            cmd = compose_exec(
                SVC_EVAL,
                f"python3 {SCRIPT_DELAN_BEST_TORQUE_AGG} "
                f"--summary_jsonl {summary_jsonl_container} "
                f"--out_dir {out_dir} "
                f"--bins {DELAN_BEST_TORQUE_BINS} "
                f"--split {DELAN_BEST_TORQUE_SPLIT} "
                f"{hp_arg}"
            )
            run_cmd(cmd, master_log)

        master_log.write("\n" + banner(["DeLaN best-model sweep finished OK"], char="#") + "\n")

    print(f"\nMASTER LOG: {master_log_path}")
    print("Done.")


if __name__ == "__main__":
    main()
