from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

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
    REPO_ROOT,
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


def _to_host_path(path: str) -> str:
    if not path:
        return path
    if path.startswith("/workspace/shared"):
        return str(REPO_ROOT / "shared") + path[len("/workspace/shared") :]
    return path


def _resample_progress(curve: np.ndarray, n_bins: int) -> np.ndarray:
    if curve is None:
        return np.full((n_bins,), np.nan, dtype=np.float32)
    arr = np.asarray(curve).reshape(-1)
    if arr.size == 0:
        return np.full((n_bins,), np.nan, dtype=np.float32)
    if arr.size == 1:
        return np.full((n_bins,), float(arr[0]), dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, num=arr.size)
    x_new = np.linspace(0.0, 1.0, num=int(n_bins))
    return np.interp(x_new, x_old, arr).astype(np.float32)


def _median_iqr(stack_list: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stack = np.stack(stack_list, axis=0)
    med = np.nanmedian(stack, axis=0)
    q25 = np.nanpercentile(stack, 25, axis=0)
    q75 = np.nanpercentile(stack, 75, axis=0)
    return med, q25, q75


def main() -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_name = _best_id(ts)
    logs_dir = Path(LOGS_DIR_HOST) / sweep_name
    logs_dir.mkdir(parents=True, exist_ok=True)

    raw_csv = f"{RAW_DIR}/{DATASET_NAME}.{IN_FORMAT}"

    master_log_path = logs_dir / f"{sweep_name}.log"

    # primary outputs include both run_tag and timestamp (requested: runtag_timestamp)
    runs_csv = Path(EVAL_DIR_HOST) / f"summary_delan_best_runs_{RUN_TAG}_{ts}.csv"
    runs_jsonl = Path(EVAL_DIR_HOST) / f"summary_delan_best_runs_{RUN_TAG}_{ts}.jsonl"
    folds_csv = Path(EVAL_DIR_HOST) / f"summary_delan_best_folds_{RUN_TAG}_{ts}.csv"
    folds_jsonl = Path(EVAL_DIR_HOST) / f"summary_delan_best_folds_{RUN_TAG}_{ts}.jsonl"
    hypers_csv = Path(EVAL_DIR_HOST) / f"summary_delan_best_hypers_{RUN_TAG}_{ts}.csv"
    hypers_jsonl = Path(EVAL_DIR_HOST) / f"summary_delan_best_hypers_{RUN_TAG}_{ts}.jsonl"
    best_model_json = Path(EVAL_DIR_HOST) / f"delan_best_model_{RUN_TAG}_{ts}.json"

    # aliases: latest per run_tag and legacy timestamp-only for compatibility
    runs_csv_tag = Path(EVAL_DIR_HOST) / f"summary_delan_best_runs_{RUN_TAG}.csv"
    runs_csv_ts = Path(EVAL_DIR_HOST) / f"summary_delan_best_runs_{ts}.csv"
    runs_jsonl_tag = Path(EVAL_DIR_HOST) / f"summary_delan_best_runs_{RUN_TAG}.jsonl"
    runs_jsonl_ts = Path(EVAL_DIR_HOST) / f"summary_delan_best_runs_{ts}.jsonl"
    folds_csv_tag = Path(EVAL_DIR_HOST) / f"summary_delan_best_folds_{RUN_TAG}.csv"
    folds_csv_ts = Path(EVAL_DIR_HOST) / f"summary_delan_best_folds_{ts}.csv"
    folds_jsonl_tag = Path(EVAL_DIR_HOST) / f"summary_delan_best_folds_{RUN_TAG}.jsonl"
    folds_jsonl_ts = Path(EVAL_DIR_HOST) / f"summary_delan_best_folds_{ts}.jsonl"
    hypers_csv_tag = Path(EVAL_DIR_HOST) / f"summary_delan_best_hypers_{RUN_TAG}.csv"
    hypers_csv_ts = Path(EVAL_DIR_HOST) / f"summary_delan_best_hypers_{ts}.csv"
    hypers_jsonl_tag = Path(EVAL_DIR_HOST) / f"summary_delan_best_hypers_{RUN_TAG}.jsonl"
    hypers_jsonl_ts = Path(EVAL_DIR_HOST) / f"summary_delan_best_hypers_{ts}.jsonl"
    best_model_json_tag = Path(EVAL_DIR_HOST) / f"delan_best_model_{RUN_TAG}.json"
    best_model_json_ts = Path(EVAL_DIR_HOST) / f"delan_best_model_{ts}.json"

    # backwards-compatible names used later in the file
    runs_csv_tag_ts = runs_csv
    runs_jsonl_tag_ts = runs_jsonl
    folds_csv_tag_ts = folds_csv
    folds_jsonl_tag_ts = folds_jsonl
    hypers_csv_tag_ts = hypers_csv
    hypers_jsonl_tag_ts = hypers_jsonl

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
            "metrics_json",
            "metrics_json_container",
            "torque_rmse_progress_npz",
            "torque_rmse_per_joint_npz",
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
            "torque_rmse_progress_med_mean",
            "torque_rmse_progress_iqr_mean",
            "torque_rmse_per_joint_med_mean",
            "torque_rmse_per_joint_iqr_mean",
            "torque_rmse_progress_npz",
            "torque_rmse_per_joint_npz",
        ]

        all_fold_rows: list[dict] = []
        all_hyper_rows: list[dict] = []
        torque_progress_by_hp: dict[str, list[np.ndarray]] = {}
        torque_joint_by_hp: dict[str, list[np.ndarray]] = {}

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

                    best_metrics_json = None
                    best_metrics_json_container = None

                    if best is not None:
                        best_dir_host = f"{MODELS_DELAN_DIR_HOST}/{best['delan_id']}"
                        copy_candidate_metrics_to_best(best_dir_host, candidates)
                        cleanup_non_best_plots(best["delan_id"], candidates)
                        best_metrics_json = best.get("metrics_json")
                        best_metrics_json_container = best.get("metrics_json_container")

                    torque_progress_npz = None
                    torque_joint_npz = None

                    if best_metrics_json or best_metrics_json_container:
                        metrics_path = _to_host_path(best_metrics_json_container or best_metrics_json)
                        if metrics_path and Path(metrics_path).exists():
                            try:
                                with open(metrics_path, "r", encoding="utf-8") as mf:
                                    metrics_data = json.load(mf)
                                artifacts = metrics_data.get("artifacts", {}) or {}
                                rmse_path = _to_host_path(artifacts.get("torque_rmse_time_npy", ""))
                                if rmse_path and Path(rmse_path).exists():
                                    try:
                                        curve = np.load(rmse_path)
                                        resampled = _resample_progress(curve, int(DELAN_BEST_TORQUE_BINS))
                                        torque_progress_by_hp.setdefault(hp_preset, []).append(resampled)
                                        seed_out_dir = (
                                            Path(EVAL_DIR_HOST)
                                            / "delan_best_npz"
                                            / f"{DATASET_NAME}__{RUN_TAG}__{ts}"
                                            / f"hp_{safe_tag(hp_preset)}"
                                            / f"d{dataset_seed}"
                                        )
                                        seed_out_dir.mkdir(parents=True, exist_ok=True)
                                        torque_progress_npz = str(seed_out_dir / "torque_rmse_progress.npz")
                                        np.savez(torque_progress_npz, curve=resampled)
                                    except Exception:
                                        pass

                                eval_key = f"eval_{DELAN_BEST_TORQUE_SPLIT}"
                                per_joint = metrics_data.get(eval_key, {}).get("torque_rmse_per_joint")
                                if isinstance(per_joint, list) and per_joint:
                                    try:
                                        vec = np.asarray(per_joint, dtype=np.float32)
                                        torque_joint_by_hp.setdefault(hp_preset, []).append(vec)
                                        seed_out_dir = (
                                            Path(EVAL_DIR_HOST)
                                            / "delan_best_npz"
                                            / f"{DATASET_NAME}__{RUN_TAG}__{ts}"
                                            / f"hp_{safe_tag(hp_preset)}"
                                            / f"d{dataset_seed}"
                                        )
                                        seed_out_dir.mkdir(parents=True, exist_ok=True)
                                        torque_joint_npz = str(seed_out_dir / "torque_rmse_per_joint.npz")
                                        np.savez(torque_joint_npz, vec=vec)
                                    except Exception:
                                        pass
                            except Exception:
                                pass

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
                        "metrics_json": best_metrics_json,
                        "metrics_json_container": best_metrics_json_container,
                        "torque_rmse_progress_npz": torque_progress_npz,
                        "torque_rmse_per_joint_npz": torque_joint_npz,
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

            agg_progress_npz = None
            agg_joint_npz = None
            prog_med_mean = prog_iqr_mean = None
            joint_med_mean = joint_iqr_mean = None

            hp_progress_curves = torque_progress_by_hp.get(hp_preset, [])
            if hp_progress_curves:
                med_c, q25_c, q75_c = _median_iqr(hp_progress_curves)
                prog_med_mean = float(np.nanmean(med_c)) if np.isfinite(np.nanmean(med_c)) else None
                prog_iqr_mean = float(np.nanmean(q75_c - q25_c)) if np.isfinite(np.nanmean(q75_c - q25_c)) else None
                hp_out = (
                    Path(EVAL_DIR_HOST)
                    / "delan_best_npz"
                    / f"{DATASET_NAME}__{RUN_TAG}__{ts}"
                    / f"hp_{safe_tag(hp_preset)}"
                )
                hp_out.mkdir(parents=True, exist_ok=True)
                agg_progress_npz = str(hp_out / "torque_rmse_progress_agg.npz")
                np.savez(agg_progress_npz, median=med_c, q25=q25_c, q75=q75_c)

            hp_joint_vecs = torque_joint_by_hp.get(hp_preset, [])
            if hp_joint_vecs:
                med_j, q25_j, q75_j = _median_iqr(hp_joint_vecs)
                joint_med_mean = float(np.nanmean(med_j)) if np.isfinite(np.nanmean(med_j)) else None
                joint_iqr_mean = float(np.nanmean(q75_j - q25_j)) if np.isfinite(np.nanmean(q75_j - q25_j)) else None
                hp_out = (
                    Path(EVAL_DIR_HOST)
                    / "delan_best_npz"
                    / f"{DATASET_NAME}__{RUN_TAG}__{ts}"
                    / f"hp_{safe_tag(hp_preset)}"
                )
                hp_out.mkdir(parents=True, exist_ok=True)
                agg_joint_npz = str(hp_out / "torque_rmse_per_joint_agg.npz")
                np.savez(agg_joint_npz, median=med_j, q25=q25_j, q75=q75_j)

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
                "torque_rmse_progress_med_mean": prog_med_mean,
                "torque_rmse_progress_iqr_mean": prog_iqr_mean,
                "torque_rmse_per_joint_med_mean": joint_med_mean,
                "torque_rmse_per_joint_iqr_mean": joint_iqr_mean,
                "torque_rmse_progress_npz": agg_progress_npz,
                "torque_rmse_per_joint_npz": agg_joint_npz,
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
            for p in (best_model_json_tag, best_model_json_ts):
                with p.open("w", encoding="utf-8") as f:
                    json.dump(best_manifest, f, indent=2)

        # copy hypers/folds/runs to aliases (keep original timestamped primaries intact)
        def _copy_all(src, dests):
            if not Path(src).exists():
                return
            for dst in dests:
                try:
                    import shutil
                    shutil.copyfile(src, dst)
                except Exception:
                    pass

        _copy_all(hypers_jsonl, (hypers_jsonl_tag, hypers_jsonl_ts))
        _copy_all(folds_jsonl, (folds_jsonl_tag, folds_jsonl_ts))
        _copy_all(runs_jsonl, (runs_jsonl_tag, runs_jsonl_ts))
        _copy_all(hypers_csv, (hypers_csv_tag, hypers_csv_ts))
        _copy_all(folds_csv, (folds_csv_tag, folds_csv_ts))
        _copy_all(runs_csv, (runs_csv_tag, runs_csv_ts))

        if DELAN_BEST_FOLD_PLOTS:
            out_dir = f"{DELAN_BEST_PLOTS_OUT_DIR}/{DATASET_NAME}__{RUN_TAG}__{ts}/folds"
            summary_jsonl_container = str(runs_jsonl_tag_ts)
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
            summary_jsonl_container = str(runs_jsonl_tag_ts)
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
            summary_jsonl_container = str(hypers_jsonl_tag_ts)
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
            summary_jsonl_container = str(runs_jsonl_tag_ts)
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
