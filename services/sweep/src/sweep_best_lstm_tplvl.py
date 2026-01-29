from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from sweep_base import (
    RAW_DIR,
    PREPROCESSED_DIR,
    PROCESSED_DIR,
    EVAL_DIR,
    EVAL_DIR_HOST,
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
    DELAN_EPOCHS,
    DELAN_MODEL_TYPE,
    DELAN_HP_FLAGS,
    SCRIPT_EXPORT_DELAN_RES,
    SVC_DELAN,
    SVC_EVAL,
    LSTM_EPOCHS,
    LSTM_BATCH,
    LSTM_UNITS,
    LSTM_DROPOUT,
    LSTM_BEST_DATASET_SEEDS,
    LSTM_BEST_FEATURE_MODES,
    LSTM_BEST_H_LIST,
    LSTM_BEST_SEEDS,
    LSTM_BEST_SCORE_LAMBDA,
    LSTM_BEST_SCORE_PENALTY,
    LSTM_BEST_DELAN_HYPERS_JSONL,
    LSTM_BEST_DELAN_FOLDS_JSONL,
    LSTM_BEST_DELAN_MODEL_JSON,
    LSTM_BEST_EVAL_SPLIT,
    LSTM_BEST_BINS,
    LSTM_BEST_RESIDUAL_AGGREGATE,
    LSTM_BEST_COMBINED_AGGREGATE,
    LSTM_BEST_BOXPLOTS,
    LSTM_BEST_PLOTS_OUT_DIR,
    LSTM_BEST_MODELS_DIR,
    LSTM_BEST_MODELS_DIR_HOST,
    SCRIPT_EVAL,
    SCRIPT_LSTM_BEST_RESIDUAL_AGG,
    SCRIPT_LSTM_BEST_COMBINED_AGG,
    SCRIPT_LSTM_METRICS_BOXPLOTS,
    REPO_ROOT,
)
from sweep_helper import (
    banner,
    run_cmd,
    run_cmd_allow_fail,
    compose_exec,
    append_csv_row,
    append_jsonl,
    safe_tag,
    pad_curve_by_epoch,
    median_iqr_curves,
    median_iqr_scalar,
)
from preprocess.sweep_preprocess_loop import comp_prep
from delan.sweep_delan_helper import read_delan_metrics
from lstm.sweep_best_lstm_loop import build_lstm_windows, train_lstm_seed, run_combined_eval


def _best_id(ts: str) -> str:
    return f"lstm_best_{DATASET_NAME}_{RUN_TAG}_{ts}"


def _read_jsonl(path: str) -> List[dict]:
    p = _to_host_path(path)
    if not Path(p).exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _to_host_path(path: str) -> str:
    if path.startswith("/workspace/shared"):
        return str(REPO_ROOT / "shared") + path[len("/workspace/shared"):]
    return path


def _delan_tag_from_ckpt(ckpt_path: str) -> str:
    stem = Path(ckpt_path).stem
    if "__" in stem:
        return stem.split("__")[-1]
    return stem


def _load_best_delan_map() -> Tuple[str | None, Dict[int, dict]]:
    hypers = _read_jsonl(LSTM_BEST_DELAN_HYPERS_JSONL)
    best_hp = None
    if hypers:
        best_hp = min(hypers, key=lambda r: r.get("score_median", float("inf"))).get("hp_preset")

    folds = _read_jsonl(LSTM_BEST_DELAN_FOLDS_JSONL)
    best_map: Dict[int, dict] = {}
    if folds and best_hp is not None:
        for r in folds:
            if r.get("hp_preset") != best_hp:
                continue
            try:
                ds = int(r.get("dataset_seed"))
            except Exception:
                continue
            best_map[ds] = r

    if not best_map:
        # fallback: single best model json (same for all dataset seeds)
        try:
            with open(_to_host_path(LSTM_BEST_DELAN_MODEL_JSON), "r", encoding="utf-8") as f:
                best_model = json.load(f)
            for ds in LSTM_BEST_DATASET_SEEDS:
                best_map[int(ds)] = {
                    "dataset_seed": int(ds),
                    "hp_preset": best_model.get("hp_preset"),
                    "best_ckpt": best_model.get("ckpt"),
                    "best_metrics_json": best_model.get("metrics_json"),
                    "delan_seed": best_model.get("delan_seed"),
                    "K": best_model.get("K"),
                }
            if best_hp is None:
                best_hp = best_model.get("hp_preset")
        except Exception:
            pass

    return best_hp, best_map


def _read_history_csv(path: str) -> Tuple[List[int], List[float], List[float]]:
    epochs: List[int] = []
    loss: List[float] = []
    val_loss: List[float] = []
    p = _to_host_path(path)
    if not Path(p).exists():
        return epochs, loss, val_loss
    try:
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    epochs.append(int(float(row.get("epoch", ""))))
                    loss.append(float(row.get("loss", "")))
                    val_loss.append(float(row.get("val_loss", "")))
                except Exception:
                    continue
    except Exception:
        return [], [], []
    return epochs, loss, val_loss


def main() -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_name = _best_id(ts)
    logs_dir = Path(LOGS_DIR_HOST) / sweep_name
    logs_dir.mkdir(parents=True, exist_ok=True)

    raw_csv = f"{RAW_DIR}/{DATASET_NAME}.{IN_FORMAT}"

    master_log_path = logs_dir / f"{sweep_name}.log"

    runs_csv = str(Path(EVAL_DIR_HOST) / f"summary_lstm_best_runs_{ts}.csv")
    runs_jsonl = str(Path(EVAL_DIR_HOST) / f"summary_lstm_best_runs_{ts}.jsonl")
    configs_csv = str(Path(EVAL_DIR_HOST) / f"summary_lstm_best_configs_{ts}.csv")
    configs_jsonl = str(Path(EVAL_DIR_HOST) / f"summary_lstm_best_configs_{ts}.jsonl")
    fh_csv = str(Path(EVAL_DIR_HOST) / f"summary_lstm_best_fh_{ts}.csv")
    fh_jsonl = str(Path(EVAL_DIR_HOST) / f"summary_lstm_best_fh_{ts}.jsonl")
    combined_csv = str(Path(EVAL_DIR_HOST) / f"summary_lstm_best_combined_{ts}.csv")
    combined_jsonl = str(Path(EVAL_DIR_HOST) / f"summary_lstm_best_combined_{ts}.jsonl")
    combined_csv_container = f"{EVAL_DIR}/summary_lstm_best_combined_{ts}.csv"
    combined_jsonl_container = f"{EVAL_DIR}/summary_lstm_best_combined_{ts}.jsonl"
    best_model_json = Path(EVAL_DIR_HOST) / f"lstm_best_model_{ts}.json"

    with master_log_path.open("w", encoding="utf-8") as master_log:
        master_log.write(banner([
            f"LSTM best-model sweep started: {ts}",
            f"sweep_name={sweep_name}",
            f"logs_dir={logs_dir}",
            f"dataset={DATASET_NAME} run_tag={RUN_TAG}",
            f"raw_csv={raw_csv}",
            f"col_format={COL_FORMAT} derive_qdd={DERIVE_QDD}",
            f"test_fraction={TEST_FRACTIONS} val_fraction={VAL_FRACTION}",
            f"dataset_seeds={LSTM_BEST_DATASET_SEEDS}",
            f"feature_modes={LSTM_BEST_FEATURE_MODES}",
            f"H_list={LSTM_BEST_H_LIST}",
            f"lstm_seeds={LSTM_BEST_SEEDS}",
            f"score_lambda={LSTM_BEST_SCORE_LAMBDA} score_penalty={LSTM_BEST_SCORE_PENALTY}",
            f"LSTM: epochs={LSTM_EPOCHS} batch={LSTM_BATCH} units={LSTM_UNITS} dropout={LSTM_DROPOUT}",
            f"delan_hypers_jsonl={LSTM_BEST_DELAN_HYPERS_JSONL}",
            f"delan_folds_jsonl={LSTM_BEST_DELAN_FOLDS_JSONL}",
            f"delan_model_json={LSTM_BEST_DELAN_MODEL_JSON}",
            f"runs_csv={runs_csv}",
            f"runs_jsonl={runs_jsonl}",
            f"configs_csv={configs_csv}",
            f"configs_jsonl={configs_jsonl}",
            f"fh_csv={fh_csv}",
            f"fh_jsonl={fh_jsonl}",
            f"combined_csv={combined_csv}",
            f"combined_jsonl={combined_jsonl}",
            f"best_model_json={best_model_json}",
        ], char="#") + "\n")

        best_hp, delan_map = _load_best_delan_map()
        if not delan_map:
            msg = "No best DeLaN mapping found; check lstm_best_delan_* paths."
            master_log.write("\n" + banner([msg], char="!") + "\n")
            master_log.flush()
            raise RuntimeError(msg)

        run_fields = [
            "timestamp",
            "dataset",
            "run_tag",
            "K",
            "test_fraction",
            "val_fraction",
            "dataset_seed",
            "feature_mode",
            "H",
            "lstm_seed",
            "lstm_out",
            "metrics_json",
            "metrics_exists",
            "epochs_ran",
            "best_epoch",
            "best_val_loss",
            "final_train_loss",
            "final_val_loss",
            "residual_rmse",
            "residual_mse",
            "rg_rmse",
            "gain",
            "gain_ratio",
            "diverged",
            "model_path",
            "scalers_path",
            "history_csv",
            "delan_seed",
            "delan_tag",
            "delan_ckpt",
            "delan_rmse_val",
            "delan_rmse_test",
            "combined_out_dir",
            "combined_metrics_json",
        ]

        config_fields = [
            "timestamp",
            "dataset",
            "run_tag",
            "K",
            "test_fraction",
            "val_fraction",
            "dataset_seed",
            "feature_mode",
            "H",
            "residual_rmse_median",
            "residual_rmse_iqr",
            "rg_rmse_median",
            "rg_rmse_iqr",
            "gain_ratio_median",
            "gain_ratio_iqr",
            "divergence_rate",
            "score",
            "best_lstm_seed",
            "best_model_path",
            "best_combined_out_dir",
            "curves_npz",
        ]

        fh_fields = [
            "timestamp",
            "dataset",
            "run_tag",
            "feature_mode",
            "H",
            "score_median",
            "score_iqr",
            "rg_rmse_median",
            "rg_rmse_iqr",
            "gain_ratio_median",
            "gain_ratio_iqr",
            "divergence_rate_median",
            "best_dataset_seed",
            "best_lstm_seed",
            "best_model_path",
        ]

        all_config_rows: List[dict] = []
        all_fh_rows: List[dict] = []

        for dataset_seed in LSTM_BEST_DATASET_SEEDS:
            dataset_seed = int(dataset_seed)
            delan_info = delan_map.get(dataset_seed)
            if not delan_info:
                master_log.write("\n" + banner([f"Missing DeLaN baseline for dataset_seed={dataset_seed}"], char="!") + "\n")
                continue

            K = int(delan_info.get("K") or 0)
            if not K:
                master_log.write("\n" + banner([f"Missing K for dataset_seed={dataset_seed}"], char="!") + "\n")
                continue

            npz_name = f"delan_{DATASET_NAME}_K{K}_seed{dataset_seed}_dataset.npz"
            npz_stem = Path(npz_name).stem
            npz_in = f"{PREPROCESSED_DIR}/{npz_stem}/{npz_stem}.npz"

            run_log_path = logs_dir / f"run_lstm_best_d{dataset_seed}_{ts}.log"
            with run_log_path.open("w", encoding="utf-8") as run_log:
                run_log.write(banner([
                    f"RUN: dataset_seed={dataset_seed} K={K} test_fraction={TEST_FRACTIONS} val_fraction={VAL_FRACTION}",
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

                delan_ckpt = delan_info.get("best_ckpt") or delan_info.get("ckpt")
                if not delan_ckpt:
                    run_log.write("\n" + banner(["Missing DeLaN checkpoint"], char="!") + "\n")
                    continue
                delan_tag = _delan_tag_from_ckpt(delan_ckpt)

                # Read DeLaN metrics for combined-eval metadata
                delan_metrics_path = delan_info.get("best_metrics_json") or delan_info.get("metrics_json")
                dm = read_delan_metrics(_to_host_path(delan_metrics_path)) if delan_metrics_path else {}
                delan_rmse_val = dm.get("val_rmse")
                delan_rmse_test = dm.get("test_rmse")
                delan_seed = delan_info.get("best_delan_seed") or delan_info.get("delan_seed")
                hp_preset = delan_info.get("hp_preset") or best_hp or "unknown"

                residual_name = f"{DATASET_NAME}__{RUN_TAG}__K{K}__residual__{delan_tag}.npz"
                res_out = f"{PROCESSED_DIR}/{residual_name}"

                run_log.write("\n" + banner(["EXPORT RESIDUALS (best DeLaN)"], char="#") + "\n")
                cmd = compose_exec(
                    SVC_DELAN,
                    f"python3 {SCRIPT_EXPORT_DELAN_RES} "
                    f"--npz_in {npz_in} "
                    f"--ckpt {delan_ckpt} "
                    f"--out {res_out}"
                )
                run_cmd(cmd, run_log)

                for feat in LSTM_BEST_FEATURE_MODES:
                    for H in LSTM_BEST_H_LIST:
                        H = int(H)
                        run_log.write("\n" + banner([f"CONFIG: feat={feat} H={H}"], char="#") + "\n")

                        windows_npz_name = (
                            f"{DATASET_NAME}__{RUN_TAG}__K{K}__d{dataset_seed}__lstm_windows_H{H}"
                            f"__feat_{feat}__{delan_tag}.npz"
                        )
                        win_out = f"{PROCESSED_DIR}/{windows_npz_name}"

                        build_lstm_windows(
                            npz_in=res_out,
                            out_npz=win_out,
                            H=H,
                            feat=feat,
                            log_file=run_log,
                        )

                        run_results: List[dict] = []
                        curves_train: List[List[float]] = []
                        curves_val: List[List[float]] = []

                        for lstm_seed in LSTM_BEST_SEEDS:
                            lstm_seed = int(lstm_seed)
                            lstm_dir_name = (
                                f"{DATASET_NAME}__{RUN_TAG}__K{K}__d{dataset_seed}__{delan_tag}"
                                f"__feat_{feat}__lstm_best_s{lstm_seed}_H{H}"
                                f"_ep{LSTM_EPOCHS}_b{LSTM_BATCH}_u{LSTM_UNITS}_do{safe_tag(LSTM_DROPOUT)}"
                            )
                            lstm_out = f"{LSTM_BEST_MODELS_DIR}/{lstm_dir_name}"
                            model_name = "residual_lstm.keras"
                            model_path = f"{lstm_out}/{model_name}"
                            scalers_path = f"{lstm_out}/scalers_H{H}.npz"

                            eval_out = f"{EVAL_DIR}/lstm_best/{DATASET_NAME}__{RUN_TAG}__{ts}/{lstm_dir_name}"

                            train_info = train_lstm_seed(
                                windows_npz=win_out,
                                lstm_out=lstm_out,
                                model_name=model_name,
                                H=H,
                                seed=lstm_seed,
                                log_file=run_log,
                            )

                            metrics = train_info.get("metrics", {})
                            diverged = bool(train_info.get("diverged"))
                            history_csv = metrics.get("history_csv")
                            epochs, loss, val_loss = _read_history_csv(history_csv) if history_csv else ([], [], [])
                            if epochs and loss:
                                curves_train.append(pad_curve_by_epoch(epochs, loss, max(epochs)))
                            if epochs and val_loss:
                                curves_val.append(pad_curve_by_epoch(epochs, val_loss, max(epochs)))

                            combined_metrics = None
                            if not diverged:
                                combined_metrics = run_combined_eval(
                                    residual_npz=res_out,
                                    model_path=model_path,
                                    scalers_path=scalers_path,
                                    eval_out=eval_out,
                                    H=H,
                                    feat=feat,
                                    K=K,
                                    test_fraction=TEST_FRACTIONS,
                                    dataset_seed=dataset_seed,
                                    lstm_seed=lstm_seed,
                                    delan_seed=delan_seed,
                                    delan_epochs=DELAN_EPOCHS,
                                    hp_preset=hp_preset,
                                    delan_rmse_val=delan_rmse_val,
                                    delan_rmse_test=delan_rmse_test,
                                    metrics_csv=combined_csv_container,
                                    metrics_json=combined_jsonl_container,
                                    split=LSTM_BEST_EVAL_SPLIT,
                                    log_file=run_log,
                                )
                                if combined_metrics is None:
                                    diverged = True

                            run_row = {
                                "timestamp": ts,
                                "dataset": DATASET_NAME,
                                "run_tag": RUN_TAG,
                                "K": K,
                                "test_fraction": TEST_FRACTIONS,
                                "val_fraction": VAL_FRACTION,
                                "dataset_seed": dataset_seed,
                                "feature_mode": feat,
                                "H": H,
                                "lstm_seed": lstm_seed,
                                "lstm_out": lstm_out,
                                "metrics_json": train_info.get("metrics_json"),
                                "metrics_exists": bool(metrics.get("exists")),
                                "epochs_ran": metrics.get("epochs_ran"),
                                "best_epoch": metrics.get("best_epoch"),
                                "best_val_loss": metrics.get("best_val_loss"),
                                "final_train_loss": metrics.get("final_train_loss"),
                                "final_val_loss": metrics.get("final_val_loss"),
                                "residual_rmse": metrics.get("rmse_total"),
                                "residual_mse": metrics.get("mse_total"),
                                "rg_rmse": combined_metrics.get("rg_rmse") if combined_metrics else None,
                                "gain": combined_metrics.get("gain") if combined_metrics else None,
                                "gain_ratio": combined_metrics.get("gain_ratio") if combined_metrics else None,
                                "diverged": bool(diverged),
                                "model_path": model_path,
                                "scalers_path": scalers_path,
                                "history_csv": history_csv,
                                "delan_seed": delan_seed,
                                "delan_tag": delan_tag,
                                "delan_ckpt": delan_ckpt,
                                "delan_rmse_val": delan_rmse_val,
                                "delan_rmse_test": delan_rmse_test,
                                "combined_out_dir": eval_out,
                                "combined_metrics_json": combined_metrics.get("metrics_json") if combined_metrics else None,
                            }
                            append_csv_row(runs_csv, run_fields, run_row)
                            append_jsonl(runs_jsonl, run_row)
                            run_results.append(run_row)

                        # align curves to common E_max
                        if curves_train:
                            emax = max(len(c) for c in curves_train)
                            curves_train = [c if len(c) == emax else pad_curve_by_epoch(list(range(1, len(c) + 1)), c.tolist(), emax) for c in curves_train]
                        if curves_val:
                            emax = max(len(c) for c in curves_val)
                            curves_val = [c if len(c) == emax else pad_curve_by_epoch(list(range(1, len(c) + 1)), c.tolist(), emax) for c in curves_val]

                        curves_npz = None
                        if curves_train or curves_val:
                            out_dir = Path(EVAL_DIR_HOST) / "lstm_best" / f"{DATASET_NAME}__{RUN_TAG}__{ts}" / "curves" / f"feat_{safe_tag(feat)}" / f"H{H}" / f"d{dataset_seed}"
                            out_dir.mkdir(parents=True, exist_ok=True)
                            curves_npz = str(out_dir / "train_val_curves.npz")
                            train_med = train_q25 = train_q75 = None
                            val_med = val_q25 = val_q75 = None
                            if curves_train:
                                train_med, train_q25, train_q75 = median_iqr_curves(curves_train)
                            if curves_val:
                                val_med, val_q25, val_q75 = median_iqr_curves(curves_val)
                            npz_kwargs = {}
                            if train_med is not None:
                                npz_kwargs["train_median"] = train_med
                                npz_kwargs["train_q25"] = train_q25
                                npz_kwargs["train_q75"] = train_q75
                            if val_med is not None:
                                npz_kwargs["val_median"] = val_med
                                npz_kwargs["val_q25"] = val_q25
                                npz_kwargs["val_q75"] = val_q75
                            if npz_kwargs:
                                import numpy as np
                                np.savez(curves_npz, **npz_kwargs)

                        non_div = [r for r in run_results if not r.get("diverged")]
                        residual_vals = [r.get("residual_rmse") for r in non_div if r.get("residual_rmse") is not None]
                        rg_vals = [r.get("rg_rmse") for r in non_div if r.get("rg_rmse") is not None]
                        gain_vals = [r.get("gain_ratio") for r in non_div if r.get("gain_ratio") is not None]

                        residual_med, residual_iqr = median_iqr_scalar(residual_vals)
                        rg_med, rg_iqr = median_iqr_scalar(rg_vals)
                        gain_med, gain_iqr = median_iqr_scalar(gain_vals)

                        total_runs = len(run_results)
                        divergence_rate = 1.0
                        if total_runs > 0:
                            divergence_rate = (total_runs - len(non_div)) / float(total_runs)

                        score = (
                            float(rg_med)
                            + float(LSTM_BEST_SCORE_LAMBDA) * float(rg_iqr)
                            + float(LSTM_BEST_SCORE_PENALTY) * float(divergence_rate)
                        )

                        best_run = None
                        if non_div:
                            best_run = min(non_div, key=lambda r: r.get("rg_rmse", float("inf")))

                        config_row = {
                            "timestamp": ts,
                            "dataset": DATASET_NAME,
                            "run_tag": RUN_TAG,
                            "K": K,
                            "test_fraction": TEST_FRACTIONS,
                            "val_fraction": VAL_FRACTION,
                            "dataset_seed": dataset_seed,
                            "feature_mode": feat,
                            "H": H,
                            "residual_rmse_median": residual_med,
                            "residual_rmse_iqr": residual_iqr,
                            "rg_rmse_median": rg_med,
                            "rg_rmse_iqr": rg_iqr,
                            "gain_ratio_median": gain_med,
                            "gain_ratio_iqr": gain_iqr,
                            "divergence_rate": divergence_rate,
                            "score": score,
                            "best_lstm_seed": best_run.get("lstm_seed") if best_run else None,
                            "best_model_path": best_run.get("model_path") if best_run else None,
                            "best_combined_out_dir": best_run.get("combined_out_dir") if best_run else None,
                            "curves_npz": curves_npz,
                        }
                        append_csv_row(configs_csv, config_fields, config_row)
                        append_jsonl(configs_jsonl, config_row)
                        all_config_rows.append(config_row)

                        master_log.write(
                            f"\n[OK] d={dataset_seed} feat={feat} H={H} score={score:.4f} log={run_log_path}\n"
                        )
                        master_log.flush()

        # Aggregate across dataset seeds: per (feature, H)
        for feat in LSTM_BEST_FEATURE_MODES:
            for H in LSTM_BEST_H_LIST:
                rows = [r for r in all_config_rows if r.get("feature_mode") == feat and int(r.get("H", -1)) == int(H)]
                if not rows:
                    continue
                score_vals = [r["score"] for r in rows if r.get("score") is not None]
                rg_vals = [r["rg_rmse_median"] for r in rows if r.get("rg_rmse_median") is not None]
                gain_vals = [r["gain_ratio_median"] for r in rows if r.get("gain_ratio_median") is not None]
                div_vals = [r["divergence_rate"] for r in rows if r.get("divergence_rate") is not None]

                score_med, score_iqr = median_iqr_scalar(score_vals)
                rg_med, rg_iqr = median_iqr_scalar(rg_vals)
                gain_med, gain_iqr = median_iqr_scalar(gain_vals)
                div_med, _ = median_iqr_scalar(div_vals)

                best_row = min(rows, key=lambda r: r.get("score", float("inf"))) if rows else None

                fh_row = {
                    "timestamp": ts,
                    "dataset": DATASET_NAME,
                    "run_tag": RUN_TAG,
                    "feature_mode": feat,
                    "H": int(H),
                    "score_median": score_med,
                    "score_iqr": score_iqr,
                    "rg_rmse_median": rg_med,
                    "rg_rmse_iqr": rg_iqr,
                    "gain_ratio_median": gain_med,
                    "gain_ratio_iqr": gain_iqr,
                    "divergence_rate_median": div_med,
                    "best_dataset_seed": best_row.get("dataset_seed") if best_row else None,
                    "best_lstm_seed": best_row.get("best_lstm_seed") if best_row else None,
                    "best_model_path": best_row.get("best_model_path") if best_row else None,
                }
                append_csv_row(fh_csv, fh_fields, fh_row)
                append_jsonl(fh_jsonl, fh_row)
                all_fh_rows.append(fh_row)

        # Final selection
        best_fh = min(all_fh_rows, key=lambda r: r.get("score_median", float("inf"))) if all_fh_rows else None
        best_config = None
        if best_fh:
            best_config = min(
                [r for r in all_config_rows if r.get("feature_mode") == best_fh["feature_mode"] and int(r.get("H")) == int(best_fh["H"])],
                key=lambda r: r.get("score", float("inf")),
                default=None,
            )

        if best_config:
            best_manifest = {
                "timestamp": ts,
                "dataset": DATASET_NAME,
                "run_tag": RUN_TAG,
                "feature_mode": best_config.get("feature_mode"),
                "H": best_config.get("H"),
                "dataset_seed": best_config.get("dataset_seed"),
                "lstm_seed": best_config.get("best_lstm_seed"),
                "model_path": best_config.get("best_model_path"),
                "combined_out_dir": best_config.get("best_combined_out_dir"),
                "score": best_config.get("score"),
                "rg_rmse_median": best_config.get("rg_rmse_median"),
                "gain_ratio_median": best_config.get("gain_ratio_median"),
            }
            best_model_json.parent.mkdir(parents=True, exist_ok=True)
            with best_model_json.open("w", encoding="utf-8") as f:
                json.dump(best_manifest, f, indent=2)

            # rerun combined evaluation to save canonical plots
            master_log.write("\n" + banner(["Best LSTM combined evaluation"], char="#") + "\n")
            best_feat = best_config.get("feature_mode")
            best_H = int(best_config.get("H"))
            best_seed = int(best_config.get("best_lstm_seed"))
            best_model = best_config.get("best_model_path")
            best_out = f"{EVAL_DIR}/lstm_best/{DATASET_NAME}__{RUN_TAG}__{ts}/best"
            # find residual from dataset_seed
            ds = int(best_config.get("dataset_seed"))
            delan_info = delan_map.get(ds, {})
            K = int(delan_info.get("K") or 0)
            delan_ckpt = delan_info.get("best_ckpt") or delan_info.get("ckpt")
            delan_tag = _delan_tag_from_ckpt(delan_ckpt) if delan_ckpt else ""
            residual_name = f"{DATASET_NAME}__{RUN_TAG}__K{K}__residual__{delan_tag}.npz"
            res_out = f"{PROCESSED_DIR}/{residual_name}"
            scalers_path = str(best_model).replace("residual_lstm.keras", f"scalers_H{best_H}.npz")

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
                f"--K {K} "
                f"--test_fraction {TEST_FRACTIONS} "
                f"--seed {best_seed}"
            )
            run_cmd_allow_fail(cmd, master_log)

        # Aggregate plots by feature_mode
        if LSTM_BEST_RESIDUAL_AGGREGATE:
            for feat in LSTM_BEST_FEATURE_MODES:
                out_dir = f"{LSTM_BEST_PLOTS_OUT_DIR}/{DATASET_NAME}__{RUN_TAG}__{ts}/residual/{safe_tag(feat)}"
                summary_jsonl_container = f"{EVAL_DIR}/summary_lstm_best_runs_{ts}.jsonl"
                master_log.write("\n" + banner([f"LSTM best residual plots feat={feat}"], char="#") + "\n")
                cmd = compose_exec(
                    SVC_EVAL,
                    f"python3 {SCRIPT_LSTM_BEST_RESIDUAL_AGG} "
                    f"--summary_jsonl {summary_jsonl_container} "
                    f"--out_dir {out_dir} "
                    f"--bins {LSTM_BEST_BINS} "
                    f"--feature {feat}"
                )
                run_cmd(cmd, master_log)

        if LSTM_BEST_COMBINED_AGGREGATE:
            for feat in LSTM_BEST_FEATURE_MODES:
                out_dir = f"{LSTM_BEST_PLOTS_OUT_DIR}/{DATASET_NAME}__{RUN_TAG}__{ts}/combined/{safe_tag(feat)}"
                summary_jsonl_container = f"{EVAL_DIR}/summary_lstm_best_combined_{ts}.jsonl"
                master_log.write("\n" + banner([f"LSTM best combined torque plots feat={feat}"], char="#") + "\n")
                cmd = compose_exec(
                    SVC_EVAL,
                    f"python3 {SCRIPT_LSTM_BEST_COMBINED_AGG} "
                    f"--summary_jsonl {summary_jsonl_container} "
                    f"--out_dir {out_dir} "
                    f"--bins {LSTM_BEST_BINS} "
                    f"--split {LSTM_BEST_EVAL_SPLIT} "
                    f"--feature {feat}"
                )
                run_cmd(cmd, master_log)

        if LSTM_BEST_BOXPLOTS:
            out_dir = f"{LSTM_BEST_PLOTS_OUT_DIR}/{DATASET_NAME}__{RUN_TAG}__{ts}/boxplots"
            master_log.write("\n" + banner(["LSTM best metrics boxplots"], char="#") + "\n")
            cmd = compose_exec(
                SVC_EVAL,
                f"python3 {SCRIPT_LSTM_METRICS_BOXPLOTS} "
                f"--lstm_root {LSTM_BEST_MODELS_DIR} "
                f"--out_dir {out_dir}"
            )
            run_cmd(cmd, master_log)

        master_log.write("\n" + banner(["LSTM best-model sweep finished OK"], char="#") + "\n")

    print(f"\nMASTER LOG: {master_log_path}")
    print("Done.")


if __name__ == "__main__":
    main()
