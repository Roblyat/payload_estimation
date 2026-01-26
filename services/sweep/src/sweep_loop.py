from __future__ import annotations

from datetime import datetime
from pathlib import Path

from sweep_base import (
    RAW_DIR,
    PREPROCESSED_DIR,
    PROCESSED_DIR,
    EVAL_DIR,
    EVAL_DIR_HOST,
    MODELS_DELAN_DIR_HOST,
    DATASET_NAME,
    RUN_TAG,
    IN_FORMAT,
    DERIVE_QDD,
    COL_FORMAT,
    LOWPASS_SIGNALS,
    LOWPASS_CUTOFF_HZ,
    LOWPASS_ORDER,
    LOWPASS_QDD_VALUES,
    TRAJ_AMOUNTS,
    TEST_FRACTIONS,
    VAL_FRACTION,
    SEEDS,
    H_LIST,
    FEATURE_MODES,
    DELAN_EPOCHS,
    DELAN_HP_PRESET,
    DELAN_HP_FLAGS,
    DELAN_MODEL_TYPE,
    LSTM_EPOCHS,
    LSTM_BATCH,
    LSTM_VAL_SPLIT,
    DELAN_ELBOW_AGGREGATE,
    DELAN_ELBOW_OUT_DIR,
    DELAN_TORQUE_AGGREGATE,
    DELAN_TORQUE_OUT_DIR,
    DELAN_TORQUE_BINS,
    DELAN_TORQUE_K_VALUES,
    LSTM_TRAINING_AGGREGATE,
    LSTM_RESIDUAL_AGGREGATE,
    LSTM_AGGREGATE_OUT_DIR,
    LSTM_AGGREGATE_BINS,
    LSTM_AGGREGATE_K_VALUES,
    LSTM_AGGREGATE_FEATURE,
    COMBINED_TORQUE_AGGREGATE,
    COMBINED_TORQUE_OUT_DIR,
    COMBINED_TORQUE_BINS,
    COMBINED_TORQUE_K_VALUES,
    COMBINED_TORQUE_FEATURE,
    SCRIPT_EXPORT_DELAN_RES,
    SCRIPT_DELAN_ELBOWS,
    SCRIPT_DELAN_TORQUE_AGG,
    SCRIPT_LSTM_TRAINING_AGG,
    SCRIPT_LSTM_RESIDUAL_AGG,
    SCRIPT_COMBINED_TORQUE_AGG,
    SVC_DELAN,
    SVC_EVAL,
)
from sweep_helper import banner, run_cmd, compose_exec, append_csv_row, append_jsonl, safe_tag
from preprocess.sweep_preprocess_loop import comp_prep
from delan.sweep_delan_loop import comp_delan, select_best
from delan.sweep_delan_helper import read_delan_metrics, copy_candidate_metrics_to_best, cleanup_non_best_plots
from lstm.sweep_lstm_loop import run_lstm_block


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = Path("logs_sweeps")
    logs_dir.mkdir(parents=True, exist_ok=True)

    raw_csv = f"{RAW_DIR}/{DATASET_NAME}.{IN_FORMAT}"
    master_logs = []

    lowpass_qdd = False
    master_log_path = logs_dir / f"sweep1_{DATASET_NAME}_{RUN_TAG}_{ts}.log"
    metrics_csv = f"{EVAL_DIR}/summary_metrics_sweep_1_{ts}.csv"
    metrics_jsonl = f"{EVAL_DIR}/summary_metrics_sweep_1_{ts}.jsonl"

    delan_summary_csv = str(Path(EVAL_DIR_HOST) / f"summary_delan_sweep_1_{ts}.csv")
    delan_summary_jsonl = str(Path(EVAL_DIR_HOST) / f"summary_delan_sweep_1_{ts}.jsonl")
    lstm_summary_csv = str(Path(EVAL_DIR_HOST) / f"summary_lstm_sweep_1_{ts}.csv")
    lstm_summary_jsonl = str(Path(EVAL_DIR_HOST) / f"summary_lstm_sweep_1_{ts}.jsonl")

    master_logs.append(master_log_path)

    with master_log_path.open("w", encoding="utf-8") as master_log:
        master_log.write(banner([
            f"Sweep1 started: {ts}",
            f"dataset={DATASET_NAME} run_tag={RUN_TAG}",
            f"raw_csv={raw_csv}",
            f"col_format={COL_FORMAT} derive_qdd={DERIVE_QDD}",
            f"traj_amounts={TRAJ_AMOUNTS} test_fracs={TEST_FRACTIONS} seeds={SEEDS}",
            f"val_fraction={VAL_FRACTION}",
            f"H={H_LIST} feature_modes={FEATURE_MODES}",
            f"lowpass_signals={LOWPASS_SIGNALS} cutoff_hz={LOWPASS_CUTOFF_HZ} order={LOWPASS_ORDER} lowpass_qdd={lowpass_qdd}",
            f"DeLaN: backend=jax type={DELAN_MODEL_TYPE} hp_preset={DELAN_HP_PRESET} epochs={DELAN_EPOCHS}",
            f"LSTM: epochs_max={LSTM_EPOCHS} batch={LSTM_BATCH} val_split={LSTM_VAL_SPLIT}",
            f"metrics_csv={metrics_csv}",
            f"metrics_json={metrics_jsonl}",
            f"delan_summary_csv(host)={delan_summary_csv}",
            f"delan_summary_json(host)={delan_summary_jsonl}",
            f"lstm_summary_csv(host)={lstm_summary_csv}",
            f"lstm_summary_json(host)={lstm_summary_jsonl}",
        ], char="#") + "\n")

        for K in TRAJ_AMOUNTS:
            for tf in TEST_FRACTIONS:
                master_log.write("\n" + banner([
                    f"SUBSWEEP: lowpass_qdd={lowpass_qdd}  K={K} test_fraction={tf}"
                ], char="#") + "\n")
                master_log.flush()

                for seed in SEEDS:
                    base_id = f"{DATASET_NAME}__{RUN_TAG}"

                    delan_npz_name = (
                        f"delan_{DATASET_NAME}_K{K}_tf{safe_tag(tf)}"
                        f"_vf{safe_tag(VAL_FRACTION)}_seed{seed}_dataset.npz"
                    )
                    npz_stem = Path(delan_npz_name).stem
                    npz_in = f"{PREPROCESSED_DIR}/{npz_stem}/{npz_stem}.npz"

                    run_log_path = logs_dir / f"run1_K{K}_tf{safe_tag(tf)}_seed{seed}_{ts}.log"
                    with run_log_path.open("w", encoding="utf-8") as run_log:
                        run_log.write(banner([
                            f"RUN: lowpass_qdd={lowpass_qdd}  K={K} test_fraction={tf} seed={seed}",
                            f"npz_in={npz_in}",
                        ], char="#") + "\n")

                        comp_prep(
                            npz_in=npz_in,
                            K=K,
                            tf=tf,
                            seed=seed,
                            lowpass_qdd=lowpass_qdd,
                            raw_csv=raw_csv,
                            log_file=run_log,
                        )

                        delan_candidates = comp_delan(
                            npz_in=npz_in,
                            K=K,
                            seed=seed,
                            log_file=run_log,
                        )
                        best = select_best(delan_candidates, run_log)

                        delan_fields = [
                            "timestamp",
                            "dataset",
                            "run_tag",
                            "lowpass_qdd",
                            "K",
                            "test_fraction",
                            "val_fraction",
                            "seed",
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
                            "selected",
                        ]
                        for cand in delan_candidates:
                            m = read_delan_metrics(cand["metrics_json"])
                            row = {
                                "timestamp": ts,
                                "dataset": DATASET_NAME,
                                "run_tag": RUN_TAG,
                                "lowpass_qdd": lowpass_qdd,
                                "K": K,
                                "test_fraction": tf,
                                "val_fraction": VAL_FRACTION,
                                "seed": seed,
                                "delan_seed": cand["delan_seed"],
                                "delan_epochs": DELAN_EPOCHS,
                                "hp_preset": DELAN_HP_PRESET,
                                "hp_flags": DELAN_HP_FLAGS,
                                "model_type": DELAN_MODEL_TYPE,
                                "ckpt": cand["ckpt"],
                                "metrics_json": cand["metrics_json"],
                                "metrics_json_container": cand.get("metrics_json_container"),
                                "metrics_exists": bool(m.get("exists", False)),
                                "val_rmse": m.get("val_rmse", cand.get("val_rmse")),
                                "val_mse": m.get("val_mse"),
                                "test_rmse": m.get("test_rmse", cand.get("test_rmse")),
                                "test_mse": m.get("test_mse"),
                                "selected": cand["delan_seed"] == best["delan_seed"],
                            }
                            append_csv_row(delan_summary_csv, delan_fields, row)
                            append_jsonl(delan_summary_jsonl, row)

                        best_dir_host = f"{MODELS_DELAN_DIR_HOST}/{best['delan_id']}"
                        copy_candidate_metrics_to_best(best_dir_host, delan_candidates)
                        cleanup_non_best_plots(best["delan_id"], delan_candidates)

                        residual_name = (
                            f"{base_id}__K{K}_tf{safe_tag(tf)}_vf{safe_tag(VAL_FRACTION)}"
                            f"__residual__{best['delan_tag']}.npz"
                        )
                        res_out = f"{PROCESSED_DIR}/{residual_name}"

                        run_log.write("\n" + banner(["4) EXPORT RESIDUALS (best only)"], char="#") + "\n")
                        cmd = compose_exec(
                            SVC_DELAN,
                            f"python3 {SCRIPT_EXPORT_DELAN_RES} "
                            f"--npz_in {npz_in} "
                            f"--ckpt {best['ckpt']} "
                            f"--out {res_out}"
                        )
                        run_cmd(cmd, run_log)

                        for H in H_LIST:
                            for feat in FEATURE_MODES:
                                run_lstm_block(
                                    base_id=base_id,
                                    K=K,
                                    tf=tf,
                                    seed=seed,
                                    best=best,
                                    H=H,
                                    feat=feat,
                                    res_out=res_out,
                                    lstm_summary_csv=lstm_summary_csv,
                                    lstm_summary_jsonl=lstm_summary_jsonl,
                                    eval_metrics_csv=metrics_csv,
                                    eval_metrics_jsonl=metrics_jsonl,
                                    ts=ts,
                                    log_file=run_log,
                                    lowpass_qdd=lowpass_qdd,
                                )

                        master_log.write(f"\n[OK] lowpass_qdd={lowpass_qdd} K={K} tf={tf} seed={seed}  log={run_log_path}\n")
                        master_log.flush()

        if DELAN_ELBOW_AGGREGATE:
            out_dir = f"{DELAN_ELBOW_OUT_DIR}/{DATASET_NAME}__{RUN_TAG}__{ts}"
            summary_jsonl_container = f"{EVAL_DIR}/summary_delan_sweep_1_{ts}.jsonl"
            master_log.write("\n" + banner(["Delan elbow aggregation"], char="#") + "\n")
            cmd = compose_exec(
                SVC_EVAL,
                f"python3 {SCRIPT_DELAN_ELBOWS} "
                f"--summary_jsonl {summary_jsonl_container} "
                f"--out_dir {out_dir} "
                f"--pad_to_epochs {DELAN_EPOCHS} "
                f"--write_best_overlays"
            )
            run_cmd(cmd, master_log)

        if DELAN_TORQUE_AGGREGATE:
            out_dir = f"{DELAN_TORQUE_OUT_DIR}/{DATASET_NAME}__{RUN_TAG}__{ts}"
            summary_jsonl_container = f"{EVAL_DIR}/summary_delan_sweep_1_{ts}.jsonl"
            master_log.write("\n" + banner(["Delan torque RMSE aggregation"], char="#") + "\n")
            k_values = ",".join(str(k) for k in DELAN_TORQUE_K_VALUES) if DELAN_TORQUE_K_VALUES else ""
            k_arg = f"--k_values {k_values} " if k_values else ""
            cmd = compose_exec(
                SVC_EVAL,
                f"python3 {SCRIPT_DELAN_TORQUE_AGG} "
                f"--summary_jsonl {summary_jsonl_container} "
                f"--out_dir {out_dir} "
                f"--bins {DELAN_TORQUE_BINS} "
                f"{k_arg}"
            )
            run_cmd(cmd, master_log)

        if LSTM_TRAINING_AGGREGATE or LSTM_RESIDUAL_AGGREGATE:
            lstm_summary_jsonl_container = f"{EVAL_DIR}/summary_lstm_sweep_1_{ts}.jsonl"
            out_dir = f"{LSTM_AGGREGATE_OUT_DIR}/{DATASET_NAME}__{RUN_TAG}__{ts}"
            feature_arg = f"--feature {LSTM_AGGREGATE_FEATURE} " if LSTM_AGGREGATE_FEATURE else ""

            if LSTM_TRAINING_AGGREGATE:
                master_log.write("\n" + banner(["LSTM training dynamics aggregation"], char="#") + "\n")
                cmd = compose_exec(
                    SVC_EVAL,
                    f"python3 {SCRIPT_LSTM_TRAINING_AGG} "
                    f"--summary_jsonl {lstm_summary_jsonl_container} "
                    f"--out_dir {out_dir} "
                    f"{feature_arg}"
                )
                run_cmd(cmd, master_log)

            if LSTM_RESIDUAL_AGGREGATE:
                master_log.write("\n" + banner(["LSTM residual RMSE aggregation"], char="#") + "\n")
                k_values = ",".join(str(k) for k in LSTM_AGGREGATE_K_VALUES) if LSTM_AGGREGATE_K_VALUES else ""
                k_arg = f"--k_values {k_values} " if k_values else ""
                cmd = compose_exec(
                    SVC_EVAL,
                    f"python3 {SCRIPT_LSTM_RESIDUAL_AGG} "
                    f"--summary_jsonl {lstm_summary_jsonl_container} "
                    f"--out_dir {out_dir} "
                    f"--bins {LSTM_AGGREGATE_BINS} "
                    f"{k_arg}"
                    f"{feature_arg}"
                )
                run_cmd(cmd, master_log)

        if COMBINED_TORQUE_AGGREGATE:
            out_dir = f"{COMBINED_TORQUE_OUT_DIR}/{DATASET_NAME}__{RUN_TAG}__{ts}"
            summary_jsonl_container = f"{EVAL_DIR}/summary_metrics_sweep_1_{ts}.jsonl"
            master_log.write("\n" + banner(["Combined torque RMSE aggregation"], char="#") + "\n")
            k_values = ",".join(str(k) for k in COMBINED_TORQUE_K_VALUES) if COMBINED_TORQUE_K_VALUES else ""
            k_arg = f"--k_values {k_values} " if k_values else ""
            feature_arg = f"--feature {COMBINED_TORQUE_FEATURE} " if COMBINED_TORQUE_FEATURE else ""
            cmd = compose_exec(
                SVC_EVAL,
                f"python3 {SCRIPT_COMBINED_TORQUE_AGG} "
                f"--summary_jsonl {summary_jsonl_container} "
                f"--out_dir {out_dir} "
                f"--bins {COMBINED_TORQUE_BINS} "
                f"{k_arg}"
                f"{feature_arg}"
            )
            run_cmd(cmd, master_log)

        master_log.write("\n" + banner(["Sweep1 finished OK"], char="#") + "\n")

    for p in master_logs:
        print(f"\nMASTER LOG: {p}")
    print("Done.")


if __name__ == "__main__":
    main()
