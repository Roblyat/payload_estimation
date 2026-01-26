from __future__ import annotations

from sweep_base import (
    SVC_PREPROCESS,
    SVC_LSTM,
    SVC_EVAL,
    PROCESSED_DIR,
    MODELS_LSTM_DIR,
    MODELS_LSTM_DIR_HOST,
    EVAL_DIR,
    DATASET_NAME,
    RUN_TAG,
    VAL_FRACTION,
    SCRIPT_BUILD_LSTM_WINDOWS,
    SCRIPT_EVAL,
    LSTM_EPOCHS,
    LSTM_BATCH,
    LSTM_VAL_SPLIT,
    LSTM_UNITS,
    LSTM_DROPOUT,
    LSTM_EPS,
    LSTM_NO_PLOTS,
    LSTM_EARLY_STOP,
    LSTM_EARLY_STOP_PATIENCE,
    LSTM_EARLY_STOP_MIN_DELTA,
    LSTM_EARLY_STOP_WARMUP_EVALS,
    DELAN_EPOCHS,
    DELAN_HP_PRESET,
)
from sweep_helper import compose_exec, run_cmd, banner, append_csv_row, append_jsonl, safe_tag
from .sweep_lstm_helper import lstm_train_cmd_patched, read_lstm_metrics


def run_lstm_block(
    *,
    base_id: str,
    K: int,
    tf: float,
    seed: int,
    best: dict,
    H: int,
    feat: str,
    res_out: str,
    lstm_summary_csv: str,
    lstm_summary_jsonl: str,
    eval_metrics_csv: str,
    eval_metrics_jsonl: str,
    ts: str,
    log_file,
    lowpass_qdd: bool,
):
    log_file.write("\n" + banner([f"BLOCK: H={H} feat={feat}"], char="#") + "\n")

    windows_npz_name = (
        f"{base_id}__K{K}_tf{safe_tag(tf)}_vf{safe_tag(VAL_FRACTION)}__lstm_windows_H{H}"
        f"__feat_{feat}__{best['delan_tag']}.npz"
    )
    win_out = f"{PROCESSED_DIR}/{windows_npz_name}"

    lstm_dir_name = (
        f"{base_id}__K{K}_tf{safe_tag(tf)}_vf{safe_tag(VAL_FRACTION)}__{best['delan_tag']}"
        f"__feat_{feat}__lstm_s{seed}_H{H}"
        f"_ep{LSTM_EPOCHS}_b{LSTM_BATCH}_u{LSTM_UNITS}_do{safe_tag(LSTM_DROPOUT)}"
    )
    lstm_out = f"{MODELS_LSTM_DIR}/{lstm_dir_name}"
    lstm_model_name = "residual_lstm.keras"
    lstm_model_path = f"{lstm_out}/{lstm_model_name}"
    lstm_scalers_path = f"{lstm_out}/scalers_H{H}.npz"

    eval_out = f"{EVAL_DIR}/{lstm_dir_name}"

    # 5) Build windows
    log_file.write("\n" + banner(["5) BUILD LSTM WINDOWS"], char="#") + "\n")
    cmd = compose_exec(
        SVC_PREPROCESS,
        f"python3 {SCRIPT_BUILD_LSTM_WINDOWS} "
        f"--in_npz {res_out} "
        f"--out_npz {win_out} "
        f"--H {H} "
        f"--features {feat}"
    )
    run_cmd(cmd, log_file)

    # 6) Train LSTM
    log_file.write("\n" + banner(["6) TRAIN LSTM"], char="#") + "\n")
    inner = lstm_train_cmd_patched(
        npz=win_out,
        out_dir=lstm_out,
        model_name=lstm_model_name,
        epochs=LSTM_EPOCHS,
        batch=LSTM_BATCH,
        val_split=LSTM_VAL_SPLIT,
        seed=seed,
        units=LSTM_UNITS,
        dropout=LSTM_DROPOUT,
        eps=LSTM_EPS,
        no_plots=LSTM_NO_PLOTS,
        early_stop=LSTM_EARLY_STOP,
        early_stop_patience=LSTM_EARLY_STOP_PATIENCE,
        early_stop_min_delta=LSTM_EARLY_STOP_MIN_DELTA,
        early_stop_warmup_evals=LSTM_EARLY_STOP_WARMUP_EVALS,
    )
    cmd = compose_exec(SVC_LSTM, inner)
    run_cmd(cmd, log_file)

    # LSTM sweep summary (host-written)
    lstm_metrics_path = f"{MODELS_LSTM_DIR_HOST}/{lstm_dir_name}/metrics_train_test_H{H}.json"
    lm = read_lstm_metrics(lstm_metrics_path)
    lstm_fields = [
        "timestamp",
        "dataset",
        "run_tag",
        "lowpass_qdd",
        "K",
        "test_fraction",
        "val_fraction",
        "seed",
        "H",
        "features",
        "lstm_out",
        "metrics_json",
        "metrics_exists",
        "epochs_ran",
        "best_epoch",
        "best_val_loss",
        "final_train_loss",
        "final_val_loss",
        "rmse_total",
        "mse_total",
        "best_model_path",
        "delan_seed",
        "delan_tag",
        "delan_rmse_val",
        "delan_rmse_test",
    ]
    lstm_row = {
        "timestamp": ts,
        "dataset": DATASET_NAME,
        "run_tag": RUN_TAG,
        "lowpass_qdd": lowpass_qdd,
        "K": K,
        "test_fraction": tf,
        "val_fraction": VAL_FRACTION,
        "seed": seed,
        "H": H,
        "features": feat,
        "lstm_out": lstm_out,
        "metrics_json": lstm_metrics_path,
        "metrics_exists": bool(lm.get("exists", False)),
        "epochs_ran": lm.get("epochs_ran"),
        "best_epoch": lm.get("best_epoch"),
        "best_val_loss": lm.get("best_val_loss"),
        "final_train_loss": lm.get("final_train_loss"),
        "final_val_loss": lm.get("final_val_loss"),
        "rmse_total": lm.get("rmse_total"),
        "mse_total": lm.get("mse_total"),
        "best_model_path": lm.get("best_model_path"),
        "delan_seed": best["delan_seed"],
        "delan_tag": best["delan_tag"],
        "delan_rmse_val": best["val_rmse"],
        "delan_rmse_test": best["test_rmse"],
    }
    append_csv_row(lstm_summary_csv, lstm_fields, lstm_row)
    append_jsonl(lstm_summary_jsonl, lstm_row)

    # 7) Combined evaluation + metrics append
    log_file.write("\n" + banner(["7) COMBINED EVALUATION"], char="#") + "\n")
    cmd = compose_exec(
        SVC_EVAL,
        f"python3 {SCRIPT_EVAL} "
        f"--residual_npz {res_out} "
        f"--model {lstm_model_path} "
        f"--scalers {lstm_scalers_path} "
        f"--out_dir {eval_out} "
        f"--H {H} "
        f"--split test "
        f"--features {feat} "
        f"--save_pred_npz "
        f"--metrics_csv {eval_metrics_csv} "
        f"--metrics_json {eval_metrics_jsonl} "
        f"--K {K} "
        f"--test_fraction {tf} "
        f"--seed {seed} "
        f"--delan_seed {best['delan_seed']} "
        f"--delan_epochs {DELAN_EPOCHS} "
        f"--hp_preset {DELAN_HP_PRESET} "
        f"--delan_rmse_val {best['val_rmse']} "
        f"--delan_rmse_test {best['test_rmse']}"
    )
    run_cmd(cmd, log_file)
