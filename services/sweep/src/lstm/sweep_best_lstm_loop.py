from __future__ import annotations

import os

from sweep_base import (
    SVC_PREPROCESS,
    SVC_LSTM,
    SVC_EVAL,
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
    REPO_ROOT,
)
from sweep_helper import compose_exec, run_cmd, run_cmd_allow_fail, banner
from .sweep_lstm_helper import lstm_train_cmd_patched, read_lstm_metrics_safe


def build_lstm_windows(*, npz_in: str, out_npz: str, H: int, feat: str, log_file) -> None:
    log_file.write("\n" + banner([f"BUILD LSTM WINDOWS H={H} feat={feat}"], char="#") + "\n")
    cmd = compose_exec(
        SVC_PREPROCESS,
        f"python3 {SCRIPT_BUILD_LSTM_WINDOWS} "
        f"--in_npz {npz_in} "
        f"--out_npz {out_npz} "
        f"--H {H} "
        f"--features {feat}"
    )
    run_cmd(cmd, log_file)


def _to_host_path(path: str) -> str:
    if path.startswith("/workspace/shared"):
        return str(REPO_ROOT / "shared") + path[len("/workspace/shared"):]
    return path


def train_lstm_seed(
    *,
    windows_npz: str,
    lstm_out: str,
    model_name: str,
    H: int,
    seed: int,
    log_file,
) -> dict:
    log_file.write("\n" + banner([f"TRAIN LSTM seed={seed} H={H}"], char="#") + "\n")
    inner = lstm_train_cmd_patched(
        npz=windows_npz,
        out_dir=lstm_out,
        model_name=model_name,
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
    ok, _ = run_cmd_allow_fail(cmd, log_file)

    metrics_json_path = f"{lstm_out}/metrics_train_test_H{H}.json"
    metrics_json_host = _to_host_path(metrics_json_path)

    metrics = read_lstm_metrics_safe(metrics_json_host)
    diverged = (not ok) or (not metrics.get("exists")) or (not metrics.get("finite"))

    return {
        "ok": bool(ok),
        "diverged": bool(diverged),
        "metrics_json": metrics_json_path,
        "metrics_json_host": metrics_json_host,
        "metrics": metrics,
    }


def run_combined_eval(
    *,
    residual_npz: str,
    model_path: str,
    scalers_path: str,
    eval_out: str,
    H: int,
    feat: str,
    K: int,
    test_fraction: float,
    dataset_seed: int,
    lstm_seed: int,
    delan_seed: int | None,
    delan_epochs: int,
    hp_preset: str,
    delan_rmse_val: float | None,
    delan_rmse_test: float | None,
    metrics_csv: str,
    metrics_json: str,
    split: str,
    log_file,
) -> dict | None:
    log_file.write("\n" + banner([f"COMBINED EVAL seed={lstm_seed} H={H}"], char="#") + "\n")
    cmd = compose_exec(
        SVC_EVAL,
        f"python3 {SCRIPT_EVAL} "
        f"--residual_npz {residual_npz} "
        f"--model {model_path} "
        f"--scalers {scalers_path} "
        f"--out_dir {eval_out} "
        f"--H {H} "
        f"--split {split} "
        f"--features {feat} "
        f"--save_pred_npz "
        f"--metrics_csv {metrics_csv} "
        f"--metrics_json {metrics_json} "
        f"--K {K} "
        f"--test_fraction {test_fraction} "
        f"--seed {lstm_seed} "
        f"--dataset_seed {dataset_seed} "
        f"--delan_seed {delan_seed if delan_seed is not None else -1} "
        f"--delan_epochs {delan_epochs} "
        f"--hp_preset {hp_preset} "
        f"--delan_rmse_val {delan_rmse_val if delan_rmse_val is not None else 'nan'} "
        f"--delan_rmse_test {delan_rmse_test if delan_rmse_test is not None else 'nan'}"
    )
    ok, _ = run_cmd_allow_fail(cmd, log_file)
    if not ok:
        return None

    metrics_json_path = f"{eval_out}/metrics_{split}_H{H}.json"
    metrics_json_host = _to_host_path(metrics_json_path)

    if not os.path.exists(metrics_json_host):
        return None
    try:
        import json
        with open(metrics_json_host, "r", encoding="utf-8") as f:
            d = json.load(f)
        m = d.get("metrics", {}) if isinstance(d, dict) else {}
        return {
            "metrics_json": metrics_json_path,
            "metrics_json_host": metrics_json_host,
            "rg_rmse": m.get("rg_rmse"),
            "gain": m.get("gain"),
            "gain_ratio": m.get("gain_ratio"),
        }
    except Exception:
        return None
