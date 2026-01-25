#!/usr/bin/env python3

# Usage:
#   python3 scripts/run_sweep_1.py

import csv
import json
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path

# ----------------------------
# USER SETTINGS (edit if needed)
# ----------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]  # repo root (…/payload_estimation)
COMPOSE = (
    f"docker compose -p payload_estimation "
    f"--project-directory {REPO_ROOT} "
    f"--env-file {REPO_ROOT}/.env "
    f"-f {REPO_ROOT}/docker-compose.yml"
)

# Services
SVC_PREPROCESS = "preprocess"
SVC_DELAN = "delan_jax"   # fixed: only delan_jax
SVC_LSTM = "lstm"
SVC_EVAL = "evaluation"

# Base paths inside shared volume (container paths)
RAW_DIR = "/workspace/shared/data/raw"
PREPROCESSED_DIR = "/workspace/shared/data/preprocessed"
PROCESSED_DIR = "/workspace/shared/data/processed"
MODELS_DELAN_DIR = "/workspace/shared/models/delan"
MODELS_LSTM_DIR = "/workspace/shared/models/lstm"
EVAL_DIR = "/workspace/shared/evaluation"

# Host path for reading DeLaN metrics.json (selection stage)
MODELS_DELAN_DIR_HOST = str(REPO_ROOT / "shared" / "models" / "delan")
MODELS_LSTM_DIR_HOST = str(REPO_ROOT / "shared" / "models" / "lstm")
EVAL_DIR_HOST = str(REPO_ROOT / "shared" / "evaluation")

# Dataset settings (your scenario)
DATASET_NAME = "UR3_Load0_cc"  # raw file: RAW_DIR/{DATASET_NAME}.csv
RUN_TAG = "A"
IN_FORMAT = "csv"
COL_FORMAT = "wide"
DERIVE_QDD = True
LOWPASS_SIGNALS = True
LOWPASS_CUTOFF_HZ = 10.0
LOWPASS_ORDER = 4
# Low-pass qdd after derivation (kept False because it showed no effect @ 100 Hz sampling).
LOWPASS_QDD = False

# Sweep
TRAJ_AMOUNTS = [8, 16, 32, 48, 64, 84, 122]
TEST_FRACTIONS = [0.2]
VAL_FRACTION = 0.1
SEEDS = [0, 1, 2]

# Window sizes and feature modes (stabilize DeLaN first)
H_LIST = [100, 150]
FEATURE_MODES = ["full"]

# DeLaN settings
DELAN_MODEL_TYPE = "structured"
DELAN_HP_PRESET = "lutter_like_256"  # or "lutter_like_256"
DELAN_HP_FLAGS = ""  # optional extra flags
DELAN_SEEDS = [0, 1, 2, 3, 4]
DELAN_EPOCHS = 50  # decoupled from K

# LSTM hyperparams (early stopping already in trainer)
LSTM_EPOCHS = 120  # max epochs; early stopping will shorten if needed
LSTM_BATCH = 64
LSTM_VAL_SPLIT = 0.1
LSTM_UNITS = 128
LSTM_DROPOUT = 0.2
LSTM_EPS = 1e-8
LSTM_NO_PLOTS = False

# Paths to scripts INSIDE the containers
SCRIPT_BUILD_DELAN_DATASET = "scripts/build_delan_dataset.py"
SCRIPT_BUILD_LSTM_WINDOWS = "scripts/build_lstm_windows.py"
SCRIPT_TRAIN_DELAN_JAX = "/workspace/delan_jax/scripts/rbyt_train_delan_jax.py"
SCRIPT_EXPORT_DELAN_RES = "/workspace/delan_jax/scripts/export_delan_residuals_jax.py"
SCRIPT_TRAIN_LSTM = "scripts/train_residual_lstm.py"
SCRIPT_EVAL = "scripts/combined_evaluation.py"


# ----------------------------
# Helper functions
# ----------------------------
def lstm_train_cmd_patched(npz: str, out_dir: str, model_name: str,
                           epochs: int, batch: int, val_split: float, seed: int,
                           units: int, dropout: float, eps: float, no_plots: bool) -> str:
    no_plots_flag = "--no_plots" if no_plots else ""

    py = (
        "import runpy\n"
        "import keras.callbacks as cb\n"
        "_Orig = cb.ModelCheckpoint\n"
        "\n"
        "class PatchedModelCheckpoint(_Orig):\n"
        "    def __init__(self, filepath, *args, **kwargs):\n"
        "        if isinstance(filepath, str) and (not filepath.endswith('.keras')) and (not filepath.endswith('.h5')):\n"
        "            filepath = filepath + '.keras'\n"
        "        super().__init__(filepath, *args, **kwargs)\n"
        "\n"
        "cb.ModelCheckpoint = PatchedModelCheckpoint\n"
        "runpy.run_path('scripts/train_residual_lstm.py', run_name='__main__')\n"
    )

    return (
        f"python3 -c {shlex.quote(py)} "
        f"--npz {npz} "
        f"--out_dir {out_dir} "
        f"--model_name {model_name} "
        f"--epochs {epochs} "
        f"--batch {batch} "
        f"--val_split {val_split} "
        f"--seed {seed} "
        f"--units {units} "
        f"--dropout {dropout} "
        f"--eps {eps} "
        f"{no_plots_flag}"
    )


def banner(lines, char="#"):
    width = max(len(s) for s in lines) if lines else 0
    bar = char * (width + 8)
    out = [bar]
    for s in lines:
        out.append(f"{char*3} {s.ljust(width)} {char*3}")
    out.append(bar)
    return "\n".join(out)


def run_cmd(cmd, log_file, also_print=True):
    if also_print:
        print(cmd)
    log_file.write("\n" + banner([cmd], char="=") + "\n")
    log_file.flush()

    p = subprocess.run(cmd, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_file.write(p.stdout + "\n")
    log_file.flush()
    if also_print:
        print(p.stdout)

    if p.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {p.returncode}")


def compose_exec(service, inner_cmd):
    return f"{COMPOSE} exec -T {service} bash -lc {shlex.quote(inner_cmd)}"


def safe_tag(x):
    return str(x).replace(".", "p")

def fmt_hp(x: float) -> str:
    if x == 0:
        return "0"
    ax = abs(x)
    if ax < 1e-2 or ax >= 1e2:
        s = f"{x:.0e}"
    else:
        s = f"{x:g}"
    s = s.replace("+", "")
    s = s.replace("e-0", "e-").replace("e+0", "e")
    return s.replace(".", "p")

def parse_hp_flags(flag_str: str) -> dict:
    if not flag_str.strip():
        return {}
    toks = shlex.split(flag_str)
    out = {}
    for i, t in enumerate(toks):
        if t == "--n_width" and i + 1 < len(toks):
            out["n_width"] = int(toks[i + 1])
        if t == "--n_depth" and i + 1 < len(toks):
            out["n_depth"] = int(toks[i + 1])
        if t == "--batch" and i + 1 < len(toks):
            out["n_minibatch"] = int(toks[i + 1])
        if t == "--lr" and i + 1 < len(toks):
            out["learning_rate"] = float(toks[i + 1])
        if t == "--wd" and i + 1 < len(toks):
            out["weight_decay"] = float(toks[i + 1])
        if t == "--activation" and i + 1 < len(toks):
            out["activation"] = str(toks[i + 1])
    return out

def hp_suffix_from_preset(preset: str, hp_flags: str) -> str:
    base = {
        "n_width": 64,
        "n_depth": 2,
        "n_minibatch": 512,
        "activation": "tanh",
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
    }
    presets = {
        "default": {},
        "fast_debug": {"n_minibatch": 256, "n_width": 64, "n_depth": 2, "learning_rate": 3e-4},
        "long_train": {"n_minibatch": 512, "n_width": 128, "n_depth": 3, "learning_rate": 1e-4, "weight_decay": 1e-5},
        "lutter_like": {"activation": "softplus", "n_minibatch": 1024, "n_width": 128, "n_depth": 2, "learning_rate": 1e-4, "weight_decay": 1e-5},
        "lutter_like_256": {"activation": "softplus", "n_minibatch": 1024, "n_width": 256, "n_depth": 2, "learning_rate": 1e-4, "weight_decay": 1e-5},
    }
    hp = dict(base)
    hp.update(presets.get(preset, {}))
    hp.update(parse_hp_flags(hp_flags))
    act = str(hp["activation"])
    return (
        f"act{act}_b{hp['n_minibatch']}_lr{fmt_hp(hp['learning_rate'])}"
        f"_wd{fmt_hp(hp['weight_decay'])}_w{hp['n_width']}_d{hp['n_depth']}"
    )

def read_delan_rmse_pair(metrics_json_path: str) -> tuple[float, float]:
    if not os.path.exists(metrics_json_path):
        return float("inf"), float("inf")
    try:
        with open(metrics_json_path, "r") as f:
            d = json.load(f)
        val_rmse = float(d.get("eval_val", {}).get("torque_rmse", float("inf")))
        test_rmse = float(d.get("eval_test", {}).get("torque_rmse", float("inf")))
        return val_rmse, test_rmse
    except Exception:
        return float("inf"), float("inf")

def read_delan_metrics(metrics_json_path: str) -> dict:
    if not os.path.exists(metrics_json_path):
        return {"exists": False}
    try:
        with open(metrics_json_path, "r") as f:
            d = json.load(f)
        return {
            "exists": True,
            "val_rmse": float(d.get("eval_val", {}).get("torque_rmse", float("inf"))),
            "val_mse": float(d.get("eval_val", {}).get("torque_mse", float("inf"))),
            "test_rmse": float(d.get("eval_test", {}).get("torque_rmse", float("inf"))),
            "test_mse": float(d.get("eval_test", {}).get("torque_mse", float("inf"))),
        }
    except Exception:
        return {"exists": False}

def read_lstm_metrics(metrics_json_path: str) -> dict:
    if not os.path.exists(metrics_json_path):
        return {"exists": False}
    try:
        with open(metrics_json_path, "r") as f:
            d = json.load(f)
        train = d.get("train", {}) if isinstance(d, dict) else {}
        eval_test = d.get("eval_test", {}) if isinstance(d, dict) else {}
        return {
            "exists": True,
            "epochs_ran": train.get("epochs_ran"),
            "best_epoch": train.get("best_epoch"),
            "best_val_loss": train.get("best_val_loss"),
            "final_train_loss": train.get("final_train_loss"),
            "final_val_loss": train.get("final_val_loss"),
            "rmse_total": eval_test.get("rmse_total"),
            "mse_total": eval_test.get("mse_total"),
            "best_model_path": eval_test.get("best_model_path"),
        }
    except Exception:
        return {"exists": False}

def append_csv_row(path: str, fieldnames: list[str], row: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)

def append_jsonl(path: str, record: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = Path("logs_sweeps")
    logs_dir.mkdir(parents=True, exist_ok=True)

    raw_csv = f"{RAW_DIR}/{DATASET_NAME}.{IN_FORMAT}"

    master_logs = []

    lowpass_qdd = LOWPASS_QDD
    lowpass_tag = f"lpqdd{'T' if lowpass_qdd else 'F'}"
    master_log_path = logs_dir / f"sweep1_{DATASET_NAME}_{RUN_TAG}_{lowpass_tag}_{ts}.log"
    metrics_csv = f"{EVAL_DIR}/summary_metrics_sweep_1_{lowpass_tag}_{ts}.csv"
    metrics_jsonl = f"{EVAL_DIR}/summary_metrics_sweep_1_{lowpass_tag}_{ts}.jsonl"

    delan_summary_csv = str(Path(EVAL_DIR_HOST) / f"summary_delan_sweep_1_{lowpass_tag}_{ts}.csv")
    delan_summary_jsonl = str(Path(EVAL_DIR_HOST) / f"summary_delan_sweep_1_{lowpass_tag}_{ts}.jsonl")
    lstm_summary_csv = str(Path(EVAL_DIR_HOST) / f"summary_lstm_sweep_1_{lowpass_tag}_{ts}.csv")
    lstm_summary_jsonl = str(Path(EVAL_DIR_HOST) / f"summary_lstm_sweep_1_{lowpass_tag}_{ts}.jsonl")

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
                    base_id = f"{DATASET_NAME}__{RUN_TAG}__{lowpass_tag}"

                    # preprocess output NPZ (unique per lowpass/K/tf/seed)
                    delan_npz_name = (
                        f"delan_{DATASET_NAME}_{lowpass_tag}_K{K}_tf{safe_tag(tf)}"
                        f"_vf{safe_tag(VAL_FRACTION)}_seed{seed}_dataset.npz"
                    )
                    npz_stem = Path(delan_npz_name).stem
                    npz_in = f"{PREPROCESSED_DIR}/{npz_stem}/{npz_stem}.npz"

                    run_log_path = logs_dir / f"run1_{lowpass_tag}_K{K}_tf{safe_tag(tf)}_seed{seed}_{ts}.log"
                    with run_log_path.open("w", encoding="utf-8") as run_log:
                        run_log.write(banner([
                            f"RUN: lowpass_qdd={lowpass_qdd}  K={K} test_fraction={tf} seed={seed}",
                            f"npz_in={npz_in}",
                        ], char="#") + "\n")

                        # ---------- 1) Preprocess ----------
                        run_log.write("\n" + banner(["1) PREPROCESS"], char="#") + "\n")
                        cmd = compose_exec(
                            SVC_PREPROCESS,
                            f"python3 {SCRIPT_BUILD_DELAN_DATASET} "
                            f"--derive_qdd_from_qd {str(DERIVE_QDD)} "
                            f"--col_format {COL_FORMAT} "
                            f"--trajectory_amount {K} "
                            f"--test_fraction {tf} "
                            f"--val_fraction {VAL_FRACTION} "
                            f"--seed {seed} "
                            f"--lowpass_signals {LOWPASS_SIGNALS} "
                            f"--lowpass_cutoff_hz {LOWPASS_CUTOFF_HZ} "
                            f"--lowpass_order {LOWPASS_ORDER} "
                            f"--lowpass_qdd {lowpass_qdd} "
                            f"--raw_csv {raw_csv} "
                            f"--out_npz {npz_in}"
                        )
                        run_cmd(cmd, run_log)

                        if True:
                            # ---------- 2) DeLaN Train (all seeds) ----------
                            hp_suffix = hp_suffix_from_preset(DELAN_HP_PRESET, DELAN_HP_FLAGS)
                            delan_candidates = []
                            for delan_seed in DELAN_SEEDS:
                                model_short = "struct"
                                delan_tag = f"delan_jax_{model_short}_s{delan_seed}_ep{DELAN_EPOCHS}_{hp_suffix}"
                                delan_id = f"{DATASET_NAME}__{RUN_TAG}__{lowpass_tag}__{delan_tag}"
                                delan_run_dir = f"{MODELS_DELAN_DIR}/{delan_id}"
                                ckpt = f"{delan_run_dir}/{delan_id}.jax"

                                run_log.write(banner([
                                    f"DELAN VARIANT: delan_seed={delan_seed}",
                                    f"ckpt={ckpt}",
                                ], char="#") + "\n")

                                run_log.write("\n" + banner(["2) DELAN TRAIN"], char="#") + "\n")
                                hp_flags = (DELAN_HP_FLAGS + " ") if DELAN_HP_FLAGS.strip() else ""
                                cmd = compose_exec(
                                    SVC_DELAN,
                                    f"python3 {SCRIPT_TRAIN_DELAN_JAX} "
                                    f"--npz {npz_in} "
                                    f"-t {DELAN_MODEL_TYPE} "
                                    f"-s {delan_seed} "
                                    f"-r 0 "
                                    f"--hp_preset {DELAN_HP_PRESET} "
                                    f"--epochs {DELAN_EPOCHS} "
                                    f"{hp_flags}"
                                    f"--save_path {ckpt}"
                                )
                                run_cmd(cmd, run_log)

                                metrics_json = f"{MODELS_DELAN_DIR_HOST}/{delan_id}/metrics.json"
                                val_rmse, test_rmse = read_delan_rmse_pair(metrics_json)
                                delan_candidates.append({
                                    "delan_seed": delan_seed,
                                    "delan_tag": delan_tag,
                                    "delan_id": delan_id,
                                    "ckpt": ckpt,
                                    "val_rmse": val_rmse,
                                    "test_rmse": test_rmse,
                                    "metrics_json": metrics_json,
                                })

                            # ---------- 3) Select best DeLaN ----------
                            delan_candidates.sort(key=lambda d: d["val_rmse"])
                            best = delan_candidates[0]
                            run_log.write("\n" + banner([
                                "3) SELECT BEST DELAN (by val RMSE)",
                                f"best_seed={best['delan_seed']} val_rmse={best['val_rmse']} test_rmse={best['test_rmse']}",
                                f"metrics_json={best['metrics_json']}",
                            ], char="#") + "\n")

                            # DeLaN sweep summary (host-written): one row per candidate, mark best
                            delan_fields = [
                                "timestamp",
                                "dataset",
                                "run_tag",
                                "lowpass_qdd",
                                "lowpass_tag",
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
                                    "lowpass_tag": lowpass_tag,
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
                                    "metrics_exists": bool(m.get("exists", False)),
                                    "val_rmse": m.get("val_rmse", cand.get("val_rmse")),
                                    "val_mse": m.get("val_mse"),
                                    "test_rmse": m.get("test_rmse", cand.get("test_rmse")),
                                    "test_mse": m.get("test_mse"),
                                    "selected": cand["delan_seed"] == best["delan_seed"],
                                }
                                append_csv_row(delan_summary_csv, delan_fields, row)
                                append_jsonl(delan_summary_jsonl, row)

                            # ---------- 4) Export residuals (best only) ----------
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

                            # ---------- 5-7) LSTM windows -> train -> eval ----------
                            for H in H_LIST:
                                for feat in FEATURE_MODES:
                                    run_log.write("\n" + banner([
                                        f"BLOCK: H={H} feat={feat}"
                                    ], char="#") + "\n")

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
                                    run_log.write("\n" + banner(["5) BUILD LSTM WINDOWS"], char="#") + "\n")
                                    cmd = compose_exec(
                                        SVC_PREPROCESS,
                                        f"python3 {SCRIPT_BUILD_LSTM_WINDOWS} "
                                        f"--in_npz {res_out} "
                                        f"--out_npz {win_out} "
                                        f"--H {H} "
                                        f"--features {feat}"
                                    )
                                    run_cmd(cmd, run_log)

                                    # 6) Train LSTM
                                    run_log.write("\n" + banner(["6) TRAIN LSTM"], char="#") + "\n")
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
                                    )
                                    cmd = compose_exec(SVC_LSTM, inner)
                                    run_cmd(cmd, run_log)

                                    # LSTM sweep summary (host-written): 1 row per (K/tf/seed/H/feat)
                                    lstm_metrics_path = (
                                        f"{MODELS_LSTM_DIR_HOST}/{lstm_dir_name}/metrics_train_test_H{H}.json"
                                    )
                                    lm = read_lstm_metrics(lstm_metrics_path)
                                    lstm_fields = [
                                        "timestamp",
                                        "dataset",
                                        "run_tag",
                                        "lowpass_qdd",
                                        "lowpass_tag",
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
                                        "lowpass_tag": lowpass_tag,
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
                                    run_log.write("\n" + banner(["7) COMBINED EVALUATION"], char="#") + "\n")
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
                                        f"--metrics_csv {metrics_csv} "
                                        f"--metrics_json {metrics_jsonl} "
                                        f"--K {K} "
                                        f"--test_fraction {tf} "
                                        f"--seed {seed} "
                                        f"--delan_seed {best['delan_seed']} "
                                        f"--delan_epochs {DELAN_EPOCHS} "
                                        f"--hp_preset {DELAN_HP_PRESET} "
                                        f"--delan_rmse_val {best['val_rmse']} "
                                        f"--delan_rmse_test {best['test_rmse']}"
                                    )
                                    run_cmd(cmd, run_log)

                        master_log.write(f"\n[OK] lowpass_qdd={lowpass_qdd} K={K} tf={tf} seed={seed}  log={run_log_path}\n")
                        master_log.flush()

            master_log.write("\n" + banner(["Sweep1 finished OK"], char="#") + "\n")

    for p in master_logs:
        print(f"\nMASTER LOG: {p}")
    print("Done.")


if __name__ == "__main__":
    main()
