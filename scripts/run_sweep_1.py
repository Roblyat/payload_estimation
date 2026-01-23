#!/usr/bin/env python3

# Usage:
#   python3 scripts/run_sweep_1.py

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

# Dataset settings (your scenario)
DATASET_NAME = "delan_UR3_Load0_combined_26"  # raw file: RAW_DIR/{DATASET_NAME}.csv
RUN_TAG = "A"
IN_FORMAT = "csv"
COL_FORMAT = "wide"
DERIVE_QDD = True

# Sweep
TRAJ_AMOUNTS = [25, 50, 75, 100, 150]
TEST_FRACTIONS = [0.2, 0.3]
VAL_FRACTION = 0.1
SEEDS = [0, 1, 2]

# Window sizes and feature modes (stabilize DeLaN first)
H_LIST = [100, 150]
FEATURE_MODES = ["full"]

# DeLaN settings
DELAN_MODEL_TYPE = "structured"
DELAN_HP_PRESET = "lutter_like"  # or "lutter_like_256"
DELAN_HP_FLAGS = ""  # optional extra flags
DELAN_SEEDS = [0, 1, 2]
DELAN_EPOCHS = 300  # decoupled from K

# LSTM hyperparams (early stopping already in trainer)
LSTM_EPOCHS = 120  # max epochs; early stopping will shorten if needed
LSTM_BATCH = 256
LSTM_VAL_SPLIT = 0.2
LSTM_UNITS = 128
LSTM_DROPOUT = 0.1
LSTM_EPS = 1e-8
LSTM_NO_PLOTS = False

# Summary metrics output (single source of truth)
METRICS_CSV = f"{EVAL_DIR}/summary_metrics_sweep_1.csv"
METRICS_JSON = f"{EVAL_DIR}/summary_metrics_sweep_1.jsonl"

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

def read_delan_rmse(metrics_json_path: str) -> float:
    if not os.path.exists(metrics_json_path):
        return float("inf")
    try:
        with open(metrics_json_path, "r") as f:
            d = json.load(f)
        if "eval_val" in d:
            return float(d.get("eval_val", {}).get("torque_rmse", float("inf")))
        return float(d.get("eval_test", {}).get("torque_rmse", float("inf")))
    except Exception:
        return float("inf")


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = Path("logs_sweeps")
    logs_dir.mkdir(parents=True, exist_ok=True)
    master_log_path = logs_dir / f"sweep1_{DATASET_NAME}_{RUN_TAG}_{ts}.log"

    raw_csv = f"{RAW_DIR}/{DATASET_NAME}.{IN_FORMAT}"

    with master_log_path.open("w", encoding="utf-8") as master_log:
        master_log.write(banner([
            f"Sweep1 started: {ts}",
            f"dataset={DATASET_NAME} run_tag={RUN_TAG}",
            f"raw_csv={raw_csv}",
            f"col_format={COL_FORMAT} derive_qdd={DERIVE_QDD}",
            f"traj_amounts={TRAJ_AMOUNTS} test_fracs={TEST_FRACTIONS} seeds={SEEDS}",
            f"val_fraction={VAL_FRACTION}",
            f"H={H_LIST} feature_modes={FEATURE_MODES}",
            f"DeLaN: backend=jax type={DELAN_MODEL_TYPE} hp_preset={DELAN_HP_PRESET} epochs={DELAN_EPOCHS}",
            f"LSTM: epochs_max={LSTM_EPOCHS} batch={LSTM_BATCH} val_split={LSTM_VAL_SPLIT}",
            f"metrics_csv={METRICS_CSV}",
            f"metrics_json={METRICS_JSON}",
        ], char="#") + "\n")

        for K in TRAJ_AMOUNTS:
            for tf in TEST_FRACTIONS:
                master_log.write("\n" + banner([
                    f"SUBSWEEP: K={K} test_fraction={tf}"
                ], char="#") + "\n")
                master_log.flush()

                for seed in SEEDS:
                    base_id = f"{DATASET_NAME}__{RUN_TAG}"

                    # preprocess output NPZ (unique per K/tf/seed)
                    delan_npz_name = f"delan_{DATASET_NAME}_K{K}_tf{safe_tag(tf)}_vf{safe_tag(VAL_FRACTION)}_seed{seed}_dataset.npz"
                    npz_in = f"{PREPROCESSED_DIR}/{delan_npz_name}"

                    run_log_path = logs_dir / f"run1_K{K}_tf{safe_tag(tf)}_seed{seed}_{ts}.log"
                    with run_log_path.open("w", encoding="utf-8") as run_log:
                        run_log.write(banner([
                            f"RUN: K={K} test_fraction={tf} seed={seed}",
                            f"npz_in={npz_in}",
                        ], char="#") + "\n")

                        # ---------- 1) Preprocess ----------
                        run_log.write("\n" + banner(["1) PREPROCESS"], char="#") + "\n")
                        cmd = compose_exec(
                            SVC_PREPROCESS,
                            f"python3 {SCRIPT_BUILD_DELAN_DATASET} "
                            f"--qdd {str(DERIVE_QDD)} "
                            f"--col_format {COL_FORMAT} "
                            f"--trajectory_amount {K} "
                            f"--test_fraction {tf} "
                            f"--val_fraction {VAL_FRACTION} "
                            f"--seed {seed} "
                            f"--raw_csv {raw_csv} "
                            f"--out_npz {npz_in}"
                        )
                        run_cmd(cmd, run_log)

                        # ---------- 2) DeLaN Train (all seeds) ----------
                        hp_suffix = hp_suffix_from_preset(DELAN_HP_PRESET, DELAN_HP_FLAGS)
                        delan_candidates = []
                        for delan_seed in DELAN_SEEDS:
                            model_short = "struct"
                            delan_tag = f"delan_jax_{model_short}_s{delan_seed}_ep{DELAN_EPOCHS}_{hp_suffix}"
                            delan_id = f"{DATASET_NAME}__{RUN_TAG}__{delan_tag}"
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
                            rmse = read_delan_rmse(metrics_json)
                            delan_candidates.append({
                                "delan_seed": delan_seed,
                                "delan_tag": delan_tag,
                                "delan_id": delan_id,
                                "ckpt": ckpt,
                                "rmse": rmse,
                                "metrics_json": metrics_json,
                            })

                        # ---------- 3) Select best DeLaN ----------
                        delan_candidates.sort(key=lambda d: d["rmse"])
                        best = delan_candidates[0]
                        run_log.write("\n" + banner([
                            "3) SELECT BEST DELAN (test RMSE proxy for val)",
                            f"best_seed={best['delan_seed']} rmse={best['rmse']}",
                            f"metrics_json={best['metrics_json']}",
                        ], char="#") + "\n")

                        # ---------- 4) Export residuals (best only) ----------
                        residual_name = f"{base_id}__K{K}_tf{safe_tag(tf)}_vf{safe_tag(VAL_FRACTION)}__residual__{best['delan_tag']}.npz"
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
                                    f"--metrics_csv {METRICS_CSV} "
                                    f"--metrics_json {METRICS_JSON} "
                                    f"--K {K} "
                                    f"--test_fraction {tf} "
                                    f"--seed {seed} "
                                    f"--delan_seed {best['delan_seed']} "
                                    f"--delan_epochs {DELAN_EPOCHS} "
                                    f"--hp_preset {DELAN_HP_PRESET} "
                                    f"--delan_rmse_val {best['rmse']}"
                                )
                                run_cmd(cmd, run_log)

                    master_log.write(f"\n[OK] K={K} tf={tf} seed={seed}  log={run_log_path}\n")
                    master_log.flush()

        master_log.write("\n" + banner(["Sweep1 finished OK"], char="#") + "\n")

    print(f"\nMASTER LOG: {master_log_path}")
    print("Done.")


if __name__ == "__main__":
    main()
