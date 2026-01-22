#!/usr/bin/env python3
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path

# ----------------------------
# USER SETTINGS (edit if needed)
# ----------------------------

COMPOSE = (
    "docker compose -p payload_estimation "
    "--project-directory /workspace "
    "--env-file /workspace/.env "
    "-f /workspace/docker-compose.yml"
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

# Dataset settings (your scenario)
DATASET_NAME = "delan_UR3_Load0_combined_26"  # raw file: RAW_DIR/{DATASET_NAME}.csv
RUN_TAG = "A"
IN_FORMAT = "csv"
COL_FORMAT = "wide"
DERIVE_QDD = True

# Sweep
TRAJ_AMOUNTS = [25, 50, 75, 100, 150]
TEST_FRACTIONS = [0.2, 0.3]
SEEDS = [0, 1, 2]  # 3 runs

H_LIST = [25, 50]
FEATURE_MODES = ["full", "tau_hat", "state", "state_tauhat"]

# Fixed choices you stated
DELAN_MODEL_TYPE = "structured"  # only structured
DELAN_HP_PRESET = "fast_debug"   # change if you want e.g. "paper" / etc.

# If you want to pass manual delan flags (optional), put them here:
DELAN_HP_FLAGS = ""  # e.g. "--n_width 64 --n_depth 2 ..."#
DELAN_SEEDS = [0, 1, 2]

# LSTM hyperparams (choose defaults; adjust later if you want)
LSTM_EPOCHS = 50
LSTM_BATCH = 256
LSTM_VAL_SPLIT = 0.2
LSTM_UNITS = 128
LSTM_DROPOUT = 0.1
LSTM_EPS = 1e-8
LSTM_NO_PLOTS = True

# Paths to scripts INSIDE the containers
# preprocess container:
SCRIPT_BUILD_DELAN_DATASET = "scripts/build_delan_dataset.py"
SCRIPT_BUILD_LSTM_WINDOWS = "scripts/build_lstm_windows.py"

# delan_jax container:
SCRIPT_TRAIN_DELAN_JAX = "/workspace/delan_jax/scripts/rbyt_train_delan_jax.py"
SCRIPT_EXPORT_DELAN_RES = "/workspace/delan_jax/scripts/export_delan_residuals_jax.py"

# lstm container:
SCRIPT_TRAIN_LSTM = "scripts/train_residual_lstm.py"

# evaluation container:
SCRIPT_EVAL = "scripts/combined_evaluation.py"


# ----------------------------
# Helper functions
# ----------------------------

def delan_epochs_for(K: int) -> int:
    # conservative scaling: small K trains faster, big K trains longer
    if K <= 25:  return 150
    if K <= 50:  return 200
    if K <= 75:  return 250
    if K <= 100: return 300
    return 400  # 150+

def lstm_epochs_for(K: int) -> int:
    if K <= 25:  return 30
    if K <= 50:  return 50
    if K <= 75:  return 60
    if K <= 100: return 80
    return 100

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
    # -T to avoid TTY issues in CI-like runs
    return f"{COMPOSE} exec -T {service} bash -lc {shlex.quote(inner_cmd)}"

def safe_tag(x):
    return str(x).replace(".", "p")

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = Path("logs_sweeps")
    logs_dir.mkdir(parents=True, exist_ok=True)
    master_log_path = logs_dir / f"sweep_{DATASET_NAME}_{RUN_TAG}_{ts}.log"

    raw_csv = f"{RAW_DIR}/{DATASET_NAME}.{IN_FORMAT}"

    with master_log_path.open("w", encoding="utf-8") as master_log:
        master_log.write(banner([
            f"Sweep started: {ts}",
            f"dataset={DATASET_NAME} run_tag={RUN_TAG}",
            f"raw_csv={raw_csv}",
            f"col_format={COL_FORMAT} derive_qdd={DERIVE_QDD}",
            f"traj_amounts={TRAJ_AMOUNTS} test_fracs={TEST_FRACTIONS} seeds={SEEDS}",
            f"H={H_LIST} feature_modes={FEATURE_MODES}",
            f"DeLaN: backend=jax type={DELAN_MODEL_TYPE} hp_preset={DELAN_HP_PRESET}",
        ], char="#") + "\n")

        for K in TRAJ_AMOUNTS:
            delan_epochs = delan_epochs_for(K)
            lstm_epochs = lstm_epochs_for(K)

            for tf in TEST_FRACTIONS:
                master_log.write("\n" + banner([
                    f"SUBSWEEP: K={K} test_fraction={tf}"
                ], char="#") + "\n")
                master_log.flush()

                for seed in SEEDS:
                    # ---------- Naming ----------
                    base_id = f"{DATASET_NAME}__{RUN_TAG}"

                    # preprocess output NPZ (unique per K/tf/seed)
                    delan_npz_name = f"delan_{DATASET_NAME}_K{K}_tf{safe_tag(tf)}_seed{seed}_dataset.npz"
                    npz_in = f"{PREPROCESSED_DIR}/{delan_npz_name}"

                    # delan tag/id
                    model_short = "struct"
                    delan_tag = f"delan_jax_{model_short}_s{delan_seed}_ep{delan_epochs}"
                    delan_id = f"{DATASET_NAME}__{RUN_TAG}__{delan_tag}"
                    delan_run_dir = f"{MODELS_DELAN_DIR}/{delan_id}"
                    ckpt = f"{delan_run_dir}/{delan_id}.jax"

                    # residual output (unique per K/tf/seed)
                    residual_name = f"{base_id}__K{K}_tf{safe_tag(tf)}__residual__{delan_tag}.npz"
                    res_out = f"{PROCESSED_DIR}/{residual_name}"

                    # per-run log
                    run_log_path = logs_dir / f"run_K{K}_tf{safe_tag(tf)}_seed{seed}_{ts}.log"
                    with run_log_path.open("w", encoding="utf-8") as run_log:
                        run_log.write(banner([
                            f"RUN: K={K} test_fraction={tf} seed={seed}",
                            f"npz_in={npz_in}",
                            f"ckpt={ckpt}",
                            f"res_out={res_out}",
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
                            f"--seed {seed} "
                            f"--raw_csv {raw_csv} "
                            f"--out_npz {npz_in}"
                        )
                        run_cmd(cmd, run_log)

                        # ---------- 2) DeLaN Train ----------
                        for delan_seed in DELAN_SEEDS:
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
                                f"--max_epoch {delan_epochs} "
                                f"{hp_flags}"
                                f"--save_path {ckpt}"
                            )
                            run_cmd(cmd, run_log)

                            # ---------- 3) Export residuals ----------
                            run_log.write("\n" + banner(["3) EXPORT RESIDUALS"], char="#") + "\n")
                            cmd = compose_exec(
                                SVC_DELAN,
                                f"python3 {SCRIPT_EXPORT_DELAN_RES} "
                                f"--npz_in {npz_in} "
                                f"--ckpt {ckpt} "
                                f"--out {res_out}"
                            )
                            run_cmd(cmd, run_log)

                            # ---------- 4-6) LSTM windows -> train -> eval ----------
                            for H in H_LIST:
                                for feat in FEATURE_MODES:
                                    run_log.write("\n" + banner([
                                        f"BLOCK: H={H} feat={feat}"
                                    ], char="#") + "\n")

                                    windows_npz_name = (
                                        f"{base_id}__K{K}_tf{safe_tag(tf)}__lstm_windows_H{H}"
                                        f"__feat_{feat}__delan_jax_seed{seed}.npz"
                                    )
                                    win_out = f"{PROCESSED_DIR}/{windows_npz_name}"

                                    # LSTM output dir naming (keep your style, but make it unique)
                                    lstm_dir_name = (
                                        f"{base_id}__K{K}_tf{safe_tag(tf)}__{delan_tag}"
                                        f"__feat_{feat}__lstm_s{seed}_H{H}"
                                        f"_ep{lstm_epochs}_b{LSTM_BATCH}_u{LSTM_UNITS}_do{safe_tag(LSTM_DROPOUT)}"
                                    )
                                    lstm_out = f"{MODELS_LSTM_DIR}/{lstm_dir_name}"
                                    lstm_model_name = "residual_lstm"
                                    lstm_model_path = f"{lstm_out}/{lstm_model_name}.pt"
                                    lstm_scalers_path = f"{lstm_out}/scalers.npz"

                                    eval_out = f"{EVAL_DIR}/{lstm_dir_name}"

                                    # 4) Build windows
                                    run_log.write("\n" + banner(["4) BUILD LSTM WINDOWS"], char="#") + "\n")
                                    cmd = compose_exec(
                                        SVC_PREPROCESS,
                                        f"python3 {SCRIPT_BUILD_LSTM_WINDOWS} "
                                        f"--in_npz {res_out} "
                                        f"--out_npz {win_out} "
                                        f"--H {H} "
                                        f"--features {feat}"
                                    )
                                    run_cmd(cmd, run_log)

                                    # 5) Train LSTM
                                    run_log.write("\n" + banner(["5) TRAIN LSTM"], char="#") + "\n")
                                    no_plots_flag = "--no_plots" if LSTM_NO_PLOTS else ""
                                    cmd = compose_exec(
                                        SVC_LSTM,
                                        f"python3 {SCRIPT_TRAIN_LSTM} "
                                        f"--npz {win_out} "
                                        f"--out_dir {lstm_out} "
                                        f"--model_name {lstm_model_name} "
                                        f"--epochs {lstm_epochs} "
                                        f"--batch {LSTM_BATCH} "
                                        f"--val_split {LSTM_VAL_SPLIT} "
                                        f"--seed {seed} "
                                        f"--units {LSTM_UNITS} "
                                        f"--dropout {LSTM_DROPOUT} "
                                        f"--eps {LSTM_EPS} "
                                        f"{no_plots_flag}"
                                    )
                                    run_cmd(cmd, run_log)

                                    # 6) Combined evaluation
                                    run_log.write("\n" + banner(["6) COMBINED EVALUATION"], char="#") + "\n")
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
                                        f"--save_pred_npz"
                                    )
                                    run_cmd(cmd, run_log)

                    # append run log path to master
                    master_log.write(f"\n[OK] K={K} tf={tf} seed={seed}  log={run_log_path}\n")
                    master_log.flush()

        master_log.write("\n" + banner(["Sweep finished OK"], char="#") + "\n")

    print(f"\nMASTER LOG: {master_log_path}")
    print("Done.")


if __name__ == "__main__":
    main()
