#!/usr/bin/env python3

#usage: python3 scripts/run_full_sweep.py

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
SEEDS = [0, 1, 2]  # 3 runs

H_LIST = [25, 50]
FEATURE_MODES = ["full", "tau_hat", "state", "state_tauhat"]

# Fixed choices you stated
DELAN_MODEL_TYPE = "structured"  # only structured
DELAN_HP_PRESET = "fast_debug"   # change if you want e.g. "paper" / etc. #scales with sample/trajectory amount

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
LSTM_NO_PLOTS = False # set True to skip plotting inside container

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
            f"val_fraction={VAL_FRACTION}",
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
                    delan_npz_name = f"delan_{DATASET_NAME}_K{K}_tf{safe_tag(tf)}_vf{safe_tag(VAL_FRACTION)}_seed{seed}_dataset.npz"
                    npz_stem = Path(delan_npz_name).stem
                    npz_in = f"{PREPROCESSED_DIR}/{npz_stem}/{npz_stem}.npz"

                    # per-run log
                    run_log_path = logs_dir / f"run_K{K}_tf{safe_tag(tf)}_seed{seed}_{ts}.log"
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
                            f"--derive_qdd_from_qd {str(DERIVE_QDD)} "
                            f"--col_format {COL_FORMAT} "
                            f"--trajectory_amount {K} "
                            f"--test_fraction {tf} "
                            f"--val_fraction {VAL_FRACTION} "
                            f"--seed {seed} "
                            f"--raw_csv {raw_csv} "
                            f"--out_npz {npz_in}"
                        )
                        run_cmd(cmd, run_log)

                    # ---------- 2) DeLaN Train ----------
                    hp_suffix = hp_suffix_from_preset(DELAN_HP_PRESET, DELAN_HP_FLAGS)
                    for delan_seed in DELAN_SEEDS:
                        # delan tag/id (depends on delan_seed)
                        model_short = "struct"
                        delan_tag = f"delan_jax_{model_short}_s{delan_seed}_ep{delan_epochs}_{hp_suffix}"
                        delan_id = f"{DATASET_NAME}__{RUN_TAG}__{delan_tag}"
                        delan_run_dir = f"{MODELS_DELAN_DIR}/{delan_id}"
                        ckpt = f"{delan_run_dir}/{delan_id}.jax"

                        # residual output (depends on delan_tag / delan_seed)
                        residual_name = f"{base_id}__K{K}_tf{safe_tag(tf)}_vf{safe_tag(VAL_FRACTION)}__residual__{delan_tag}.npz"
                        res_out = f"{PROCESSED_DIR}/{residual_name}"

                        run_log.write(banner([
                            f"DELAN VARIANT: delan_seed={delan_seed}",
                            f"ckpt={ckpt}",
                            f"res_out={res_out}",
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
                            f"--epochs {delan_epochs} "
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
                                    f"{base_id}__K{K}_tf{safe_tag(tf)}_vf{safe_tag(VAL_FRACTION)}__lstm_windows_H{H}"
                                    f"__feat_{feat}__{delan_tag}.npz"
                                )
                                win_out = f"{PROCESSED_DIR}/{windows_npz_name}"

                                # LSTM output dir naming (keep your style, but make it unique)
                                lstm_dir_name = (
                                    f"{base_id}__K{K}_tf{safe_tag(tf)}_vf{safe_tag(VAL_FRACTION)}__{delan_tag}"
                                    f"__feat_{feat}__lstm_s{seed}_H{H}"
                                    f"_ep{lstm_epochs}_b{LSTM_BATCH}_u{LSTM_UNITS}_do{safe_tag(LSTM_DROPOUT)}"
                                )
                                lstm_out = f"{MODELS_LSTM_DIR}/{lstm_dir_name}"
                                lstm_model_name = lstm_model_name = "residual_lstm.keras"
                                lstm_model_path = f"{lstm_out}/{lstm_model_name}"        # already includes .keras
                                lstm_scalers_path = f"{lstm_out}/scalers_H{H}.npz"

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
                                inner = lstm_train_cmd_patched(
                                    npz=win_out,
                                    out_dir=lstm_out,
                                    model_name=lstm_model_name,
                                    epochs=lstm_epochs,
                                    batch=LSTM_BATCH,
                                    val_split=LSTM_VAL_SPLIT,
                                    seed=4,
                                    units=LSTM_UNITS,
                                    dropout=LSTM_DROPOUT,
                                    eps=LSTM_EPS,
                                    no_plots=LSTM_NO_PLOTS,
                                )

                                cmd = compose_exec(SVC_LSTM, inner)

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
