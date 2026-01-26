from __future__ import annotations

from pathlib import Path

from cfg.sweep_config import default_sweep_config

CFG = default_sweep_config()

# ----------------------------
# USER SETTINGS (edit in services/sweep/cfg/sweep_config.py)
# ----------------------------

REPO_ROOT = CFG.repo_root / "payload_estimation"
COMPOSE = (
    f"docker compose -p payload_estimation "
    f"--project-directory {REPO_ROOT} "
    f"--env-file {REPO_ROOT}/.env "
    f"-f {REPO_ROOT}/docker-compose.yml"
)

# Services
SVC_PREPROCESS = "preprocess"
SVC_DELAN = "delan_jax"
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
DATASET_NAME = CFG.dataset_name  # raw file: RAW_DIR/{DATASET_NAME}.csv
RUN_TAG = CFG.run_tag
IN_FORMAT = CFG.in_format
COL_FORMAT = CFG.col_format
DERIVE_QDD = CFG.derive_qdd
LOWPASS_SIGNALS = CFG.lowpass_signals
LOWPASS_CUTOFF_HZ = CFG.lowpass_cutoff_hz
LOWPASS_ORDER = CFG.lowpass_order
LOWPASS_QDD_VALUES = CFG.lowpass_qdd_values

# Sweep
TRAJ_AMOUNTS = CFG.traj_amounts
TEST_FRACTIONS = CFG.test_fractions
VAL_FRACTION = CFG.val_fraction
SEEDS = CFG.seeds

# Window sizes and feature modes
H_LIST = CFG.h_list
FEATURE_MODES = CFG.feature_modes

# DeLaN settings
DELAN_MODEL_TYPE = CFG.delan_model_type
DELAN_HP_PRESET = CFG.delan_hp_preset
DELAN_HP_FLAGS = CFG.delan_hp_flags
DELAN_SEEDS = CFG.delan_seeds
DELAN_EPOCHS = CFG.delan_epochs
DELAN_EVAL_EVERY = CFG.delan_eval_every
DELAN_LOG_EVERY = CFG.delan_log_every
DELAN_EARLY_STOP = CFG.delan_early_stop
DELAN_EARLY_STOP_PATIENCE = CFG.delan_early_stop_patience
DELAN_EARLY_STOP_MIN_DELTA = CFG.delan_early_stop_min_delta
DELAN_EARLY_STOP_WARMUP_EVALS = CFG.delan_early_stop_warmup_evals

# LSTM hyperparams
LSTM_EPOCHS = CFG.lstm_epochs
LSTM_BATCH = CFG.lstm_batch
LSTM_VAL_SPLIT = CFG.lstm_val_split
LSTM_UNITS = CFG.lstm_units
LSTM_DROPOUT = CFG.lstm_dropout
LSTM_EPS = CFG.lstm_eps
LSTM_NO_PLOTS = CFG.lstm_no_plots
LSTM_EARLY_STOP = CFG.lstm_early_stop
LSTM_EARLY_STOP_PATIENCE = CFG.lstm_early_stop_patience
LSTM_EARLY_STOP_MIN_DELTA = CFG.lstm_early_stop_min_delta
LSTM_EARLY_STOP_WARMUP_EVALS = CFG.lstm_early_stop_warmup_evals

# Plot/cleanup behavior
LOGS_DIR_HOST = CFG.logs_root_dir
if not Path(LOGS_DIR_HOST).is_absolute():
    LOGS_DIR_HOST = str(REPO_ROOT / LOGS_DIR_HOST)
CLEANUP_NON_BEST_PLOTS = CFG.cleanup_non_best_plots
DELAN_ELBOW_AGGREGATE = CFG.delan_elbow_aggregate
DELAN_ELBOW_OUT_DIR = CFG.delan_elbow_out_dir
DELAN_TORQUE_AGGREGATE = CFG.delan_torque_aggregate
DELAN_TORQUE_OUT_DIR = CFG.delan_torque_out_dir
DELAN_TORQUE_BINS = CFG.delan_torque_bins
DELAN_TORQUE_K_VALUES = CFG.delan_torque_k_values
LSTM_TRAINING_AGGREGATE = CFG.lstm_training_aggregate
LSTM_RESIDUAL_AGGREGATE = CFG.lstm_residual_aggregate
LSTM_AGGREGATE_OUT_DIR = CFG.lstm_aggregate_out_dir
LSTM_AGGREGATE_BINS = CFG.lstm_aggregate_bins
LSTM_AGGREGATE_K_VALUES = CFG.lstm_aggregate_k_values
LSTM_AGGREGATE_FEATURE = CFG.lstm_aggregate_feature
LSTM_AGGREGATE_PAD_TO_EPOCHS = CFG.lstm_aggregate_pad_to_epochs
if LSTM_AGGREGATE_PAD_TO_EPOCHS is None:
    LSTM_AGGREGATE_PAD_TO_EPOCHS = LSTM_EPOCHS
LSTM_AGGREGATE_ALIGN = CFG.lstm_aggregate_align
COMBINED_TORQUE_AGGREGATE = CFG.combined_torque_aggregate
COMBINED_TORQUE_OUT_DIR = CFG.combined_torque_out_dir
COMBINED_TORQUE_BINS = CFG.combined_torque_bins
COMBINED_TORQUE_K_VALUES = CFG.combined_torque_k_values
COMBINED_TORQUE_FEATURE = CFG.combined_torque_feature

# Paths to scripts INSIDE the containers
SCRIPT_BUILD_DELAN_DATASET = "scripts/build_delan_dataset.py"
SCRIPT_BUILD_LSTM_WINDOWS = "scripts/build_lstm_windows.py"
SCRIPT_TRAIN_DELAN_JAX = "/workspace/delan_jax/scripts/rbyt_train_delan_jax.py"
SCRIPT_EXPORT_DELAN_RES = "/workspace/delan_jax/scripts/export_delan_residuals_jax.py"
SCRIPT_TRAIN_LSTM = "scripts/train_residual_lstm.py"
SCRIPT_EVAL = "scripts/combined_evaluation.py"
SCRIPT_DELAN_ELBOWS = "scripts/delan_training_dynamics_aggregate.py"
SCRIPT_DELAN_TORQUE_AGG = "scripts/delan_torque_rmse_aggregate.py"
SCRIPT_LSTM_TRAINING_AGG = "scripts/lstm_training_dynamics_aggregate.py"
SCRIPT_LSTM_RESIDUAL_AGG = "scripts/lstm_residual_rmse_aggregate.py"
SCRIPT_COMBINED_TORQUE_AGG = "scripts/combined_torque_rmse_aggregate.py"
