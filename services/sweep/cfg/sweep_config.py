from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import List, Optional


@dataclass
class SweepConfig:
    repo_root: Path

    # Dataset settings
    dataset_name: str = "UR3_Load0_cc"
    run_tag: str = "best5x10L0"
    in_format: str = "csv"
    col_format: str = "wide"
    derive_qdd: bool = True
    lowpass_signals: bool = True
    lowpass_cutoff_hz: float = 10.0
    lowpass_order: int = 4
    lowpass_qdd_values: bool = False

    # Sweep
    traj_amounts: List[int] = field(default_factory=lambda: [8, 16, 32, 48, 64, 86, 122]) ## K-Domination-Story
    test_fractions = 0.2
    val_fraction = 0.1
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2]) ## K-Domination-Story

    # Window sizes and feature modes
    h_list: List[int] = field(default_factory=lambda: [100]) ## K-Domination-Story
    feature_modes: List[str] = field(default_factory=lambda: ["full"]) ## K-Domination-Story

    # DeLaN settings
    delan_model_type: str = "structured"
    delan_hp_preset: str = "lutter_like_256"
    delan_hp_flags: str = ""
    delan_seeds: List[int] = field(default_factory=lambda: [0, 1]) #[0, 1, 2, 3, 4]) ## Best DeLaN Model & K-Domination-Story
    delan_epochs: int = 200
    delan_eval_every: int = 1
    delan_log_every: int = 1
    delan_early_stop = True
    delan_early_stop_patience: int = 10
    delan_early_stop_min_delta: float = 0.0
    delan_early_stop_warmup_evals: int = 0

    # DeLaN best-model sweep
    delan_best_k_max: int = 84
    delan_best_dataset_seeds: List[int] = field(default_factory=lambda: [0, 1]) #[0, 1, 2, 3, 4])
    # delan_best_hp_presets: List[str] = field(default_factory=lambda: 
    #                                      [
    #     "lutter_like_128",
    #     "lutter_like_256",
    #     "lutter_like_256_d3",
    #     "lutter_like_256_lr5e5",
    #     "lutter_like_256_wd1e4",
    # ])
    delan_best_hp_presets: List[str] = field(default_factory=lambda: 
                                         [
        "lutter_like_128",
        "lutter_like_256",
    ])    
    delan_best_score_lambda: float = 0.5
    delan_best_score_penalty: float = 10.0
    delan_best_fold_plots: bool = True
    delan_best_hp_curves: bool = True
    delan_best_scatter_plots: bool = True
    delan_best_torque_aggregate: bool = True
    delan_best_torque_bins: int = 200
    delan_best_torque_split: str = "test"
    # best torque hp_presets just for plot alginment. no relation to any algorithm logic
    # delan_best_torque_hp_presets: List[str] = field(default_factory=list)
    # delan_best_torque_hp_presets = [
    #     "lutter_like_128",
    #     "lutter_like_256",
    #     "lutter_like_256_d3",
    #     "lutter_like_256_lr5e5",
    #     "lutter_like_256_wd1e4"
    #     ]
    delan_best_torque_hp_presets = [
        "lutter_like_128",
        "lutter_like_256",
        ]    
    delan_best_plots_out_dir: str = "/workspace/shared/evaluation/delan_best"

    # LSTM hyperparams
    lstm_epochs: int = 120
    lstm_batch: int = 64
    lstm_val_split: float = 0.1
    lstm_units: int = 128
    lstm_dropout: float = 0.2
    lstm_eps: float = 1e-8
    lstm_no_plots: bool = False
    lstm_early_stop: bool = True
    lstm_early_stop_patience: int = 20
    lstm_early_stop_min_delta: float = 0.0
    lstm_early_stop_warmup_evals: int = 10

    # LSTM best-model sweep
    lstm_best_dataset_seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    lstm_best_feature_modes: List[str] = field(default_factory=lambda: ["full"]) #["full", "state", "tau_hat", "state_tauhat"])
    lstm_best_h_list: List[int] = field(default_factory=lambda: [50]) #[50, 100, 150])
    lstm_best_seeds: List[int] = field(default_factory=lambda: [0]) #[0, 1, 2])
    lstm_best_score_lambda: float = 0.5
    lstm_best_score_penalty: float = 10.0
    lstm_best_delan_hypers_jsonl: str = "/workspace/shared/evaluation/summary_delan_best_hypers_20260129_154853.jsonl"
    lstm_best_delan_folds_jsonl: str = "/workspace/shared/evaluation/summary_delan_best_folds_20260129_154853.jsonl"
    lstm_best_delan_model_json: str = "/workspace/shared/evaluation/delan_best_model_20260129_154853.json"
    lstm_best_eval_split: str = "test"
    lstm_best_bins: int = 200
    lstm_best_residual_aggregate: bool = True
    lstm_best_combined_aggregate: bool = True
    lstm_best_boxplots: bool = True
    lstm_best_scatter_legend: bool = True
    lstm_best_models_dir: str = "/workspace/shared/models/lstm/best"
    lstm_best_plots_out_dir: str = "/workspace/shared/evaluation/lstm_best"

    # Plot/cleanup behavior
    logs_root_dir: str = "shared/logs"
    cleanup_non_best_plots: bool = True
    delan_elbow_aggregate: bool = True
    delan_elbow_out_dir: str = "/workspace/shared/evaluation/delan_elbows"
    delan_torque_aggregate: bool = True
    delan_torque_out_dir: str = "/workspace/shared/evaluation/delan_torque"
    delan_torque_bins: int = 200
    delan_torque_k_values: List[int] = field(default_factory=list)
    lstm_training_aggregate: bool = True
    lstm_residual_aggregate: bool = True
    lstm_aggregate_out_dir: str = "/workspace/shared/evaluation/lstm_aggregate"
    lstm_aggregate_bins: int = 200
    lstm_aggregate_k_values: List[int] = field(default_factory=list)
    lstm_aggregate_feature: Optional[str] = None
    lstm_aggregate_pad_to_epochs: Optional[int] = None
    lstm_aggregate_align: str = "max"
    combined_torque_aggregate: bool = True
    combined_torque_out_dir: str = "/workspace/shared/evaluation/combined_torque"
    combined_torque_bins: int = 200
    combined_torque_k_values: List[int] = field(default_factory=list)
    combined_torque_feature: Optional[str] = None


def default_sweep_config() -> SweepConfig:
    # sweep_config.py -> cfg -> sweep -> services -> payload_estimation -> repo root
    repo_root = Path(__file__).resolve().parents[4]
    cfg = SweepConfig(repo_root=repo_root)

    def _env_str(name: str, default: str) -> str:
        val = os.getenv(name)
        return val if val else default

    def _env_list_int(name: str, default: List[int]) -> List[int]:
        val = os.getenv(name)
        if not val:
            return default
        try:
            if val.strip().startswith("["):
                import json
                parsed = json.loads(val)
                return [int(x) for x in parsed]
            return [int(t.strip()) for t in val.split(",") if t.strip()]
        except Exception:
            return default

    cfg.dataset_name = _env_str("SWEEP_DATASET_NAME", cfg.dataset_name)
    cfg.run_tag = _env_str("SWEEP_RUN_TAG", cfg.run_tag)
    cfg.lstm_best_delan_hypers_jsonl = _env_str(
        "LSTM_BEST_DELAN_HYPERS_JSONL", cfg.lstm_best_delan_hypers_jsonl
    )
    cfg.lstm_best_delan_folds_jsonl = _env_str(
        "LSTM_BEST_DELAN_FOLDS_JSONL", cfg.lstm_best_delan_folds_jsonl
    )
    cfg.lstm_best_delan_model_json = _env_str(
        "LSTM_BEST_DELAN_MODEL_JSON", cfg.lstm_best_delan_model_json
    )
    cfg.lstm_best_dataset_seeds = _env_list_int(
        "LSTM_BEST_DATASET_SEEDS", cfg.lstm_best_dataset_seeds
    )

    return cfg
