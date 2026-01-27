from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class SweepConfig:
    repo_root: Path

    # Dataset settings
    dataset_name: str = "UR3_Load0_cc"
    run_tag: str = "trajKdom"
    in_format: str = "csv"
    col_format: str = "wide"
    derive_qdd: bool = True
    lowpass_signals: bool = True
    lowpass_cutoff_hz: float = 10.0
    lowpass_order: int = 4
    lowpass_qdd_values: bool = False

    # Sweep
    traj_amounts: List[int] = field(default_factory=lambda: [8, 16, 32, 48, 64, 84, 122])
    test_fractions = 0.2
    val_fraction = 0.1
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])

    # Window sizes and feature modes
    h_list: List[int] = field(default_factory=lambda: [100])
    feature_modes: List[str] = field(default_factory=lambda: ["full"])

    # DeLaN settings
    delan_model_type: str = "structured"
    delan_hp_preset: str = "lutter_like_256"
    delan_hp_flags: str = ""
    delan_seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    delan_epochs: int = 200
    delan_eval_every: int = 1
    delan_log_every: int = 1
    delan_early_stop = True
    delan_early_stop_patience: int = 10
    delan_early_stop_min_delta: float = 0.0
    delan_early_stop_warmup_evals: int = 0

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
    lstm_early_stop_warmup_evals: int = 5

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
    return SweepConfig(repo_root=repo_root)
