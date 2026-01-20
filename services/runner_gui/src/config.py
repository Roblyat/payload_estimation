#/runner_gui/src/config.py

from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class AppConfig:
    # Base directories (container paths)
    BASE_PREPROCESSED: str = "/workspace/shared/data/preprocessed"
    BASE_PROCESSED: str = "/workspace/shared/data/processed"
    BASE_MODELS_DELAN: str = "/workspace/shared/models/delan"
    BASE_MODELS_LSTM: str = "/workspace/shared/models/lstm"
    BASE_EVALUATION: str = "/workspace/shared/evaluation"

    # DeLaN backend options
    DELAN_BACKENDS: tuple[str, ...] = ("jax", "torch")

    DELAN_SERVICE: dict[str, str] = None
    DELAN_TRAIN_SCRIPT: dict[str, str] = None
    DELAN_EXPORT_SCRIPT: dict[str, str] = None
    DELAN_CKPT_EXT: dict[str, str] = None


    # Shared Compose invocation used by GUI
    COMPOSE: str = (
        "docker compose -p payload_estimation "
        "--project-directory /workspace "
        "--env-file /workspace/.env "
        "-f /workspace/docker-compose.yml"
    )

    FEATURE_MODES: tuple[str, ...] = ("full", "tau_hat", "state", "state_tauhat")

    FEATURE_HELP: dict[str, str] = None  # filled in __post_init__

    def __post_init__(self):
        object.__setattr__(
            self,
            "FEATURE_HELP",
            {
                "full": "x_k = [q, qd, qdd, tau_hat] (dim=24)",
                "tau_hat": "x_k = [tau_hat] (dim=6)",
                "state": "x_k = [q, qd, qdd] (dim=18)",
                "state_tauhat": "x_k = [qd, qdd, tau_hat] (dim=18)",
            },
        )

        object.__setattr__(
            self,
            "DELAN_SERVICE",
            {
                "jax": "delan_jax",
                "torch": "delan_torch",
            },
        )
        object.__setattr__(
            self,
            "DELAN_TRAIN_SCRIPT",
            {
                "jax": "/workspace/delan_jax/scripts/rbyt_train_delan_jax.py",
                "torch": "/workspace/delan_torch/scripts/rbyt_train_delan_torch.py",
            },
        )
        object.__setattr__(
            self,
            "DELAN_EXPORT_SCRIPT",
            {
                "jax": "/workspace/delan_jax/scripts/export_delan_residuals_jax.py",
                "torch": "/workspace/delan_torch/scripts/export_delan_residuals_torch.py",
            },
        )
        object.__setattr__(
            self,
            "DELAN_CKPT_EXT",
            {
                "jax": "jax",
                "torch": "torch",
            },
        )