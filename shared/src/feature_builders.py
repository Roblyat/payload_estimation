# payload_estimation/shared/src/feature_builders.py
from __future__ import annotations
import numpy as np

FEATURE_MODES = ("full", "tau_hat", "state", "state_tauhat")
FEATURE_CHOICES = FEATURE_MODES

def build_features(
    q: np.ndarray,
    qd: np.ndarray,
    qdd: np.ndarray,
    tau_hat: np.ndarray,
    mode: str = "full",
) -> np.ndarray:
    """
    Inputs: (T, dof) each
    Output: (T, D) depending on mode
    """
    if mode not in FEATURE_MODES:
        raise ValueError(f"Unknown feature mode '{mode}'. Valid: {FEATURE_MODES}")

    # Defensive: ensure 2D
    q = np.asarray(q)
    qd = np.asarray(qd)
    qdd = np.asarray(qdd)
    tau_hat = np.asarray(tau_hat)

    if mode == "full":
        feat = np.concatenate([q, qd, qdd, tau_hat], axis=1)
    elif mode == "tau_hat":
        feat = tau_hat
    elif mode == "state":
        feat = np.concatenate([q, qd, qdd], axis=1)
    elif mode == "state_tauhat":
        feat = np.concatenate([qd, qdd, tau_hat], axis=1)
    else:
        raise ValueError(f"Unhandled feature mode '{mode}'")

    return feat.astype(np.float32, copy=False)

def feature_dim(dof: int, mode: str) -> int:
    if mode == "full":
        return 4 * dof
    if mode == "tau_hat":
        return 1 * dof
    if mode in ("state", "state_tauhat"):
        return 3 * dof
    raise ValueError(f"Unknown feature mode '{mode}'")