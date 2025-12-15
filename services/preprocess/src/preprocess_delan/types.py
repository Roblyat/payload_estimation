from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Trajectory:
    label: str
    t: np.ndarray     # (T,)
    q: np.ndarray     # (T, dof)
    qd: np.ndarray    # (T, dof)
    qdd: np.ndarray   # (T, dof)
    tau: np.ndarray   # (T, dof)
