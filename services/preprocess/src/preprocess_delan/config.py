from dataclasses import dataclass
from typing import Sequence

@dataclass(frozen=True)
class DelanPreprocessConfig:
    # joints (UR5 default)
    dof_joints: Sequence[str] = (
        "ur5_shoulder_pan_joint",
        "ur5_shoulder_lift_joint",
        "ur5_elbow_joint",
        "ur5_wrist_1_joint",
        "ur5_wrist_2_joint",
        "ur5_wrist_3_joint",
    )

    # CSV column names
    col_time: str = "Time"
    col_joint: str = "Joint Name"
    col_pos: str = "Position"
    col_vel: str = "Velocity"
    col_acc: str = "Acceleration"
    col_tau: str = "Effort"

    # Trajectory inference (fallback if no explicit trajectory id column exists)
    time_gap_seconds: float = 0.25

    # Split
    test_fraction: float = 0.2
    random_seed: int = 0
