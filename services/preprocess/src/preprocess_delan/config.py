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

    # segmentation
    segment_mode: str = "fixed_length"   # "time_gap" or "fixed_length"
    frames_per_trajectory: int = 75      # 75 frames ≈ ~200 traj for your file

    time_gap_seconds: float = 0.25       # still available if you use "time_gap"

    # Trajectory inference (fallback if no explicit trajectory id column exists)
    time_gap_seconds: float = 0.25

    # Split
    test_fraction: float = 0.2
    random_seed: int = 0
