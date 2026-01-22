from dataclasses import dataclass
from typing import Sequence
from typing import Optional

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

    # input format
    # - "long": rows are (Time, Joint Name, Position, Velocity, Acceleration, Effort)
    # - "wide": rows are frames with q1..q6, dq1..dq6, Iq1..Iq6, etc. (converted to long internally)
    
    input_format: str = "long"  # "long" or "wide"
    
    derive_qdd_from_dq: bool = True

    # (wide only) optional trajectory/frame id column in the wide CSV.
    # If present, it will be converted to "trajectory_id" and used directly (no re-segmentation).
    wide_traj_id_col: str = "ID"

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

    # Trajectory inference (fallback if no explicit trajectory id column exists)
    time_gap_seconds: float = 0.25

    # Split
    test_fraction: float = 0.2
    random_seed: int = 0

    # Optional: limit number of trajectories used to build the dataset.
    # If None or <=0: use all trajectories.
    trajectory_amount: Optional[int] = None