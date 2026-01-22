import pandas as pd
import numpy as np

class RawCSVLoader:
    """Load raw CSVs and (optionally) adapt them to the pipeline's expected *long* format.
    The rest of the preprocess pipeline assumes a long-format dataframe with columns:
        Time, Joint Name, Position, Velocity, Acceleration, Effort
    For other dataset layouts (e.g., wide q1..q6), we provide dedicated loader helpers
    that convert into this long format.
    """
    def load_example(self, path: str) -> pd.DataFrame:
        """Example dataset: already in long format."""
        return pd.read_csv(path)
    def load_dataset1(self, path: str, cfg) -> pd.DataFrame:
        """Dataset1: wide format with columns like t1, q1..q6, dq1..dq6, Iq1..Iq6.
        Keeps only time, q, dq, Iq, and converts to the long format expected by the pipeline.
        - Time      := t1
        - Position  := q*
        - Velocity  := dq*
        - Effort    := Iq*   (raw motor current; interpreted downstream as tau proxy)
        - Accel     := derived from dq* if cfg.derive_qdd_from_dq else zeros
        """
        dfw = pd.read_csv(path, skipinitialspace=True)
        # Required columns
        time_col = "t1"
        traj_col = getattr(cfg, "wide_traj_id_col", "ID")
        q_cols = [f"q{i}" for i in range(1, 7)]
        dq_cols = [f"dq{i}" for i in range(1, 7)]
        iq_cols = [f"Iq{i}" for i in range(1, 7)]
        required = [time_col, *q_cols, *dq_cols, *iq_cols]
        # trajectory id is optional but strongly recommended
        if traj_col:
            required.append(traj_col)

        missing = [c for c in required if c not in dfw.columns]
        if missing:
            raise ValueError(f"Dataset1 CSV is missing required columns: {missing}")
        # Work on a copy with only the columns we need
        keep_cols = [time_col, *q_cols, *dq_cols, *iq_cols]
        if traj_col:
            keep_cols.append(traj_col)
        dfw = dfw[keep_cols].copy()
        # Drop exact duplicate rows (your sample shows repeated identical lines)
        dfw = dfw.drop_duplicates()
        t = dfw[time_col].to_numpy(dtype=float)
        traj_id = None
        if traj_col:
            # keep as int if possible
            traj_id = pd.to_numeric(dfw[traj_col], errors="coerce").fillna(0).astype(int).to_numpy()
        q = dfw[q_cols].to_numpy(dtype=float)      # (N, 6)
        dq = dfw[dq_cols].to_numpy(dtype=float)    # (N, 6)
        iq = dfw[iq_cols].to_numpy(dtype=float)    # (N, 6)
        # Derive acceleration qdd if requested
        if getattr(cfg, "derive_qdd_from_dq", True) and len(t) >= 2:
            # Use time-based gradient if time is strictly increasing; else fallback to uniform step
            if np.all(np.diff(t) > 0):
                qdd = np.stack([np.gradient(dq[:, j], t) for j in range(dq.shape[1])], axis=1)
            else:
                qdd = np.stack([np.gradient(dq[:, j]) for j in range(dq.shape[1])], axis=1)
        else:
            qdd = np.zeros_like(dq)
        # Convert wide frames to long rows: one row per (time, joint)
        long_parts = []
        for j, joint_name in enumerate(cfg.dof_joints):
            out = {
                cfg.col_time: t,
                cfg.col_joint: joint_name,
                cfg.col_pos: q[:, j],
                cfg.col_vel: dq[:, j],
                cfg.col_acc: qdd[:, j],
                cfg.col_tau: iq[:, j],
            }
            # If we have an explicit trajectory/frame id, keep it for the rest of the pipeline
            if traj_id is not None:
                out["trajectory_id"] = traj_id
            long_parts.append(pd.DataFrame(out))
        return pd.concat(long_parts, ignore_index=True)
    def load(self, path: str, cfg) -> pd.DataFrame:
        """Dispatch to the correct loader based on cfg.input_format."""
        fmt = getattr(cfg, "input_format", "long")
        if fmt == "wide":
            return self.load_dataset1(path, cfg)
        return self.load_example(path)
