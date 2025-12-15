import numpy as np
import pandas as pd

class WidePivotBuilder:
    """Build wide arrays (T, dof) from long-format joint logs."""

    def __init__(self, cfg):
        self.cfg = cfg

    def build_for_trajectory(self, df_traj: pd.DataFrame):
        # Use sorted unique timestamps for row index
        t = np.sort(df_traj[self.cfg.col_time].unique())

        def pivot(col: str) -> np.ndarray:
            wide = (
                df_traj.pivot_table(
                    index=self.cfg.col_time,
                    columns=self.cfg.col_joint,
                    values=col,
                    aggfunc="mean",
                )
                .reindex(index=t, columns=list(self.cfg.dof_joints))
            )
            arr = wide.to_numpy()
            if np.isnan(arr).any():
                raise ValueError(
                    f"NaNs after pivot for '{col}'. Missing joint samples or inconsistent logging."
                )
            return arr

        q = pivot(self.cfg.col_pos)
        qd = pivot(self.cfg.col_vel)
        qdd = pivot(self.cfg.col_acc)
        tau = pivot(self.cfg.col_tau)

        return t, q, qd, qdd, tau
