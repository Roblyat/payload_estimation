import pandas as pd
import numpy as np

class TimeGapSegmenter:
    """Infer trajectory boundaries by time gaps (fallback approach)."""

    def __init__(self, time_col: str, gap_s: float):
        self.time_col = time_col
        self.gap_s = gap_s

    def add_trajectory_id(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(self.time_col).copy()
        dt = df[self.time_col].diff().fillna(0.0)
        df["trajectory_id"] = (dt > self.gap_s).cumsum().astype(int)
        return df

class FixedLengthSegmenter:
    """
    Split a continuous log into trajectories by fixed number of *frames*.
    One frame = one unique timestamp. All joints at that timestamp share the same traj id.
    """

    def __init__(self, time_col: str, frames_per_trajectory: int):
        self.time_col = time_col
        self.frames_per_trajectory = int(frames_per_trajectory)

    def add_trajectory_id(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Map each unique time to a frame index (0..T-1)
        unique_t = np.sort(df[self.time_col].unique())
        frame_idx_map = pd.Series(np.arange(len(unique_t), dtype=int), index=unique_t)

        frame_idx = frame_idx_map.loc[df[self.time_col]].to_numpy()
        df["trajectory_id"] = (frame_idx // self.frames_per_trajectory).astype(int)
        return df