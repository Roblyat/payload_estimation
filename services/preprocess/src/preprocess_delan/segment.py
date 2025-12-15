import pandas as pd

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
