import pandas as pd
from typing import Sequence

class JointSelector:
    def __init__(self, joints: Sequence[str], joint_col: str):
        self.joints = list(joints)
        self.joint_col = joint_col

    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[df[self.joint_col].isin(self.joints)].copy()
