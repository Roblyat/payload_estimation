import pandas as pd

class RawCSVLoader:
    def load(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path)
