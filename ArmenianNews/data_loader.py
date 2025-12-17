import pandas as pd
from pathlib import Path
from typing import List
import warnings


class ArmenianNewsDataLoader:
    def __init__(self):
        pass

    def load_data(self, file_paths: List[str]) -> pd.DataFrame:
        dfs = []

        for file_path in file_paths:
            try:
                df_temp = pd.read_csv(file_path).iloc[:, [1, 2, 3]]
                dfs.append(df_temp)
                print(f"Loaded {len(df_temp)} rows from {file_path}")
            except Exception as e:
                warnings.warn(f"Error loading {file_path}: {e}")
                continue

        if not dfs:
            raise ValueError("No data loaded from any file")

        return pd.concat(dfs, ignore_index=True)

    def load_from_directory(self, directory: str, pattern: str = "*.csv") -> pd.DataFrame:
        directory_path = Path(directory)
        file_paths = list(directory_path.glob(pattern))

        if not file_paths:
            raise FileNotFoundError(f"No CSV files found in {directory}")

        return self.load_data([str(fp) for fp in file_paths])