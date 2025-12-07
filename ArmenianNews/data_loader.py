import pandas as pd
from pathlib import Path
from typing import List, Optional
import warnings

class ArmenianNewsDataLoader:
    """Оптимизированная загрузка данных новостей на армянском языке"""
    
    def __init__(self, sample_size: int = 100):
        self.sample_size = sample_size
        
    def load_data_optimized(self, file_paths: List[str]) -> pd.DataFrame:
        """Оптимизированная загрузка данных"""
        dfs = []
        
        for file_path in file_paths:
            try:
                df_temp = pd.read_csv(file_path).iloc[:self.sample_size, [1, 2, 3]]
                dfs.append(df_temp)
                print(f"Loaded {len(df_temp)} rows from {file_path}")
            except Exception as e:
                warnings.warn(f"Error loading {file_path}: {e}")
                continue
        
        if not dfs:
            raise ValueError("No data loaded from any file")
        
        return pd.concat(dfs, ignore_index=True)
    
    def load_from_directory(self, directory: str, pattern: str = "*.csv") -> pd.DataFrame:
        """Загрузка всех CSV файлов из директории"""
        directory_path = Path(directory)
        file_paths = list(directory_path.glob(pattern))
        
        if not file_paths:
            raise FileNotFoundError(f"No CSV files found in {directory}")
        
        return self.load_data_optimized([str(fp) for fp in file_paths])