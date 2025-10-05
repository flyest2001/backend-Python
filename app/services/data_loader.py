# app/services/data_loader.py
import pandas as pd
import numpy as np
from ..config import AppConfig

def load_and_normalize_data(config: AppConfig) -> np.ndarray:
    """Loads, cleans, and normalizes the dataset."""
    try:
        df = pd.read_csv(config.data_file_path).iloc[:, :28]
        print(f"Loaded dataset from {config.data_file_path}")
    except FileNotFoundError:
        print(f"'{config.data_file_path}' not found. Trying fallback...")
        try:
            df = pd.read_csv(config.fallback_data_file_path)
        except FileNotFoundError:
            print("No dataset found. Using random dummy data.")
            return np.random.rand(20000, 30)

    if 'date' in df.columns:
        df = df.drop(columns=['date'])
    
    raw_data = df.values.astype(np.float64)

    # Normalize data to [0, 1] range
    min_vals = raw_data.min(axis=0)
    max_vals = raw_data.max(axis=0)
    range_vals = max_vals - min_vals
    # Avoid division by zero for columns with constant value
    range_vals[range_vals == 0] = 1e-9 
    
    normalized_data = (raw_data - min_vals) / range_vals
    
    print(f"Dataset loaded and normalized. Shape: {normalized_data.shape}")
    return normalized_data