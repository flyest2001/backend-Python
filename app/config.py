# app/config.py
from dataclasses import dataclass

@dataclass
class SimulationConfig:
    """Core AURA parameters for the simulation."""
    threshold: float = 0.98
    duration: int = 40
    n_way_comparison: int = 2
    shadow_mode_probability: float = 0.05

@dataclass
class HybridModelConfig:
    """Parameters for the self-adapting hybrid model."""
    fidelity_threshold: float = 0.97
    max_timesteps_since_retrain: int = 2880  # e.g., retrain every 2 days (1 step/min)
    collection_period: int = 200

@dataclass
class AppConfig:
    """General application configuration."""
    data_file_path: str = 'GlobalWeather.csv '  # Main dataset
    fallback_data_file_path: str = 'advanced_iot.csv' # Example if you add another fallback