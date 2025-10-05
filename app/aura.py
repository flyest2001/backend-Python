import numpy as np
from numba import njit

@njit
def aura_index(readings, n):
    if n < 2: return 0.0
    total = np.sum(readings)
    if total <= 1e-9: return 0.0
    denominator = n * (np.sin(np.pi / n)**2)
    if denominator < 1e-9: return 0.0
    numerator = np.sum(np.sin(np.pi * readings / total)**2)
    return numerator / denominator
