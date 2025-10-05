# app/services/simulation_core.py
import numpy as np
from numba import njit

@njit
def generalized_redundancy_metric(readings: np.ndarray, n: int) -> float:
    """Calculates the AURA redundancy metric for a combination of sensor readings."""
    if n < 2: return 0.0
    total = np.sum(readings)
    if total <= 1e-9: return 0.0
    denominator = n * (np.sin(np.pi / n)**2)
    if denominator < 1e-9: return 0.0
    numerator = np.sum(np.sin(np.pi * readings / total)**2)
    return numerator / denominator

@njit
def run_simulation_for_learner(
    threshold_R: float, 
    duration: int, 
    n_way: int, 
    data_input: np.ndarray
) -> tuple[float, float]:
    """
    Runs a self-contained simulation to evaluate parameters.
    Returns: (power_saved_percentage, fidelity_score)
    """
    n_sensors = data_input.shape[1]
    current_timesteps = data_input.shape[0]
    if current_timesteps < 2: return 0.0, 1.0
    
    deactivated_storage = np.full((n_sensors, 3), -1, dtype=np.int32)
    num_deactivated = 0
    sensor_noise_variance = np.zeros(n_sensors, dtype=np.float64)
    last_readings = data_input[0].copy()
    power_saved, total_squared_error, fidelity_count = 0.0, 0.0, 0.0

    for t in range(1, current_timesteps):
        # --- Update sensor deactivation status ---
        temp_storage = np.full((n_sensors, 3), -1, dtype=np.int32)
        temp_num_deactivated = 0
        is_sensor_off = np.zeros(n_sensors, dtype=np.bool_)
        
        active_deactivations = 0
        for k in range(num_deactivated):
            deactivated_id, _, end_time = deactivated_storage[k]
            if t < end_time:
                temp_storage[active_deactivations] = deactivated_storage[k]
                is_sensor_off[deactivated_id] = True
                active_deactivations += 1
        deactivated_storage, num_deactivated = temp_storage, active_deactivations
        power_saved += num_deactivated
        
        # --- Update noise variance ---
        readings = data_input[t]
        deltas = readings - last_readings
        sensor_noise_variance = 0.99 * sensor_noise_variance + 0.01 * (deltas**2)
        last_readings = readings.copy()
        
        # --- AURA Logic ---
        active_sensor_indices = np.where(~is_sensor_off)[0]
        if len(active_sensor_indices) < n_way: continue

        # Efficiently iterate through combinations
        indices = np.arange(n_way)
        while True:
            combo_indices = active_sensor_indices[indices]
            combo_readings = readings[combo_indices]
            aura_index = generalized_redundancy_metric(combo_readings, n_way)

            if aura_index > threshold_R:
                combo_noise = sensor_noise_variance[combo_indices]
                max_noise_idx_in_combo = np.argmax(combo_noise)
                sensor_to_deactivate_id = combo_indices[max_noise_idx_in_combo]

                if not is_sensor_off[sensor_to_deactivate_id]:
                    peer_indices = np.delete(combo_indices, max_noise_idx_in_combo)
                    estimated_reading = np.mean(readings[peer_indices])
                    true_reading = readings[sensor_to_deactivate_id]
                    total_squared_error += (true_reading - estimated_reading)**2
                    fidelity_count += 1
                    
                    deactivated_storage[num_deactivated] = [sensor_to_deactivate_id, -1, t + duration]
                    is_sensor_off[sensor_to_deactivate_id] = True
                    num_deactivated += 1
            
            # Next combination logic (itertools.combinations in Numba)
            i = n_way - 1
            while i >= 0 and indices[i] == i + len(active_sensor_indices) - n_way: i -= 1
            if i < 0: break
            indices[i] += 1
            for j in range(i + 1, n_way): indices[j] = indices[j - 1] + 1

    mse = total_squared_error / fidelity_count if fidelity_count > 0 else 0.0
    fidelity_score = max(0.0, 1.0 - mse)
    power_saved_percentage = power_saved / (n_sensors * current_timesteps) if current_timesteps > 0 else 0.0
    return power_saved_percentage, fidelity_score