import numpy as np
from scipy.optimize import differential_evolution
from numba import njit

from . import state
from .aura import aura_index

@njit
def run_simulation_for_learner(threshold_R, duration, n_way, data_input):
    n_sensors = data_input.shape[1]
    current_timesteps = data_input.shape[0]
    if current_timesteps < 2: return 0.0, 1.0
    
    deactivated_storage = np.full((n_sensors, 3), -1, dtype=np.int32)
    num_deactivated = 0
    sensor_noise_variance = np.zeros(n_sensors, dtype=np.float64)
    last_readings = data_input[0].copy()
    power_saved, total_squared_error, fidelity_count = 0.0, 0.0, 0.0

    for t in range(1, current_timesteps):
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
        
        readings = data_input[t]
        deltas = readings - last_readings
        sensor_noise_variance = 0.99 * sensor_noise_variance + 0.01 * (deltas**2)
        last_readings = readings.copy()
        
        active_sensor_indices = np.where(~is_sensor_off)[0]
        
        if len(active_sensor_indices) < n_way: continue

        indices = np.arange(n_way)
        while True:
            combo_indices = active_sensor_indices[indices]
            combo_readings = readings[combo_indices]
            aura_index = aura_index(combo_readings, n_way)

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
            
            i = n_way - 1
            while i >= 0 and indices[i] == i + len(active_sensor_indices) - n_way: i -= 1
            if i < 0: break
            indices[i] += 1
            for j in range(i + 1, n_way): indices[j] = indices[j - 1] + 1

    mse = total_squared_error / fidelity_count if fidelity_count > 0 else 0.0
    fidelity_score = max(0.0, 1.0 - mse)
    power_saved_percentage = power_saved / (n_sensors * current_timesteps) if current_timesteps > 0 else 0.0
    return power_saved_percentage, fidelity_score

def objective_function(params, training_data, n_way):
    threshold_R, duration = params[0], int(round(params[1]))
    power_saved, fidelity = run_simulation_for_learner(threshold_R, duration, n_way, training_data)
    # The negative sign is because optimizers minimize, and we want to maximize this score
    return -((fidelity ** 10) * (power_saved ** 0.1))

def learner_task(collected_data, n_way):
    with state.SIMULATION_STATE["lock"]:
        state.SIMULATION_STATE["learner_status"] = "running"
    
    split_index = int(len(collected_data) * 0.8)
    train_set, test_set = collected_data[:split_index], collected_data[split_index:]

    if not len(train_set) or not len(test_set):
        print("Not enough collected data for train/test split. Aborting learner.")
        with state.SIMULATION_STATE["lock"]:
            state.SIMULATION_STATE["learner_status"] = "idle"
        return

    print(f"Learner started. Training on {len(train_set)}, testing on {len(test_set)}.")
    
    bounds = [(0.9, 1.0), (10.0, 200.0)]
    result = differential_evolution(objective_function, bounds, args=(train_set, n_way), maxiter=30, popsize=10, tol=0.02, disp=False)
    new_threshold, new_duration = result.x[0], int(round(result.x[1]))

    _, final_fidelity = run_simulation_for_learner(new_threshold, new_duration, n_way, test_set)

    with state.SIMULATION_STATE["lock"]:
        state.SIMULATION_STATE["threshold"] = new_threshold
        state.SIMULATION_STATE["duration"] = new_duration
        state.SIMULATION_STATE["last_learner_fidelity"] = final_fidelity
        state.SIMULATION_STATE["learner_status"] = "idle"
        # Reset shadow stats after retraining
        state.SIMULATION_STATE["shadow_fidelity_error"] = 0.0
        state.SIMULATION_STATE["shadow_fidelity_count"] = 0
        state.SIMULATION_STATE["shadow_samples"] = []
        print(f"Learner finished. New params deployed: T={new_threshold:.4f}, D={new_duration}. Tested Fidelity: {final_fidelity:.4f}")
