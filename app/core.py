import time
import threading
import numpy as np
from itertools import combinations

from . import state
from . import learner
from .aura import aura_index

def operator_loop():
    while True:
        time.sleep(0.05)
        with state.SIMULATION_STATE["lock"]:
            if not state.SIMULATION_STATE["is_running"]:
                break
            
            t = state.SIMULATION_STATE["timestep"]
            if state.SIMULATION_STATE["data"] is None or t >= len(state.SIMULATION_STATE["data"]) - 1:
                state.SIMULATION_STATE["is_running"] = False
                state.SIMULATION_STATE["current_phase"] = "finished"
                break

            # --- Hybrid Model Logic ---
            # 1. Check if a retrain should be triggered
            current_fidelity = 1.0
            if state.SIMULATION_STATE["shadow_fidelity_count"] > 0:
                mse = state.SIMULATION_STATE["shadow_fidelity_error"] / state.SIMULATION_STATE["shadow_fidelity_count"]
                current_fidelity = max(0, 1.0 - mse)
            
            time_since_last_retrain = t - state.SIMULATION_STATE["last_retrain_timestep"]
            
            is_learner_busy = state.learner_thread is not None and state.learner_thread.is_alive()
            
            # Trigger conditions
            trigger_by_fidelity = current_fidelity < state.SIMULATION_STATE["hybrid_fidelity_threshold"]
            trigger_by_interval = time_since_last_retrain > state.SIMULATION_STATE["hybrid_max_timesteps_since_retrain"]
            
            if (trigger_by_fidelity or trigger_by_interval) and not is_learner_busy and state.SIMULATION_STATE["current_phase"] != "collecting":
                state.SIMULATION_STATE["current_phase"] = "collecting"
                state.SIMULATION_STATE["last_retrain_timestep"] = t # Mark the start of the collection cycle
                print(f"Retrain triggered at timestep {t}. Reason: Fidelity drop ({trigger_by_fidelity}), Interval exceeded ({trigger_by_interval}).")

            # 2. Execute logic based on the current phase
            phase = state.SIMULATION_STATE["current_phase"]
            
            if phase == "collecting":
                for s in state.SIMULATION_STATE["sensors"]:
                    s["is_off"] = False
                
                collection_progress = t - state.SIMULATION_STATE["last_retrain_timestep"]
                if collection_progress >= state.SIMULATION_STATE["collection_period"]:
                    # Collection finished, start learner and switch back to shadow mode
                    start_learn = max(0, t - state.SIMULATION_STATE["collection_period"] + 1)
                    data_chunk = state.SIMULATION_STATE["data"][start_learn:t+1]
                    n_way = state.SIMULATION_STATE["n_way_comparison"]
                    state.learner_thread = threading.Thread(target=learner.learner_task, args=(data_chunk, n_way))
                    state.learner_thread.start()
                    state.SIMULATION_STATE["current_phase"] = "shadow_op"
                    state.SIMULATION_STATE["last_retrain_timestep"] = t # Update to mark end of this cycle
            
            elif phase == "shadow_op":
                # Standard shadow mode operation
                num_deactivated = sum(1 for s in state.SIMULATION_STATE["sensors"] if s["is_off"] and t < s["end_time"])
                for s in state.SIMULATION_STATE["sensors"]:
                    if s["is_off"] and t >= s["end_time"]:
                        s["is_off"] = False

                readings = state.SIMULATION_STATE["data"][t]
                last_readings = state.SIMULATION_STATE["data"][t - 1] if t > 0 else readings
                
                active_sensors = [s for s in state.SIMULATION_STATE["sensors"] if not s["is_off"]]
                for s in active_sensors:
                    delta = readings[s["id"]] - last_readings[s["id"]]
                    s["noise_variance"] = 0.99 * s["noise_variance"] + 0.01 * (delta ** 2)

                if len(active_sensors) >= state.SIMULATION_STATE["n_way_comparison"]:
                    for combo in combinations(active_sensors, state.SIMULATION_STATE["n_way_comparison"]):
                        if aura_index(np.array([readings[s["id"]] for s in combo]), state.SIMULATION_STATE["n_way_comparison"]) > state.SIMULATION_STATE["threshold"]:
                            noisiest_sensor = max(combo, key=lambda s: s["noise_variance"])
                            if not noisiest_sensor["is_off"]:
                                if np.random.rand() < state.SIMULATION_STATE["shadow_mode_probability"]:
                                    # Undercover quality check
                                    peer_readings = [readings[s["id"]] for s in combo if s["id"] != noisiest_sensor["id"]]
                                    estimated = np.mean(np.array(peer_readings)) if peer_readings else readings[noisiest_sensor["id"]]
                                    true = readings[noisiest_sensor["id"]]
                                    state.SIMULATION_STATE["shadow_fidelity_error"] += (true - estimated) ** 2
                                    state.SIMULATION_STATE["shadow_fidelity_count"] += 1
                                    state.SIMULATION_STATE["shadow_samples"].append((true, estimated))
                                else:
                                    # Normal deactivation
                                    noisiest_sensor["is_off"] = True
                                    noisiest_sensor["end_time"] = t + state.SIMULATION_STATE["duration"]
                                    num_deactivated += 1
                
                state.SIMULATION_STATE["total_power_saved_steps"] += num_deactivated
            
            state.SIMULATION_STATE["timestep"] += 1
