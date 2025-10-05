# app/services/simulation_manager.py
import threading
import time
import numpy as np
from itertools import combinations
from scipy.optimize import differential_evolution

from ..config import SimulationConfig, HybridModelConfig
from .simulation_core import generalized_redundancy_metric, run_simulation_for_learner

class SimulationManager:
    def __init__(self, sim_config: SimulationConfig, hybrid_config: HybridModelConfig):
        self.sim_config = sim_config
        self.hybrid_config = hybrid_config
        self.lock = threading.Lock()
        self.operator_thread = None
        self.learner_thread = None
        self.is_data_loaded = False # Flag to track data loading status
        self._reset_state()

    def _reset_state(self, preserve_data=False):
        """Resets the simulation to its initial state."""
        data = self.data if preserve_data else None
        total_sensors = self.total_sensors if preserve_data else 0
        
        # Set data loaded status based on whether data is being preserved
        self.is_data_loaded = True if preserve_data else False

        # Simulation state
        self.is_running = False
        self.timestep = 0
        self.current_phase = "shadow_op"
        self.learner_status = "idle"

        # Parameters
        self.threshold = self.sim_config.threshold
        self.duration = self.sim_config.duration
        self.n_way_comparison = self.sim_config.n_way_comparison
        self.shadow_mode_probability = self.sim_config.shadow_mode_probability
        
        # Hybrid model state
        self.last_retrain_timestep = 0

        # Performance Metrics
        self.total_power_saved_steps = 0
        self.last_learner_fidelity = 1.0
        self.shadow_fidelity_error = 0.0
        self.shadow_fidelity_count = 0
        self.shadow_samples = []

        # Data and Sensors
        self.data = data
        self.total_sensors = total_sensors
        self.sensors = [
            {"id": i, "is_off": False, "end_time": -1, "noise_variance": 0.0}
            for i in range(total_sensors)
        ]

    def load_data(self, data: np.ndarray):
        """Loads data, initializes sensors, and flags that data is ready."""
        with self.lock:
            self.data = data
            self.total_sensors = data.shape[1]
            self._reset_state(preserve_data=True)
            self.is_data_loaded = True # Explicitly set to true after loading
            print("--- Data loaded and simulation manager is ready. ---")

    def start(self, params: dict):
        """Starts the simulation with new parameters."""
        with self.lock:
            if self.is_running:
                return {"message": "Simulation is already running."}
            
            if not self.is_data_loaded:
                return {"message": "Cannot start, data is not loaded yet."}
            
            # Reset state but keep the loaded data
            self._reset_state(preserve_data=True)

            # Update parameters from the request
            self.threshold = params.get('threshold', self.sim_config.threshold)
            self.duration = params.get('duration', self.sim_config.duration)
            self.n_way_comparison = params.get('n_way_comparison', self.sim_config.n_way_comparison)
            self.shadow_mode_probability = params.get('shadow_mode_probability', self.sim_config.shadow_mode_probability)

            self.is_running = True
            if not self.operator_thread or not self.operator_thread.is_alive():
                self.operator_thread = threading.Thread(target=self._operator_loop, daemon=True)
                self.operator_thread.start()
            
            return {"message": "Simulation started"}

    def pause(self):
        """Pauses the simulation."""
        with self.lock:
            self.is_running = False
        print("Simulation paused.")
        return {"message": "Simulation paused"}

    def reset(self):
        """Stops and resets the simulation completely."""
        with self.lock:
            self.is_running = False
        
        if self.operator_thread and self.operator_thread.is_alive():
            self.operator_thread.join(timeout=1.0)
        
        with self.lock:
            self._reset_state(preserve_data=True)
        print("Simulation reset.")
        return {"message": "Simulation reset"}

    def get_status(self) -> dict:
        """Returns the current status of the simulation."""
        with self.lock:
            if self.shadow_fidelity_count > 0:
                mse = self.shadow_fidelity_error / self.shadow_fidelity_count
                fidelity = max(0, 1.0 - mse)
            else:
                fidelity = self.last_learner_fidelity

            active_count = sum(1 for s in self.sensors if not s["is_off"])
            total_possible_steps = self.timestep * self.total_sensors
            power_saved_percent = (self.total_power_saved_steps / total_possible_steps) * 100 if total_possible_steps > 0 else 0

            readings = []
            if self.data is not None and self.timestep < len(self.data):
                readings = self.data[self.timestep].tolist()

            return {
                "is_running": self.is_running,
                "timestep": self.timestep,
                "current_phase": self.current_phase,
                "learner_status": self.learner_status,
                "threshold": self.threshold,
                "duration": self.duration,
                "n_way_comparison": self.n_way_comparison,
                "total_sensors": self.total_sensors,
                "active_sensors": active_count,
                "power_saved_percent": power_saved_percent,
                "fidelity": fidelity,
                "last_learner_fidelity": self.last_learner_fidelity,
                "collected_shadow_samples": len(self.shadow_samples),
                "current_readings": readings,
            }
            
    def _operator_loop(self):
        """The main simulation loop running in a background thread."""
        print("--- Operator loop has started ---")
        while True:
            time.sleep(0.05)
            with self.lock:
                if not self.is_running: break
                
                if self.timestep % 100 == 0:
                    print(f"Operator loop ticking at timestep: {self.timestep}")

                if self.data is None or self.timestep >= len(self.data) - 1:
                    self.is_running = False
                    self.current_phase = "finished"
                    print("--- Simulation finished: End of data reached. ---")
                    break
                
                self._check_and_trigger_retrain()
                
                if self.current_phase == "collecting":
                    self._execute_collection_step()
                elif self.current_phase == "shadow_op":
                    self._execute_shadow_op_step()

                self.timestep += 1
            
    def _check_and_trigger_retrain(self):
        is_learner_busy = self.learner_thread is not None and self.learner_thread.is_alive()
        if is_learner_busy or self.current_phase == "collecting":
            return

        current_fidelity = self.get_status()["fidelity"]
        time_since_retrain = self.timestep - self.last_retrain_timestep
        
        trigger_by_fidelity = current_fidelity < self.hybrid_config.fidelity_threshold
        trigger_by_interval = time_since_retrain > self.hybrid_config.max_timesteps_since_retrain

        if trigger_by_fidelity or trigger_by_interval:
            self.current_phase = "collecting"
            self.last_retrain_timestep = self.timestep
            print(f"Retrain triggered at t={self.timestep}. Fidelity Drop: {trigger_by_fidelity}, Interval Exceeded: {trigger_by_interval}.")
            
    def _execute_collection_step(self):
        for s in self.sensors:
            s["is_off"] = False
        
        collection_progress = self.timestep - self.last_retrain_timestep
        if collection_progress >= self.hybrid_config.collection_period:
            start_idx = max(0, self.timestep - self.hybrid_config.collection_period + 1)
            data_chunk = self.data[start_idx : self.timestep + 1]
            
            self.learner_thread = threading.Thread(
                target=self._learner_task, args=(data_chunk, self.n_way_comparison), daemon=True
            )
            self.learner_thread.start()
            
            self.current_phase = "shadow_op"
            self.last_retrain_timestep = self.timestep
    
    def _execute_shadow_op_step(self):
        t = self.timestep
        readings = self.data[t]
        last_readings = self.data[t - 1] if t > 0 else readings

        for s in self.sensors:
            if s["is_off"] and t >= s["end_time"]:
                s["is_off"] = False
        
        active_sensors = [s for s in self.sensors if not s["is_off"]]
        
        for s in active_sensors:
            delta = readings[s["id"]] - last_readings[s["id"]]
            s["noise_variance"] = 0.99 * s["noise_variance"] + 0.01 * (delta ** 2)
            
        if len(active_sensors) >= self.n_way_comparison:
            for combo_sensors in combinations(active_sensors, self.n_way_comparison):
                combo_readings = np.array([readings[s["id"]] for s in combo_sensors])
                if generalized_redundancy_metric(combo_readings, self.n_way_comparison) > self.threshold:
                    noisiest_sensor = max(combo_sensors, key=lambda s: s["noise_variance"])
                    
                    if not noisiest_sensor["is_off"]:
                        if np.random.rand() < self.shadow_mode_probability:
                            self._perform_shadow_check(readings, combo_sensors, noisiest_sensor)
                        else:
                            noisiest_sensor["is_off"] = True
                            noisiest_sensor["end_time"] = t + self.duration

        num_deactivated_total = sum(1 for s in self.sensors if s["is_off"])
        self.total_power_saved_steps += num_deactivated_total
        
    def _perform_shadow_check(self, readings, combo_sensors, noisiest_sensor):
        peer_readings = [readings[s["id"]] for s in combo_sensors if s["id"] != noisiest_sensor["id"]]
        estimated = np.mean(peer_readings) if peer_readings else readings[noisiest_sensor["id"]]
        true = readings[noisiest_sensor["id"]]
        
        self.shadow_fidelity_error += (true - estimated) ** 2
        self.shadow_fidelity_count += 1
        if len(self.shadow_samples) < 1000:
            self.shadow_samples.append((true, estimated))
            
    def _learner_task(self, collected_data: np.ndarray, n_way: int):
        with self.lock:
            self.learner_status = "running"
        
        split_idx = int(len(collected_data) * 0.8)
        train_set, test_set = collected_data[:split_idx], collected_data[split_idx:]

        if not len(train_set) or not len(test_set):
            print("Learner aborted: Not enough data for train/test split.")
            with self.lock: self.learner_status = "idle"
            return

        print(f"Learner started. Training on {len(train_set)}, testing on {len(test_set)}.")
        
        def objective_function(params, data):
            power_saved, fidelity = run_simulation_for_learner(params[0], int(round(params[1])), n_way, data)
            return -((fidelity ** 10) * (power_saved ** 0.1))

        bounds = [(0.9, 1.0), (10, 200)]
        result = differential_evolution(objective_function, bounds, args=(train_set,), maxiter=30, popsize=10, tol=0.02, disp=False)
        
        new_threshold, new_duration = result.x[0], int(round(result.x[1]))
        _, final_fidelity = run_simulation_for_learner(new_threshold, new_duration, n_way, test_set)

        with self.lock:
            self.threshold = new_threshold
            self.duration = new_duration
            self.last_learner_fidelity = final_fidelity
            self.learner_status = "idle"
            self.shadow_fidelity_error = 0.0
            self.shadow_fidelity_count = 0
            self.shadow_samples = []
            
            print(f"Learner finished. New params: T={new_threshold:.4f}, D={new_duration}. Fidelity: {final_fidelity:.4f}")
