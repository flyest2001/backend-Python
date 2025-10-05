import threading
import pandas as pd
import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from . import state
from . import core
from .state import get_default_state

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], # Adjust for your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import io
import requests

# ... (keep existing imports)

@app.on_event("startup")
async def startup_event():
    def load_data():
        # --- IMPORTANT ---
        # Replace these with the actual public URLs where you've hosted your CSV files.
        # For example, you can use GitHub Raw, AWS S3, Vercel Blob, etc.
        primary_data_url = "https://raw.githubusercontent.com/Aitsam-Ahad/Datasets/main/GlobalWeather.csv"
        fallback_data_url = "https://raw.githubusercontent.com/Aitsam-Ahad/Datasets/main/advanced_iot.csv"

        try:
            # Try loading the new dataset first from the URL
            print(f"Attempting to download primary dataset from: {primary_data_url}")
            response = requests.get(primary_data_url)
            response.raise_for_status()  # Raise an exception for bad status codes
            df = pd.read_csv(io.StringIO(response.text)).iloc[:,:28]
            print("Primary dataset downloaded and loaded successfully.")
            
            if 'date' in df.columns:
                df = df.drop(columns=['date'])
            raw_data = df.values
        except (requests.exceptions.RequestException, pd.errors.EmptyDataError) as e:
            print(f"Failed to load primary dataset: {e}. Trying fallback.")
            try:
                # Fallback to the original dataset from the URL
                print(f"Attempting to download fallback dataset from: {fallback_data_url}")
                response = requests.get(fallback_data_url)
                response.raise_for_status()
                df_sensor = pd.read_csv(io.StringIO(response.text)).iloc[:,:28]
                print("Fallback dataset downloaded and loaded successfully.")
                
                raw_data = df_sensor.drop(columns=['date']).values
            except (requests.exceptions.RequestException, pd.errors.EmptyDataError) as e:
                print(f"Failed to load fallback dataset: {e}. Using dummy data.")
                raw_data = np.random.rand(20000, 28)
        
        # Normalize data
        min_vals, max_vals = raw_data.min(axis=0), raw_data.max(axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1e-6
        normalized_data = (raw_data - min_vals) / range_vals
        
        # DYNAMICALLY configure the simulation state
        num_sensors = normalized_data.shape[1]
        print(f"Dataset loaded and normalized. Detected {num_sensors} sensors. Shape: {normalized_data.shape}")
        
        with state.SIMULATION_STATE["lock"]:
            state.SIMULATION_STATE["data"] = normalized_data
            state.SIMULATION_STATE["total_sensors"] = num_sensors
            state.SIMULATION_STATE["sensors"] = [{"id": i, "is_off": False, "master_id": -1, "end_time": -1, "noise_variance": 0} for i in range(num_sensors)]

    threading.Thread(target=load_data).start()

@app.get('/status')
async def get_status():
    with state.SIMULATION_STATE["lock"]:
        state_copy = {k: v for k, v in state.SIMULATION_STATE.items() if k not in ["lock", "data", "shadow_samples"]}
        
        if state_copy["shadow_fidelity_count"] > 0:
            mse = state_copy["shadow_fidelity_error"] / state_copy["shadow_fidelity_count"]
            state_copy["fidelity"] = max(0, 1.0 - mse)
        else:
            state_copy["fidelity"] = state_copy["last_learner_fidelity"]
        
        active_count = sum(1 for s in state.SIMULATION_STATE["sensors"] if not s["is_off"])
        t_step = state_copy["timestep"]
        total_sensors = state_copy["total_sensors"]
        total_possible = t_step * total_sensors if t_step > 0 else 1
        power_saved_percent = (state_copy["total_power_saved_steps"] / total_possible) * 100 if total_possible > 0 else 0
        
        response = {**state_copy}
        response["active_sensors"] = active_count
        response["power_saved_percent"] = power_saved_percent
        response["collected_shadow_samples"] = len(state.SIMULATION_STATE["shadow_samples"])
        
        current_readings = []
        if state.SIMULATION_STATE.get("data") is not None and t_step < len(state.SIMULATION_STATE["data"]):
            current_readings = state.SIMULATION_STATE["data"][t_step].tolist()
        response["current_readings"] = current_readings
        
        return response

@app.post('/start')
async def start_simulation(request: Request):
    with state.SIMULATION_STATE["lock"]:
        if not state.SIMULATION_STATE["is_running"]:
            params = await request.json()
            
            # Preserve essential loaded data properties during reset
            loaded_data = state.SIMULATION_STATE.get("data")
            total_sensors = state.SIMULATION_STATE.get("total_sensors", 0)
            
            # Get a fresh default state
            new_state = get_default_state()
            state.SIMULATION_STATE.clear()
            state.SIMULATION_STATE.update(new_state)
            
            # Restore the essential properties
            state.SIMULATION_STATE["data"] = loaded_data
            state.SIMULATION_STATE["total_sensors"] = total_sensors
            if total_sensors > 0:
                state.SIMULATION_STATE["sensors"] = [{"id": i, "is_off": False, "master_id": -1, "end_time": -1, "noise_variance": 0} for i in range(total_sensors)]

            state.SIMULATION_STATE.update({
                "threshold": params.get('threshold', state.SIMULATION_STATE['threshold']),
                "duration": params.get('duration', state.SIMULATION_STATE['duration']),
                "n_way_comparison": params.get('n_way_comparison', state.SIMULATION_STATE['n_way_comparison']),
                "shadow_mode_probability": params.get('shadow_mode_probability', state.SIMULATION_STATE['shadow_mode_probability']),
                "hybrid_fidelity_threshold": params.get('hybrid_fidelity_threshold', state.SIMULATION_STATE['hybrid_fidelity_threshold']),
                "hybrid_max_timesteps_since_retrain": params.get('hybrid_max_timesteps_since_retrain', state.SIMULATION_STATE['hybrid_max_timesteps_since_retrain']),
                "is_running": True
            })
            
            if state.operator_thread is None or not state.operator_thread.is_alive():
                state.operator_thread = threading.Thread(target=core.operator_loop)
                state.operator_thread.start()
            
            print("Simulation started with Hybrid Model.")
    return {"message": "Simulation started"}

@app.post('/pause')
async def pause_simulation():
    with state.SIMULATION_STATE["lock"]:
        state.SIMULATION_STATE["is_running"] = False
    return {"message": "Simulation paused"}

@app.post('/reset')
async def reset_simulation():
    with state.SIMULATION_STATE["lock"]:
        state.SIMULATION_STATE["is_running"] = False
        if state.operator_thread and state.operator_thread.is_alive():
            state.operator_thread.join(timeout=1.0)
        
        # Preserve essential loaded data properties
        loaded_data = state.SIMULATION_STATE.get("data")
        total_sensors = state.SIMULATION_STATE.get("total_sensors", 0)
        
        # Get a fresh default state
        new_state = get_default_state()
        state.SIMULATION_STATE.clear()
        state.SIMULATION_STATE.update(new_state)
        
        # Restore the essential properties
        state.SIMULATION_STATE["data"] = loaded_data
        state.SIMULATION_STATE["total_sensors"] = total_sensors
        if total_sensors > 0:
            state.SIMULATION_STATE["sensors"] = [{"id": i, "is_off": False, "master_id": -1, "end_time": -1, "noise_variance": 0} for i in range(total_sensors)]
    return {"message": "Simulation reset"}