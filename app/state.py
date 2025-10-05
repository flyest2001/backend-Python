import threading




def get_default_state():
    return {
        "is_running": False,
        "timestep": 0,
        "current_phase": "shadow_op", # Start in shadow mode
        
        # Core AURA parameters
        "threshold": 0.98, 
        "duration": 40,
        "n_way_comparison": 2,

        # Shadow mode configuration
        "shadow_mode_probability": 0.05,
        
        # Hybrid model triggers
        "hybrid_fidelity_threshold": 0.97,
        "hybrid_max_timesteps_since_retrain": 2880, # e.g., retrain at least every 2 days if 1 step = 1 minute
        "last_retrain_timestep": 0,
        "collection_period": 200,

        # Performance metrics
        "total_power_saved_steps": 0,
        "last_learner_fidelity": 1.0,
        "shadow_fidelity_error": 0.0,
        "shadow_fidelity_count": 0,
        "shadow_samples": [],
        
        # Sensor info will be populated dynamically after data load
        "sensors": [],
        "total_sensors": 0,
        "data": None,
        "lock": threading.Lock(),
        "learner_status": "idle"
    }

SIMULATION_STATE = get_default_state()
operator_thread = None
learner_thread = None
