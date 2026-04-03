import numpy as np
import random
import json
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from start import IntelliCloudPredictor

# Action space mapping
# Index 0: Tiny   (0.25 Cores, 256MB)
# Index 1: Medium (0.50 Cores, 512MB)  - stored in small_vm_profiles.json
# Index 2: Large  (1.00 Cores, 1024MB) - stored in medium_vm_profiles.json

CONTAINERS = [
    {"name": "Tiny",   "cpu": 0.25, "memory": 256,  "tier": "tiny"},
    {"name": "Medium", "cpu": 0.50, "memory": 512,  "tier": "small"},
    {"name": "Large",  "cpu": 1.00, "memory": 1024, "tier": "medium"},
]

class VMEnvironment:
    """
    High-Fidelity RL Environment for IntelliCloud container scheduling.
    Uses 10,000 observations to calculate REAL rewards based on Power and Time.
    """

    def __init__(self, dataset_path="dataset/task_profiles_clean_final.json"):
        print("Initializing High-Fidelity RL Environment...")
        self.predictor = IntelliCloudPredictor()

        # 1. Load the 1,000 unique task profiles (for state extraction)
        with open(dataset_path, "r") as f:
            self.tasks = json.load(f)

        # 2. Skip legacy fidelity lookup since we use a predictive math model now.
        self.fidelity_data = {} 

        # 3. Pre-compute states for all 1,000 unique tasks
        print("   → Pre-computing states for 1,000 unique tasks...")
        self._raw_states = []
        for task in self.tasks:
            raw, _ = self._extract_raw(task)
            self._raw_states.append(raw)

        self._raw_states = np.array(self._raw_states, dtype=np.float32)
        
        # Fit a MinMaxScaler on ALL task states
        self.state_scaler = MinMaxScaler()
        self.state_scaler.fit(self._raw_states)

        self.current_task_idx = 0
        self._order = list(range(len(self.tasks)))
        print(f"✓ Environment ready! Training pool: {len(self.tasks)} unique tasks.")

    def _extract_raw(self, task):
        """Return (raw_13_array, prediction_result) for a task."""
        res = self.predictor.predict_energy_efficiency(task, include_shap=False)
        if "error" in res:
            return np.zeros(13), res

        f = res["features"]
        size_enc = {"SMALL": 0, "MEDIUM": 1, "LARGE": 2}
        features_12 = [
            f["input_size_mb"],
            f["cpu_usage_cores_absolute"],
            f["memory_usage_mb"],
            f["execution_time_normalized"],
            f["instruction_count"],
            f["network_io_mb"],
            f["power_consumption_watts"],
            size_enc.get(f["task_size_category"], 1),
            f["latent_f1"],
            f["latent_f2"],
            f["latent_f3"],
            f["latent_f4"],
        ]
        rf_class = res["prediction"]["energy_efficiency_class"]
        raw = np.array(features_12 + [rf_class], dtype=np.float32)
        return raw, res

    def _normalise(self, raw_state):
        return self.state_scaler.transform(raw_state.reshape(1, -1)).flatten()

    def reset(self):
        random.shuffle(self._order)
        self.current_task_idx = 0
        idx = self._order[self.current_task_idx]
        return self._normalise(self._raw_states[idx])

    def step(self, action):
        idx = self._order[self.current_task_idx]
        task = self.tasks[idx]
        signature = task.get("task_signature")
        container = CONTAINERS[action]
        target_tier = container["tier"]

        # ── Reward: Balanced Pareto comparison across all 3 VMs ──────────────
        try:
            res    = self.predictor.predict_energy_efficiency(task, include_shap=False)
            f      = res["features"]
            size_mb = float(f.get("input_size_mb", 1.0))
            task_t  = task.get("task_type", "")
            priority = task.get("priority", "medium")

            # ── Simulate compute time ─────────────────────────────────────────
            if 'vid' in task_t:
                mbps_per_core = 0.1   # video: Re-calibrated to reality (12MB = 120s @ 1.0c)
            elif 'img' in task_t:
                mbps_per_core = 16.0   # image: fast
            elif 'aud' in task_t:
                mbps_per_core = 7.0    
            elif 'pdf' in task_t:
                mbps_per_core = 12.0
            else:
                mbps_per_core = 10.0

            speed    = container["cpu"] * mbps_per_core
            # Add a 1.0s fixed overhead (Docker spawn + ffmpeg load)
            actual_processing_time = size_mb / max(0.01, speed)
            time_sec = actual_processing_time + 1.0

            # ── Simulate energy: Joules = power * time ────────────────────────
            idle_power = {"tiny": 20.0, "small": 40.0, "medium": 80.0}.get(target_tier, 40.0)
            dyn_power  = idle_power + 20.0 * container["cpu"]
            joules     = dyn_power * time_sec

            # ── Priority-aware latency weight (Even Heavier for latency focus) ─────────────────────────────────
            p_mult = {"low": 1.0, "medium": 5.0, "high": 15.0, "critical": 40.0}.get(priority, 5.0)

            # ── Shaped reward ─────────────────────────────────────────────────
            reward = 100.0  # Higher base success bonus
            reward -= (time_sec * p_mult)        # latency penalty 
            reward -= (joules / 500.0)           # energy penalty (minimal focus)

            # --- SMART SLA Logic ---
            SLA_LIMIT = {"low": 90, "medium": 45, "high": 20, "critical": 10}.get(priority, 45)
            
            if time_sec > SLA_LIMIT:
                # 🐯 THE TIGER RULE: If we are already using the BEST possible tier (2),
                # we don't penalize for time because we literally can't go faster.
                if target_tier == "medium":
                    reward -= 10.0   # small 'pity' penalty
                else:
                    reward -= 300.0  # MASSIVE FATAL penalty for any other tier that is too slow

            # Efficiency bonus: Only for extremely fast Tiny tasks
            if target_tier == "tiny" and time_sec < 10.0:
                reward += 20.0   
            elif target_tier == "medium":
                reward += 30.0   # Bonus for actually picking the strongest VM for heavy load

        except Exception as e:
            reward = -5.0

        # ── Advance ──────────────────────────────────────────────
        self.current_task_idx += 1
        done = self.current_task_idx >= len(self._order)

        if done:
            next_state = np.zeros(13, dtype=np.float32)
        else:
            next_idx = self._order[self.current_task_idx]
            next_state = self._normalise(self._raw_states[next_idx])

        return next_state, reward, done, {"task_signature": signature, "reward": reward}

    @property
    def state_dim(self):
        return 13

    @property
    def action_dim(self):
        return len(CONTAINERS)
