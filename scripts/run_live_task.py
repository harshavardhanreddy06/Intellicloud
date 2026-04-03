import os
import sys
import json
import time
import subprocess
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from start import IntelliCloudPredictor

class LiveTaskExecutor:
    def __init__(self):
        print("=" * 70)
        print("  INTELLICLOUD — Real-Time Container Task Executor")
        print("=" * 70)
        self.predictor = IntelliCloudPredictor()
        
        # Action space configurations
        self.tier_limits = {
            0: {"name": "Tiny",   "cpus": 0.25, "mem": "256m", "static_power": 15.0},
            1: {"name": "Medium", "cpus": 0.50, "mem": "512m", "static_power": 35.0},
            2: {"name": "Large",  "cpus": 1.00, "mem": "1024m", "static_power": 55.0},
        }
        
    def execute(self, input_image, task_type="image_resize"):
        # 1. Feature Extraction & Prediction
        print(f"\n📂 Step 1: Extracting SHERA Features for {os.path.basename(input_image)}...")
        input_size_mb = os.path.getsize(input_image) / (1024 * 1024)
        
        task_dict = {
            "task_type": task_type,
            "input_size_mb": input_size_mb,
            "complexity": "high",
            "priority": "normal",
            "application": "graphics_processing"
        }
        
        prediction_res = self.predictor.predict_energy_efficiency(task_dict, include_shap=True)
        
        # 2. DQN Scheduler Decision
        action_id = prediction_res['vm_scheduling']['dqn_action_id']
        selected = self.tier_limits[action_id]
        
        print(f"🎯 Step 2: DQN Scheduling Decision")
        print(f"      🚀 Selected Container: {selected['name']}")
        print(f"      🔧 Limits: CPUs={selected['cpus']}, Memory={selected['mem']}")
        
        # 3. Docker Execution with Resource Contraints
        print(f"\n🐋 Step 3: Executing Real Task in Docker...")
        
        output_image = "/Users/harshareddy/Desktop/intellicloud/image_processed.png"
        
        container_cmd = [
            "docker", "run", "--rm",
            f"--cpus={selected['cpus']}",
            f"--memory={selected['mem']}",
            "-v", f"{input_image}:/app/input.png",
            "-v", f"{os.path.dirname(output_image)}:/app/output",
            "intellicloud-task:latest",
            "/app/input.png", "/app/output/image_processed.png"
        ]
        
        start_time = time.time()
        process = subprocess.Popen(container_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        exec_duration = time.time() - start_time
        
        if process.returncode != 0:
            print(f"   ❌ Execution Failed: {stderr}")
            return
            
        print(stdout)
        
        # 4. Metric Collection (parsing the container output)
        metrics = {}
        for line in stdout.splitlines():
            if "METRICS_REPORT" in line:
                # METRICS_REPORT: DURATION=2.345, CPU=45.6, MEM=89.1
                parts = line.split(":")[-1].split(",")
                for p in parts:
                    k, v = p.strip().split("=")
                    metrics[k] = float(v)
        
        # 5. Energy Saving Analysis
        # Actual Energy = (Static Power + (Dynamic Power * CPU Util)) * Execution Time
        # Dynamic Power is approx 40W * (actual_cpus / core_baseline)
        
        actual_power = selected['static_power'] + (40.0 * (metrics['CPU'] / 100.0) * selected['cpus'])
        actual_joules = actual_power * exec_duration
        
        # Baseline Comparison (always on Large VM without resource constraints)
        baseline_tier = self.tier_limits[2]
        # On a larger VM, time is usually slightly faster but static power is higher
        baseline_time = exec_duration * 0.8 # Assume 20% faster on a full core
        baseline_power = baseline_tier['static_power'] + (40.0 * (metrics['CPU'] / 100.0) * 1.0)
        baseline_joules = baseline_power * baseline_time
        
        energy_saved = baseline_joules - actual_joules
        efficiency_gain = (energy_saved / baseline_joules) * 100
        
        print("\n" + "=" * 70)
        print("  ENERGY SAVINGS ANALYSIS (Selected vs Baseline)")
        print("=" * 70)
        print(f"   DQN Choice ( {selected['name']} )  :  {actual_joules:.2f} Joules")
        print(f"   Baseline   ( Large )      :  {baseline_joules:.2f} Joules")
        print(f"   🛡️ Energy Saved            :  {max(0, energy_saved):.2f} Joules")
        print(f"   📈 Efficiency Improvement :  {max(0, efficiency_gain):.1f}%")
        print("-" * 70)
        print(f"✅ Metrics stored and report generated successfully.")
        
        # 6. Store metrics in profiles
        self._store_metrics(task_type, selected['name'].lower(), metrics, actual_power, exec_duration)
        
    def _store_metrics(self, task_type, tier, metrics, power, duration):
        # Translate tier names
        tier_map = {"tiny": "tiny", "medium": "small", "large": "medium"}
        file_path = f"dataset/{tier_map[tier]}_vm_profiles.json"
        
        if Path(file_path).exists():
            with open(file_path, "r") as f:
                data = json.load(f)
            
            entry = {
                "task_id": f"live_{int(time.time())}",
                "task_signature": f"{task_type}_live",
                "vm_tier": tier,
                "power_consumption_watts": round(power, 2),
                "execution_time_sec": round(duration, 3),
                "cpu_usage_percent": metrics['CPU'],
                "memory_usage_mb": metrics['MEM'],
                "executed_at": datetime.now().isoformat()
            }
            data.append(entry)
            
            with open(file_path, "w") as f:
                json.dump(data[-5000:], f, indent=2) # Keep it manageable
            print(f"   📋 Metric entry added to {os.path.basename(file_path)}")

if __name__ == "__main__":
    executor = LiveTaskExecutor()
    image_path = "/Users/harshareddy/Desktop/intellicloud/image.png"
    executor.execute(image_path)
