import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from start import IntelliCloudPredictor
from src.rl_scheduler.dqn_agent import DQNAgent

class IntelliCloudScheduler:
    """Full IntelliCloud Scheduler using SHERA + DQN."""
    def __init__(self, dqn_model_path="models/dqn_scheduler.pth"):
        print("\n⚡ Initializing IntelliCloud Scheduler (SHERA + DQN) ⚡")
        self.predictor = IntelliCloudPredictor()
        
        # Action space mapping
        self.containers = [
            {"name": "Tiny", "cpu": 0.25, "memory": 256},
            {"name": "Medium", "cpu": 0.5, "memory": 512},
            {"name": "Large", "cpu": 1.0, "memory": 1024}
        ]
        
        # Load or create DQN agent
        self.agent = DQNAgent(state_dim=13, action_dim=3)
        if Path(dqn_model_path).exists():
            self.agent.load(dqn_model_path)
            print(f"   ✓ DQN Model loaded from: {dqn_model_path}")
        else:
            print("   ⚠️  No pre-trained DQN model found. Using initial weights.")
            
        print("✓ Scheduler is ready for production VM decisions.")

    def schedule_task(self, task_dict):
        """Complete workflow: Task → SHERA → DQN → Decision."""
        start_time = datetime.now()
        
        # 1. Prediction (RF + Encoder)
        res = self.predictor.predict_energy_efficiency(task_dict)
        if "error" in res:
            return {"error": "Prediction failed", "status": "failed"}

        # 2. State Preparation for DQN (13 dims)
        f = res['features']
        features_12 = [
            f['input_size_mb'], f['cpu_usage_cores_absolute'], f['memory_usage_mb'],
            f['execution_time_normalized'], f['instruction_count'], f['network_io_mb'],
            f['power_consumption_watts'], 
            0 if f['task_size_category'] == 'SMALL' else 1 if f['task_size_category'] == 'MEDIUM' else 2,
            f['latent_f1'], f['latent_f2'], f['latent_f3'], f['latent_f4']
        ]
        rf_class = res['prediction']['energy_efficiency_class']
        state = np.array(features_12 + [rf_class], dtype=np.float32)

        # 3. DQN Decision (Action)
        action = self.agent.select_action(state, training=False)
        selected_container = self.containers[action]
        
        # 4. Feedback Logic (Simulated for inference)
        req_cpu = float(task_dict.get('cpu_usage_cores_absolute', 0))
        req_mem = float(task_dict.get('memory_usage_mb', 0))
        
        is_viable = (req_cpu <= selected_container['cpu'] and req_mem <= selected_container['memory'])
        
        # 5. Build Final Response
        response = {
            "task_id": task_dict.get("task_id", "unknown"),
            "task_info": task_dict,
            "prediction": res['prediction'],
            "scheduling_decision": {
                "container_name": selected_container['name'],
                "allocated_cpu": selected_container['cpu'],
                "allocated_memory_mb": selected_container['memory'],
                "decision_confidence": "high",  # DQN-based
                "is_viable": is_viable,
                "reasoning": f"DQN selected {selected_container['name']} based on state features."
            },
            "timestamp": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            "processing_time_ms": (datetime.now() - start_time).total_seconds() * 1000,
            "shaping_report": res['prediction']['explanation_image']
        }
        
        return response

# ----------------------------------------------------------------------------
# Demo execution
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    scheduler = IntelliCloudScheduler()
    
    # Load sample tasks
    tasks_file = "tasks.json"
    try:
        with open(tasks_file, 'r') as f:
            tasks_data = json.load(f)
    except:
        tasks_data = [{"task_type": "text_search", "input_size_mb": 45, "cpu_usage_cores_absolute": 0.45, "memory_usage_mb": 128}]

    print("\nProcessing Tasks through IntelliCloud Scheduler:")
    print("-" * 60)
    for i, task in enumerate(tasks_data[:3], 1):
        print(f"\n📋 Task {i}: {task.get('task_type', 'unknown')}")
        result = scheduler.schedule_task(task)
        
        decision = result['scheduling_decision']
        print(f"   🎯 Decision  : {decision['container_name']} Container")
        print(f"   🧠 Reasoning : {decision['reasoning']}")
        print(f"   ✅ Viable    : {'YES' if decision['is_viable'] else 'NO (Capacity exceeded)'}")
        print(dim := f"   🖼️  SHAP Plot: {result['shaping_report']}")
