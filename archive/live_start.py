import torch
import numpy as np
import json
import time
from pathlib import Path
from src.rl_scheduler.live_environment import LiveVMEnvironment
from src.rl_scheduler.dqn_agent import DQNAgent
from start import IntelliCloudPredictor

def run_live_scheduler(tasks_file="tasks.json", episodes=1, episodes_per_task=1):
    """
    Main loop for IntelliCloud DQN Live Scheduler.
    Loads tasks, lets DQN select VM, executes in Docker, collects metrics, and learns.
    """
    print("\n" + "=" * 80)
    print("🚀 INTELLICLOUD LIVE SCHEDULER: END-TO-END WORKFLOW")
    print("=" * 80)

    # 1. Initialize Environment and Predictor
    print("\n1. Initializing Live Environment (Docker connectivity)...")
    env = LiveVMEnvironment(tasks_file=tasks_file)
    n_tasks = len(env.tasks)
    print(f"   ✓ Tasks to process: {n_tasks}")

    # 2. Initialize Agent with latest weights (if any)
    print("\n2. Initializing DQN Agent with existing weights...")
    agent = DQNAgent(state_dim=13, action_dim=3)
    dqn_path = Path("models/dqn_scheduler.pth")
    if dqn_path.exists():
        agent.load(str(dqn_path))
        print(f"   ✓ DQN model loaded from {dqn_path}")
    else:
        print("   ⚠️ No trained DQN model found — using initial weights")
    
    # Pre-train state scaler from all task states if exist (from start.py)
    # predictor already exists in env.predictor
    
    # 3. Main Live Execution and Learning Loop
    print("\n3. Starting Live Execution and Training...")
    print("-" * 50)
    
    total_reward = 0.0
    completed_tasks = 0
    start_time = time.time()

    for i, task in enumerate(env.tasks):
        print(f"\n📋 Task {i+1}: {task['task_type']} ({task.get('input_size_mb', 'unknown')} MB)")
        
        # 3.1 Get SHERA Observation (13-dim)
        state = env._get_observation(task)
        
        # 3.2 DQN Action Selection (Greedy for deployment, but we can do epsilon-greedy if we want exploration)
        action = agent.select_action(state, training=True)
        tier_label = env.container_configs[action]['tier'].upper()
        
        # 3.3 Execute in Docker and get real feedback
        next_state, reward, done, profile = env.step(action)
        total_reward += reward
        completed_tasks += 1
        
        # 3.4 DQN LEARNING (Binary update: Store transition and perform one train_step)
        # This is the "Live Learning" part
        agent.store_transition(state, action, reward, next_state, done)
        agent.train_step()
        
        # 3.5 Regular Target Network Update
        if (i+1) % 5 == 0:
            agent.update_target_network()
            print(f"   ♻️ DQN target network synced.")

    # 4. Finalize
    print("\n" + "=" * 80)
    print(f"✅ LIVE WORKFLOW COMPLETE!")
    print(f"   Completed Tasks: {completed_tasks}")
    print(f"   Total Reward: {total_reward:.2f}")
    print(f"   Total Duration: {(time.time() - start_time)/60:.2f} min")
    
    # Save the updated model
    agent.save("models/dqn_scheduler.pth")
    print(f"   ✓ DQN model updated and saved to: models/dqn_scheduler.pth")
    print(f"   ✓ Datasets updated in dataset/...")
    print("-" * 80)
    print("Workflow Recap: Task → Encoder → RF → SHAP → DQN (live) → Docker → Feedback → DQN Update")
    print("=" * 80)

if __name__ == '__main__':
    run_live_scheduler()
