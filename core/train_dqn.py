"""
DQN Training Script for IntelliCloud Container Scheduler
Trains the agent to pick the best container (Tiny / Medium / Large)
for each incoming task, balancing resource efficiency vs. viability.
"""
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rl_scheduler.dqn_agent import DQNAgent
from src.rl_scheduler.environment import VMEnvironment

import joblib


def train_dqn(episodes=200, target_update=10):
    """Train DQN with the full dataset as one episode."""

    print("=" * 60)
    print("  DQN TRAINING — IntelliCloud Container Scheduler")
    print("=" * 60)

    # 1. Build environment (pre-computes + normalises states)
    env = VMEnvironment()
    n_tasks = len(env.tasks)
    print(f"  Tasks in dataset : {n_tasks}")
    print(f"  Episodes         : {episodes}")
    print(f"  State dim        : {env.state_dim}")
    print(f"  Action dim       : {env.action_dim}")

    # 2. Create agent
    agent = DQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        lr=5e-4,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.99,
        epsilon_min=0.05,
    )
    agent.batch_size = 32

    # 3. Logging
    rewards_history = []
    avg_rewards = []
    action_counts_history = []

    start = datetime.now()
    print("-" * 60)

    for ep in range(1, episodes + 1):
        state = env.reset()
        ep_reward = 0.0
        ep_actions = [0, 0, 0]

        for _ in range(n_tasks):
            action = agent.select_action(state, training=True)
            ep_actions[action] += 1

            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.train_step()

            state = next_state
            ep_reward += reward
            if done:
                break

        if ep % target_update == 0:
            agent.update_target_network()

        rewards_history.append(ep_reward)
        avg = np.mean(rewards_history[-20:])
        avg_rewards.append(avg)
        action_counts_history.append(ep_actions)

        if ep % 10 == 0 or ep == 1:
            t_pct = ep_actions[0] / n_tasks * 100
            m_pct = ep_actions[1] / n_tasks * 100
            l_pct = ep_actions[2] / n_tasks * 100
            print(
                f"  Ep {ep:>4}/{episodes} | "
                f"Reward {ep_reward:>8.1f} | "
                f"Avg(20) {avg:>8.1f} | "
                f"ε {agent.epsilon:.3f} | "
                f"T{t_pct:4.0f}% M{m_pct:4.0f}% L{l_pct:4.0f}%"
            )

    duration = datetime.now() - start
    print("-" * 60)
    print(f"  Training done in {duration}")

    # 4. Save model + state scaler (needed at inference)
    os.makedirs("models", exist_ok=True)
    agent.save("models/dqn_scheduler.pth")
    joblib.dump(env.state_scaler, "models/dqn_state_scaler.pkl")
    print("  ✓ DQN model  → models/dqn_scheduler.pth")
    print("  ✓ State scaler → models/dqn_state_scaler.pkl")

    # 5. Plot
    os.makedirs("results", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(rewards_history, alpha=0.3, label="Episode Reward")
    ax1.plot(avg_rewards, linewidth=2, label="20-ep Moving Avg")
    ax1.set_ylabel("Total Reward")
    ax1.set_title("DQN Training: Container Scheduler Reward Curve")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    counts = np.array(action_counts_history)
    ax2.stackplot(
        range(len(counts)),
        counts[:, 0], counts[:, 1], counts[:, 2],
        labels=["Tiny (0.25c/256MB)", "Medium (0.5c/512MB)", "Large (1c/1024MB)"],
        colors=["#2ecc71", "#f39c12", "#e74c3c"],
        alpha=0.8,
    )
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("# Tasks")
    ax2.set_title("Container Selection Distribution Over Training")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/dqn_training_rewards.png", dpi=150)
    plt.close()
    print("  ✓ Plot → results/dqn_training_rewards.png")
    print("=" * 60)


if __name__ == "__main__":
    train_dqn(episodes=100)
