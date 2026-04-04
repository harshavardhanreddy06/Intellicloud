# DQN Scheduler

The Deep Q-Network (DQN) Scheduler is the final stage of the IntelliCloud intelligent pipeline. It determines the most efficient container tier (VM) for a given task.

## 1. Action Space (3 Tiers)
The DQN Agent selects from three possible deployment tiers:
- **Action 0: Tiny** (0.25 Cores, 1024MB Memory)
- **Action 1: Medium** (0.50 Cores, 2048MB Memory)
- **Action 2: Large** (1.00 Cores, 4096MB Memory)

The agent's goal is to pick the smallest (most energy-efficient) tier that can still meet the task's Service Level Agreement (SLA).

## 2. State Space (13 Dimensions)
The input to the DQN Agent is a 13-dimensional vector:
- **12 SHERA Features**: 8 from the feature extractor + 4 from the autoencoder.
- **1 RF Class**: The energy efficiency class (1-5) predicted by the Random Forest model.

## 3. How Points (Rewards) are Assigned
The DQN agent learns by maximizing its **cumulative reward** over time. The reward function is defined in `src/rl_scheduler/environment.py`:

### The Reward Formula
`Reward = 100.0 - (Processing Time * Priority Multiplier) - (Energy Consumption / 500.0)`

- **Base Success Bonus (+100.0)**: Initial points for successfully considering a task.
- **Latency Penalty (`Time * P_Mult`)**: Negative points based on processing time.
  - **Priority Multiplier (`P_Mult`)**:
    - Low: 1.0x
    - Medium: 5.0x
    - High: 15.0x
    - **Critical: 40.0x** (High penalty for slow processing!)
- **Energy Penalty (`Joules / 500.0`)**: Points subtracted for power consumed (Watt-seconds).

### Special Rules and Bonuses
- **SLA Penalty (-300.0)**: A massive penalty is applied if the task's processing time exceeds its **SLA Limit** (e.g., 10s for critical tasks).
  - **The Tiger Rule**: To prevent unfair penalization, the agent is **not** penalized for time if it has already selected the **Large (Tier 2)** VM (since it physically cannot go faster).
- **Efficiency Bonus (+20.0)**: Awarded if the agent handles a task in under 10 seconds using the **Tiny** VM.
- **Smart Resource Bonus (+30.0)**: Awarded for choosing the **Large** VM when the load is heavy (preventing SLA violations).

## 4. DQN Architecture
- **Policy Network**: 13-dim input -> 64-dim hidden -> 64-dim hidden -> 3-dim output (Q-values).
- **Optimizer**: Adam with 0.001 learning rate.
- **Experience Replay**: Stores historical (state, action, reward, next_state) tuples to break correlations during training.

## 5. Summary of Model Files
- `models/dqn_scheduler.pth`: Trained neural network weights for the DQN agent.
- `models/dqn_state_scaler.pkl`: The MinMaxScaler used to normalize the 13-dim state vector for the agent.
