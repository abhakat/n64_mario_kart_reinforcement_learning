# N64 Mario Kart Reinforcement Learning - Learning & Implementation Guide

A progressive RL curriculum with integrated implementation tasks, designed for someone with ML/neural network experience who is new to reinforcement learning. Theory and hands-on coding are interleaved so you build as you learn.

---

## Setup: M1 Mac Local Environment

Before starting the checkpoints, set up your development environment.

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.9+ installed
- Git installed
- Docker Desktop installed

### Initial Setup

**1. Create project directory and virtual environment**
```bash
mkdir -p ~/mario_kart_reinforcement_learning
cd ~/mario_kart_reinforcement_learning
python3 -m venv venv
source venv/bin/activate
```

**2. Install core dependencies**
```bash
pip install --upgrade pip
pip install gymnasium stable-baselines3 torch torchvision numpy matplotlib tensorboard jupyter
```

**3. Verify MPS (Metal Performance Shaders) availability**
```python
# test_mps.py
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

if torch.backends.mps.is_available():
    device = torch.device("mps")
    x = torch.randn(3, 3, device=device)
    print(f"Tensor on MPS: {x.device}")
    print("MPS is working!")
else:
    print("MPS not available, will use CPU")
```

**4. Create requirements.txt**
```
gymnasium>=0.29.0
stable-baselines3>=2.0.0
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
matplotlib>=3.7.0
tensorboard>=2.14.0
jupyter>=1.0.0
opencv-python>=4.8.0
```

**5. Initialize git repository**
```bash
git init
echo "venv/
__pycache__/
*.pyc
logs/
models/
*.z64
*.n64
.DS_Store
" > .gitignore
git add .
git commit -m "Initial project structure"
```

### M1 Mac Notes

- **Use MPS for GPU acceleration**: PyTorch supports Apple's Metal Performance Shaders
- **Stable Baselines 3 + MPS**: SB3 will automatically use MPS if available
- **Docker on M1**: Use ARM64 images when possible for better performance
- **Memory**: M1 Macs share RAM between CPU and GPU - monitor memory usage

### Verification Checklist

- [ ] `python --version` shows 3.9+
- [ ] `docker --version` shows Docker installed
- [ ] Virtual environment activates without errors
- [ ] MPS test script shows "MPS is working!"
- [ ] Can import gymnasium and stable_baselines3

---

## Checkpoint 1: From Supervised to Reinforcement Learning

### Theory (30-45 min)

#### How RL Differs from Supervised Learning

You're familiar with supervised learning: given labeled data (X, y), learn a function f(X) → y. Reinforcement learning flips this paradigm entirely:

| Supervised Learning | Reinforcement Learning |
|---------------------|------------------------|
| Dataset provided upfront | Agent generates its own data through interaction |
| Labels tell you the correct answer | Rewards only tell you "how good" (not "what's correct") |
| IID assumption (samples independent) | Sequential, correlated experiences |
| Minimize loss on held-out test set | Maximize cumulative future reward |

#### The Agent-Environment Loop

```
┌─────────────────────────────────────────────┐
│                                             │
│    ┌─────────┐         ┌─────────────┐      │
│    │  Agent  │─action──▶│ Environment │      │
│    │         │◀─state───│             │      │
│    │         │◀─reward──│             │      │
│    └─────────┘         └─────────────┘      │
│                                             │
└─────────────────────────────────────────────┘

At each timestep t:
1. Agent observes state sₜ
2. Agent selects action aₜ
3. Environment returns reward rₜ and new state sₜ₊₁
4. Repeat
```

#### Key Concepts

**No Labeled Dataset**: In Mario Kart, nobody tells you "at this exact frame, press left." The agent must discover good driving through trial and error.

**Delayed Rewards**: You might make a great pass on lap 1, but only win the race 2 minutes later. How do you credit that early decision? This is the **credit assignment problem**.

**Exploration vs Exploitation**: Should you try the path you know works (exploit) or discover potentially better routes (explore)? Too much exploitation = stuck in local optima. Too much exploration = never converge.

#### Mario Kart Connection

In Mario Kart, you won't create a dataset of "good driving frames." Instead:
- State = pixels from the game screen (+ optionally RAM data)
- Action = controller inputs (accelerate, steer, etc.)
- Reward = designed by you (speed, progress, lap time)
- The agent learns by playing millions of frames

### Hands-On: Random Agent on CartPole

**Task 1.1**: Create and run a random agent

Create `checkpoints/01_random_agent.py`:
```python
import gymnasium as gym

# Create the CartPole environment
env = gym.make("CartPole-v1", render_mode="human")

# Run 5 episodes with random actions
for episode in range(5):
    observation, info = env.reset()
    total_reward = 0
    step = 0

    while True:
        # Random action (no learning yet!)
        action = env.action_space.sample()

        # Take the action
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        if terminated or truncated:
            print(f"Episode {episode + 1}: {step} steps, reward = {total_reward}")
            break

env.close()
```

**What to observe**:
1. The pole falls quickly with random actions (usually < 50 steps)
2. Reward is +1 for each step the pole stays up
3. The observation is 4 numbers: [cart_pos, cart_vel, pole_angle, pole_angular_vel]
4. Actions are discrete: 0 = push left, 1 = push right

**Task 1.2**: Extend the exercise
- [ ] Print the observation at each step - see how values change
- [ ] Count how many times the agent "survives" > 100 steps (rare!)
- [ ] Modify to run 100 episodes and compute average reward

### Checkpoint 1 Quiz

1. **In RL, where does the training data come from?**
   - a) A pre-collected dataset
   - b) The agent's own interactions with the environment
   - c) Human demonstrations
   - d) Random sampling

2. **What is the "credit assignment problem"?**
   - a) Deciding which agent gets credit for a team win
   - b) Figuring out which earlier actions led to a later reward
   - c) Assigning credit cards to agents
   - d) Calculating the reward function

3. **In the CartPole example, what does the reward represent?**
   - a) The angle of the pole
   - b) +1 for each timestep the pole stays balanced
   - c) Distance traveled by the cart
   - d) Whether the episode ended successfully

4. **Why can't we just use supervised learning for Mario Kart?**
   - a) It's too slow
   - b) We don't have labeled data of correct actions for each frame
   - c) Neural networks don't work with games
   - d) The state space is too small

<details>
<summary>Answers</summary>

1. **b)** The agent generates data through interaction
2. **b)** Determining which past actions caused future rewards
3. **b)** +1 per timestep while pole is balanced (survival reward)
4. **b)** No labeled dataset of "correct" driving decisions exists

</details>

---

## Checkpoint 2: MDPs and the Bellman Equation

### Theory (30-45 min)

#### Markov Decision Process (MDP)

An MDP formally defines an RL problem with 5 components:

**M = (S, A, P, R, γ)**

| Symbol | Meaning | Mario Kart Example |
|--------|---------|-------------------|
| S | State space | Game frames, RAM values |
| A | Action space | {accelerate, brake, left, right, ...} |
| P | Transition probability P(s'|s,a) | Game physics (deterministic) |
| R | Reward function R(s,a,s') | Speed, progress, penalties |
| γ | Discount factor (0 < γ ≤ 1) | Typically 0.99 |

**Markov Property**: The future depends only on the current state, not the history. This is why we often "stack" frames - to encode velocity information in the state.

#### Value Functions

**State Value V(s)**: Expected cumulative reward starting from state s

```
V(s) = E[rₜ + γrₜ₊₁ + γ²rₜ₊₂ + ... | sₜ = s]
     = E[Σ γᵏrₜ₊ₖ | sₜ = s]
```

**Action Value Q(s,a)**: Expected cumulative reward taking action a in state s

```
Q(s,a) = E[rₜ + γrₜ₊₁ + γ²rₜ₊₂ + ... | sₜ = s, aₜ = a]
```

#### The Bellman Equation

The key insight: value functions satisfy a recursive relationship.

```
V(s) = R(s) + γ * max_a Σ P(s'|s,a) * V(s')
```

In words: "The value of a state equals the immediate reward plus the discounted value of the best next state."

For Q-values:
```
Q(s,a) = R(s,a) + γ * Σ P(s'|s,a) * max_a' Q(s',a')
```

#### Discount Factor γ

Why discount future rewards?

```
γ = 0.99: Values 100 steps ahead by 0.99¹⁰⁰ ≈ 0.37
γ = 0.95: Values 100 steps ahead by 0.95¹⁰⁰ ≈ 0.006
γ = 1.00: No discounting (can cause infinite values in continuing tasks)
```

- High γ (0.99): Agent plans far ahead (good for Mario Kart - lap strategy matters)
- Low γ (0.9): Agent is more "myopic" (good for quick reactions)

#### Q-Learning Algorithm

```
Initialize Q(s,a) arbitrarily
For each episode:
    s = initial state
    While not terminal:
        a = ε-greedy action from Q(s,·)     # Explore vs exploit
        Take action a, observe r, s'
        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]  # Bellman update
        s ← s'
```

### Hands-On: Tabular Q-Learning on FrozenLake

**Task 2.1**: Implement Q-learning from scratch

Create `checkpoints/02_q_learning.py`:
```python
import gymnasium as gym
import numpy as np

# FrozenLake: navigate a 4x4 frozen lake without falling in holes
env = gym.make("FrozenLake-v1", is_slippery=False)

# Initialize Q-table (16 states × 4 actions)
Q = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameters
alpha = 0.1      # Learning rate
gamma = 0.99     # Discount factor
epsilon = 0.1    # Exploration rate
episodes = 10000

# Training loop
rewards_per_episode = []

for episode in range(episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(Q[state])        # Exploit

        # Take action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Q-learning update (Bellman equation!)
        best_next = np.max(Q[next_state]) if not done else 0
        Q[state, action] += alpha * (reward + gamma * best_next - Q[state, action])

        state = next_state
        total_reward += reward

    rewards_per_episode.append(total_reward)

    # Print progress every 1000 episodes
    if (episode + 1) % 1000 == 0:
        avg_reward = np.mean(rewards_per_episode[-1000:])
        print(f"Episode {episode + 1}: Avg reward = {avg_reward:.3f}")

# Test the learned policy
print("\nLearned Q-table (showing best action per state):")
print(np.argmax(Q, axis=1).reshape(4, 4))
print("\nAction mapping: 0=Left, 1=Down, 2=Right, 3=Up")

# Visualize one episode
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")
state, _ = env.reset()
done = False
while not done:
    action = np.argmax(Q[state])
    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
env.close()
```

**Expected output**:
- Average reward approaches 1.0 (goal reached)
- Q-table shows a clear path from start (0,0) to goal (3,3)

**Task 2.2**: Experiments
- [ ] Set `is_slippery=True` - how does stochastic transitions affect learning?
- [ ] Change γ to 0.5 - does the agent still find optimal path?
- [ ] Set ε = 0 - what happens without exploration?

### Checkpoint 2 Quiz

1. **What does γ = 0.99 mean?**
   - a) 99% chance of random action
   - b) A reward 100 steps in the future is worth about 37% of an immediate reward
   - c) Learning rate is 0.99
   - d) 99% of actions are optimal

2. **In Q(s,a), what does the function output?**
   - a) The immediate reward for taking action a in state s
   - b) The probability of action a in state s
   - c) The expected cumulative discounted reward from (s,a) onwards
   - d) Whether action a is legal in state s

3. **Why do we use ε-greedy exploration?**
   - a) To make training faster
   - b) To ensure we discover potentially better actions, not just exploit known ones
   - c) To reduce memory usage
   - d) To satisfy the Markov property

4. **The Bellman equation says V(s) = R(s) + γ * max V(s'). Why "max"?**
   - a) We assume the agent acts optimally going forward
   - b) We want the maximum possible reward
   - c) To ensure convergence
   - d) It's just a convention

<details>
<summary>Answers</summary>

1. **b)** 0.99¹⁰⁰ ≈ 0.37, so distant rewards are worth less
2. **c)** Expected total discounted future reward from that state-action pair
3. **b)** Balance exploitation of known good actions with exploration of unknowns
4. **a)** Optimal value assumes optimal future actions

</details>

---

## Checkpoint 3: Policy Gradients (The Key Insight)

### Theory (30-45 min)

#### Value-Based vs Policy-Based Methods

So far, Q-learning learns a value function and derives the policy from it:
```
π(s) = argmax_a Q(s,a)
```

**Policy gradient methods** learn the policy directly:
```
π_θ(a|s) = P(action = a | state = s, parameters = θ)
```

#### Why Policy Gradients?

| Value-Based (Q-Learning) | Policy-Based |
|--------------------------|--------------|
| Deterministic policy | Stochastic policy (probabilities) |
| Discrete actions only | Handles continuous actions naturally |
| Can oscillate, hard to converge | Smoother optimization |
| Finds locally optimal policy | Can represent any policy |

**For Mario Kart**: Steering is continuous (-1 to +1), making policy gradients a natural fit.

#### The Policy Gradient Theorem

The goal: maximize expected cumulative reward J(θ)

```
J(θ) = E_τ~π_θ [R(τ)]
```

Where τ is a trajectory (sequence of states, actions, rewards).

The gradient (how to update θ):
```
∇J(θ) = E_τ [Σₜ ∇log π_θ(aₜ|sₜ) · Gₜ]
```

Where Gₜ = Σₖ γᵏrₜ₊ₖ is the return from time t.

**Intuition**:
- Increase probability of actions that led to high returns
- Decrease probability of actions that led to low returns
- Weighted by how good the outcome was

#### REINFORCE Algorithm

```
Initialize θ randomly
For each episode:
    Generate trajectory τ = (s₀,a₀,r₀, s₁,a₁,r₁, ...)
    For t = 0 to T:
        Gₜ = Σₖ γᵏrₜ₊ₖ              # Compute return
        θ ← θ + α · ∇log π_θ(aₜ|sₜ) · Gₜ   # Gradient ascent
```

#### Actor-Critic: The Best of Both Worlds

REINFORCE has high variance (returns can vary a lot). Solution: use a value function as a "baseline."

**Actor**: The policy π_θ(a|s) - decides what actions to take
**Critic**: The value function V_φ(s) - evaluates how good states are

Update rule:
```
Advantage: A(s,a) = Q(s,a) - V(s) ≈ r + γV(s') - V(s)
Actor: θ ← θ + α · ∇log π_θ(a|s) · A(s,a)
Critic: φ ← φ - α · ∇(V_φ(s) - (r + γV(s')))²
```

**Advantage** tells us: "Was this action better or worse than average for this state?"

### Hands-On: REINFORCE on CartPole

**Task 3.1**: Implement REINFORCE from scratch

Create `checkpoints/03_reinforce.py`:
```python
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# Use MPS if available (M1 Mac), otherwise CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Simple policy network
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.network(x)

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        probs = self.forward(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

# Training setup
env = gym.make("CartPole-v1")
policy = PolicyNetwork(state_dim=4, action_dim=2).to(device)
optimizer = optim.Adam(policy.parameters(), lr=0.01)
gamma = 0.99

def compute_returns(rewards, gamma):
    """Compute discounted returns for each timestep."""
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns).to(device)
    # Normalize returns (reduces variance)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns

# Training loop
episode_rewards = []

for episode in range(500):
    state, _ = env.reset()
    log_probs = []
    rewards = []

    # Generate episode
    done = False
    while not done:
        action, log_prob = policy.get_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        log_probs.append(log_prob)
        rewards.append(reward)
        state = next_state

    episode_rewards.append(sum(rewards))

    # REINFORCE update
    returns = compute_returns(rewards, gamma)

    # Policy gradient loss (negative because we want gradient ASCENT)
    loss = 0
    for log_prob, G in zip(log_probs, returns):
        loss -= log_prob * G

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print progress
    if (episode + 1) % 50 == 0:
        avg_reward = np.mean(episode_rewards[-50:])
        print(f"Episode {episode + 1}: Avg reward = {avg_reward:.1f}")

env.close()

# Test the trained policy
print("\nTesting trained policy...")
env = gym.make("CartPole-v1", render_mode="human")
for _ in range(3):
    state, _ = env.reset()
    total_reward = 0
    done = False
    while not done:
        with torch.no_grad():
            action, _ = policy.get_action(state)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
    print(f"Test episode reward: {total_reward}")
env.close()
```

**Expected output**:
- Rewards should increase from ~20 to ~200-500 over training
- Test episodes should achieve 200+ (often max 500)

**Task 3.2**: Experiments
- [ ] Remove return normalization - observe higher variance
- [ ] Try different learning rates (0.001, 0.1)
- [ ] Change hidden_dim to 32 and 256 - compare learning speed

### Checkpoint 3 Quiz

1. **What does π_θ(a|s) represent?**
   - a) The value of taking action a in state s
   - b) The probability of choosing action a given state s
   - c) The reward for action a in state s
   - d) The optimal action in state s

2. **Why are policy gradients better for continuous action spaces?**
   - a) They use less memory
   - b) They can output probabilities over a continuous distribution
   - c) They train faster
   - d) They don't need neural networks

3. **In REINFORCE, we multiply ∇log π(a|s) by the return G. Why?**
   - a) To normalize the gradients
   - b) To weight updates by how good the outcome was
   - c) To ensure convergence
   - d) It's required by PyTorch

4. **What is the "Advantage" in Actor-Critic?**
   - a) The total episode reward
   - b) How much better an action was compared to the average for that state
   - c) The learning rate
   - d) The policy's confidence in an action

<details>
<summary>Answers</summary>

1. **b)** It's a probability distribution over actions conditioned on state
2. **b)** Can parameterize continuous distributions (e.g., Gaussian) for actions
3. **b)** Actions leading to high returns get higher probability increases
4. **b)** A(s,a) = Q(s,a) - V(s) measures "how much better than average"

</details>

---

## Checkpoint 4: PPO - The Workhorse Algorithm

### Theory (30-45 min)

#### The Problem with Vanilla Policy Gradients

Policy gradient methods (like REINFORCE) can be unstable:
- Large gradient steps can drastically change the policy
- A bad update can make the policy much worse
- Hard to recover from bad updates (you're collecting data with the bad policy!)

#### Trust Region Methods

Idea: Limit how much the policy can change in one update.

**TRPO** (Trust Region Policy Optimization) explicitly constrains the KL divergence:
```
maximize L(θ) subject to KL(π_old || π_new) ≤ δ
```

But TRPO requires complex second-order optimization (computing Hessians).

#### PPO: Practical Trust Regions

PPO achieves similar stability with a simpler approach: **clipping**.

**PPO Clipped Objective**:
```
L(θ) = E[min(rₜ(θ)Aₜ, clip(rₜ(θ), 1-ε, 1+ε)Aₜ)]

where rₜ(θ) = π_θ(aₜ|sₜ) / π_old(aₜ|sₜ)  # Probability ratio
```

**Intuition**:
- rₜ(θ) = 1 means new policy = old policy for this action
- rₜ(θ) > 1 means new policy MORE likely to take this action
- rₜ(θ) < 1 means new policy LESS likely to take this action
- Clipping to [1-ε, 1+ε] prevents too large changes

#### Generalized Advantage Estimation (GAE)

How to compute advantages? GAE balances bias and variance:

```
δₜ = rₜ + γV(sₜ₊₁) - V(sₜ)           # TD error

Aₜ = δₜ + (γλ)δₜ₊₁ + (γλ)²δₜ₊₂ + ...  # GAE
```

- λ = 0: High bias, low variance (just TD error)
- λ = 1: Low bias, high variance (Monte Carlo returns)
- λ = 0.95: Good practical default

#### Key PPO Hyperparameters

| Parameter | Typical Value | Effect |
|-----------|---------------|--------|
| `clip_range` | 0.2 | How much policy can change per update |
| `learning_rate` | 3e-4 | Step size for optimization |
| `n_steps` | 2048 | Steps to collect before each update |
| `batch_size` | 64 | Minibatch size for updates |
| `n_epochs` | 10 | Passes over collected data |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE λ for advantage estimation |
| `ent_coef` | 0.0-0.01 | Entropy bonus (encourages exploration) |

### Hands-On: PPO on LunarLander with Stable Baselines 3

**Task 4.1**: Train PPO using Stable Baselines 3

Create `checkpoints/04_ppo_lunarlander.py`:
```python
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
import numpy as np

# Create vectorized environment (runs multiple envs in parallel)
env = make_vec_env("LunarLander-v3", n_envs=4)

# Create PPO agent with default hyperparameters
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    verbose=1,
    tensorboard_log="./logs/"
)

# Train for 100k steps
print("Training PPO on LunarLander...")
model.learn(total_timesteps=100_000)

# Evaluate the trained agent
print("\nEvaluating trained agent...")
eval_env = gym.make("LunarLander-v3", render_mode="human")
rewards = []

for episode in range(10):
    obs, _ = eval_env.reset()
    total_reward = 0
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = eval_env.step(action)
        total_reward += reward
        done = terminated or truncated

    rewards.append(total_reward)
    print(f"Episode {episode + 1}: Reward = {total_reward:.1f}")

print(f"\nMean reward: {np.mean(rewards):.1f} (+/- {np.std(rewards):.1f})")
eval_env.close()

# Save the model
model.save("ppo_lunarlander")
print("\nModel saved to ppo_lunarlander.zip")
```

**Expected output**:
- Training shows increasing `ep_rew_mean` (episode reward mean)
- After 100k steps, rewards should be > 200 (landing successfully)
- Evaluation shows the lander touching down smoothly

**Task 4.2**: Hyperparameter Experiments

Create `checkpoints/04_ppo_experiments.py`:
```python
# Experiment 1: Effect of clip_range
for clip in [0.1, 0.2, 0.3]:
    model = PPO("MlpPolicy", env, clip_range=clip, verbose=0)
    model.learn(total_timesteps=50_000)
    # Evaluate and record

# Experiment 2: Effect of learning_rate
for lr in [1e-4, 3e-4, 1e-3]:
    model = PPO("MlpPolicy", env, learning_rate=lr, verbose=0)
    model.learn(total_timesteps=50_000)
    # Evaluate and record

# Experiment 3: Effect of n_steps (how much data per update)
for n_steps in [512, 1024, 2048]:
    model = PPO("MlpPolicy", env, n_steps=n_steps, verbose=0)
    model.learn(total_timesteps=50_000)
    # Evaluate and record
```

**Task 4.3**: TensorBoard visualization
```bash
tensorboard --logdir ./logs/
```
Navigate to http://localhost:6006 to see learning curves.

### Checkpoint 4 Quiz

1. **What does the "clip" in PPO's clipped objective do?**
   - a) Removes outlier data points
   - b) Limits how much the policy can change in one update
   - c) Clips gradient magnitudes
   - d) Removes low-reward trajectories

2. **If rₜ(θ) = 1.5 and clip_range = 0.2, what value is used?**
   - a) 1.5
   - b) 1.2 (clipped to 1 + 0.2)
   - c) 0.8
   - d) 1.0

3. **What is the purpose of GAE (Generalized Advantage Estimation)?**
   - a) To compute the policy gradient
   - b) To balance bias-variance in advantage estimation
   - c) To clip the objective
   - d) To parallelize training

4. **Why does PPO use multiple passes (n_epochs) over collected data?**
   - a) To save memory
   - b) To get more gradient updates from limited environment interactions
   - c) To increase exploration
   - d) To satisfy the Markov property

<details>
<summary>Answers</summary>

1. **b)** Clipping the probability ratio limits policy change magnitude
2. **b)** 1.5 is clipped to 1.2 (max is 1 + ε = 1.2)
3. **b)** λ parameter trades off bias (λ=0) vs variance (λ=1)
4. **b)** Sample efficiency - environment interaction is expensive

</details>

---

## Checkpoint 5: CNNs for Visual RL (You Know This!)

### Theory (30-45 min)

#### Connecting to Your CNN Knowledge

You already know CNNs:
- Convolutional layers detect features (edges → shapes → objects)
- Pooling reduces dimensionality
- Fully connected layers make final predictions

In visual RL, the CNN is the **feature extractor** that processes raw pixels into a representation the policy can use.

#### The NatureCNN Architecture (from DQN paper)

```
Input: 84 × 84 × 4 (grayscale, 4 stacked frames)
        ↓
Conv2D: 32 filters, 8×8, stride 4, ReLU
        ↓
Conv2D: 64 filters, 4×4, stride 2, ReLU
        ↓
Conv2D: 64 filters, 3×3, stride 1, ReLU
        ↓
Flatten: 3136 features
        ↓
Linear: 512, ReLU
        ↓
Linear: action_dim (policy head)
        ↓
Linear: 1 (value head)
```

This architecture is battle-tested and works well for many visual RL tasks.

#### Frame Stacking: Encoding Motion

A single frame doesn't capture velocity! Stack 4 consecutive frames:

```
Frame Stack (4 frames):
┌─────────────────────────────────┐
│ t-3 │ t-2 │ t-1 │  t  │  →  CNN
└─────────────────────────────────┘
  ↑                        ↑
 Past                   Present

The CNN learns to detect motion by comparing frames.
```

**Why 4 frames?**
- 2 frames: Only velocity, no acceleration
- 4 frames: Velocity + acceleration patterns
- More frames: Diminishing returns, more memory

#### Preprocessing Pipeline

Raw game frames → neural network input:

```python
# Typical preprocessing for visual RL
1. RGB to Grayscale    # 3 channels → 1 channel (3x less memory)
2. Resize to 84×84     # Standard size (256×224 → 84×84)
3. Normalize to [0,1]  # Divide by 255
4. Stack 4 frames      # Capture motion
5. Frame skip (4)      # Take action every 4th frame (speeds up training)

Final shape: (4, 84, 84) or (84, 84, 4) depending on convention
```

### Hands-On: PPO with CNN on Atari Pong

**Task 5.1**: Install Atari dependencies
```bash
pip install "gymnasium[atari,accept-rom-license]"
```

**Task 5.2**: Train CNN policy

Create `checkpoints/05_ppo_pong.py`:
```python
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.env_util import make_atari_env
import numpy as np

# Create Atari environment with standard wrappers
# This handles: NoopReset, MaxAndSkip, EpisodicLife, FireReset,
#               ClipReward, Resize(84x84), Grayscale
env = make_atari_env("PongNoFrameskip-v4", n_envs=4, seed=42)

# Add frame stacking (4 frames)
env = VecFrameStack(env, n_stack=4)

print(f"Observation shape: {env.observation_space.shape}")
# Expected: (4, 84, 84) - 4 stacked 84x84 grayscale frames

# Create PPO with CNN policy
model = PPO(
    "CnnPolicy",        # Uses NatureCNN architecture
    env,
    learning_rate=2.5e-4,
    n_steps=128,        # Lower for Atari (faster updates)
    batch_size=256,
    n_epochs=4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.1,     # Smaller clip for Atari
    ent_coef=0.01,      # Entropy bonus for exploration
    verbose=1,
    tensorboard_log="./logs/pong/"
)

# Train (Pong typically needs ~1M steps for good play)
print("Training PPO with CnnPolicy on Pong...")
print("Note: This will take a while! Start with 200k for a checkpoint.")
model.learn(total_timesteps=200_000)

# Save checkpoint
model.save("ppo_pong_200k")
print("Checkpoint saved!")

# Quick evaluation
print("\nEvaluating...")
eval_env = make_atari_env("PongNoFrameskip-v4", n_envs=1, seed=123)
eval_env = VecFrameStack(eval_env, n_stack=4)

obs = eval_env.reset()
total_reward = 0
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = eval_env.step(action)
    total_reward += reward[0]

print(f"Evaluation reward: {total_reward}")
# Score of +21 means perfect play, -21 means opponent won
# After 200k steps, expect around -10 to 0

eval_env.close()
```

**M1 Mac Note**: Atari training on M1 will use CPU by default for the emulation. MPS acceleration helps with the neural network forward/backward passes but the Atari emulation itself runs on CPU. Training times will be longer than on a dedicated GPU machine.

**Task 5.3**: Visualizing CNN Features (Optional)

```python
import torch
import matplotlib.pyplot as plt

# Get the CNN feature extractor from the trained model
cnn = model.policy.features_extractor

# Get a sample observation
obs = env.reset()
obs_tensor = torch.tensor(obs).float()

# Extract features
with torch.no_grad():
    features = cnn(obs_tensor)

print(f"CNN output shape: {features.shape}")
# Expected: (n_envs, 512) - 512 features per environment

# Visualize the 4 stacked frames
fig, axes = plt.subplots(1, 4, figsize=(12, 3))
for i in range(4):
    axes[i].imshow(obs[0, i], cmap='gray')
    axes[i].set_title(f'Frame {i}')
    axes[i].axis('off')
plt.suptitle('Stacked Frames Input to CNN')
plt.tight_layout()
plt.savefig('frame_stack_visualization.png')
print("Saved frame_stack_visualization.png")
```

### Checkpoint 5 Quiz

1. **Why do we stack 4 frames for visual RL?**
   - a) To increase resolution
   - b) To encode motion/velocity information
   - c) To reduce memory usage
   - d) To match the NatureCNN architecture requirement

2. **What is the purpose of frame skipping (taking action every 4 frames)?**
   - a) To reduce computational cost and speed up training
   - b) To increase frame rate
   - c) To prevent overfitting
   - d) To satisfy the Markov property

3. **Why convert to grayscale?**
   - a) Games look better in grayscale
   - b) Reduces input size from 3 channels to 1 (faster, less memory)
   - c) CNNs can't process color
   - d) Color information is never useful

4. **In "CnnPolicy", what does the CNN learn to detect?**
   - a) Object boundaries only
   - b) Task-relevant visual features (ball position, paddle, motion)
   - c) Text on screen
   - d) Exact pixel values to memorize

<details>
<summary>Answers</summary>

1. **b)** Single frames don't show which direction things are moving
2. **a)** Each action is repeated 4 frames, reducing computation 4x
3. **b)** 3x less data to process; for many games color isn't critical
4. **b)** The CNN learns features useful for the task through training

</details>

---

## Checkpoint 6: Gymnasium - The Standard Interface

### Theory (30-45 min)

#### The Gymnasium API

Gymnasium (formerly OpenAI Gym) defines a standard interface all RL environments implement:

```python
import gymnasium as gym

env = gym.make("CartPole-v1")

# Reset to initial state
observation, info = env.reset(seed=42)

# Main loop
for _ in range(1000):
    action = env.action_space.sample()  # Your policy goes here
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

#### The Step Return Values

```python
observation, reward, terminated, truncated, info = env.step(action)
```

| Value | Type | Meaning |
|-------|------|---------|
| `observation` | np.array | Current state (pixels, numbers, etc.) |
| `reward` | float | Reward for the action taken |
| `terminated` | bool | Episode ended (goal reached, died, etc.) |
| `truncated` | bool | Episode cut short (time limit, etc.) |
| `info` | dict | Extra debugging information |

**terminated vs truncated**:
- `terminated`: Natural episode end (Mario falls off track = terminated)
- `truncated`: Artificial end (max 1000 steps reached = truncated)

#### Spaces: Defining Observations and Actions

```python
# Observation space examples
env.observation_space

# Discrete: integers from 0 to n-1
gym.spaces.Discrete(4)  # {0, 1, 2, 3}

# Box: continuous bounded arrays
gym.spaces.Box(low=0, high=255, shape=(84, 84, 4), dtype=np.uint8)  # Image
gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)  # Continuous state

# Action space examples
env.action_space

gym.spaces.Discrete(4)  # 4 discrete actions
gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)  # Continuous control

# Sample random valid action
action = env.action_space.sample()
```

#### Wrappers: Modular Environment Modification

Wrappers let you modify environments without changing the original code:

```python
class MyWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        # Modify action before sending to env
        obs, reward, term, trunc, info = self.env.step(action)
        # Modify return values
        return obs, reward, term, trunc, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Modify initial observation
        return obs, info
```

### Hands-On: Create Custom Wrappers

**Task 6.1**: Implement custom wrappers

Create `src/wrappers/custom_wrappers.py`:
```python
import gymnasium as gym
import numpy as np
from gymnasium import ObservationWrapper, ActionWrapper, RewardWrapper, Wrapper
from gymnasium.spaces import Box

# 1. Logging Wrapper - Logs every action taken
class LoggingWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_history = []
        self.reward_history = []

    def step(self, action):
        self.action_history.append(action)
        obs, reward, term, trunc, info = self.env.step(action)
        self.reward_history.append(reward)
        return obs, reward, term, trunc, info

    def reset(self, **kwargs):
        if self.action_history:
            print(f"Episode summary: {len(self.action_history)} steps, "
                  f"total reward: {sum(self.reward_history):.2f}")
        self.action_history = []
        self.reward_history = []
        return self.env.reset(**kwargs)

# 2. Reward Scaling Wrapper
class ScaledRewardWrapper(RewardWrapper):
    def __init__(self, env, scale=0.1):
        super().__init__(env)
        self.scale = scale

    def reward(self, reward):
        return reward * self.scale

# 3. Observation Normalization Wrapper
class NormalizeObservationWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Assuming continuous observation space
        low = self.observation_space.low
        high = self.observation_space.high
        self.mean = (low + high) / 2
        self.scale = (high - low) / 2

        # Update observation space to normalized range
        self.observation_space = Box(
            low=-1, high=1,
            shape=env.observation_space.shape,
            dtype=np.float32
        )

    def observation(self, obs):
        return ((obs - self.mean) / self.scale).astype(np.float32)

# 4. Frame Skip Wrapper (for visual environments)
class FrameSkipWrapper(Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0
        for _ in range(self.skip):
            obs, reward, term, trunc, info = self.env.step(action)
            total_reward += reward
            if term or trunc:
                break
        return obs, total_reward, term, trunc, info


# Test the wrappers
if __name__ == "__main__":
    print("Testing wrappers on CartPole...")

    # Stack wrappers (order matters!)
    env = gym.make("CartPole-v1")
    env = LoggingWrapper(env)
    env = ScaledRewardWrapper(env, scale=0.1)
    env = NormalizeObservationWrapper(env)

    obs, info = env.reset()
    print(f"Normalized observation: {obs}")
    print(f"Observation range: [{obs.min():.2f}, {obs.max():.2f}]")

    # Run an episode
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc

    env.reset()  # Triggers logging output
    env.close()
```

**Task 6.2**: Test wrapper composition

```python
# Demonstrate wrapper chaining
from gymnasium.wrappers import TimeLimit, RecordVideo, RecordEpisodeStatistics

env = gym.make("CartPole-v1")
env = RecordEpisodeStatistics(env)  # Track episode stats
env = TimeLimit(env, max_episode_steps=200)  # Limit episode length

for episode in range(3):
    obs, info = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc

    # RecordEpisodeStatistics adds these to info on episode end
    if "episode" in info:
        print(f"Episode {episode + 1}: "
              f"reward={info['episode']['r']:.0f}, "
              f"length={info['episode']['l']}")
```

### Checkpoint 6 Quiz

1. **What's the difference between `terminated` and `truncated`?**
   - a) They mean the same thing
   - b) `terminated` is natural episode end, `truncated` is artificial cutoff
   - c) `terminated` means success, `truncated` means failure
   - d) `truncated` is for visual environments only

2. **Why use wrappers instead of modifying the environment directly?**
   - a) Wrappers are faster
   - b) Modularity - mix and match transformations without changing source
   - c) The environment code is read-only
   - d) Wrappers use less memory

3. **What does `env.action_space.sample()` return?**
   - a) The optimal action
   - b) A random valid action from the action space
   - c) All possible actions
   - d) The most recent action

4. **If observation is Box(low=0, high=255, shape=(84,84,4)), what does it represent?**
   - a) 4 grayscale 84x84 images (pixel values 0-255)
   - b) An 84x84 RGB image
   - c) 4 discrete actions
   - d) A single 84-dimensional vector

<details>
<summary>Answers</summary>

1. **b)** `terminated` = goal/death/natural end; `truncated` = time limit/forced end
2. **b)** Wrappers are composable and reusable across environments
3. **b)** Random action from the defined action space
4. **a)** 4 stacked grayscale frames, each 84x84 with values 0-255

</details>

---

## Checkpoint 7: Emulator Setup & Integration

**This checkpoint ends with a critical decision point about whether to continue on M1 Mac or switch to cloud (RunPod).**

### Theory (30-45 min)

#### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       Your Training Loop                     │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  PPO Agent (Stable Baselines 3)                         ││
│  │    ↓ action                          ↑ (obs, reward)    ││
│  └─────────────────────────────────────────────────────────┘│
│                            ↓                    ↑            │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  gym-mupen64plus (Gymnasium Wrapper)                    ││
│  │    - Implements step(), reset()                         ││
│  │    - Reads frames from emulator                         ││
│  │    - Sends controller inputs                            ││
│  │    - Computes rewards from RAM                          ││
│  └─────────────────────────────────────────────────────────┘│
│                            ↓                    ↑            │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  Mupen64Plus Emulator (in Docker container)             ││
│  │    - N64 hardware emulation                             ││
│  │    - ROM execution                                      ││
│  │    - Memory (RAM) state                                 ││
│  │    - Video output                                       ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

#### Mupen64Plus

Mupen64Plus is an open-source N64 emulator with a plugin architecture:

- **Core**: CPU emulation, memory management
- **Video Plugin**: Graphics rendering (we'll use a simple plugin for speed)
- **Input Plugin**: Controller input (we override this!)
- **Audio Plugin**: Sound (can be disabled for speed)

Key features for RL:
- **Savestates**: Save/load exact game state
- **Frame advance**: Step one frame at a time
- **Memory access**: Read game RAM directly

#### Docker: Why Containerize?

The emulator setup involves many dependencies:
- Mupen64Plus core + plugins
- SDL libraries for graphics
- X11 for display
- Python bindings

Docker packages all of this reproducibly:
```bash
# Instead of hours of setup, just:
docker build -t mario-kart-rl .
docker run -it mario-kart-rl python train.py
```

#### M1 Mac Docker Considerations

Running Docker on M1 Mac with x86 emulation (Rosetta 2) adds overhead. The emulator will run slower than on native x86 hardware. This is why the decision point exists - if performance is too slow, cloud GPUs become necessary.

### Hands-On: Emulator Setup

**Task 7.1**: Clone the repository
```bash
git clone https://github.com/bzier/gym-mupen64plus.git
cd gym-mupen64plus
```

**Task 7.2**: Acquire the ROM
- You need a Mario Kart 64 ROM file (`.z64` or `.n64`)
- If you own the game, you can dump your cartridge
- Place the ROM in the designated directory

**Task 7.3**: Build the Docker container
```bash
# On M1 Mac, you may need to use platform flag
docker build --platform linux/amd64 -t gym-mupen64plus .
```

**Task 7.4**: Test emulator launches

```bash
# Headless mode (recommended for M1 Mac)
docker run --platform linux/amd64 -it \
    -v $(pwd)/roms:/roms \
    gym-mupen64plus \
    xvfb-run python -c "import gym_mupen64plus; print('Headless import successful!')"
```

**Task 7.5**: Run random agent and measure FPS

Create `scripts/test_emulator_speed.py`:
```python
import gymnasium as gym
import gym_mupen64plus
import time

# Create Mario Kart environment
env = gym.make("Mario-Kart-Luigi-Raceway-v0")

obs, info = env.reset()
print(f"Observation shape: {obs.shape}")
print(f"Action space: {env.action_space}")

# Run random actions and measure FPS
num_steps = 500
start_time = time.time()

for step in range(num_steps):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

elapsed = time.time() - start_time
fps = num_steps / elapsed

print(f"\n{'='*50}")
print(f"Performance Test Results")
print(f"{'='*50}")
print(f"Total steps: {num_steps}")
print(f"Elapsed time: {elapsed:.2f} seconds")
print(f"FPS: {fps:.1f}")
print(f"{'='*50}")

env.close()
```

Run it:
```bash
docker run --platform linux/amd64 -it \
    -v $(pwd):/app \
    -v $(pwd)/roms:/roms \
    gym-mupen64plus \
    xvfb-run python /app/scripts/test_emulator_speed.py
```

### ★ DECISION POINT: M1 Mac vs Cloud (RunPod)

After running the FPS test, evaluate your results:

| FPS | Recommendation |
|-----|----------------|
| **>15 FPS** | Continue locally on M1 Mac |
| **10-15 FPS** | Borderline - local is possible but slow |
| **<10 FPS** | Switch to cloud (RunPod recommended) |

#### If FPS > 15: Continue Locally

You can proceed with all remaining checkpoints on your M1 Mac. Training will be slower than a dedicated GPU machine, but feasible for learning and experimentation.

**Local setup advantages**:
- No cloud costs
- Immediate feedback
- Easier debugging

**Proceed to Checkpoint 8.**

#### If FPS < 15: Set Up RunPod Sync Workflow

If performance is too slow locally, set up a cloud workflow:

**1. Create RunPod account**
- Go to https://runpod.io
- Create account and add payment method
- Recommended GPU: RTX 3090 or A4000 (good price/performance)

**2. Set up sync workflow**

Create `scripts/sync_to_runpod.sh`:
```bash
#!/bin/bash
# Sync local changes to RunPod pod

RUNPOD_IP="your-pod-ip"
RUNPOD_PORT="your-ssh-port"
PROJECT_DIR="n64_mario_kart_reinforcement_learning"

# Sync code (excluding large files)
rsync -avz --progress \
    --exclude 'venv/' \
    --exclude 'logs/' \
    --exclude 'models/' \
    --exclude '*.z64' \
    --exclude '*.n64' \
    --exclude '__pycache__/' \
    . root@${RUNPOD_IP}:~/${PROJECT_DIR}/ \
    -e "ssh -p ${RUNPOD_PORT}"

echo "Sync complete!"
```

Create `scripts/sync_from_runpod.sh`:
```bash
#!/bin/bash
# Sync results back from RunPod

RUNPOD_IP="your-pod-ip"
RUNPOD_PORT="your-ssh-port"
PROJECT_DIR="n64_mario_kart_reinforcement_learning"

# Sync models and logs back
rsync -avz --progress \
    root@${RUNPOD_IP}:~/${PROJECT_DIR}/models/ \
    ./models/ \
    -e "ssh -p ${RUNPOD_PORT}"

rsync -avz --progress \
    root@${RUNPOD_IP}:~/${PROJECT_DIR}/logs/ \
    ./logs/ \
    -e "ssh -p ${RUNPOD_PORT}"

echo "Results synced!"
```

**3. Workflow**:
1. Develop and test locally (wrappers, reward functions)
2. Sync code to RunPod: `./scripts/sync_to_runpod.sh`
3. SSH to pod and run training
4. Sync results back: `./scripts/sync_from_runpod.sh`
5. Analyze results locally with TensorBoard

**Proceed to Checkpoint 8 (run training commands on RunPod instead of locally).**

### Checkpoint 7 Quiz

1. **Why do we use Docker for the emulator setup?**
   - a) Docker is faster than native
   - b) Reproducible environment with all dependencies packaged
   - c) Docker is required by gym-mupen64plus
   - d) To run multiple emulators

2. **What is X11 forwarding used for?**
   - a) Faster emulation
   - b) Displaying the emulator's video output on your screen
   - c) Network communication
   - d) Memory access

3. **In gym-mupen64plus, where do actions go?**
   - a) Directly to the game
   - b) To the input plugin, which simulates controller input
   - c) To a file
   - d) To the video plugin

4. **What is a savestate?**
   - a) A backup of the ROM
   - b) A snapshot of the complete emulator state that can be loaded
   - c) The final save at game completion
   - d) The initial game state

<details>
<summary>Answers</summary>

1. **b)** Docker ensures consistent dependencies across systems
2. **b)** X11 forwards the graphical display to your local screen
3. **b)** Actions go to the input plugin which emulates controller
4. **b)** Complete emulator state snapshot for save/load

</details>

---

## Checkpoint 8: RAM Reading & Memory Mapping

### Theory (30-45 min)

#### Why Read RAM?

The game screen shows you what's happening, but RAM tells you the exact values:

| Information | From Screen | From RAM |
|-------------|-------------|----------|
| Speed | Estimate from motion | Exact: 45.7 km/h |
| Position | Rough estimate | Exact: X=1234, Y=5678 |
| Lap progress | "Lap 2/3" text | 67.3% complete |
| Collision | Visual jitter | Collision flag = 1 |

**RAM data is essential for reward functions!**

#### N64 Memory Layout

The N64 has 4MB (or 8MB with expansion pak) of RAM. Games store data at specific addresses:

```
Memory Map (simplified):
0x00000000 - 0x003FFFFF : RDRAM (main memory)
0x10000000 - 0x1FBFFFFF : Cartridge ROM
0x1FC00000 - 0x1FC007BF : PIF ROM (boot)

Game data is in RDRAM at addresses developers chose.
```

#### Mario Kart 64 RAM Addresses

From the Hack64 community RAM map:

```
Player 1 Data (approximate addresses, verify for your ROM version):
- Speed:         0x800F6BBC (float, km/h)
- X Position:    0x800F6BA0 (float)
- Y Position:    0x800F6BA4 (float)
- Z Position:    0x800F6BA8 (float)
- Lap Count:     0x800F6CD4 (byte)
- Race Position: 0x800F6CD8 (byte, 0=1st, 7=8th)
- Checkpoint:    0x800F6CE0 (word, for progress tracking)

Race State:
- Timer:         0x800DC510 (frames since race start)
- Race Started:  0x800DC51C (flag)
```

**Important**: Addresses can vary between ROM versions (US, JP, EU). Always verify!

#### Data Types and Endianness

N64 uses big-endian byte ordering:

```python
# Reading a 4-byte integer at address 0x800F6BBC
memory = b'\x42\x20\x00\x00'  # Raw bytes

# Big-endian interpretation
import struct
value = struct.unpack('>f', memory)[0]  # '>' means big-endian, 'f' means float
print(value)  # 40.0

# Common format specifiers:
# >b : signed byte
# >B : unsigned byte
# >h : signed short (2 bytes)
# >H : unsigned short (2 bytes)
# >i : signed int (4 bytes)
# >I : unsigned int (4 bytes)
# >f : float (4 bytes)
```

### Hands-On: Implement RAM Reader

**Task 8.1**: Create the RAM reader class

Create `src/rewards/ram_reader.py`:
```python
import struct
from typing import Tuple, Dict

class MarioKartRAMReader:
    """Read game state from Mario Kart 64 RAM."""

    # RAM addresses (verify for your ROM version!)
    ADDRESSES = {
        # Player 1 state
        'speed': 0x800F6BBC,
        'pos_x': 0x800F6BA0,
        'pos_y': 0x800F6BA4,
        'pos_z': 0x800F6BA8,
        'lap': 0x800F6CD4,
        'position': 0x800F6CD8,  # Race position (0-7)
        'checkpoint': 0x800F6CE0,

        # Race state
        'timer': 0x800DC510,
        'race_started': 0x800DC51C,
    }

    def __init__(self, memory_reader):
        """
        Args:
            memory_reader: Object with read(address, size) method
        """
        self.memory = memory_reader

    def _read_byte(self, address: int) -> int:
        data = self.memory.read(address, 1)
        return struct.unpack('>B', data)[0]

    def _read_short(self, address: int) -> int:
        data = self.memory.read(address, 2)
        return struct.unpack('>H', data)[0]

    def _read_int(self, address: int) -> int:
        data = self.memory.read(address, 4)
        return struct.unpack('>I', data)[0]

    def _read_float(self, address: int) -> float:
        data = self.memory.read(address, 4)
        return struct.unpack('>f', data)[0]

    def get_speed(self) -> float:
        """Get current speed in km/h."""
        return self._read_float(self.ADDRESSES['speed'])

    def get_position(self) -> Tuple[float, float, float]:
        """Get (x, y, z) position."""
        x = self._read_float(self.ADDRESSES['pos_x'])
        y = self._read_float(self.ADDRESSES['pos_y'])
        z = self._read_float(self.ADDRESSES['pos_z'])
        return (x, y, z)

    def get_lap(self) -> int:
        """Get current lap number (1-indexed)."""
        return self._read_byte(self.ADDRESSES['lap']) + 1

    def get_race_position(self) -> int:
        """Get race position (1-8)."""
        return self._read_byte(self.ADDRESSES['position']) + 1

    def get_checkpoint(self) -> int:
        """Get checkpoint counter for progress tracking."""
        return self._read_int(self.ADDRESSES['checkpoint'])

    def get_timer_frames(self) -> int:
        """Get race timer in frames (60 fps)."""
        return self._read_int(self.ADDRESSES['timer'])

    def get_timer_seconds(self) -> float:
        """Get race timer in seconds."""
        return self.get_timer_frames() / 60.0

    def is_race_started(self) -> bool:
        """Check if race has begun (after countdown)."""
        return self._read_byte(self.ADDRESSES['race_started']) != 0

    def get_full_state(self) -> Dict:
        """Get complete game state dictionary."""
        return {
            'speed': self.get_speed(),
            'position': self.get_position(),
            'lap': self.get_lap(),
            'race_position': self.get_race_position(),
            'checkpoint': self.get_checkpoint(),
            'timer': self.get_timer_seconds(),
            'race_started': self.is_race_started(),
        }
```

**Task 8.2**: Create test with mock memory

Create `tests/test_ram_reader.py`:
```python
import struct
from src.rewards.ram_reader import MarioKartRAMReader

class MockMemory:
    """Simulated memory for testing."""
    def __init__(self):
        self.data = {}
        # Set up some test values
        self._write_float(0x800F6BBC, 45.5)  # Speed
        self._write_float(0x800F6BA0, 1234.0)  # X
        self._write_float(0x800F6BA4, 100.0)  # Y
        self._write_float(0x800F6BA8, 5678.0)  # Z
        self._write_byte(0x800F6CD4, 1)  # Lap (0-indexed)
        self._write_byte(0x800F6CD8, 2)  # Position (0-indexed)

    def _write_byte(self, address, value):
        self.data[address] = struct.pack('>B', value)

    def _write_float(self, address, value):
        self.data[address] = struct.pack('>f', value)

    def read(self, address, size):
        return self.data.get(address, b'\x00' * size)


if __name__ == "__main__":
    print("Testing RAM reader with mock memory...")

    mock_mem = MockMemory()
    reader = MarioKartRAMReader(mock_mem)

    print(f"Speed: {reader.get_speed():.1f} km/h")
    print(f"Position: {reader.get_position()}")
    print(f"Lap: {reader.get_lap()}")
    print(f"Race position: {reader.get_race_position()}")

    print("\nFull state:")
    for key, value in reader.get_full_state().items():
        print(f"  {key}: {value}")
```

**Task 8.3**: Manual testing with emulator (when available)

Create `scripts/manual_control.py`:
```python
# This script displays RAM values while you play manually
# Useful for verifying RAM addresses match your ROM version

import gymnasium as gym
import gym_mupen64plus
from src.rewards.ram_reader import MarioKartRAMReader

env = gym.make("Mario-Kart-Luigi-Raceway-v0")
# Assuming env has a memory_reader property

obs, info = env.reset()

# Display RAM values periodically
for step in range(1000):
    action = 0  # No action, or connect to keyboard input
    obs, reward, term, trunc, info = env.step(action)

    if step % 60 == 0:  # Every second at 60fps
        # Access RAM through environment (implementation-specific)
        # print(f"Speed: {reader.get_speed():.1f} km/h")
        pass

    if term or trunc:
        break

env.close()
```

### Checkpoint 8 Quiz

1. **Why read game state from RAM instead of just using the screen?**
   - a) RAM is faster to read
   - b) RAM provides exact numerical values (speed, position, progress)
   - c) Screen reading is not allowed
   - d) RAM uses less memory

2. **N64 uses big-endian byte order. For bytes [0x42, 0x28, 0x00, 0x00], what float is this?**
   - a) 0.0
   - b) 42.0 (Hint: 0x42280000 = 42.0 in IEEE 754)
   - c) 66.0
   - d) 16416.0

3. **Why might RAM addresses differ between ROM versions?**
   - a) Different byte order
   - b) Compiler and linker place data at different addresses
   - c) Region differences in text take different space
   - d) Both b and c

4. **What information is most useful for a reward function?**
   - a) Current lap number only
   - b) Combination of speed, progress, and collision state
   - c) Player name
   - d) Timer display

<details>
<summary>Answers</summary>

1. **b)** RAM has exact values; screen would require computer vision
2. **b)** 0x42280000 is 42.0 in big-endian IEEE 754 float
3. **d)** Both different compilation and region differences affect layout
4. **b)** Multiple signals give richer reward shaping

</details>

---

## Checkpoint 9: Reward Engineering (The Art)

### Theory (30-45 min)

#### The Importance of Reward Design

The reward function is how you communicate your goal to the agent. Get it wrong, and:
- Agent finds unintended shortcuts ("reward hacking")
- Agent gets stuck in local optima
- Training never converges

**You're not programming behavior; you're designing incentives.**

#### Sparse vs Dense Rewards

**Sparse Rewards**: Only reward at major events
```python
# Sparse: only reward finishing
reward = 1.0 if finished_race else 0.0
```
- Problem: Agent rarely experiences reward → slow/no learning
- Mario Kart: Takes minutes to finish; too sparse!

**Dense Rewards**: Reward at every timestep
```python
# Dense: reward for making progress
reward = speed * progress_delta
```
- Guides agent continuously
- Risk: Can lead to reward hacking if poorly designed

#### Reward Shaping

Add intermediate rewards that guide toward the goal:

```python
# Potential-based shaping (theoretically sound)
Φ(s) = distance_to_finish(s)
shaped_reward = base_reward + γ * Φ(s') - Φ(s)
```

This encourages progress without changing the optimal policy!

#### Common Reward Components for Racing

| Component | Formula | Purpose |
|-----------|---------|---------|
| Speed | `+speed / max_speed` | Encourage fast driving |
| Progress | `+Δcheckpoint` | Reward advancing on track |
| Lap completion | `+100 on lap complete` | Milestone bonus |
| Collision | `-10 on crash` | Discourage hitting walls |
| Off-track | `-5 when off road` | Stay on track |
| Time penalty | `-0.1 per step` | Encourage finishing quickly |
| Finish bonus | `+1000 on finish` | Ultimate goal |

#### Common Pitfalls

**Reward Hacking Examples**:
- Reward for speed → Agent drives in circles (fast but no progress)
- Reward for progress → Agent clips through walls
- Reward for survival → Agent stops moving (can't die if not moving)

**Solutions**:
- Combine multiple objectives
- Add penalties for unwanted behavior
- Use curriculum learning (start easy)

### Hands-On: Implement Multiple Reward Functions

**Task 9.1**: Create reward functions

Create `src/rewards/reward_functions.py`:
```python
from abc import ABC, abstractmethod
from typing import Dict, Any

class RewardFunction(ABC):
    """Base class for reward functions."""

    @abstractmethod
    def compute(self, state: Dict, next_state: Dict, action: Any) -> float:
        """Compute reward from state transition."""
        pass

    def reset(self):
        """Reset any episode-specific state."""
        pass


class SpeedReward(RewardFunction):
    """Reward based on speed only."""

    def __init__(self, max_speed: float = 80.0):
        self.max_speed = max_speed

    def compute(self, state: Dict, next_state: Dict, action: Any) -> float:
        speed = next_state.get('speed', 0)
        return speed / self.max_speed  # Normalized to [0, 1]


class ProgressReward(RewardFunction):
    """Reward based on track progress."""

    def __init__(self, checkpoint_value: float = 1.0):
        self.checkpoint_value = checkpoint_value
        self.last_checkpoint = None

    def compute(self, state: Dict, next_state: Dict, action: Any) -> float:
        current_checkpoint = next_state.get('checkpoint', 0)

        if self.last_checkpoint is None:
            self.last_checkpoint = current_checkpoint
            return 0.0

        progress = current_checkpoint - self.last_checkpoint
        self.last_checkpoint = current_checkpoint

        # Handle lap wrap-around (checkpoints reset)
        if progress < -100:  # Arbitrary threshold for wrap detection
            progress = 0  # Ignore wrap-around in reward

        return progress * self.checkpoint_value

    def reset(self):
        self.last_checkpoint = None


class CompositeReward(RewardFunction):
    """Combine multiple reward functions with weights."""

    def __init__(self, components: Dict[str, tuple]):
        """
        Args:
            components: Dict mapping name -> (RewardFunction, weight)
        """
        self.components = components

    def compute(self, state: Dict, next_state: Dict, action: Any) -> float:
        total = 0.0
        for name, (func, weight) in self.components.items():
            reward = func.compute(state, next_state, action)
            total += weight * reward
        return total

    def reset(self):
        for name, (func, _) in self.components.items():
            func.reset()


class MarioKartRewardV1(RewardFunction):
    """
    Version 1: Speed + Progress
    Simple reward that encourages fast forward movement.
    """

    def __init__(self):
        self.composite = CompositeReward({
            'speed': (SpeedReward(max_speed=80.0), 0.5),
            'progress': (ProgressReward(checkpoint_value=1.0), 0.5),
        })

    def compute(self, state: Dict, next_state: Dict, action: Any) -> float:
        return self.composite.compute(state, next_state, action)

    def reset(self):
        self.composite.reset()


class MarioKartRewardV2(RewardFunction):
    """
    Version 2: Speed + Progress + Collision Penalty
    Adds penalty for hitting walls/obstacles.
    """

    def __init__(self):
        self.last_speed = None
        self.composite = CompositeReward({
            'speed': (SpeedReward(max_speed=80.0), 0.3),
            'progress': (ProgressReward(checkpoint_value=1.0), 0.5),
        })

    def compute(self, state: Dict, next_state: Dict, action: Any) -> float:
        base_reward = self.composite.compute(state, next_state, action)

        # Collision detection via sudden speed drop
        current_speed = next_state.get('speed', 0)
        collision_penalty = 0.0

        if self.last_speed is not None:
            speed_drop = self.last_speed - current_speed
            if speed_drop > 20:  # Sudden drop indicates collision
                collision_penalty = -0.5

        self.last_speed = current_speed
        return base_reward + collision_penalty

    def reset(self):
        self.last_speed = None
        self.composite.reset()


class MarioKartRewardV3(RewardFunction):
    """
    Version 3: Full reward function with lap bonuses.
    """

    def __init__(self):
        self.last_lap = None
        self.last_speed = None
        self.composite = CompositeReward({
            'speed': (SpeedReward(max_speed=80.0), 0.2),
            'progress': (ProgressReward(checkpoint_value=0.8), 0.4),
        })

    def compute(self, state: Dict, next_state: Dict, action: Any) -> float:
        base_reward = self.composite.compute(state, next_state, action)

        # Lap completion bonus
        current_lap = next_state.get('lap', 1)
        lap_bonus = 0.0

        if self.last_lap is not None and current_lap > self.last_lap:
            lap_bonus = 10.0  # Big bonus for completing a lap

        self.last_lap = current_lap

        # Collision penalty
        current_speed = next_state.get('speed', 0)
        collision_penalty = 0.0

        if self.last_speed is not None:
            speed_drop = self.last_speed - current_speed
            if speed_drop > 20:
                collision_penalty = -0.3

        self.last_speed = current_speed

        # Time penalty (encourages finishing faster)
        time_penalty = -0.01

        return base_reward + lap_bonus + collision_penalty + time_penalty

    def reset(self):
        self.last_lap = None
        self.last_speed = None
        self.composite.reset()
```

**Task 9.2**: Test reward functions

Create `tests/test_reward_functions.py`:
```python
from src.rewards.reward_functions import (
    MarioKartRewardV1, MarioKartRewardV2, MarioKartRewardV3
)

# Simulated state transitions
states = [
    {'speed': 0, 'checkpoint': 0, 'lap': 1},
    {'speed': 30, 'checkpoint': 5, 'lap': 1},
    {'speed': 50, 'checkpoint': 10, 'lap': 1},
    {'speed': 60, 'checkpoint': 15, 'lap': 1},
    {'speed': 10, 'checkpoint': 16, 'lap': 1},  # Collision!
    {'speed': 55, 'checkpoint': 25, 'lap': 1},
    {'speed': 60, 'checkpoint': 0, 'lap': 2},   # New lap
]

reward_functions = {
    'V1 (Speed+Progress)': MarioKartRewardV1(),
    'V2 (V1+Collision)': MarioKartRewardV2(),
    'V3 (Full)': MarioKartRewardV3(),
}

for name, rf in reward_functions.items():
    print(f"\n{name}:")
    rf.reset()
    total = 0
    for i in range(len(states) - 1):
        r = rf.compute(states[i], states[i+1], None)
        total += r
        event = ""
        if states[i+1]['speed'] < states[i]['speed'] - 15:
            event = " (collision!)"
        if states[i+1]['lap'] > states[i]['lap']:
            event = " (new lap!)"
        print(f"  Step {i+1}: reward = {r:+.3f}{event}")
    print(f"  Total: {total:.3f}")
```

### Checkpoint 9 Quiz

1. **What is "reward hacking"?**
   - a) Modifying the reward code illegally
   - b) Agent finding unintended ways to maximize reward
   - c) Hacking into the game's reward system
   - d) Using cheats to get higher rewards

2. **Why are sparse rewards problematic for Mario Kart?**
   - a) They're too computationally expensive
   - b) Episodes are long; agent rarely sees reward signal
   - c) The game doesn't support sparse rewards
   - d) They cause overfitting

3. **An agent keeps driving in circles. Which reward component might help?**
   - a) Higher speed reward
   - b) Progress/checkpoint reward
   - c) Lower collision penalty
   - d) Longer episodes

4. **Why start with a simple reward function before adding components?**
   - a) Simple functions train faster
   - b) Easier to debug and understand agent behavior
   - c) Complex functions cause crashes
   - d) The agent prefers simple rewards

<details>
<summary>Answers</summary>

1. **b)** Agent exploits loopholes to maximize reward in unintended ways
2. **b)** Rare reward signal makes credit assignment very difficult
3. **b)** Progress reward incentivizes forward track movement
4. **b)** Easier to identify what's working and what's not

</details>

---

## Checkpoint 10: Training at Scale & Debugging

### Theory (30-45 min)

#### TensorBoard Monitoring

Key metrics to watch:

```
Episode Reward Mean (ep_rew_mean)
├── Should trend upward
├── Plateau = might need hyperparameter tuning
└── Sudden drop = something broke

Policy Loss (policy_loss)
├── Should decrease initially, then stabilize
└── Wild oscillations = learning rate too high

Value Loss (value_loss)
├── Should decrease as value function improves
└── Very high = value estimates are bad

Entropy (entropy_loss)
├── Should decrease slowly over time
├── Too fast = agent becoming deterministic too quickly
└── Add entropy bonus to slow down

FPS (fps)
├── Training speed
└── Low FPS = bottleneck somewhere (CPU, I/O, emulator)
```

#### Common Failure Modes

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Reward stays flat | Sparse reward, bad exploration | Denser reward, more entropy |
| Reward oscillates wildly | Learning rate too high | Lower learning rate |
| Agent does nothing | Reward hacking by staying still | Time penalty, progress reward |
| Agent repeats same action | Deterministic policy too early | Entropy bonus |
| Training very slow | CPU bottleneck | More parallel environments |
| NaN in loss | Gradient explosion | Gradient clipping, lower LR |

#### Debugging Strategies

1. **Sanity check**: Does random agent get some reward?
2. **Visual inspection**: Watch the agent play - what is it doing?
3. **Reward logging**: Print rewards per component
4. **Gradient analysis**: Check for NaN, very large/small gradients
5. **Simplify**: Test on easier environment first (e.g., single track section)

### Hands-On: Full Training Run

**Task 10.1**: Create training script

Create `scripts/train.py`:
```python
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
import os

# Configuration
CONFIG = {
    'total_timesteps': 500_000,
    'n_envs': 4,
    'eval_freq': 25_000,
    'checkpoint_freq': 50_000,
    'log_dir': './logs/',
    'model_dir': './models/',
    'eval_episodes': 10,
}

# Create directories
os.makedirs(CONFIG['log_dir'], exist_ok=True)
os.makedirs(CONFIG['model_dir'], exist_ok=True)

def create_env():
    """Create and wrap the environment."""
    # For Mario Kart:
    # env = gym.make("Mario-Kart-Luigi-Raceway-v0")
    # env = YourWrapperStack(env)

    # For testing, use LunarLander:
    env = gym.make("LunarLander-v3")
    return env

# Create training and evaluation environments
train_env = make_vec_env(create_env, n_envs=CONFIG['n_envs'])
eval_env = make_vec_env(create_env, n_envs=1)

# Set up callbacks
checkpoint_callback = CheckpointCallback(
    save_freq=CONFIG['checkpoint_freq'] // CONFIG['n_envs'],
    save_path=CONFIG['model_dir'],
    name_prefix="ppo_checkpoint",
    verbose=1
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=CONFIG['model_dir'],
    log_path=CONFIG['log_dir'],
    eval_freq=CONFIG['eval_freq'] // CONFIG['n_envs'],
    n_eval_episodes=CONFIG['eval_episodes'],
    deterministic=True,
    verbose=1
)

callbacks = CallbackList([checkpoint_callback, eval_callback])

# Create PPO model
model = PPO(
    "MlpPolicy",  # Use "CnnPolicy" for visual input
    train_env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1,
    tensorboard_log=CONFIG['log_dir']
)

# Train!
print(f"Starting training for {CONFIG['total_timesteps']} timesteps...")
print(f"TensorBoard logs: {CONFIG['log_dir']}")
print(f"Run: tensorboard --logdir {CONFIG['log_dir']}")

model.learn(
    total_timesteps=CONFIG['total_timesteps'],
    callback=callbacks,
    progress_bar=True
)

# Save final model
final_path = os.path.join(CONFIG['model_dir'], "final_model")
model.save(final_path)
print(f"\nTraining complete! Final model saved to {final_path}")

train_env.close()
eval_env.close()
```

**Task 10.2**: Create evaluation script

Create `scripts/evaluate.py`:
```python
import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import argparse

def evaluate(model_path, env_name, n_episodes=10, render=False):
    """Evaluate a trained model."""

    # Load model
    model = PPO.load(model_path)

    # Create environment
    render_mode = "human" if render else None
    env = gym.make(env_name, render_mode=render_mode)

    rewards = []
    episode_lengths = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        rewards.append(total_reward)
        episode_lengths.append(steps)
        print(f"Episode {episode + 1}: reward = {total_reward:.1f}, steps = {steps}")

    print(f"\nResults over {n_episodes} episodes:")
    print(f"  Mean reward: {np.mean(rewards):.1f} (+/- {np.std(rewards):.1f})")
    print(f"  Mean length: {np.mean(episode_lengths):.1f} steps")

    env.close()
    return rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to saved model")
    parser.add_argument("--env", default="LunarLander-v3", help="Environment name")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--render", action="store_true", help="Render environment")
    args = parser.parse_args()

    evaluate(args.model, args.env, args.episodes, args.render)
```

**Task 10.3**: Monitor with TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir ./logs/

# In browser, look for:
# 1. rollout/ep_rew_mean - should trend up
# 2. train/policy_loss - should decrease then stabilize
# 3. train/value_loss - should decrease
# 4. train/entropy_loss - should decrease slowly
```

### Checkpoint 10 Quiz

1. **In TensorBoard, `ep_rew_mean` is flat after 100k steps. What should you try?**
   - a) Train longer (patience)
   - b) Denser rewards, different hyperparameters, or verify environment
   - c) Use a smaller network
   - d) Remove all callbacks

2. **Your training keeps crashing with NaN loss. What's the likely cause?**
   - a) Not enough GPU memory
   - b) Learning rate too high causing gradient explosion
   - c) Environment is buggy
   - d) Model file is corrupted

3. **Why save checkpoints during training?**
   - a) To use less disk space
   - b) To resume if training crashes and to compare different stages
   - c) TensorBoard requires checkpoints
   - d) The model trains faster with checkpoints

4. **The agent's entropy drops to near-zero quickly. What does this mean?**
   - a) Training is going well
   - b) Agent became deterministic too fast, reducing exploration
   - c) The reward function is wrong
   - d) The environment is too easy

<details>
<summary>Answers</summary>

1. **b)** Flat reward means no learning signal; check rewards and hyperparameters
2. **b)** NaN typically comes from numerical instability in gradients
3. **b)** Checkpoints enable recovery and analysis of training progress
4. **b)** Low entropy means less exploration; increase `ent_coef`

</details>

---

## Next Steps After This Curriculum

Congratulations! After completing these 10 checkpoints, you'll have:
- Solid theoretical foundation in RL
- Practical experience with standard benchmarks
- Working Mario Kart environment setup
- Custom reward functions
- Full training pipeline with monitoring

**From here, iterate on**:
1. Different reward function designs
2. Hyperparameter tuning
3. Curriculum learning (easier → harder tracks)
4. More training time (1M+ steps)
5. Advanced techniques (multi-agent, self-play, etc.)

---

## Resources

### Documentation

| Resource | URL | Purpose |
|----------|-----|---------|
| gym-mupen64plus | https://github.com/bzier/gym-mupen64plus | Gymnasium wrapper |
| Stable Baselines 3 | https://stable-baselines3.readthedocs.io/ | PPO implementation |
| Gymnasium | https://gymnasium.farama.org/ | Environment API |
| Hack64 RAM Map | https://hack64.net/wiki/doku.php?id=mario_kart_64:ram_map | Memory addresses |

### Papers

| Paper | Purpose |
|-------|---------|
| [PPO Paper](https://arxiv.org/abs/1707.06347) | Original PPO algorithm |
| [NatureCNN (DQN Paper)](https://arxiv.org/abs/1312.5602) | CNN architecture |
| [NeuralKart](http://cs229.stanford.edu/proj2016/report/NeuroKartRealTimeMarioKart64AIusingCNN.pdf) | Prior Mario Kart RL work |

### Tutorials

| Resource | Purpose |
|----------|---------|
| [Spinning Up](https://spinningup.openai.com/) | RL theory |
| [SB3 Tutorial](https://stable-baselines3.readthedocs.io/en/master/guide/quickstart.html) | Getting started with SB3 |

---

## Quick Reference

### Key Libraries
```python
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
```

### Basic Training Loop
```python
env = gym.make("YourEnv-v0")
model = PPO("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
model.save("model_name")
```

### Evaluation
```python
model = PPO.load("model_name")
obs, _ = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, term, trunc, info = env.step(action)
    if term or trunc:
        break
```

### Key Hyperparameters
| Parameter | Default | Range to try |
|-----------|---------|--------------|
| learning_rate | 3e-4 | 1e-4 to 1e-3 |
| n_steps | 2048 | 512 to 4096 |
| batch_size | 64 | 32 to 256 |
| n_epochs | 10 | 3 to 20 |
| clip_range | 0.2 | 0.1 to 0.3 |
| ent_coef | 0.0 | 0.0 to 0.1 |

---

## Troubleshooting Quick Reference

| Problem | Likely Cause | Solution |
|---------|--------------|----------|
| Import error | Package not installed | `pip install -r requirements.txt` |
| Docker build fails | Missing dependencies | Check Dockerfile, rebuild from scratch |
| "Cannot open display" | X11 not configured | Export DISPLAY, or use xvfb |
| ROM not found | Wrong path | Check ROM location and permissions |
| Training stuck | Learning rate too high/low | Try 3e-4, add entropy bonus |
| NaN in loss | Gradient explosion | Lower learning rate, add gradient clipping |
| Reward flat | Sparse reward | Add dense shaping |
| Agent circles | No progress reward | Add checkpoint-based reward |
| Out of GPU memory | Batch too large | Reduce batch_size or n_steps |
| Slow on M1 Mac | x86 Docker emulation | Consider RunPod for training |

---

Good luck with your Mario Kart AI!
