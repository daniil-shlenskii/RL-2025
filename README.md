# Reinforcement Learning Mini Projects

This repository contains small reinforcement learning (RL) projects, including a Q-learning agent for the `CliffWalking-v0` environment and a custom 1D point environment with obstacles.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage and Code Overview](#usage-and-code-overview)
- [Directory Structure](#directory-structure)
- [License](#license)

---

## Project Overview

### 1. Q-Learning Agent for CliffWalking-v0
The Q-learning agent is trained to navigate the `CliffWalking-v0` environment using a tabular Q-learning algorithm.

### 2. 1D Point Environment with Obstacles

The `PointEnv` is a custom environment where a point moves in a 1D space with obstacles, rendered using `pygame`.

---

## Installation

1. **Clone the repository:**

```bash
   git clone https://github.com/your-username/reinforcement-learning-mini-projects.git
   cd reinforcement-learning-mini-projects
```
2. **Set up a virtual environment (optional but recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. **Install the required dependencies:**
```bash
pip install gymnasium numpy pygame
```

## Usage and Code Overview

### Q-Learning Agent for CliffWalking-v0

The Q-learning agent is implemented in `src/agents/tabular/qlearning.py`  and trained/evaluated in  `run/1_tabular.py` .

1. **Train and evaluate the agent:**
```bash
python run/1_tabular.py
```

This script:
- Initializes the `CliffWalking-v0`  environment.
- Trains the Q-learning agent.
- Evaluates the agent and renders the environment.

Key code snippet from `src/agents/tabular/qlearning.py`.

```bash
class QLearningAgent(Agent):
    def __init__(self, env: gym.Env, discount_factor: float = 0.99, learning_rate: float = 0.1, epsilon: float = 0.1, episodes: int = 1000, seed: int = 42):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.seed = seed
        self.episodes = episodes
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))
```

### 1D Point Environment with Obstacles

The PointEnv is implemented in `src/envs/custom_envs/point.py`  and demonstrated in `run/render_point_env.py`.


1. **Run the PointEnv visualization:**
```bash
python run/render_point_env.py
```

This script:
- Initializes the PointEnv environment.
- Takes random actions and renders the point's movement using pygame.

Key code snippet from `src/envs/custom_envs/point.py`.


```bash
from typing import Optional
import gymnasium as gym

class PointEnv(gym.Env):
    x_start: float = 0.
    screen_dim: int = 500

    def __init__(self, size: float = 10., render_mode: Optional[str] = None):
        self.size = size
        self.observation_space = gym.spaces.Box(-1., 1., shape=(1,))
        self.action_space = gym.spaces.Box(-1., 1., shape=(1,))
        self._x_location = self.x_start
        self._terminated = False
        self.render_mode = render_mode
```

## Directory Structure

```bash
.
├── run/
│   ├── 1_tabular.py              # Q-learning agent training and evaluation
│   └── render_point_env.py        # PointEnv visualization
├── src/
│   ├── agents/                    # RL agent implementations
│   │   ├── tabular/               # Tabular methods (e.g., Q-learning)
│   │   │   ├── __init__.py
│   │   │   └── qlearning.py
│   │   ├── __init__.py
│   │   └── base.py                # Base agent class
│   └── envs/                      # Custom environments
│       ├── custom_envs/           # Custom environments (e.g., PointEnv)
│       │   ├── __init__.py
│       │   ├── cardenv.py         # Card environment
│       │   ├── cardenv_1d.py      # 1D card environment
│       │   ├── obstacles.py       # Obstacles implementation
│       │   └── point.py           # Point environment
│       └── __init__.py
├── .gitignore                    # Git ignore file
└── README.md                     # This file
```
