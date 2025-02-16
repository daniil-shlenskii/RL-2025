# Reinforcement Learning Mini Projects

This repository contains small reinforcement learning (RL) projects implemented using Python and the `gymnasium` library. The projects include a Q-learning agent for the `CliffWalking-v0` environment and a custom `PointEnv` environment for visualizing random actions.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Q-Learning Agent](#q-learning-agent)
  - [Point Environment](#point-environment)
- [Directory Structure](#directory-structure)
- [License](#license)

---

## Project Overview

### 1. Q-Learning Agent
The `Q-learning` agent is implemented for the `CliffWalking-v0` environment from the `gymnasium` library. The agent learns to navigate the environment using the Q-learning algorithm, balancing exploration and exploitation.

### 2. Point Environment
The `PointEnv` is a custom environment where a point moves randomly within a bounded space. The environment is rendered using `pygame`, allowing visualization of the point's movement.

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

## Usage

### Q-Learning Agent

The Q-learning agent is trained and evaluated in the run/1_tabular.py script.

1. **Train and evaluate the agent:**
```bash
python run/1_tabular.py
```
This script:
- Initializes the CliffWalking-v0 environment.
- Trains the Q-learning agent.
- Evaluates the agent and renders the environment.

### Point Environment
The PointEnv environment is demonstrated in the run/render_point_env.py script.

1. **Run the PointEnv visualization::**
```bash
python run/render_point_env.py
```

This script:

- Initializes the PointEnv environment.
- Takes random actions and renders the point's movement using pygame.

  ## Directory Structure

```bash
.
├── run/
│   ├── 1_tabular.py              # Q-learning agent training and evaluation
│   └── render_point_env.py        # PointEnv visualization
├── src/
│   ├── agents/                    # RL agent implementations
│   │   ├── base.py                # Base agent class
│   │   └── tabular/               # Tabular methods (e.g., Q-learning)
│   └── envs/                      # Custom environments
│       └── custom_envs/           # Custom environments (e.g., PointEnv)
├── .gitignore                    # Git ignore file
└── README.md                     # This file
```
