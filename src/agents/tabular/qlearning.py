import gymnasium as gym
import numpy as np

from src.agents.base import Agent


class QLearningAgent(Agent):
    def __init__(
        self,
        #
        env: gym.Env,
        discount_factor: float = 0.99,
        #
        learning_rate: float = 0.1,
        epsilon: float = 0.1,
        #
        seed: int = 42
    ):
        self.env = env
        self.learning_rate = learning_rate  # Alpha
        self.discount_factor = discount_factor  # Gamma
        self.epsilon = epsilon  # Exploration factor
        self.seed = seed  # Random seed for reproducibility

        # Initialize Q-table with zeros
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))

    def sample_actions(self, states: np.ndarray) -> np.ndarray:
        """ Choose actions using epsilon-greedy policy for exploration. """
        actions = []
        for state in states:
            if np.random.rand() < self.epsilon:
                action = self.env.action_space.sample()  # Random action (exploration)
            else:
                action = np.argmax(self.Q[state, :])  # Best action (exploitation)
            actions.append(action)
        return np.array(actions)

    def eval_actions(self, states: np.ndarray) -> np.ndarray:
        """ Choose actions by exploiting the learned Q-table (greedy policy). """
        actions = [np.argmax(self.Q[state, :]) for state in states]
        return np.array(actions)

    def train_step(self, episode: int):
        state, _ = self.env.reset(seed=self.seed + episode)
        done, truncated = False, False
        total_reward = 0
        while not (done or truncated):
            action = self.sample_actions([state])[0]
            next_state, reward, done, truncated, _ = self.env.step(action)
            self.Q[state, action] += self.learning_rate * (
                reward + self.discount_factor * np.max(self.Q[next_state, :]) - self.Q[state, action]
            )
            state = next_state
            total_reward += reward 
        return total_reward

    def save(self, save_path: str):
        super().save(save_path)
        with open(f"{save_path}/Q.npy", "wb") as file:
            np.save(file, self.Q)

    def load(self, save_path: str):
        self.Q = np.load(f"{save_path}/Q.npy")
