from abc import ABC, abstractmethod

import gymnasium as gym
import numpy as np


class Agent(ABC):
    seed: int
    env: gym.Env

    @abstractmethod
    def sample_actions(self, states: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def eval_actions(self, observations: np.ndarray,) -> np.ndarray:
        pass

    def evaluate(self, num_episodes: int) -> dict[str, float]:
        env = gym.wrappers.RecordEpisodeStatistics(self.env, buffer_length=num_episodes)
        for i in range(num_episodes):
            observation, _, done, truncated = *env.reset(seed=self.seed+i), False, False
            while not (done or truncated):
                action = self.eval_actions([observation])[0]
                next_observation, _, done, truncated, _ = env.step(action)
                observation = next_observation

        stats = {"return": np.mean(env.return_queue), "length": np.mean(env.length_queue)}
        return stats
