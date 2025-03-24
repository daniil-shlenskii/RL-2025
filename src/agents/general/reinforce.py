import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from src.agents.base import Agent


class ReinforceAgent(Agent):
    def __init__(
        self,
        env: gym.Env,
        policy_net: nn.Module,
        gamma: float = 0.99,
        episodes: int = 1000,
        seed: int = 42,
    ):
        self.env = env
        self.gamma = gamma
        self.seed = seed
        self.episodes = episodes

        self.policy_net = policy_net

    def sample_actions(self, states: torch.Tensor) -> torch.Tensor:
        distr =  self.policy_net.get_distribution(states)
        actions = distr.sample()
        log_probs = distr.log_prob(actions)
        return actions, log_probs

    def eval_actions(self, states: np.ndarray) -> np.ndarray:
        states = torch.tensor(states)
        actions =  self.policy_net.mode(states)
        return actions.detach().numpy()

    def update_policy_net(self, rewards: list, log_probs: list):
        discounted_rewards = [rewards[-1]]
        for reward in reversed(rewards[:-1]):
            discounted_rewards.append(reward + self.gamma * discounted_rewards[-1])
        discounted_rewards = torch.tensor(list(reversed(discounted_rewards)))

        log_probs = torch.stack(log_probs)
        policy_loss = torch.sum(-log_probs * discounted_rewards)

        self.policy_net.optimizer.zero_grad()
        policy_loss.backward()
        self.policy_net.optimizer.step()

    def train_step(self, episode: int):
        state, _ = self.env.reset(seed=self.seed + episode)
        done, truncated = False, False
        rewards, log_probs = [], []
        while not (done or truncated):
            action, log_prob = self.sample_actions(torch.tensor(state))
            next_state, reward, done, truncated, _ = self.env.step(action.item())
            rewards.append(reward)
            log_probs.append(log_prob)
            state = next_state

        self.update_policy_net(rewards, log_probs)
        return np.sum(rewards)

    def save(self, save_path: str):
        super().save(save_path)
        torch.save(self.policy_net.state_dict(), f"{save_path}/checkpoint.pt")

    def load(self, save_path: str):
        self.policy_net.load_state_dict(torch.load(f"{save_path}/checkpoint.pt", weights_only=True))
