import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from src.agents.base import Agent


class PPOAgent(Agent):
    def __init__(
        self,
        env: gym.Env,
        policy_net: nn.Module,
        value_net: nn.Module,
        gamma: float = 0.99,
        episodes: int = 1000,
        seed: int = 42,
        #
        clip_epsilon=0.2,
        update_epochs=4,
        batch_size=64,
        device: str = "cpu",
    ):
        self.env = env
        self.gamma = gamma
        self.seed = seed
        self.episodes = episodes

        self.policy_net = policy_net
        self.value_net = value_net

        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.device = device

    def sample_actions(self, states: torch.Tensor) -> torch.Tensor:
        distr =  self.policy_net.get_distribution(states)
        actions = distr.sample()
        log_probs = distr.log_prob(actions)
        return actions, log_probs

    def eval_actions(self, states: np.ndarray) -> np.ndarray:
        states = torch.tensor(states)
        actions =  self.policy_net.mode(states)
        return actions.detach().numpy()

    def _collect_trajectory(self, seed: int):
        states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []

        state, _ = self.env.reset(seed=seed)
        done = False
        truncated = False
        total_reward = 0

        while not (done or truncated):
            state_tensor = torch.FloatTensor(state)[None].to(self.device)
            action, log_prob = self.sample_actions(state_tensor)
            value = self.value_net(state_tensor)

            next_state, reward, done, truncated, _ = self.env.step(action.item())

            states.append(state)
            actions.append(action[0])
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value[0])
            dones.append(done)

            state = next_state
            total_reward += reward

        return states, actions, rewards, log_probs, values, dones

    def train_step(self, episode: int):
        # sample trajectory
        states, actions, rewards, log_probs, values, dones = self._collect_trajectory(seed=self.seed+episode)

        # compute returns
        returns, advantage = [], 0.
        for r, v, done in reversed(list(zip(rewards, values, dones))):
            if done:
                advantage = 0
            advantage = r + self.gamma * advantage - v
            returns.insert(0, advantage + v)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        log_probs = torch.FloatTensor(log_probs).to(self.device)
        values = torch.FloatTensor(values).to(self.device)

        # main update logic
        for _ in range(self.update_epochs):
            for i in range(0, len(states), self.batch_size):
                batch_states = states[i:i + self.batch_size]
                batch_actions = actions[i:i + self.batch_size]
                batch_returns = returns[i:i + self.batch_size]
                batch_log_probs = log_probs[i:i + self.batch_size]
                batch_values = values[i:i + self.batch_size]

                new_distr = self.policy_net.get_distribution(batch_states)
                new_log_probs = new_distr.log_prob(batch_actions)
                new_values = self.value_net(batch_states).squeeze(1)

                ratio = torch.exp(new_log_probs - batch_log_probs)
                advantage = batch_returns - batch_values
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                actor_loss = -torch.min(ratio * advantage, clipped_ratio * advantage).mean()
                critic_loss = (new_values - batch_returns).pow(2).mean()

                self.policy_net.optimizer.zero_grad()
                actor_loss.backward()
                self.policy_net.optimizer.step()

                self.value_net.optimizer.zero_grad()
                critic_loss.backward()
                self.value_net.optimizer.step()

        total_reward = np.sum(rewards)
        return total_reward

    def save(self, save_path: str):
        super().save(save_path)
        torch.save(self.policy_net.state_dict(), f"{save_path}/policy_net.pt")
        torch.save(self.value_net.state_dict(), f"{save_path}/value_net.pt")

    def load(self, save_path: str):
        self.policy_net.load_state_dict(torch.load(f"{save_path}/policy_net.pt"))
        self.value_net.load_state_dict(torch.load(f"{save_path}/value_net.pt"))
