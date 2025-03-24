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
        #
        gae_lambda: float = 0.95, # from original paper
        clip_epsilon: float = 0.2, # from original paper
        #
        update_epochs: int = 3,
        n_train_trajectories: int = 3,
        batch_size: int = 128,
        #
        gamma: float = 0.99,
        seed: int = 42,
        device: str = "cpu",
    ):
        self.env = env
        self.policy_net = policy_net
        self.value_net = value_net

        self.gae_lambda = gae_lambda 
        self.clip_epsilon = clip_epsilon

        self.update_epochs = update_epochs
        self.n_train_trajectories = n_train_trajectories
        self.batch_size = batch_size

        self.gamma = gamma
        self.seed = seed
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

    def train_step(self, episode: int):
        # sample trajectory
        states, actions, rewards, returns, log_probs, advantages =\
            self._collect_dataset(seed=self.seed+episode)

        # main update logic
        for _ in range(self.update_epochs):
            for i in range(0, len(states), self.batch_size):
                # sample batches
                batch_states = states[i:i + self.batch_size]
                batch_actions = actions[i:i + self.batch_size]
                batch_returns = returns[i:i + self.batch_size]
                batch_log_probs = log_probs[i:i + self.batch_size]
                batch_advantages = advantages[i:i + self.batch_size]

                # infere new policy
                new_distr = self.policy_net.get_distribution(batch_states)
                new_batch_log_probs = new_distr.log_prob(batch_actions)
                new_batch_values = self.value_net(batch_states).squeeze(1)

                # update actor and critic
                self._update_actor(batch_log_probs, new_batch_log_probs, batch_advantages)
                self._update_critic(batch_returns, new_batch_values)

        total_reward = np.sum(rewards) / self.n_train_trajectories
        return total_reward

    def _update_actor(self, batch_log_probs: torch.Tensor, new_batch_log_probs: torch.Tensor, batch_advantages: torch.Tensor):
        ratio = torch.exp(new_batch_log_probs - batch_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        surr1 = ratio * batch_advantages
        surr2 = clipped_ratio * batch_advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        self.policy_net.optimizer.zero_grad()
        actor_loss.backward()
        self.policy_net.optimizer.step()

    def _update_critic(self, batch_returns: torch.Tensor, new_batch_values: torch.Tensor):
        critic_loss = (new_batch_values - batch_returns).pow(2).mean()

        self.value_net.optimizer.zero_grad()
        critic_loss.backward()
        self.value_net.optimizer.step()

    def _collect_dataset(self, seed: int):
        states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []

        # collect trajectory stats
        for _ in range(self.n_train_trajectories):
            state, _ = self.env.reset(seed=seed)
            done = False
            truncated = False

            while not (done or truncated):
                state_tensor = torch.FloatTensor(state)[None].to(self.device)
                action, log_prob = self.sample_actions(state_tensor)
                value = self.value_net(state_tensor)

                next_state, reward, done, truncated, _ = self.env.step(action.item())

                states.append(state)
                actions.append(action[0])
                rewards.append(reward)
                log_probs.append(log_prob[0])
                values.append(value[0])
                dones.append(done)

                state = next_state

        # compute returns and advantages
        returns, advantages = self._compute_gae_and_returns(rewards, dones, values)

        # convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        log_probs = torch.FloatTensor(log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)

        return states, actions, rewards, returns, log_probs, advantages

    def _compute_gae_and_returns(self, rewards: list, dones: list, values: list):
        returns_list = [rewards[-1]]
        gae_list = [0.]
        for t_reversed, reward in enumerate(reversed(rewards[:-1])):
            t = len(rewards) - t_reversed - 2

            # returns
            returns_list.append(reward + self.gamma * returns_list[-1] * (1 - dones[t]))

            # gae
            delta = reward + self.gamma * values[t + 1] - values[t]
            gae_list.append(delta + self.gamma * self.gae_lambda * gae_list[-1] * (1 - dones[t]))

        returns_list.reverse()
        gae_list.reverse()
        return returns_list, gae_list

    def save(self, save_path: str):
        super().save(save_path)
        torch.save(self.policy_net.state_dict(), f"{save_path}/policy_net.pt")
        torch.save(self.value_net.state_dict(), f"{save_path}/value_net.pt")

    def load(self, save_path: str):
        self.policy_net.load_state_dict(torch.load(f"{save_path}/policy_net.pt"))
        self.value_net.load_state_dict(torch.load(f"{save_path}/value_net.pt"))
