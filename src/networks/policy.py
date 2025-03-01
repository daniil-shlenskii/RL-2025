import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import MLP


class DiscretePolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list[int]=[], lr: float=3e-4): 
        super().__init__()
        self.net = MLP(obs_dim, action_dim, hidden_dims)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        log_probs = self.net(state)
        return log_probs

    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        log_probs = self.forward(state)
        probs = F.softmax(log_probs, dim=-1)

        distr = torch.distributions.Categorical(probs=probs)
        actions = distr.sample()
        log_probs = distr.log_prob(actions)
        return actions, log_probs 

    def eval_actions(self, state: torch.Tensor) -> torch.Tensor:
        log_probs = self.forward(state)
        actions = torch.argmax(log_probs, dim=-1)
        return actions
