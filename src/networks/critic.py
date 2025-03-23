import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import MLP


class StateCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden_dims: list[int]=[], lr: float=3e-4): 
        super().__init__()
        self.net = MLP(obs_dim, 1, hidden_dims)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        values = self.net(state)
        return values
