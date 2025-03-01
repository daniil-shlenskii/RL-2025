import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dims: list[int]=[]):
        super().__init__()
        hidden_dims = [in_dim, *hidden_dims, out_dim]
        layers = []
        for i, (dim_in, dim_out) in enumerate(zip(hidden_dims[:-1], hidden_dims[1:])):
            layers.append(nn.Linear(dim_in, dim_out))
            if i < len(hidden_dims[:-1]) - 1:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
