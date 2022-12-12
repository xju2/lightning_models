"""Residual Network (ResNet) implementation Based on MLPs."""
try:
    from itertools import pairwise
except ImportError:
    from more_itertools import pairwise

import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
            )

    def forward(self, x) -> torch.Tensor:
        return self.model(x) + x
    

def build_layers(block, input_dim, hidden_dims, output_dim):
    layers = [block(input_dim, hidden_dims[0])]
    for l0, l1 in pairwise(hidden_dims):
        layers.append(block(l0, l1))
    layers.append(nn.Linear(hidden_dims[-1], output_dim))
    return layers

  
class ResMLPModule(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim,
                 dropout = 0.0,         # not used.
                 layer_norm = True,    # not used.
                 last_activation = None):
        super().__init__()
        
        block = ResidualBlock
        layers = build_layers(block, input_dim, hidden_dims, output_dim)
        if last_activation is not None:
            layers.append(last_activation)

        self.model = nn.Sequential(*layers)
        
        
    def forward(self, x) -> torch.Tensor:
        return self.model(x)
