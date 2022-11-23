"""Multilayer Perceptron (MLP) module."""
from typing import List
try:
    from itertools import pairwise
except ImportError:
    from more_itertools import pairwise

import torch
import torch.nn as nn
import torch.nn.functional as F



class MyMLPMoudle(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        layer_norm: bool = True,
        dropout: float = 0.0,
        # <TODO, add activation function>
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.layer_norm = layer_norm
        self.dropout = dropout

        # build the model
        self.model = nn.Sequential(*self._build_layers())
        
    def forward(self, x) -> torch.Tensor:
        return self.model(x)
    

    def _build_layers(self) -> List[nn.Module]:
        layer_list = [torch.nn.Linear(self.input_dim, self.hidden_dims[0])]
        
        
        for l0,l1 in pairwise(self.hidden_dims):
            layer_list.append(torch.nn.Linear(l0, l1))

            if self.layer_norm:
                layer_list.append(torch.nn.LayerNorm(l1))

            layer_list.append(torch.nn.LeakyReLU(0.2))
            
            if self.dropout > 0:
                layer_list.append(torch.nn.Dropout(self.dropout))

        layer_list.append(torch.nn.Linear(self.hidden_dims[-1], self.output_dim))
        return layer_list
    
    
if __name__ == "__main__":
    model = MyMLPMoudle(784, [256, 256, 256], 10)
    print(model)