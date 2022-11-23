"""Multilayer Perceptron (MLP) module."""
from typing import List, Optional
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
        last_activation: Optional[torch.nn.Module] = None,
        # <TODO, add activation function>
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.layer_norm = layer_norm
        self.dropout = dropout
        self.last_activation = last_activation
        self.leaky_ratio = 0.2
        # activations = {
        #     'relu': nn.ReLU(),
        #     'sigmoid': nn.Sigmoid(),
        #     'tanh': nn.Tanh()
        # }
        self.last_act = None if last_activation is None else last_activation
            
        # build the model
        self.model = nn.Sequential(*self._build_layers())
        
    def forward(self, x) -> torch.Tensor:
        return self.model(x)
    

    def _build_layers(self) -> List[nn.Module]:
        layer_list = [torch.nn.Linear(self.input_dim, self.hidden_dims[0]),
            torch.nn.LeakyReLU(self.leaky_ratio)]
        
        
        for l0,l1 in pairwise(self.hidden_dims):
            layer_list.append(torch.nn.Linear(l0, l1))

            if self.layer_norm:
                layer_list.append(torch.nn.LayerNorm(l1))

            layer_list.append(torch.nn.LeakyReLU(self.leaky_ratio))
            
            if self.dropout > 0:
                layer_list.append(torch.nn.Dropout(self.dropout))

        layer_list.append(torch.nn.Linear(self.hidden_dims[-1], self.output_dim))
        if self.last_act is not None:
            layer_list.append(self.last_act)

        return layer_list
    
    
if __name__ == "__main__":
    model = MyMLPMoudle(784, [256, 256, 256], 10)
    print(model)