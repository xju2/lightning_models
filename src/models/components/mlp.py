"""Multilayer Perceptron (MLP) module."""
from typing import List, Optional
try:
    from itertools import pairwise
except ImportError:
    from more_itertools import pairwise

import torch
import torch.nn as nn
import torch.nn.functional as F

def build_linear_layers(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    layer_norm: bool = True,
    dropout: float = 0.0,
    last_activation: Optional[torch.nn.Module] = None,
    leaky_ratio: float = 0.2,
    ) -> List[nn.Module]:
    
    layer_list = [torch.nn.Linear(input_dim, hidden_dims[0]),
        torch.nn.LeakyReLU(leaky_ratio)]
    
    
    for l0,l1 in pairwise(hidden_dims):
        layer_list.append(torch.nn.Linear(l0, l1))

        if layer_norm:
            layer_list.append(torch.nn.LayerNorm(l1))

        layer_list.append(torch.nn.LeakyReLU(leaky_ratio))
        
        if dropout > 0:
            layer_list.append(torch.nn.Dropout(dropout))

    layer_list.append(torch.nn.Linear(hidden_dims[-1], output_dim))
    if last_activation is not None:
        layer_list.append(last_activation)

    return layer_list

class MLPModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        layer_norm: bool = True,
        dropout: float = 0.0,
        last_activation: Optional[torch.nn.Module] = None,
    ):
        super().__init__()
                    
        # build the linear model
        self.model = nn.Sequential(*build_linear_layers(
            input_dim, hidden_dims, output_dim, layer_norm, dropout, last_activation)
        )
        
    def forward(self, x) -> torch.Tensor:
        return self.model(x)


class MLPTypeEmbeddingModule(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

            
        # build the linear model
        self.model = nn.Sequential(*build_linear_layers(
            embedding_dim, hidden_dims, output_dim, True, dropout)
        )
        
    def forward(self, x) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.view(-1)
        embeds = self.embeddings(x).view(batch_size, -1)
        return  self.model(embeds)


class MLPWithEmbeddingModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        word_embedding_dim: int,
        num_words: int,
        encoder_dims: List[int],
        decoder_dims: List[int],
        output_dim: int,
        last_activation: Optional[torch.nn.Module] = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.normal_mlp = MLPModule(input_dim, encoder_dims, encoder_dims[-1])
        
        self.type_mlp = MLPTypeEmbeddingModule(
            vocab_size, word_embedding_dim, 
            encoder_dims, encoder_dims[-1], dropout=dropout)
        
        self.decoder = MLPModule(
            encoder_dims[-1]*(1+num_words),
            decoder_dims, output_dim, last_activation=last_activation)
        
    def forward(self, x, type_ids) -> torch.Tensor:
        normal_embeds = self.normal_mlp(x)
        num_particles = type_ids.shape[1]
        
        ## same MLP acting on different particles
        type_embeds = [self.type_mlp(type_ids[:, i]) for i in range(num_particles)]
        
        decoder_embds = self.decoder(torch.cat([normal_embeds]+type_embeds, dim=1))
        return decoder_embds
    
if __name__ == "__main__":
    model = MLPModule(784, [256, 256, 256], 10)
    print(model)