import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

# helpers

def exists(val):
    return val is not None

# helper classes

# main class

class Perceiver(nn.Module):
    def __init__(
        self,
        depth,
        cross_attn_dim = 512,
        num_latents = 6,
        cross_attn_heads = 1,
        cross_attn_dim_head = 64,
        latent_attn_dim = 512,
        latent_attn_heads = 8,
        latent_attn_dim_head = 64,
        num_classes = 1000,
        attn_dropout = 0.,
        ff_dropout = 0.
    ):
        super().__init__()

    def forward(self, x):
        return x
