from math import pi, log
from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

from perceiver_pytorch.perceiver_pytorch import exists, default, cache_fn, fourier_encode, RMSNorm, PreNorm, FeedForward, Attention

# main class

class Perceiver(nn.Module):
    def __init__(
        self,
        *,
        num_freq_bands,
        depth,
        max_freq,
        freq_base = 2,
        input_channels = 3,
        input_axis = 2,
        num_latents = 512,
        cross_dim = 512,
        latent_dim = 512,
        cross_heads = 1,
        latent_heads = 8,
        cross_dim_head = 64,
        latent_dim_head = 64,
        num_classes = 1000,
        attn_dropout = 0.,
        ff_dropout = 0.,
        weight_tie_layers = False
    ):
        super().__init__()
        self.input_axis = input_axis
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands
        self.freq_base = freq_base

        input_dim = input_axis * ((num_freq_bands * 2) + 1) + input_channels

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.pos_emb = nn.Parameter(torch.randn(num_latents, latent_dim))

        get_cross_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, input_dim, heads = cross_heads, dim_head = cross_dim_head, dropout = attn_dropout), context_dim = input_dim)
        get_cross_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))

        get_rev_cross_attn = lambda: PreNorm(input_dim, Attention(input_dim, latent_dim, heads = cross_heads, dim_head = cross_dim_head, dropout = attn_dropout), context_dim = latent_dim)
        get_rev_cross_ff = lambda: PreNorm(input_dim, FeedForward(input_dim, dropout = ff_dropout))

        get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads = latent_heads, dim_head = latent_dim_head, dropout = attn_dropout))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))

        if weight_tie_layers:
            get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(cache_fn, (get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                get_cross_attn(),
                get_cross_ff(),
                get_rev_cross_attn(),
                get_rev_cross_ff(),
                get_latent_attn(),
                get_latent_ff()
            ]))

        self.to_logits = nn.Sequential(
            RMSNorm(latent_dim),
            nn.Linear(latent_dim, num_classes)
        )

    def forward(self, data, mask = None):
        b, *axis, _, device = *data.shape, data.device
        assert len(axis) == self.input_axis, 'input data must have the right number of axis'

        # calculate fourier encoded positions in the range of [-1, 1], for all axis

        axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps = size, device = device), axis))
        pos = torch.stack(torch.meshgrid(*axis_pos), dim = -1)
        enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands, base = self.freq_base)
        enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
        enc_pos = repeat(enc_pos, '... -> b ...', b = b)

        # concat to channels of data and flatten axis

        data = torch.cat((data, enc_pos), dim = -1)
        data = rearrange(data, 'b ... d -> b (...) d')

        x = self.latents + self.pos_emb
        x = repeat(x, 'n d -> b n d', b = b)

        for i, (cross_attn, cross_ff, rev_cross_attn, rev_cross_ff, latent_attn, latent_ff) in enumerate(self.layers):
            is_last = i == (len(self.layers) - 1)

            x = cross_attn(x, context = data, mask = mask) + x
            x = cross_ff(x) + x

            if not is_last:
                data = rev_cross_attn(data, context = x) + data
                data = rev_cross_ff(data) + data

            x = latent_attn(x) + x
            x = latent_ff(x) + x

        x = x.mean(dim = -2)
        return self.to_logits(x)
