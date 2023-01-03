import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

from perceiver_pytorch.perceiver_pytorch import exists, default, cache_fn, fourier_encode, PreNorm, FeedForward, Attention

# latent mixer

def Mixer(seq_len, mult = 4, dropout = 0.):
    return nn.Sequential(
        nn.Conv1d(seq_len, seq_len * mult, 1),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Conv1d(seq_len * mult, seq_len, 1)
    )

# main class

class Perceiver(nn.Module):
    def __init__(
        self,
        *,
        num_freq_bands,
        depth,
        max_freq,
        input_channels = 3,
        input_axis = 2,
        num_latents = 512,
        latent_dim = 512,
        cross_heads = 1,
        latent_heads = 8,
        cross_dim_head = 64,
        latent_dim_head = 64,
        num_classes = 1000,
        attn_dropout = 0.,
        ff_dropout = 0.,
        weight_tie_layers = False,
        **kwargs
    ):
        super().__init__()
        self.input_axis = input_axis
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands

        input_dim = input_axis * ((num_freq_bands * 2) + 1) + input_channels

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        get_cross_attn  = lambda: PreNorm(latent_dim, Attention(latent_dim, input_dim, heads = cross_heads, dim_head = cross_dim_head, dropout = attn_dropout), context_dim = input_dim)
        get_latent_attn = lambda: PreNorm(latent_dim, Mixer(num_latents, dropout = ff_dropout))
        get_cross_ff    = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))
        get_latent_ff   = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))

        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(cache_fn, (get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            self.layers.append(nn.ModuleList([
                get_cross_attn(**cache_args),
                get_cross_ff(**cache_args),
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

        self.to_logits = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, num_classes)
        )

    def forward(self, data, mask = None):
        b, *axis, _, device = *data.shape, data.device
        assert len(axis) == self.input_axis, 'input data must have the right number of axis'

        # calculate fourier encoded positions in the range of [-1, 1], for all axis

        axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps = size, device = device), axis))
        pos = torch.stack(torch.meshgrid(*axis_pos, indexing = 'ij'), dim = -1)
        enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands)
        enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
        enc_pos = repeat(enc_pos, '... -> b ...', b = b)

        # concat to channels of data and flatten axis

        data = torch.cat((data, enc_pos), dim = -1)
        data = rearrange(data, 'b ... d -> b (...) d')

        x = repeat(self.latents, 'n d -> b n d', b = b)

        for cross_attn, cross_ff, latent_attn, latent_ff in self.layers:
            x = cross_attn(x, context = data, mask = mask) + x
            x = cross_ff(x) + x
            x = latent_attn(x) + x
            x = latent_ff(x) + x

        x = x.mean(dim = -2)
        return self.to_logits(x)
