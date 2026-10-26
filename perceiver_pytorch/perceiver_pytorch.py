from __future__ import annotations
from math import pi, log
from functools import wraps, partial

import torch
from torch import nn, stack, cat, Tensor, is_tensor
from torch.nn import Module, ModuleList, Linear
import torch.nn.functional as F

from einops import rearrange, repeat, einsum
from einops.layers.torch import Reduce

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def l1norm(t, dim = -1, eps = 1e-8):
    return F.normalize(t, p = 1, dim = dim, eps = eps)

def cache_fn(f):
    cache = dict()
    @wraps(f)
    def cached_fn(*args, _cache = True, key = None, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if key in cache:
            return cache[key]
        result = f(*args, **kwargs)
        cache[key] = result
        return result
    return cached_fn

def fourier_encode(x, max_freq, num_bands = 4):
    x = rearrange(x, '... -> ... 1')
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.linspace(1., max_freq / 2, num_bands, device = device, dtype = dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi
    x = cat([x.sin(), x.cos()], dim = -1)
    x = cat((x, orig_x), dim = -1)
    return x

# constants

LinearNoBias = partial(Linear, bias = False)

# helper classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GEGLU(Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(
        self,
        query_dim,
        context_dim = None,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        queries_contribute_kv = False
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        if queries_contribute_kv:
            assert context_dim == query_dim, 'queries_contribute_kv requires context_dim to equal query_dim'

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.queries_contribute_kv = queries_contribute_kv

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context = None, mask = None, inverted_attention = False):
        h = self.heads

        q = self.to_q(x)
        has_context = exists(context)
        context = default(context, x)

        if self.queries_contribute_kv and has_context:
            # queries join the key / value set
            kv = cat((x, context), dim = 1)

            if exists(mask):
                mask = cat((mask.new_ones(mask.shape[0], x.shape[1]), mask), dim = -1)
        else:
            kv = context

        k, v = self.to_kv(kv).chunk(2, dim = -1)

        q, k, v = (rearrange(t, 'b n (h d) -> (b h) n d', h = h) for t in (q, k, v))

        sim = einsum(q, k, 'b i d, b j d -> b i j') * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of

        if inverted_attention:
            attn = sim.softmax(dim = -2)

            if exists(mask):
                attn = attn.masked_fill(~mask, 0.)

            attn = l1norm(attn)
        else:
            attn = sim.softmax(dim = -1)

        attn = self.dropout(attn)

        # aggregate

        out = einsum(attn, v, 'b i j, b j d -> b i d')
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

# enformer attention residual
# attention residuals proposed by Guangyu (Nathan) Chen et al. with Kimi team (https://arxiv.org/abs/2603.15031)
# using the attention pool designed by Žiga Avsec et al. in Enformer (https://www.nature.com/articles/s41592-021-01252-x)

class EnformerAttentionResidual(Module):
    def __init__(
        self,
        dim,
        *,
        rank = 64
    ):
        super().__init__()
        self.to_attn_logits = nn.Sequential(
            nn.RMSNorm(dim),
            LinearNoBias(dim, rank),
            LinearNoBias(rank, dim)
        )

    def forward(
        self,
        block_outputs: list[Tensor] | Tensor
    ):
        block_outputs = list(block_outputs)
        past_layers = rearrange(block_outputs, 'l b n d -> b n l d')

        logits = self.to_attn_logits(past_layers)
        attn = logits.softmax(dim = -2)

        return einsum(past_layers, attn, 'b n l d, b n l d -> b n d')

# block

class PerceiverBlock(Module):
    def __init__(
        self,
        *,
        dim,
        context_dim = None,
        cross_heads = 1,
        latent_heads = 8,
        cross_dim_head = 64,
        latent_dim_head = 64,
        self_per_cross_attn = 1,
        attn_dropout = 0.,
        ff_dropout = 0.,
        has_cross_attn = True,
        queries_contribute_kv = False
    ):
        super().__init__()
        self.has_cross_attn = has_cross_attn

        if has_cross_attn:
            context_dim = default(context_dim, dim)
            self.cross_attn = PreNorm(
                dim,
                Attention(
                    dim,
                    context_dim,
                    heads = cross_heads,
                    dim_head = cross_dim_head,
                    dropout = attn_dropout,
                    queries_contribute_kv = queries_contribute_kv
                ),
                context_dim = context_dim
            )
            self.cross_ff = PreNorm(dim, FeedForward(dim, dropout = ff_dropout))
        else:
            self.cross_attn = None
            self.cross_ff = None

        self.self_attns = ModuleList([])

        for _ in range(self_per_cross_attn):
            self.self_attns.append(ModuleList([
                PreNorm(dim, Attention(dim, heads = latent_heads, dim_head = latent_dim_head, dropout = attn_dropout)),
                PreNorm(dim, FeedForward(dim, dropout = ff_dropout))
            ]))

    def forward(
        self,
        x,
        context = None,
        mask = None,
        inverted_attention = False
    ):
        if self.has_cross_attn:
            x = self.cross_attn(x, context = context, mask = mask, inverted_attention = inverted_attention) + x
            x = self.cross_ff(x) + x

        for self_attn, self_ff in self.self_attns:
            x = self_attn(x) + x
            x = self_ff(x) + x

        return x

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
        fourier_encode_data = True,
        self_per_cross_attn = 1,
        final_classifier_head = True,
        inverted_cross_attn: bool | tuple[bool, ...] = False,
        attn_residual = False,
        attn_residual_rank = 64
    ):
        """The shape of the final attention mechanism will be:
        depth * (cross attention -> self_per_cross_attn * self attention)

        Args:
          num_freq_bands: Number of freq bands, with original value (2 * K + 1)
          depth: Depth of net.
          max_freq: Maximum frequency, hyperparameter depending on how
              fine the data is.
          freq_base: Base for the frequency
          input_channels: Number of channels for each token of the input.
          input_axis: Number of axes for input data (2 for images, 3 for video)
          num_latents: Number of latents, or induced set points, or centroids.
              Different papers giving it different names.
          latent_dim: Latent dimension.
          cross_heads: Number of heads for cross attention. Paper said 1.
          latent_heads: Number of heads for latent self attention, 8.
          cross_dim_head: Number of dimensions per cross attention head.
          latent_dim_head: Number of dimensions per latent self attention head.
          num_classes: Output number of classes.
          attn_dropout: Attention dropout
          ff_dropout: Feedforward dropout
          weight_tie_layers: Whether to weight tie layers (optional).
          fourier_encode_data: Whether to auto-fourier encode the data, using
              the input_axis given. defaults to True, but can be turned off
              if you are fourier encoding the data yourself.
          self_per_cross_attn: Number of self attention blocks per cross attn.
          final_classifier_head: mean pool and project embeddings to number of classes (num_classes) at the end
        """
        super().__init__()
        self.input_axis = input_axis
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands

        self.has_attn_residual = attn_residual

        # inverted attention
        # slot attention: https://arxiv.org/abs/2006.15055
        # inverted attention: https://openreview.net/forum?id=3H8j14mA3X

        if isinstance(inverted_cross_attn, bool):
            inverted_cross_attn = (inverted_cross_attn,) * depth

        assert len(inverted_cross_attn) == depth, f'inverted_cross_attn tuple must have length of depth ({depth})'
        self.inverted_cross_attn = inverted_cross_attn

        # fourier

        self.fourier_encode_data = fourier_encode_data
        fourier_channels = (input_axis * ((num_freq_bands * 2) + 1)) if fourier_encode_data else 0
        input_dim = fourier_channels + input_channels

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        get_block = lambda: PerceiverBlock(
            dim = latent_dim,
            context_dim = input_dim,
            cross_heads = cross_heads,
            latent_heads = latent_heads,
            cross_dim_head = cross_dim_head,
            latent_dim_head = latent_dim_head,
            self_per_cross_attn = self_per_cross_attn,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout
        )
        get_attn_residual = lambda: (EnformerAttentionResidual(latent_dim, rank = attn_residual_rank) if attn_residual else None)

        get_block, get_attn_residual = map(cache_fn, (get_block, get_attn_residual))

        self.layers = ModuleList([])
        self.attn_residuals = ModuleList([])

        for i in range(depth):
            cache_args = {'_cache': i > 0 and weight_tie_layers}
            self.layers.append(get_block(**cache_args))
            self.attn_residuals.append(get_attn_residual(**cache_args))

        self.to_logits = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, num_classes)
        ) if final_classifier_head else nn.Identity()

    def forward(
        self,
        data,
        mask = None,
        return_embeddings = False,
        block_outputs: list[Tensor] | None = None,
        return_hiddens = False
    ):
        b, *axis, _, device, dtype = *data.shape, data.device, data.dtype
        assert len(axis) == self.input_axis, 'input data must have the right number of axis'

        if self.fourier_encode_data:
            # calculate fourier encoded positions in the range of [-1, 1], for all axis

            axis_pos = [torch.linspace(-1., 1., steps = size, device = device, dtype = dtype) for size in axis]
            pos = stack(torch.meshgrid(*axis_pos, indexing = 'ij'), dim = -1)
            enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands)
            enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
            enc_pos = repeat(enc_pos, '... -> b ...', b = b)

            data = cat((data, enc_pos), dim = -1)

        # concat to channels of data and flatten axis

        data = rearrange(data, 'b ... d -> b (...) d')

        x = repeat(self.latents, 'n d -> b n d', b = b)

        # keep the hidden states only if they are needed for the attention residual, or explicitly requested

        keep_hiddens = return_hiddens or self.has_attn_residual or exists(block_outputs)

        if keep_hiddens:
            block_outputs = [block_outputs] if is_tensor(block_outputs) else list(default(block_outputs, [x]))

        # layers

        for i, (block, attn_residual) in enumerate(zip(self.layers, self.attn_residuals)):
            x = block(
                x,
                context = data,
                mask = mask,
                inverted_attention = self.inverted_cross_attn[i]
            )

            # keep the hidden states for the attention residual

            if keep_hiddens:
                block_outputs.append(x)

            # attention residual

            if exists(attn_residual):
                x = attn_residual(block_outputs)

        # allow for fetching embeddings

        out = x if return_embeddings else self.to_logits(x)

        if not return_hiddens:
            return out

        return out, block_outputs
