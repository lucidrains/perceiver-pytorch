from __future__ import annotations
from math import pi, log
from functools import wraps, partial

import torch
from torch import nn, einsum, Tensor, is_tensor
import torch.nn.functional as F

from einops import rearrange, repeat

from perceiver_pytorch.perceiver_pytorch import PerceiverBlock

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def l1norm(t, dim = -1, eps = 1e-8):
    return F.normalize(t, p = 1, dim = dim, eps = eps)

# constants

LinearNoBias = partial(nn.Linear, bias = False)

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

# structured dropout, more effective than traditional attention dropouts

def dropout_seq(seq, mask, dropout):
    b, n, *_, device = *seq.shape, seq.device
    logits = torch.randn(b, n, device = device)

    if exists(mask):
        logits = logits.masked_fill(~mask, -torch.finfo(logits.dtype).max)

    keep_prob = 1. - dropout
    num_keep = max(1,  int(keep_prob * n))
    keep_indices = logits.topk(num_keep, dim = 1).indices

    batch_indices = torch.arange(b, device = device)
    batch_indices = rearrange(batch_indices, 'b -> b 1')

    seq = seq[batch_indices, keep_indices]

    if exists(mask):
        seq_counts = mask.sum(dim = -1)
        seq_keep_counts = torch.ceil(seq_counts * keep_prob).int()
        keep_mask = torch.arange(num_keep, device = device) < rearrange(seq_keep_counts, 'b -> b 1')

        mask = mask[batch_indices, keep_indices] & keep_mask

    return seq, mask

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

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context = None, mask = None, inverted_attention = False):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

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

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

# enformer attention residual
# attention residuals proposed by Guangyu (Nathan) Chen et al. with Kimi team (https://arxiv.org/abs/2603.15031)
# using the attention pool designed by Žiga Avsec et al. in Enformer (https://www.nature.com/articles/s41592-021-01252-x)

class EnformerAttentionResidual(nn.Module):
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

        return einsum('b n l d, b n l d -> b n d', past_layers, attn)

# main class

class PerceiverIO(nn.Module):
    def __init__(
        self,
        *,
        depth,
        dim,
        queries_dim,
        logits_dim = None,
        num_latents = 512,
        latent_dim = 512,
        cross_heads = 1,
        latent_heads = 8,
        cross_dim_head = 64,
        latent_dim_head = 64,
        weight_tie_layers = False,
        decoder_ff = False,
        seq_dropout_prob = 0.,
        attn_residual = False,
        attn_residual_rank = 64,
        inverted_cross_attn = False
    ):
        super().__init__()
        self.seq_dropout_prob = seq_dropout_prob

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        self.cross_attend_block = PerceiverBlock(
            dim = latent_dim,
            context_dim = dim,
            cross_heads = cross_heads,
            latent_heads = latent_heads,
            cross_dim_head = cross_dim_head,
            latent_dim_head = latent_dim_head,
            self_per_cross_attn = 0
        )

        self.has_attn_residual = attn_residual
        self.inverted_cross_attn = inverted_cross_attn

        get_latent_block = lambda: PerceiverBlock(
            dim = latent_dim,
            cross_heads = cross_heads,
            latent_heads = latent_heads,
            cross_dim_head = cross_dim_head,
            latent_dim_head = latent_dim_head,
            has_cross_attn = False
        )
        get_attn_residual = lambda: (EnformerAttentionResidual(latent_dim, rank = attn_residual_rank) if attn_residual else None)

        get_latent_block, get_attn_residual = map(cache_fn, (get_latent_block, get_attn_residual))

        self.layers = nn.ModuleList([])
        self.attn_residuals = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}

        for i in range(depth):
            self.layers.append(get_latent_block(**cache_args))
            self.attn_residuals.append(get_attn_residual(**cache_args))

        self.decoder_cross_attn = PreNorm(queries_dim, Attention(queries_dim, latent_dim, heads = cross_heads, dim_head = cross_dim_head), context_dim = latent_dim)
        self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None

        self.to_logits = nn.Linear(queries_dim, logits_dim) if exists(logits_dim) else nn.Identity()

    def forward(
        self,
        data,
        mask = None,
        queries = None,
        block_outputs: list[Tensor] | None = None,
        return_hiddens = False
    ):
        b, *_, device = *data.shape, data.device

        x = repeat(self.latents, 'n d -> b n d', b = b)

        # structured dropout (as done in perceiver AR https://arxiv.org/abs/2202.07765)

        if self.training and self.seq_dropout_prob > 0.:
            data, mask = dropout_seq(data, mask, self.seq_dropout_prob)

        # cross attention only happens once for Perceiver IO

        x = self.cross_attend_block(
            x,
            context = data,
            mask = mask,
            inverted_attention = self.inverted_cross_attn
        )

        # keep the hidden states only if they are needed for the attention residual, or explicitly requested

        keep_hiddens = return_hiddens or self.has_attn_residual or exists(block_outputs)

        if keep_hiddens:
            block_outputs = [block_outputs] if is_tensor(block_outputs) else list(default(block_outputs, [x]))

        # layers

        for block, attn_residual in zip(self.layers, self.attn_residuals):
            x = block(x)

            if keep_hiddens:
                block_outputs.append(x)

            if exists(attn_residual):
                x = attn_residual(block_outputs)

        if not exists(queries):
            if not return_hiddens:
                return x

            return x, block_outputs

        # make sure queries contains batch dimension

        if queries.ndim == 2:
            queries = repeat(queries, 'n d -> b n d', b = b)

        # cross attend from decoder queries to latents

        latents = self.decoder_cross_attn(queries, context = x)

        # optional decoder feedforward

        if exists(self.decoder_ff):
            latents = latents + self.decoder_ff(latents)

        # final linear out

        out = self.to_logits(latents)

        if not return_hiddens:
            return out

        return out, block_outputs

# Perceiver LM example

class PerceiverLM(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        max_seq_len,
        **kwargs
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.perceiver_io = PerceiverIO(
            dim = dim,
            queries_dim = dim,
            logits_dim = num_tokens,
            **kwargs
        )

    def forward(
        self,
        x,
        mask = None
    ):
        n, device = x.shape[1], x.device
        x = self.token_emb(x)

        pos_emb = self.pos_emb(torch.arange(n, device = device))
        pos_emb = rearrange(pos_emb, 'n d -> () n d')
        x = x + pos_emb

        logits = self.perceiver_io(x, mask = mask, queries = x)
        return logits
