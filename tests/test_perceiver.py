import pytest
import torch

from perceiver_pytorch import Perceiver, PerceiverBlock, PerceiverIO

param = pytest.mark.parametrize

def test_perceiver_block():
    block = PerceiverBlock(
        dim = 16,
        context_dim = 16,
        cross_heads = 2,
        latent_heads = 4,
        cross_dim_head = 8,
        latent_dim_head = 8,
        self_per_cross_attn = 2
    )

    latents = torch.randn(2, 9, 16)
    data = torch.randn(2, 30, 16)

    mask = torch.ones(2, 30, dtype = torch.bool)
    mask[1, 20:] = False

    out = block(latents, context = data, mask = mask)
    assert out.shape == (2, 9, 16)

    # masked data must not affect the latents

    changed_data = data.clone()
    changed_data[1, 20:] += 100.

    changed_out = block(latents, context = changed_data, mask = mask)
    assert torch.allclose(out, changed_out)

    # self attention only

    self_block = PerceiverBlock(dim = 16, has_cross_attn = False)

    self_out = self_block(latents)
    assert self_out.shape == (2, 9, 16)

def test_perceiver_block_queries_contribute_kv():
    block = PerceiverBlock(
        dim = 16,
        context_dim = 16,
        cross_heads = 2,
        latent_heads = 4,
        cross_dim_head = 8,
        latent_dim_head = 8,
        queries_contribute_kv = True
    )

    latents = torch.randn(2, 9, 16)
    data = torch.randn(2, 30, 16)
    mask = torch.zeros(2, 30, dtype = torch.bool)

    out = block(latents, context = data, mask = mask)
    assert torch.isfinite(out).all()

    # queries are the only valid keys / values

    changed_data = data + 100.
    changed_out = block(latents, context = changed_data, mask = mask)
    assert torch.allclose(out, changed_out)

    with pytest.raises(AssertionError):
        PerceiverBlock(dim = 16, context_dim = 8, queries_contribute_kv = True)

@param('attn_residual', (False, True))
@param('inverted_cross_attn', [
    False,
    True,
    (True, False)
])
def test_perceiver(attn_residual, inverted_cross_attn):
    model = Perceiver(
        input_channels = 3,
        input_axis = 2,
        num_freq_bands = 3,
        max_freq = 5.,
        depth = 2,
        num_latents = 16,
        latent_dim = 16,
        cross_heads = 1,
        latent_heads = 2,
        cross_dim_head = 8,
        latent_dim_head = 8,
        num_classes = 10,
        attn_dropout = 0.,
        ff_dropout = 0.,
        weight_tie_layers = False,
        inverted_cross_attn = inverted_cross_attn,
        attn_residual = attn_residual
    )

    img = torch.randn(1, 32, 32, 3)
    out = model(img)

    assert out.shape == (1, 10), 'output shape must be correct'

@param('attn_residual', (False, True))
@param('inverted_cross_attn', (False, True))
def test_perceiver_io(attn_residual, inverted_cross_attn):
    model = PerceiverIO(
        dim = 32,
        queries_dim = 32,
        logits_dim = 10,
        depth = 2,
        num_latents = 16,
        latent_dim = 16,
        cross_heads = 1,
        latent_heads = 2,
        cross_dim_head = 8,
        latent_dim_head = 8,
        attn_residual = attn_residual,
        inverted_cross_attn = inverted_cross_attn
    )

    seq = torch.randn(1, 64, 32)
    queries = torch.randn(16, 32)
    out = model(seq, queries = queries)

    assert out.shape == (1, 16, 10), 'output shape must be correct'
