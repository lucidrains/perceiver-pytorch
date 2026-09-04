import torch
import pytest
from perceiver_pytorch import Perceiver, PerceiverIO

param = pytest.mark.parametrize

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

@pytest.mark.parametrize('module_path', [
    'perceiver_pytorch.gated',
    'perceiver_pytorch.mixed_latents',
])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_perceiver_variants_preserve_low_precision(module_path, dtype):
    import importlib

    Perceiver = importlib.import_module(module_path).Perceiver

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
        ff_dropout = 0.
    ).to(dtype)

    img = torch.randn(1, 32, 32, 3, dtype = dtype)
    out = model(img)

    assert out.shape == (1, 10), 'output shape must be correct'
    assert out.dtype == dtype, 'a low-precision input must not be upcast by the fourier position grid'
