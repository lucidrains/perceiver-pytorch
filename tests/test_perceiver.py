import torch
import pytest
from perceiver_pytorch import Perceiver

@pytest.mark.parametrize('inverted_cross_attn', [
    False,
    True,
    (True, False)
])
def test_perceiver(inverted_cross_attn):
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
        inverted_cross_attn = inverted_cross_attn
    )

    img = torch.randn(1, 32, 32, 3)
    out = model(img)

    assert out.shape == (1, 10), 'output shape must be correct'
