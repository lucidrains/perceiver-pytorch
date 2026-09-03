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


def test_weight_tied_latent_transformers_include_first_layer():
    model = Perceiver(
        input_channels = 3,
        input_axis = 2,
        num_freq_bands = 1,
        max_freq = 2.,
        depth = 3,
        num_latents = 4,
        latent_dim = 8,
        cross_heads = 1,
        latent_heads = 1,
        cross_dim_head = 8,
        latent_dim_head = 8,
        num_classes = 2,
        self_per_cross_attn = 2,
        weight_tie_layers = True,
    )

    # Each position in the repeated latent transformer stack must point to
    # one shared module, including the first depth layer.
    for block_ind in range(2):
        latent_attn = [layer[2][block_ind][0] for layer in model.layers]
        latent_ff = [layer[2][block_ind][1] for layer in model.layers]
        assert all(module is latent_attn[0] for module in latent_attn)
        assert all(module is latent_ff[0] for module in latent_ff)


def test_untied_latent_transformers_remain_independent():
    model = Perceiver(
        input_channels = 3,
        input_axis = 2,
        num_freq_bands = 1,
        max_freq = 2.,
        depth = 2,
        num_latents = 4,
        latent_dim = 8,
        cross_heads = 1,
        latent_heads = 1,
        cross_dim_head = 8,
        latent_dim_head = 8,
        num_classes = 2,
        weight_tie_layers = False,
    )

    assert model.layers[0][2][0][0] is not model.layers[1][2][0][0]
    assert model.layers[0][2][0][1] is not model.layers[1][2][0][1]
