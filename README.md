<img src="./perceiver.png" width="600px"></img>

## Perceiver - Pytorch

Implementation of <a href="https://arxiv.org/abs/2103.03206">Perceiver</a>, General Perception with Iterative Attention, in Pytorch

## Install

```bash
$ pip install perceiver-pytorch
```

## Usage

```python
import torch
from perceiver_pytorch.perceiver_pytorch import Perceiver

model = Perceiver(
	num_fourier_features = 6,    # number of fourier features, with original value (2 * K + 1)
    depth = 48,                  # depth of net, in paper, they went deep, making up for lack of attention
    num_latents = 6,             # number of latents, or induced set points, or centroids. different papers giving it different names
    cross_dim = 512,             # cross attention dimension
    latent_dim = 512,            # latent dimension
    cross_heads = 1,             # number of heads for cross attention. paper said 1
    latent_heads = 8,            # number of heads for latent self attention, 8
    cross_dim_head = 64,
    latent_dim_head = 64,
    num_classes = 1000,          # output number of classes
    attn_dropout = 0.,
    ff_dropout = 0.,
    weight_tie_layers = False    # whether to weight tie layers (optional, as indicated in the diagram)
)

img = torch.randn(1, 224 * 224) # 1 imagenet image, pixelized

model(img) # (1, 1000)
```
## Citations

```bibtex
@misc{jaegle2021perceiver,
    title   = {Perceiver: General Perception with Iterative Attention},
    author  = {Andrew Jaegle and Felix Gimeno and Andrew Brock and Andrew Zisserman and Oriol Vinyals and Joao Carreira},
    year    = {2021},
    eprint  = {2103.03206},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```
