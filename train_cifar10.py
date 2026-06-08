# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "fire",
#     "torch",
#     "torchvision",
#     "accelerate",
#     "wandb",
#     "einops",
# ]
# ///

import os
import fire
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from einops import rearrange
from accelerate import Accelerator
from perceiver_pytorch import Perceiver

# constants

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

# helpers

def eval_model(model, testloader, criterion):
    model.eval()
    total_loss = 0.
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs = rearrange(inputs, 'b c h w -> b h w c')
            batch = targets.shape[0]
            logits = model(inputs)
            loss = criterion(logits, targets)

            total_loss += loss.item() * batch
            correct += (logits.argmax(dim = -1) == targets).sum().item()
            total += batch

    model.train()
    return total_loss / total, correct / total

# train

def train(
    # training
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    eval_every: int = 100,
    # perceiver
    depth: int = 3,
    num_latents: int = 128,
    latent_dim: int = 128,
    cross_heads: int = 1,
    latent_heads: int = 4,
    cross_dim_head: int = 64,
    latent_dim_head: int = 64,
    attn_dropout: float = 0.1,
    ff_dropout: float = 0.1,
    self_per_cross_attn: int = 2,
    weight_tie_layers: bool = False,
    inverted_cross_attn = False
):
    accelerator = Accelerator(log_with = 'wandb')
    accelerator.print(f'device: {accelerator.device}')

    accelerator.init_trackers(
        project_name = 'perceiver-cifar10',
        config = dict(
            epochs = epochs,
            batch_size = batch_size,
            lr = lr,
            weight_decay = weight_decay,
            depth = depth,
            num_latents = num_latents,
            latent_dim = latent_dim,
            cross_heads = cross_heads,
            latent_heads = latent_heads,
            cross_dim_head = cross_dim_head,
            latent_dim_head = latent_dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            self_per_cross_attn = self_per_cross_attn,
            weight_tie_layers = weight_tie_layers,
            inverted_cross_attn = inverted_cross_attn
        )
    )

    # data

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    os.makedirs('./data', exist_ok = True)

    trainset = datasets.CIFAR10(root = './data', train = True, download = True, transform = transform_train)
    testset  = datasets.CIFAR10(root = './data', train = False, download = True, transform = transform_test)

    trainloader = DataLoader(trainset, batch_size = batch_size, shuffle = True, num_workers = 4, pin_memory = True)
    testloader  = DataLoader(testset, batch_size = batch_size, shuffle = False, num_workers = 4, pin_memory = True)

    # model

    model = Perceiver(
        input_channels = 3,
        input_axis = 2,
        num_freq_bands = 6,
        max_freq = 10.,
        depth = depth,
        num_latents = num_latents,
        latent_dim = latent_dim,
        cross_heads = cross_heads,
        latent_heads = latent_heads,
        cross_dim_head = cross_dim_head,
        latent_dim_head = latent_dim_head,
        num_classes = 10,
        attn_dropout = attn_dropout,
        ff_dropout = ff_dropout,
        weight_tie_layers = weight_tie_layers,
        fourier_encode_data = True,
        self_per_cross_attn = self_per_cross_attn,
        inverted_cross_attn = inverted_cross_attn
    )

    # optimizer

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr = lr, weight_decay = weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr = lr, steps_per_epoch = len(trainloader), epochs = epochs)

    # prepare

    model, optimizer, trainloader, testloader, scheduler = accelerator.prepare(
        model, optimizer, trainloader, testloader, scheduler
    )

    # train

    step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.
        epoch_correct = 0
        epoch_total = 0

        for inputs, targets in trainloader:
            inputs = rearrange(inputs, 'b c h w -> b h w c')

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), max_norm = 1.0)

            optimizer.step()
            scheduler.step()

            batch_loss = loss.item()
            preds = logits.argmax(dim = -1)
            batch_acc = (preds == targets).float().mean().item()
            batch = targets.shape[0]

            epoch_loss += batch_loss * batch
            epoch_correct += (preds == targets).sum().item()
            epoch_total += batch

            step += 1

            accelerator.log({
                'train/loss': batch_loss,
                'train/acc': batch_acc,
                'lr': scheduler.get_last_lr()[0],
            }, step = step)

            # periodic eval

            if step % eval_every == 0:
                val_loss, val_acc = eval_model(model, testloader, criterion)

                accelerator.log({
                    'val/loss': val_loss,
                    'val/acc': val_acc,
                }, step = step)

                accelerator.print(f'step {step} (epoch {epoch}) | val loss: {val_loss:.4f}, val acc: {val_acc:.4f}')

        # epoch summary

        epoch_loss /= epoch_total
        epoch_acc = epoch_correct / epoch_total

        accelerator.log({
            'train/epoch_loss': epoch_loss,
            'train/epoch_acc': epoch_acc,
        }, step = step)

        accelerator.print(f'epoch {epoch}/{epochs} | train loss: {epoch_loss:.4f}, train acc: {epoch_acc:.4f}')

    accelerator.end_training()

if __name__ == '__main__':
    fire.Fire(train)
