#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Train ConvSpikingNet on MNIST — targeting 99%+ accuracy.

Usage:
    pip install sc-neurocore[research] torchvision
    python examples/mnist_conv_train.py --epochs 30 --device cuda

Key techniques for high accuracy:
  - ConvSpikingNet (conv layers, not just dense)
  - Learnable beta + threshold per layer
  - Membrane readout (more gradient signal than spike count)
  - Cosine annealing LR schedule
  - Weight decay regularisation
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sc_neurocore.training import ConvSpikingNet
from sc_neurocore.training.surrogate import fast_sigmoid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--timesteps", type=int, default=25)
    parser.add_argument("--beta", type=float, default=0.95)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-dir", default="./data")
    args = parser.parse_args()

    transform = transforms.Compose(
        [
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_ds = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(args.data_dir, train=False, transform=test_transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=2)

    model = ConvSpikingNet(
        n_output=10,
        beta=args.beta,
        surrogate_fn=fast_sigmoid,
        learn_beta=True,
        learn_threshold=True,
    ).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"ConvSpikingNet | {n_params:,} params | T={args.timesteps} | device={args.device}")
    print("-" * 70)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)

            # Repeat across time: (T, batch, 1, 28, 28)
            x = images.unsqueeze(0).repeat(args.timesteps, 1, 1, 1, 1)

            spike_sum, mem_sum = model(x)

            # Membrane readout (better gradients than spike count)
            loss = criterion(mem_sum, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (mem_sum.argmax(1) == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_acc = 100.0 * correct / total
        train_loss = total_loss / total

        # Evaluate
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(args.device)
                labels = labels.to(args.device)
                x = images.unsqueeze(0).repeat(args.timesteps, 1, 1, 1, 1)
                _, mem_sum = model(x)
                test_correct += (mem_sum.argmax(1) == labels).sum().item()
                test_total += labels.size(0)

        test_acc = 100.0 * test_correct / test_total
        elapsed = time.perf_counter() - t0

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "conv_spiking_net_best.pt")

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"loss={train_loss:.4f} | train={train_acc:.2f}% | "
            f"test={test_acc:.2f}% | best={best_acc:.2f}% | "
            f"lr={lr:.6f} | {elapsed:.1f}s"
        )

    print(f"\nBest test accuracy: {best_acc:.2f}%")
    print("Model saved to conv_spiking_net_best.pt")


if __name__ == "__main__":
    main()
