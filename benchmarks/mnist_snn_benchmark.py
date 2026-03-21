# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — MNIST SNN training benchmark

"""Reproducible MNIST benchmark for SC-NeuroCore SNN training.

Architecture: 784 → 128 → 128 → 10 (SpikingNet, 2 hidden layers)
Training: 10 epochs, T=25 timesteps, Adam lr=2e-3, cosine annealing
Target: >97% test accuracy (published: 99.49% with ConvSNN + tuning)

Usage:
    python benchmarks/mnist_snn_benchmark.py
    python benchmarks/mnist_snn_benchmark.py --epochs 20 --timesteps 50
"""

from __future__ import annotations

import argparse
import time

import torch
from torch.utils.data import DataLoader

from sc_neurocore.training.loops import auto_device, evaluate, train_epoch
from sc_neurocore.training.snn_modules import SpikingNet


def main():
    parser = argparse.ArgumentParser(description="SC-NeuroCore MNIST SNN benchmark")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--timesteps", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--learn-beta", action="store_true")
    parser.add_argument("--learn-threshold", action="store_true")
    args = parser.parse_args()

    device = auto_device()
    print(f"Device: {device}")

    # Load MNIST
    try:
        from torchvision import datasets, transforms

        transform = transforms.ToTensor()
        train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
        test_ds = datasets.MNIST("./data", train=False, transform=transform)
    except ImportError:
        print("torchvision required: pip install torchvision")
        return

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    # Model
    model = SpikingNet(
        n_input=784,
        n_hidden=args.hidden,
        n_output=10,
        n_layers=args.layers,
        beta=args.beta,
        learn_beta=args.learn_beta,
        learn_threshold=args.learn_threshold,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.layers} hidden layers × {args.hidden}, {n_params} params")
    print(f"Training: {args.epochs} epochs, T={args.timesteps}, lr={args.lr}")
    print()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    t0 = time.time()

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, args.timesteps, device=device
        )
        test_loss, test_acc = evaluate(model, test_loader, args.timesteps, device=device)
        scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc

        print(
            f"Epoch {epoch + 1:2d}/{args.epochs} | "
            f"Train: {train_acc * 100:.1f}% (loss {train_loss:.4f}) | "
            f"Test: {test_acc * 100:.1f}% (loss {test_loss:.4f}) | "
            f"Best: {best_acc * 100:.1f}%"
        )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. Best test accuracy: {best_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
