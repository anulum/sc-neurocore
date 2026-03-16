#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Train an SNN on MNIST with surrogate gradients.

Usage:
    pip install sc-neurocore[research] torchvision
    python examples/mnist_surrogate/train.py [--epochs 10] [--device cuda]

Typical results (CPU, 10 epochs, T=25): ~95% test accuracy.
"""

from __future__ import annotations

import argparse
import time

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sc_neurocore.training import SpikingNet, evaluate, train_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--timesteps", type=int, default=25)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-dir", default="./data")
    args = parser.parse_args()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_ds = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(args.data_dir, train=False, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    model = SpikingNet(
        n_input=784,
        n_hidden=args.hidden,
        n_output=10,
        n_layers=2,
        beta=args.beta,
    ).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"SC-NeuroCore SNN Training | {n_params} params")
    print(f"Device: {args.device} | T={args.timesteps} | beta={args.beta}")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            args.timesteps,
            device=args.device,
        )
        test_loss, test_acc = evaluate(
            model,
            test_loader,
            args.timesteps,
            device=args.device,
        )
        dt = time.perf_counter() - t0
        print(
            f"Epoch {epoch:2d}/{args.epochs} | "
            f"train {train_acc:.1%} loss {train_loss:.4f} | "
            f"test {test_acc:.1%} loss {test_loss:.4f} | "
            f"{dt:.1f}s"
        )

    sc_weights = model.to_sc_weights()
    print(f"\nSC deployment: {len(sc_weights)} weight matrices exported to [0,1] range")
    for i, w in enumerate(sc_weights):
        print(f"  Layer {i}: {tuple(w.shape)}")


if __name__ == "__main__":
    main()
