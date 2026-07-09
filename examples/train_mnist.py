#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

# SC-NeuroCore — Train SNN on MNIST with surrogate gradients
#
# Usage:
#   python examples/train_mnist.py              # feedforward SNN
#   python examples/train_mnist.py --conv       # convolutional SNN
#   python examples/train_mnist.py --epochs 10  # more epochs
#
# Requires: pip install sc-neurocore[research] torchvision

from __future__ import annotations

import argparse
import time

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sc_neurocore.training import (
    ConvSpikingNet,
    SpikingNet,
    auto_device,
    evaluate,
    model_info,
    spike_count_loss,
    train_epoch,
)


def main():
    parser = argparse.ArgumentParser(description="SC-NeuroCore MNIST SNN training")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--timesteps", type=int, default=25)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--conv", action="store_true", help="Use ConvSpikingNet")
    parser.add_argument("--learn-beta", action="store_true")
    parser.add_argument("--learn-threshold", action="store_true")
    parser.add_argument("--data-dir", default="./data")
    args = parser.parse_args()

    device = auto_device()
    print(f"Device: {device}")

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_ds = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(args.data_dir, train=False, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=0)

    if args.conv:
        model = ConvSpikingNet(
            n_output=10,
            beta=args.beta,
            learn_beta=args.learn_beta,
            learn_threshold=args.learn_threshold,
        ).to(device)
        flatten = False
    else:
        model = SpikingNet(
            n_input=784,
            n_hidden=args.hidden,
            n_output=10,
            beta=args.beta,
            learn_beta=args.learn_beta,
            learn_threshold=args.learn_threshold,
        ).to(device)
        flatten = True

    info = model_info(model)
    print(f"Model: {type(model).__name__}")
    print(f"Parameters: {info['trainable_params']:,} trainable")
    print(f"Spiking cells: {info['spiking_cells']} ({', '.join(info['cell_types'])})")
    if info["learnable_dynamics"]:
        print(f"Learnable dynamics: {info['learnable_dynamics']}")
    print(f"Timesteps: {args.timesteps}, Epochs: {args.epochs}")
    print()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            args.timesteps,
            loss_fn=spike_count_loss,
            device=device,
            max_grad_norm=1.0,
            flatten_input=flatten,
        )
        test_loss, test_acc = evaluate(
            model,
            test_loader,
            args.timesteps,
            loss_fn=spike_count_loss,
            device=str(device),
            flatten_input=flatten,
        )
        scheduler.step()
        dt = time.time() - t0
        best_acc = max(best_acc, test_acc)

        print(
            f"Epoch {epoch:2d}/{args.epochs} | "
            f"Train: {train_acc:.2%} loss={train_loss:.4f} | "
            f"Test: {test_acc:.2%} loss={test_loss:.4f} | "
            f"{dt:.1f}s"
        )

    print(f"\nBest test accuracy: {best_acc:.2%}")

    sc_weights = model.to_sc_weights()
    total_params = sum(w["weight"].numel() for w in sc_weights)
    print(f"SC weight export: {len(sc_weights)} layers, {total_params:,} weights in [0,1]")


if __name__ == "__main__":
    main()
