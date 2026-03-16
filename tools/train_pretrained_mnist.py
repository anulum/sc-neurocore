# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Train ConvSpikingNet on MNIST and save pretrained weights

"""Train ConvSpikingNet on MNIST and save pretrained weights.

Usage:
    python tools/train_pretrained_mnist.py [--epochs 5] [--output weights/conv_spiking_net_mnist.pt]
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sc_neurocore.training.snn_modules import ConvSpikingNet
from sc_neurocore.training.losses import spike_count_loss


def train_conv_epoch(model, loader, optimizer, T, device):
    model.train()
    total_loss = correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        x = images.unsqueeze(0).expand(T, -1, -1, -1, -1)
        spike_counts, _ = model(x)
        loss = spike_count_loss(spike_counts, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.shape[0]
        correct += (spike_counts.argmax(1) == labels).sum().item()
        total += labels.shape[0]
    return total_loss / total, correct / total


@torch.no_grad()
def eval_conv(model, loader, T, device):
    model.eval()
    total_loss = correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        x = images.unsqueeze(0).expand(T, -1, -1, -1, -1)
        spike_counts, _ = model(x)
        loss = spike_count_loss(spike_counts, labels)
        total_loss += loss.item() * labels.shape[0]
        correct += (spike_counts.argmax(1) == labels).sum().item()
        total += labels.shape[0]
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--timesteps", type=int, default=25)
    parser.add_argument("--output", default="weights/conv_spiking_net_mnist.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=0)

    model = ConvSpikingNet(n_output=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ConvSpikingNet: {n_params:,} params | device={device} | T={args.timesteps}")

    t0 = time.time()
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_conv_epoch(
            model, train_loader, optimizer, args.timesteps, device
        )
        test_loss, test_acc = eval_conv(model, test_loader, args.timesteps, device)
        best_acc = max(best_acc, test_acc)
        print(
            f"Epoch {epoch:2d} | train {train_acc:.1%} loss={train_loss:.4f} | test {test_acc:.1%} loss={test_loss:.4f}"
        )
    elapsed = time.time() - t0

    sc_weights = model.to_sc_weights()
    print(f"\nSC export: {len(sc_weights)} weight matrices normalized to [0,1]")
    print(f"Training time: {elapsed:.1f}s | Best test accuracy: {best_acc:.1%}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "sc_weights": [w.cpu().numpy() for w in sc_weights],
            "test_accuracy": float(test_acc),
            "best_accuracy": float(best_acc),
            "n_params": n_params,
            "architecture": "ConvSpikingNet(n_output=10)",
        },
        out,
    )

    meta = {
        "model_name": "ConvSpikingNet",
        "dataset": "MNIST",
        "test_accuracy": round(float(test_acc), 4),
        "best_accuracy": round(float(best_acc), 4),
        "n_parameters": n_params,
        "architecture": "Conv2d(1->32,5)->LIF->Pool->Conv2d(32->64,5)->LIF->Pool->FC(1024->128)->LIF->FC(128->10)->LIF",
        "training_date": time.strftime("%Y-%m-%d"),
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "timesteps": args.timesteps,
            "optimizer": "Adam",
            "surrogate": "atan",
        },
        "device": device,
        "training_time_s": round(elapsed, 1),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
    }
    meta_path = out.with_suffix(".metadata.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Saved: {out} + {meta_path}")


if __name__ == "__main__":
    main()
