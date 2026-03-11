# SPDX-License-Identifier: AGPL-3.0-or-later
"""Load pretrained ConvSpikingNet and classify MNIST digits.

Demonstrates:
1. Loading pretrained weights from weights/ directory
2. Running inference with surrogate-gradient-trained SNN
3. Extracting SC-normalized weights for bitstream deployment

Requires: torch, torchvision
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import torch
    from torchvision import datasets, transforms
except ImportError:
    raise SystemExit("pip install torch torchvision")

from sc_neurocore.training.snn_modules import ConvSpikingNet


def load_pretrained(weights_dir: Path | None = None):
    if weights_dir is None:
        weights_dir = Path(__file__).resolve().parent.parent / "weights"
    ckpt_path = weights_dir / "conv_spiking_net_mnist.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"{ckpt_path} not found. Run: python tools/train_pretrained_mnist.py"
        )
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = ConvSpikingNet(n_output=10)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def main():
    model, ckpt = load_pretrained()
    print(f"Loaded ConvSpikingNet (test accuracy: {ckpt['best_accuracy']:.1%})")
    print(f"  Parameters: {ckpt['n_params']:,}")
    print(f"  SC weight matrices: {len(ckpt['sc_weights'])}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)

    correct = 0
    n_test = 100
    T = 25
    with torch.no_grad():
        for i in range(n_test):
            img, label = test_ds[i]
            x = img.unsqueeze(0).unsqueeze(0).expand(T, -1, -1, -1, -1)
            spike_counts, _ = model(x)
            pred = spike_counts.argmax(1).item()
            correct += pred == label

    print(f"\nInference on {n_test} test images: {correct}/{n_test} correct ({correct/n_test:.0%})")

    sc_weights = model.to_sc_weights()
    for i, w in enumerate(sc_weights):
        print(f"  SC weight {i}: shape={tuple(w.shape)}, range=[{w.min():.3f}, {w.max():.3f}]")


if __name__ == "__main__":
    main()
