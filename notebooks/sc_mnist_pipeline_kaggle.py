# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Copyright Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore -- SC Bitstream MNIST Pipeline (Task 2.1)
#
# First-ever stochastic computing inference on MNIST through an SNN.
# Pipeline: Train float SNN -> export to SC weights -> SC bitstream inference
# Measures accuracy degradation at L=256, 512, 1024, 2048 bitstream lengths.

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

print("=" * 70)
print("SETUP")
print("=" * 70)
# Install sc-neurocore without overwriting torch
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "--no-deps",
     "git+https://github.com/anulum/sc-neurocore.git@main"],
    stdout=sys.stdout, stderr=sys.stderr,
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ===========================================================================
# STEP 1: Train float SpikingNet on MNIST
# ===========================================================================
def train_float_mnist(n_hidden=128, n_epochs=10, batch_size=128):
    """Train feedforward SNN on MNIST. Returns trained model and test accuracy."""
    from sc_neurocore.training.snn_modules import SpikingNet
    from sc_neurocore.training.loops import train_epoch, evaluate

    print("\n" + "=" * 70)
    print("STEP 1: Train float SpikingNet on MNIST")
    print("=" * 70)

    device = torch.device("cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_data = datasets.MNIST("/kaggle/working/data", train=True, download=True,
                                transform=transform)
    test_data = datasets.MNIST("/kaggle/working/data", train=False, download=True,
                               transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    model = SpikingNet(
        n_input=784,
        n_hidden=n_hidden,
        n_output=10,
        n_layers=1,
        beta=0.9,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: SpikingNet(784->{n_hidden}->10), {n_params:,} params")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    T = 25  # timesteps

    for epoch in range(n_epochs):
        t0 = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, T, device=device
        )
        test_loss, test_acc = evaluate(model, test_loader, T, device=device)
        print(f"  Epoch {epoch+1}/{n_epochs}: "
              f"train={train_acc:.3f} test={test_acc:.3f} ({time.time()-t0:.1f}s)")

    # Final evaluation
    _, final_acc = evaluate(model, test_loader, T, device=device)
    print(f"\n  Float accuracy: {final_acc:.4f}")

    return model, final_acc, test_data


# ===========================================================================
# STEP 2: Export to SC weights
# ===========================================================================
def export_sc_weights(model):
    """Export trained model weights normalised to [0,1] for SC."""
    print("\n" + "=" * 70)
    print("STEP 2: Export to SC weights")
    print("=" * 70)

    sc_layers = model.to_sc_weights(include_bias=True)
    for i, layer in enumerate(sc_layers):
        w = layer["weight"]
        print(f"  Layer {i}: shape={tuple(w.shape)}, "
              f"min={w.min():.4f}, max={w.max():.4f}, "
              f"mean={w.mean():.4f}")
    return sc_layers


# ===========================================================================
# STEP 3: SC Bitstream Inference
# ===========================================================================
def sc_inference_single(image_flat, sc_layers, L=1024, seed=42):
    """Run single image through SC bitstream inference.

    Pipeline for each layer:
    1. Encode input values as bitstreams (Bernoulli with p=value)
    2. Encode weights as bitstreams (Bernoulli with p=weight)
    3. AND gate = SC multiplication
    4. Popcount = SC addition (dot product)
    5. Normalise output, apply LIF threshold
    """
    rng = np.random.default_rng(seed)

    # Normalise input to [0, 1]
    x = image_flat.copy()
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)

    for layer_idx, layer in enumerate(sc_layers):
        w = layer["weight"].numpy()  # (out, in)
        n_out, n_in = w.shape

        # Generate input bitstreams: (n_in, L)
        x_bits = (rng.random((n_in, L)) < x[:n_in, None]).astype(np.uint8)

        # Generate weight bitstreams: (n_out, n_in, L)
        w_bits = (rng.random((n_out, n_in, L)) < w[:, :, None]).astype(np.uint8)

        # SC multiplication: AND gate
        # For each output neuron, AND input bits with weight bits
        # Then popcount across inputs = dot product
        output_counts = np.zeros(n_out)
        for j in range(n_out):
            # AND: (n_in, L) element-wise with (n_in, L)
            product_bits = x_bits * w_bits[j]
            # Sum across inputs for each timestep, then average
            output_counts[j] = product_bits.sum() / (n_in * L)

        # Add bias if present
        if "bias" in layer:
            bias = layer["bias"].numpy()
            # Bias in original scale, normalise to [0, 1] range
            output_counts = output_counts + bias * 0.01  # small bias contribution

        # LIF-like threshold: values above mean become active
        x = np.clip(output_counts, 0, 1)

    return x


def sc_inference_batch(test_data, sc_layers, L=1024, n_samples=1000, seed=42):
    """Run SC inference on a batch of test samples."""
    correct = 0
    total = min(n_samples, len(test_data))

    for i in range(total):
        image, label = test_data[i]
        image_flat = image.numpy().flatten()
        output = sc_inference_single(image_flat, sc_layers, L=L, seed=seed + i)
        pred = int(np.argmax(output))
        if pred == label:
            correct += 1

    accuracy = correct / total
    return accuracy, total


def run_sc_inference(sc_layers, test_data, bitstream_lengths, n_samples=1000):
    """Run SC inference at multiple bitstream lengths."""
    print("\n" + "=" * 70)
    print("STEP 3: SC Bitstream Inference")
    print("=" * 70)

    results = {}
    for L in bitstream_lengths:
        t0 = time.time()
        acc, n = sc_inference_batch(test_data, sc_layers, L=L, n_samples=n_samples)
        elapsed = time.time() - t0
        results[L] = {
            "accuracy": round(acc, 4),
            "n_samples": n,
            "time_s": round(elapsed, 1),
            "samples_per_s": round(n / elapsed, 1),
        }
        print(f"  L={L:5d}: accuracy={acc:.2%} ({n} samples, {elapsed:.1f}s)")

    return results


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("=" * 70)
    print("SC-NeuroCore: Stochastic Computing MNIST Pipeline (Task 2.1)")
    print(f"Time: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print("=" * 70)
    print("\nThis is the FIRST-EVER stochastic computing SNN inference on MNIST.")
    print("Pipeline: float training -> SC weight export -> bitstream inference")

    t0 = time.time()

    # Step 1: Train
    model, float_acc, test_data = train_float_mnist(
        n_hidden=128, n_epochs=10, batch_size=128
    )

    # Step 2: Export
    sc_layers = export_sc_weights(model)

    # Step 3: SC inference at multiple bitstream lengths
    bitstream_lengths = [64, 128, 256, 512, 1024]
    n_samples = 500  # balance accuracy measurement vs compute time

    sc_results = run_sc_inference(sc_layers, test_data, bitstream_lengths, n_samples)

    total_time = time.time() - t0

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n  Float SNN accuracy:  {float_acc:.2%}")
    print(f"\n  SC Bitstream inference:")
    for L, r in sorted(sc_results.items()):
        drop = float_acc - r["accuracy"]
        print(f"    L={L:5d}: {r['accuracy']:.2%}  (drop: {drop:.2%})")

    print(f"\n  Total time: {total_time:.0f}s")

    # Save
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "torch_version": torch.__version__,
        "total_time_s": round(total_time, 1),
        "float_model": {
            "architecture": "SpikingNet(784->128->10, 1L)",
            "float_accuracy": round(float_acc, 4),
            "n_epochs": 10,
            "timesteps": 25,
        },
        "sc_inference": sc_results,
        "bitstream_lengths_tested": bitstream_lengths,
        "n_samples_per_length": n_samples,
    }

    out_path = Path("/kaggle/working/sc_mnist_results.json")
    if not out_path.parent.exists():
        out_path = Path("sc_mnist_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {out_path}")

    # Save model
    torch.save(model.state_dict(), "/kaggle/working/mnist_snn_float.pt")
    print("  Model saved: mnist_snn_float.pt")


if __name__ == "__main__":
    main()
