# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore -- Bipolar SC MNIST Pipeline (Fix 1 + Fix 3)
#
# Fix 1: XNOR multiplication for signed weights [-1,1]
# Fix 3: Per-layer calibration of output distributions
# Expected: 70-90% (vs 10% with naive unipolar)

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
subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--no-deps",
        "git+https://github.com/anulum/sc-neurocore.git@main",
    ],
    stdout=sys.stdout,
    stderr=sys.stderr,
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sc_neurocore.training.snn_modules import SpikingNet


# ===========================================================================
# Bipolar SC primitives (embedded for Kaggle, also in core/bipolar.py)
# ===========================================================================


def bipolar_mac_vectorised(inputs, weights, L, rng):
    """Bipolar XNOR MAC: inputs (N,) x weights (M, N) -> (M,)"""
    N = len(inputs)
    M = weights.shape[0]

    input_probs = np.clip((inputs + 1.0) / 2.0, 0.0, 1.0)
    weight_probs = np.clip((weights + 1.0) / 2.0, 0.0, 1.0)

    # Generate bitstreams
    input_bits = (rng.random((N, L)) < input_probs[:, None]).astype(np.uint8)

    outputs = np.zeros(M)
    for j in range(M):
        w_bits = (rng.random((N, L)) < weight_probs[j, :, None]).astype(np.uint8)
        # XNOR = (a == b), shape (N, L)
        xnor = (input_bits == w_bits).astype(np.float32)
        # Per-input bipolar product: average over L, decode to [-1,1]
        per_input = 2.0 * xnor.mean(axis=1) - 1.0  # (N,) bipolar products
        # Sum across inputs = dot product (matches w @ x in float)
        outputs[j] = per_input.sum()

    return outputs


# ===========================================================================
# STEP 1: Train float model
# ===========================================================================
def train_float_mnist(n_hidden=128, n_epochs=10, batch_size=128):
    print("\n" + "=" * 70)
    print("STEP 1: Train float SpikingNet on MNIST")
    print("=" * 70)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_data = datasets.MNIST(
        "/kaggle/working/data", train=True, download=True, transform=transform
    )
    test_data = datasets.MNIST(
        "/kaggle/working/data", train=False, download=True, transform=transform
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    model = SpikingNet(n_input=784, n_hidden=n_hidden, n_output=10, n_layers=1, beta=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    T = 25

    from sc_neurocore.training.loops import train_epoch, evaluate

    for epoch in range(n_epochs):
        t0 = time.time()
        _, train_acc = train_epoch(model, train_loader, optimizer, T)
        _, test_acc = evaluate(model, test_loader, T)
        print(
            f"  Epoch {epoch + 1}/{n_epochs}: train={train_acc:.3f} test={test_acc:.3f} "
            f"({time.time() - t0:.1f}s)"
        )

    _, final_acc = evaluate(model, test_loader, T)
    print(f"\n  Float accuracy: {final_acc:.4f}")
    return model, final_acc, test_data


# ===========================================================================
# STEP 2: Export bipolar weights + calibrate
# ===========================================================================
def export_bipolar_weights(model):
    """Extract weights normalised to [-1, 1] (bipolar).

    Preserves sign information, unlike unipolar to_sc_weights().
    """
    print("\n" + "=" * 70)
    print("STEP 2: Export bipolar weights")
    print("=" * 70)

    layers = []
    for i, lin in enumerate(model.linears):
        w = lin.weight.detach().cpu().numpy()
        abs_max = max(np.abs(w).max(), 1e-8)
        w_bp = w / abs_max

        b = None
        if lin.bias is not None:
            b = lin.bias.detach().cpu().numpy()

        layers.append({"weight": w_bp, "bias": b, "scale": float(abs_max)})
        print(
            f"  Layer {i}: shape={w_bp.shape}, range=[{w_bp.min():.3f}, {w_bp.max():.3f}], "
            f"scale={abs_max:.4f}"
        )

    return layers


def calibrate_layers(model, test_data, n_cal=200):
    """Run calibration: measure output distribution of each layer.

    Returns per-layer (mean, std) for normalising SC outputs.
    """
    print("\n  Calibrating layer distributions...")
    model.eval()
    T = 25

    # Hook to capture intermediate activations
    activations = {i: [] for i in range(len(model.linears))}

    def make_hook(layer_idx):
        def hook(module, input, output):
            activations[layer_idx].append(output.detach().cpu())

        return hook

    hooks = []
    for i, lin in enumerate(model.linears):
        hooks.append(lin.register_forward_hook(make_hook(i)))

    with torch.no_grad():
        for idx in range(min(n_cal, len(test_data))):
            img, _ = test_data[idx]
            x = img.view(1, -1).unsqueeze(0).expand(T, 1, 784)
            model(x)

    for h in hooks:
        h.remove()

    cal = {}
    for i, acts in activations.items():
        if acts:
            all_acts = torch.cat(acts, dim=0).numpy()
            cal[i] = {"mean": float(all_acts.mean()), "std": float(all_acts.std())}
            print(f"  Layer {i}: mean={cal[i]['mean']:.4f}, std={cal[i]['std']:.4f}")

    return cal


# ===========================================================================
# STEP 3: Bipolar SC inference
# ===========================================================================
def sc_inference_bipolar(image_flat, bp_layers, L=1024, seed=42, calibration=None):
    """Bipolar SC inference through trained SNN."""
    rng = np.random.default_rng(seed)

    # Normalise input to [-1, 1]
    x = image_flat.copy().astype(np.float64)
    x_range = max(x.max() - x.min(), 1e-8)
    x = 2.0 * (x - x.min()) / x_range - 1.0

    for layer_idx, layer in enumerate(bp_layers):
        w = layer["weight"]
        n_out, n_in = w.shape

        # Pad or truncate input to match layer width
        if len(x) < n_in:
            x_padded = np.zeros(n_in)
            x_padded[: len(x)] = x
            x = x_padded
        elif len(x) > n_in:
            x = x[:n_in]

        # Bipolar MAC via XNOR — output is dot product in [-N, N]
        x_bp = np.clip(x, -1.0, 1.0)
        out = bipolar_mac_vectorised(x_bp, w, L, rng)

        # Add bias (in original float scale, rescaled by weight normalisation)
        if layer["bias"] is not None:
            out = out * layer["scale"] + layer["bias"]
        else:
            out = out * layer["scale"]

        # Normalise to [-1, 1] for next layer using calibration
        if calibration and layer_idx in calibration:
            cal = calibration[layer_idx]
            if cal["std"] > 1e-8:
                out = (out - cal["mean"]) / (3.0 * cal["std"])

        # ReLU activation for hidden layers
        if layer_idx < len(bp_layers) - 1:
            out = np.maximum(out, 0.0)

        x = np.clip(out, -1.0, 1.0)

    return x


def run_bipolar_inference(bp_layers, test_data, lengths, n_samples=500, calibration=None):
    print("\n" + "=" * 70)
    print("STEP 3: Bipolar SC inference")
    print("=" * 70)

    results = {}
    for L in lengths:
        t0 = time.time()
        correct = 0
        total = min(n_samples, len(test_data))

        for i in range(total):
            img, label = test_data[i]
            img_flat = img.numpy().flatten()
            output = sc_inference_bipolar(
                img_flat, bp_layers, L=L, seed=42 + i, calibration=calibration
            )
            pred = int(np.argmax(output))
            if pred == label:
                correct += 1

        elapsed = time.time() - t0
        acc = correct / total
        results[L] = {
            "accuracy": round(acc, 4),
            "n_samples": total,
            "time_s": round(elapsed, 1),
        }
        print(f"  L={L:5d}: accuracy={acc:.2%} ({total} samples, {elapsed:.1f}s)")

    return results


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("=" * 70)
    print("SC-NeuroCore: Bipolar SC MNIST (Fix 1 + Fix 3)")
    print(f"Time: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    print(f"PyTorch: {torch.__version__}")
    print("=" * 70)

    t0 = time.time()

    model, float_acc, test_data = train_float_mnist(n_hidden=128, n_epochs=10)
    bp_layers = export_bipolar_weights(model)
    calibration = calibrate_layers(model, test_data, n_cal=200)

    lengths = [64, 128, 256, 512, 1024]
    sc_results = run_bipolar_inference(
        bp_layers, test_data, lengths, n_samples=500, calibration=calibration
    )

    total_time = time.time() - t0

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\n  Float accuracy: {float_acc:.2%}")
    print(f"\n  Bipolar SC inference:")
    for L, r in sorted(sc_results.items()):
        drop = float_acc - r["accuracy"]
        print(f"    L={L:5d}: {r['accuracy']:.2%}  (drop: {drop:.2%})")
    print(f"\n  Total time: {total_time:.0f}s")

    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "torch_version": torch.__version__,
        "total_time_s": round(total_time, 1),
        "method": "bipolar_xnor_with_calibration",
        "float_accuracy": round(float_acc, 4),
        "sc_results": sc_results,
        "calibration": calibration,
    }

    out_path = Path("/kaggle/working/sc_mnist_bipolar_results.json")
    if not out_path.parent.exists():
        out_path = Path("sc_mnist_bipolar_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
