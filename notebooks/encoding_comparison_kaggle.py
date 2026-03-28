# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Copyright Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore -- 7 Encoding Comparison on MNIST (Task 2.3)
#
# Compare all 7 temporal spike encodings on MNIST classification.
# Each encoding converts pixel values to spike trains, then a
# feedforward SNN classifies the digit.

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
    [sys.executable, "-m", "pip", "install", "-q", "--no-deps",
     "git+https://github.com/anulum/sc-neurocore.git@main"],
    stdout=sys.stdout, stderr=sys.stderr,
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from sc_neurocore.encoding.encoders import (
    rate_encode, latency_encode, phase_encode,
    burst_encode, rank_order_encode,
)
from sc_neurocore.training.snn_modules import SpikingNet


# ===========================================================================
# Encoding functions
# ===========================================================================
ENCODINGS = {
    "rate": lambda vals, T, seed=42: rate_encode(vals, T, seed=seed),
    "latency": lambda vals, T, seed=42: latency_encode(vals, T),
    "phase": lambda vals, T, seed=42: phase_encode(vals, T, n_phases=8),
    "burst": lambda vals, T, seed=42: burst_encode(vals, T, max_burst=5),
    "rank_order": lambda vals, T, seed=42: rank_order_encode(vals, T),
    "direct": lambda vals, T, seed=42: np.tile(vals[None, :], (T, 1)).astype(np.float32),
    "repeat_binary": lambda vals, T, seed=42: np.tile(
        (vals > 0.5).astype(np.int8)[None, :], (T, 1)
    ),
}


def encode_dataset(images, labels, encoding_name, T=25, n_samples=2000, seed=42):
    """Encode MNIST images with a given encoding scheme.

    Returns: (encoded_tensor [n_samples, T, 784], labels_tensor [n_samples])
    """
    enc_fn = ENCODINGS[encoding_name]
    n = min(n_samples, len(labels))
    encoded = np.zeros((n, T, 784), dtype=np.float32)

    for i in range(n):
        img = images[i].numpy().flatten()
        img_norm = np.clip(img, 0, 1)
        spikes = enc_fn(img_norm, T, seed=seed + i)
        # spikes shape: (T, N) or similar
        if spikes.shape[0] == T:
            encoded[i] = spikes[:T, :784].astype(np.float32)
        else:
            encoded[i, :min(spikes.shape[0], T)] = spikes[:T, :784].astype(np.float32)

    return (
        torch.tensor(encoded, dtype=torch.float32),
        torch.tensor(labels[:n].numpy(), dtype=torch.long),
    )


# ===========================================================================
# Training
# ===========================================================================
def train_with_encoding(encoding_name, T=25, n_hidden=128, n_epochs=5,
                        batch_size=128, n_train=5000, n_test=1000):
    """Train SpikingNet with a specific encoding on MNIST."""
    print(f"\n  --- {encoding_name} encoding ---")

    transform = transforms.ToTensor()
    train_data = datasets.MNIST("/kaggle/working/data", train=True, download=True,
                                transform=transform)
    test_data = datasets.MNIST("/kaggle/working/data", train=False, download=True,
                               transform=transform)

    t0 = time.time()

    # Encode
    X_train, y_train = encode_dataset(
        train_data.data.float() / 255.0, train_data.targets,
        encoding_name, T=T, n_samples=n_train, seed=42,
    )
    X_test, y_test = encode_dataset(
        test_data.data.float() / 255.0, test_data.targets,
        encoding_name, T=T, n_samples=n_test, seed=1000,
    )
    encode_time = time.time() - t0

    # Spike count stats
    train_spikes = X_train.sum().item() / n_train
    test_spikes = X_test.sum().item() / n_test

    # DataLoaders
    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False
    )

    model = SpikingNet(n_input=784, n_hidden=n_hidden, n_output=10, n_layers=1, beta=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # Train
    t0 = time.time()
    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            # X_batch: (batch, T, 784) -> (T, batch, 784)
            X_batch = X_batch.permute(1, 0, 2)
            spike_counts, _ = model(X_batch)
            loss = loss_fn(spike_counts, y_batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    train_time = time.time() - t0

    # Evaluate
    model.eval()
    correct, total = 0, 0
    t0 = time.time()
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.permute(1, 0, 2)
            spike_counts, _ = model(X_batch)
            correct += (spike_counts.argmax(1) == y_batch).sum().item()
            total += y_batch.size(0)
    eval_time = time.time() - t0

    accuracy = correct / total

    result = {
        "encoding": encoding_name,
        "test_accuracy": round(accuracy, 4),
        "n_train": n_train,
        "n_test": n_test,
        "T_timesteps": T,
        "avg_spikes_per_train_sample": round(train_spikes, 1),
        "avg_spikes_per_test_sample": round(test_spikes, 1),
        "encode_time_s": round(encode_time, 2),
        "train_time_s": round(train_time, 2),
        "eval_time_s": round(eval_time, 3),
        "n_epochs": n_epochs,
    }

    print(f"    Accuracy: {accuracy:.2%}  |  Spikes/sample: {test_spikes:.0f}  |  "
          f"Encode: {encode_time:.1f}s  Train: {train_time:.1f}s")

    return result


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("=" * 70)
    print("SC-NeuroCore: 7-Encoding Comparison on MNIST (Task 2.3)")
    print(f"Time: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    print(f"PyTorch: {torch.__version__}")
    print("=" * 70)

    t0 = time.time()

    results = []
    for enc_name in ENCODINGS:
        r = train_with_encoding(
            enc_name, T=25, n_hidden=128, n_epochs=5,
            n_train=5000, n_test=1000,
        )
        results.append(r)

    total_time = time.time() - t0

    # Summary
    print("\n" + "=" * 70)
    print("ENCODING COMPARISON RESULTS")
    print("=" * 70)
    print(f"\n  {'Encoding':<16} {'Accuracy':>10} {'Spikes/sample':>15} {'Total time':>12}")
    print("  " + "-" * 55)

    # Sort by accuracy
    results.sort(key=lambda x: x["test_accuracy"], reverse=True)
    for r in results:
        total = r["encode_time_s"] + r["train_time_s"] + r["eval_time_s"]
        print(f"  {r['encoding']:<16} {r['test_accuracy']:>9.2%} "
              f"{r['avg_spikes_per_test_sample']:>14.0f} {total:>11.1f}s")

    print(f"\n  Total time: {total_time:.0f}s")

    # Save
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "torch_version": torch.__version__,
        "total_time_s": round(total_time, 1),
        "model": "SpikingNet(784->128->10, 1L, beta=0.9)",
        "T_timesteps": 25,
        "n_train": 5000,
        "n_test": 1000,
        "n_epochs": 5,
        "results": results,
    }

    out_path = Path("/kaggle/working/encoding_comparison_results.json")
    if not out_path.parent.exists():
        out_path = Path("encoding_comparison_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
