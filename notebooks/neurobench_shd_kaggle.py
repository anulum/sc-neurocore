# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Copyright Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore -- NeuroBench SHD benchmark (Task 1.5)
#
# Trains a feedforward SNN on Spiking Heidelberg Digits (SHD) and reports
# NeuroBench-style metrics: accuracy, parameters, synaptic ops, latency.
# Expected accuracy: 75-85% (SOTA around 95% with attention/recurrence).

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Install dependencies
# ---------------------------------------------------------------------------
print("=" * 70)
print("SETUP")
print("=" * 70)
# Install sc-neurocore without deps to avoid overwriting Kaggle's CUDA torch
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "--no-deps",
     "git+https://github.com/anulum/sc-neurocore.git@main"],
    stdout=sys.stdout, stderr=sys.stderr,
)
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "h5py"],
    stdout=sys.stdout, stderr=sys.stderr,
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Download SHD dataset
# ---------------------------------------------------------------------------
SHD_TRAIN_URL = "https://zenkelab.org/datasets/shd_train.h5.gz"
SHD_TEST_URL = "https://zenkelab.org/datasets/shd_test.h5.gz"
DATA_DIR = Path("/kaggle/working/data/shd")


def download_shd():
    """Download and decompress SHD h5 files."""
    import gzip
    import urllib.request

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for url, fname in [(SHD_TRAIN_URL, "shd_train.h5"), (SHD_TEST_URL, "shd_test.h5")]:
        h5_path = DATA_DIR / fname
        if h5_path.exists():
            print(f"  {fname} already exists ({h5_path.stat().st_size / 1e6:.1f} MB)")
            continue
        gz_path = DATA_DIR / f"{fname}.gz"
        print(f"  Downloading {fname}...")
        urllib.request.urlretrieve(url, gz_path)
        print(f"  Decompressing {fname}...")
        with gzip.open(gz_path, "rb") as f_in, open(h5_path, "wb") as f_out:
            while True:
                chunk = f_in.read(1 << 20)
                if not chunk:
                    break
                f_out.write(chunk)
        gz_path.unlink()
        print(f"  {fname}: {h5_path.stat().st_size / 1e6:.1f} MB")


# ---------------------------------------------------------------------------
# Load and preprocess SHD
# ---------------------------------------------------------------------------
N_CHANNELS = 700
N_CLASSES = 20
T_MAX = 100  # bin into 100 timesteps (10ms bins over 1s)


def load_shd_binned(train: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Load SHD and bin spikes into fixed-length tensors (T_MAX, 700)."""
    import h5py

    fname = "shd_train.h5" if train else "shd_test.h5"
    h5_path = DATA_DIR / fname
    print(f"  Loading {fname}...")

    with h5py.File(h5_path, "r") as f:
        spike_times = f["spikes"]["times"]
        spike_units = f["spikes"]["units"]
        labels = f["labels"][:]
        n_samples = len(labels)

        # Pre-allocate
        data = np.zeros((n_samples, T_MAX, N_CHANNELS), dtype=np.float32)

        for i in range(n_samples):
            times = np.asarray(spike_times[i])
            units = np.asarray(spike_units[i])
            if len(times) == 0:
                continue
            # Bin: 10ms bins (times are in seconds, max ~1s)
            bin_idx = np.clip((times * 1000 / 10).astype(int), 0, T_MAX - 1)
            unit_idx = np.clip(units.astype(int), 0, N_CHANNELS - 1)
            data[i, bin_idx, unit_idx] = 1.0

    print(f"  Loaded {n_samples} samples, shape {data.shape}")
    return data, labels.astype(np.int64)


# ---------------------------------------------------------------------------
# SNN Model (uses sc_neurocore SpikingNet)
# ---------------------------------------------------------------------------
def build_model(n_hidden=256, n_layers=2, beta=0.9):
    """Build feedforward SNN for SHD classification."""
    from sc_neurocore.training.snn_modules import SpikingNet
    return SpikingNet(
        n_input=N_CHANNELS,
        n_hidden=n_hidden,
        n_output=N_CLASSES,
        n_layers=n_layers,
        beta=beta,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_and_evaluate(
    n_hidden=256,
    n_layers=2,
    beta=0.9,
    lr=1e-3,
    n_epochs=30,
    batch_size=128,
):
    # Force CPU: Kaggle P100 is sm_60, PyTorch 2.10 requires sm_70+
    device = torch.device("cpu")
    print(f"\n  Device: {device}")

    # Load data
    X_train, y_train = load_shd_binned(train=True)
    X_test, y_test = load_shd_binned(train=False)

    train_ds = TensorDataset(
        torch.tensor(X_train).permute(1, 0, 2),  # (T, N, 700)
        torch.tensor(y_train),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test).permute(1, 0, 2),
        torch.tensor(y_test),
    )

    # DataLoader expects (batch, ...) so we need custom collation
    # Actually SpikingNet.forward expects (T, batch, n_input)
    # So we need to permute after batching
    def collate_fn(batch):
        xs, ys = zip(*batch)
        # Each x is (T, features) -- stack on dim=1 -> (T, batch, features)
        return torch.stack(xs, dim=1), torch.stack(ys)

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        ),
        batch_size=batch_size, shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long),
        ),
        batch_size=batch_size, shuffle=False,
    )

    model = build_model(n_hidden, n_layers, beta).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_hidden}h × {n_layers}L, {n_params:,} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0
    history = []

    for epoch in range(n_epochs):
        t0 = time.time()

        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        total_spikes = 0

        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            # data: (batch, T, 700) -> (T, batch, 700)
            data = data.permute(1, 0, 2)

            spike_counts, mem_acc = model(data)
            loss = loss_fn(spike_counts, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * targets.size(0)
            train_correct += (spike_counts.argmax(1) == targets).sum().item()
            train_total += targets.size(0)
            total_spikes += spike_counts.sum().item()

        scheduler.step()

        # Eval
        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                data = data.permute(1, 0, 2)
                spike_counts, _ = model(data)
                test_correct += (spike_counts.argmax(1) == targets).sum().item()
                test_total += targets.size(0)

        train_acc = train_correct / train_total
        test_acc = test_correct / test_total
        avg_spikes = total_spikes / train_total
        elapsed = time.time() - t0

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "/kaggle/working/shd_best.pt")

        history.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss / train_total, 4),
            "train_acc": round(train_acc, 4),
            "test_acc": round(test_acc, 4),
            "avg_spikes_per_sample": round(avg_spikes, 1),
            "time_s": round(elapsed, 1),
        })

        print(f"  Epoch {epoch+1:2d}/{n_epochs}: "
              f"loss={train_loss/train_total:.4f} "
              f"train={train_acc:.3f} test={test_acc:.3f} "
              f"spikes={avg_spikes:.0f} "
              f"best={best_acc:.3f} "                f"({elapsed:.1f}s)")

    # Final evaluation with timing
    model.load_state_dict(torch.load("/kaggle/working/shd_best.pt", weights_only=True))
    model.eval()

    t_start = time.time()
    test_correct, test_total, total_synaptic_ops = 0, 0, 0
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            data = data.permute(1, 0, 2)
            spike_counts, _ = model(data)
            test_correct += (spike_counts.argmax(1) == targets).sum().item()
            test_total += targets.size(0)
            # Synaptic ops ~ spikes * fan_out per layer
            total_synaptic_ops += spike_counts.sum().item() * n_hidden

    inference_time = time.time() - t_start
    final_acc = test_correct / test_total

    return {
        "task": "SHD",
        "n_classes": N_CLASSES,
        "n_channels": N_CHANNELS,
        "T_bins": T_MAX,
        "model": f"SpikingNet({n_hidden}h, {n_layers}L)",
        "n_parameters": n_params,
        "best_test_accuracy": round(best_acc, 4),
        "final_test_accuracy": round(final_acc, 4),
        "total_epochs": n_epochs,
        "inference_time_s": round(inference_time, 3),
        "inference_samples_per_s": round(test_total / inference_time, 1),
        "avg_synaptic_ops_per_sample": round(total_synaptic_ops / test_total, 0),
        "history": history,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("SC-NeuroCore NeuroBench SHD Benchmark (Task 1.5)")
    print(f"Time: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}"
          f"{' - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")
    print("=" * 70)

    t0 = time.time()

    # Download SHD
    print("\nDownloading SHD dataset...")
    download_shd()

    # Train with default config
    print("\nTraining SpikingNet on SHD...")
    results = train_and_evaluate(
        n_hidden=256,
        n_layers=2,
        beta=0.9,
        lr=1e-3,
        n_epochs=30,
        batch_size=128,
    )

    # Also try 128h for comparison
    print("\nTraining smaller model (128h)...")
    results_small = train_and_evaluate(
        n_hidden=128,
        n_layers=2,
        beta=0.9,
        lr=1e-3,
        n_epochs=20,
        batch_size=128,
    )

    total_time = time.time() - t0

    # Summary
    print("\n" + "=" * 70)
    print("NEUROBENCH SHD RESULTS")
    print("=" * 70)
    for r in [results, results_small]:
        print(f"\n  {r['model']}:")
        print(f"    Parameters: {r['n_parameters']:,}")
        print(f"    Best test accuracy: {r['best_test_accuracy']:.2%}")
        print(f"    Inference: {r['inference_samples_per_s']:.0f} samples/s")
        print(f"    Synaptic ops/sample: {r['avg_synaptic_ops_per_sample']:.0f}")

    print(f"\n  Total time: {total_time:.0f}s")

    # Save
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "torch_version": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "total_time_s": round(total_time, 1),
        "models": {
            "256h_2L": results,
            "128h_2L": results_small,
        },
    }

    out_path = Path("/kaggle/working/neurobench_shd_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
