# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore -- SC-Aware Training + Bipolar Inference (Fix 2 eval)
#
# Train SCAwareLIFNet (with SC noise injection) on MNIST,
# then run bipolar SC inference. Compares with standard SpikingNet.

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
sys.stdout.flush()
try:
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
    print("  sc-neurocore installed")
except Exception as e:
    print(f"  WARNING: sc-neurocore install failed: {e}")
    print("  Continuing with embedded code only")
sys.stdout.flush()

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ===========================================================================
# Bipolar MAC (same as bipolar_kaggle v2)
# ===========================================================================
def bipolar_mac(inputs, weights, L, rng):
    N, M = len(inputs), weights.shape[0]
    input_probs = np.clip((inputs + 1.0) / 2.0, 0.0, 1.0)
    weight_probs = np.clip((weights + 1.0) / 2.0, 0.0, 1.0)
    input_bits = (rng.random((N, L)) < input_probs[:, None]).astype(np.uint8)
    outputs = np.zeros(M)
    for j in range(M):
        w_bits = (rng.random((N, L)) < weight_probs[j, :, None]).astype(np.uint8)
        per_input = 2.0 * (input_bits == w_bits).astype(np.float32).mean(axis=1) - 1.0
        outputs[j] = per_input.sum()
    return outputs


# ===========================================================================
# SC-Aware LIF Net (embedded for Kaggle, mirrors qat/torch_qat.py)
# ===========================================================================
try:
    from sc_neurocore.training.snn_modules import LIFCell, atan_surrogate

    print("  LIFCell imported from sc_neurocore")
except ImportError:
    print("  WARNING: sc_neurocore import failed, using embedded LIFCell")
    # Embedded fallback
    from typing import Callable, Tuple

    def atan_surrogate(x, alpha=2.0):
        return 0.5 + (1.0 / 3.14159) * torch.atan(alpha * x)

    class LIFCell(nn.Module):
        def __init__(self, beta=0.9, surrogate_fn=None, **kw):
            super().__init__()
            self.register_buffer("_beta", torch.tensor(beta))
            self.register_buffer("_th", torch.tensor(1.0))
            self.sfn = surrogate_fn or atan_surrogate

        @property
        def beta(self):
            return self._beta

        @property
        def threshold(self):
            return self._th

        def forward(self, current, v):
            v_next = self.beta * v + current
            spike = self.sfn(v_next - self.threshold)
            v_next = v_next - spike.detach() * self.threshold
            return spike, v_next


sys.stdout.flush()


class SCAwareLinear(nn.Module):
    def __init__(self, in_f, out_f, L=256, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_f, out_f, bias=bias)
        self.L = L
        with torch.no_grad():
            self.linear.weight.clamp_(-1.0, 1.0)

    def forward(self, x):
        w = self.linear.weight.clamp(-1.0, 1.0)
        if self.training:
            p = (w + 1.0) / 2.0
            noise = torch.randn_like(w) * (p * (1.0 - p) / self.L).sqrt()
            w = w + noise
        return nn.functional.linear(x, w, self.linear.bias)


class SCAwareLIFNet(nn.Module):
    def __init__(self, n_in, n_hid, n_out, n_layers=1, L=256, beta=0.9):
        super().__init__()
        self.n_out = n_out
        sizes = [n_in] + [n_hid] * n_layers + [n_out]
        self.linears = nn.ModuleList(
            SCAwareLinear(sizes[i], sizes[i + 1], L=L) for i in range(len(sizes) - 1)
        )
        self.lifs = nn.ModuleList(
            LIFCell(beta=beta, surrogate_fn=atan_surrogate) for _ in range(len(sizes) - 1)
        )

    def forward(self, x):
        T, batch, _ = x.shape
        device = x.device
        v = [torch.zeros(batch, lin.linear.out_features, device=device) for lin in self.linears]
        spike_sum = torch.zeros(batch, self.n_out, device=device)
        mem_sum = torch.zeros(batch, self.n_out, device=device)
        for t in range(T):
            h = x[t]
            for i in range(len(self.linears)):
                h = self.linears[i](h)
                spike, v[i] = self.lifs[i](h, v[i])
                h = spike
            spike_sum += spike
            mem_sum += v[-1]
        return spike_sum, mem_sum


# ===========================================================================
# Training
# ===========================================================================
def train_model(model, train_loader, test_loader, n_epochs, T, tag=""):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        model.train()
        t0 = time.time()
        correct, total = 0, 0
        for data, targets in train_loader:
            data = data.view(data.size(0), -1).unsqueeze(0).expand(T, data.size(0), 784)
            spikes, _ = model(data)
            loss = loss_fn(spikes, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            correct += (spikes.argmax(1) == targets).sum().item()
            total += targets.size(0)

        model.eval()
        t_correct, t_total = 0, 0
        with torch.no_grad():
            for data, targets in test_loader:
                data = data.view(data.size(0), -1).unsqueeze(0).expand(T, data.size(0), 784)
                spikes, _ = model(data)
                t_correct += (spikes.argmax(1) == targets).sum().item()
                t_total += targets.size(0)

        print(
            f"  [{tag}] Epoch {epoch + 1}/{n_epochs}: "
            f"train={correct / total:.3f} test={t_correct / t_total:.3f} "
            f"({time.time() - t0:.1f}s)"
        )

    return t_correct / t_total


def sc_inference(model, test_data, L, n_samples=300):
    """Run bipolar SC inference using model's clamped weights."""
    layers = []
    for lin in model.linears:
        w = lin.linear.weight.detach().clamp(-1.0, 1.0).cpu().numpy()
        abs_max = max(np.abs(w).max(), 1e-8)
        b = lin.linear.bias.detach().cpu().numpy() if lin.linear.bias is not None else None
        layers.append({"weight": w / abs_max, "bias": b, "scale": float(abs_max)})

    # Calibrate
    model.eval()
    T = 25
    cal = {}
    activations = {i: [] for i in range(len(model.linears))}
    hooks = []
    for i, lin in enumerate(model.linears):

        def make_hook(idx):
            def hook(m, inp, out):
                activations[idx].append(out.detach().cpu())

            return hook

        hooks.append(lin.register_forward_hook(make_hook(i)))

    with torch.no_grad():
        for idx in range(min(100, len(test_data))):
            img, _ = test_data[idx]
            x = img.view(1, -1).unsqueeze(0).expand(T, 1, 784)
            model(x)
    for h in hooks:
        h.remove()
    for i, acts in activations.items():
        if acts:
            a = torch.cat(acts).numpy()
            cal[i] = {"mean": float(a.mean()), "std": float(a.std())}

    # SC inference
    correct = 0
    total = min(n_samples, len(test_data))
    rng = np.random.default_rng(42)

    for i in range(total):
        img, label = test_data[i]
        x = img.numpy().flatten().astype(np.float64)
        x = 2.0 * (x - x.min()) / max(x.max() - x.min(), 1e-8) - 1.0

        for li, layer in enumerate(layers):
            w = layer["weight"]
            n_out, n_in = w.shape
            if len(x) < n_in:
                xp = np.zeros(n_in)
                xp[: len(x)] = x
                x = xp
            elif len(x) > n_in:
                x = x[:n_in]

            out = bipolar_mac(np.clip(x, -1, 1), w, L, rng)
            out = out * layer["scale"]
            if layer["bias"] is not None:
                out += layer["bias"]
            if li in cal and cal[li]["std"] > 1e-8:
                out = (out - cal[li]["mean"]) / (3.0 * cal[li]["std"])
            if li < len(layers) - 1:
                out = np.maximum(out, 0.0)
            x = np.clip(out, -1.0, 1.0)

        if int(np.argmax(x)) == label:
            correct += 1

    return correct / total


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("=" * 70)
    print("SC-NeuroCore: SC-Aware Training + Bipolar Inference (Fix 2)")
    print(f"Time: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    print(f"PyTorch: {torch.__version__}")
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
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

    T = 25
    results = {}
    t0 = time.time()

    # 1. Standard SpikingNet (baseline) — use SCAwareLIFNet with large L as proxy
    try:
        from sc_neurocore.training.snn_modules import SpikingNet
    except ImportError:
        SpikingNet = None
    print("\n--- Standard SpikingNet (no SC awareness) ---")
    if SpikingNet is not None:
        std_model = SpikingNet(784, 128, 10, n_layers=1, beta=0.9)
    else:
        # Fallback: SCAwareLIFNet with very large L (effectively no noise)
        std_model = SCAwareLIFNet(784, 128, 10, n_layers=1, L=100000, beta=0.9)
    std_float_acc = train_model(std_model, train_loader, test_loader, 10, T, "Standard")
    print(f"  Float accuracy: {std_float_acc:.4f}")

    # 2. SC-Aware with L=256
    print("\n--- SCAwareLIFNet (L=256) ---")
    sc256 = SCAwareLIFNet(784, 128, 10, n_layers=1, L=256, beta=0.9)
    sc256_float = train_model(sc256, train_loader, test_loader, 10, T, "SC-L256")
    print(f"  Float accuracy: {sc256_float:.4f}")

    # 3. SC-Aware with L=1024
    print("\n--- SCAwareLIFNet (L=1024) ---")
    sc1024 = SCAwareLIFNet(784, 128, 10, n_layers=1, L=1024, beta=0.9)
    sc1024_float = train_model(sc1024, train_loader, test_loader, 10, T, "SC-L1024")
    print(f"  Float accuracy: {sc1024_float:.4f}")

    # SC inference on all three
    print("\n--- Bipolar SC Inference (L=256, 512, 1024) ---")
    for name, model in [("standard", std_model), ("sc_L256", sc256), ("sc_L1024", sc1024)]:
        model_results = {}
        for L in [256, 512, 1024]:
            t1 = time.time()
            acc = sc_inference(model, test_data, L=L, n_samples=300)
            elapsed = time.time() - t1
            model_results[L] = {"accuracy": round(acc, 4), "time_s": round(elapsed, 1)}
            print(f"  [{name}] L={L}: SC accuracy={acc:.2%} ({elapsed:.1f}s)")
        results[name] = model_results

    total_time = time.time() - t0

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  Float accuracies:")
    print(f"    Standard:  {std_float_acc:.2%}")
    print(f"    SC-L256:   {sc256_float:.2%}")
    print(f"    SC-L1024:  {sc1024_float:.2%}")
    print(f"\n  SC inference at L=1024:")
    for name in results:
        print(f"    {name}: {results[name][1024]['accuracy']:.2%}")
    print(f"\n  Total time: {total_time:.0f}s")

    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "torch_version": torch.__version__,
        "total_time_s": round(total_time, 1),
        "float_accuracies": {
            "standard": round(std_float_acc, 4),
            "sc_L256": round(sc256_float, 4),
            "sc_L1024": round(sc1024_float, 4),
        },
        "sc_inference": results,
    }

    out_path = Path("/kaggle/working/sc_aware_mnist_results.json")
    if not out_path.parent.exists():
        out_path = Path("sc_aware_mnist_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback

        print("\n" + "=" * 70)
        print("FATAL ERROR")
        print("=" * 70)
        traceback.print_exc()
        sys.exit(1)
