#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Head-to-head SNN surrogate gradient training benchmark.

Compares SC-NeuroCore vs Norse vs snnTorch on MNIST with identical
architectures: 784 -> 128 -> 128 -> 10, T=25, beta=0.9, Adam lr=2e-3.

Usage:
    python benchmarks/surrogate_training_bench.py [--epochs 10] [--json results.json]
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ── Architecture constants ──────────────────────────────────────────────────
N_INPUT = 784
N_HIDDEN = 128
N_OUTPUT = 10
N_LAYERS = 2
BETA = 0.9
BATCH_SIZE = 128
LR = 2e-3
T = 25


@dataclass
class EpochResult:
    epoch: int
    train_loss: float
    train_acc: float
    test_loss: float
    test_acc: float
    wall_s: float


@dataclass
class BenchResult:
    framework: str
    version: str
    device: str
    n_params: int
    epochs: List[EpochResult] = field(default_factory=list)
    total_train_s: float = 0.0
    final_test_acc: float = 0.0
    avg_epoch_s: float = 0.0
    error: str = ""


def get_loaders(data_dir: str = "./data"):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test = datasets.MNIST(data_dir, train=False, transform=transform)
    return (
        DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True),
        DataLoader(test, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True),
    )


# ── SC-NeuroCore ────────────────────────────────────────────────────────────


def bench_scneurocore(train_loader, test_loader, n_epochs: int, device: str) -> BenchResult:
    from sc_neurocore.training import SpikingNet, train_epoch, evaluate
    import sc_neurocore

    model = SpikingNet(N_INPUT, N_HIDDEN, N_OUTPUT, N_LAYERS, beta=BETA).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n_params = sum(p.numel() for p in model.parameters())

    result = BenchResult(
        framework="sc-neurocore",
        version=sc_neurocore.__version__,
        device=device,
        n_params=n_params,
    )

    t_total = time.perf_counter()
    for ep in range(1, n_epochs + 1):
        t0 = time.perf_counter()
        tr_loss, tr_acc = train_epoch(model, train_loader, opt, T, device=device)
        te_loss, te_acc = evaluate(model, test_loader, T, device=device)
        dt = time.perf_counter() - t0
        result.epochs.append(EpochResult(ep, tr_loss, tr_acc, te_loss, te_acc, dt))
        print(f"  [sc-neurocore] Epoch {ep}/{n_epochs} | test {te_acc:.1%} | {dt:.1f}s")

    result.total_train_s = time.perf_counter() - t_total
    result.final_test_acc = result.epochs[-1].test_acc
    result.avg_epoch_s = result.total_train_s / n_epochs
    return result


# ── SC-NeuroCore (learnable) ──────────────────────────────────────────────


def bench_scneurocore_learnable(
    train_loader, test_loader, n_epochs: int, device: str
) -> BenchResult:
    from sc_neurocore.training import SpikingNet, train_epoch, evaluate
    import sc_neurocore

    model = SpikingNet(
        N_INPUT,
        N_HIDDEN,
        N_OUTPUT,
        N_LAYERS,
        beta=BETA,
        learn_beta=True,
        learn_threshold=True,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n_params = sum(p.numel() for p in model.parameters())

    result = BenchResult(
        framework="sc-neurocore-learnable",
        version=sc_neurocore.__version__,
        device=device,
        n_params=n_params,
    )

    t_total = time.perf_counter()
    for ep in range(1, n_epochs + 1):
        t0 = time.perf_counter()
        tr_loss, tr_acc = train_epoch(model, train_loader, opt, T, device=device, max_grad_norm=1.0)
        te_loss, te_acc = evaluate(model, test_loader, T, device=device)
        dt = time.perf_counter() - t0
        result.epochs.append(EpochResult(ep, tr_loss, tr_acc, te_loss, te_acc, dt))
        print(f"  [sc-learnable] Epoch {ep}/{n_epochs} | test {te_acc:.1%} | {dt:.1f}s")

    result.total_train_s = time.perf_counter() - t_total
    result.final_test_acc = result.epochs[-1].test_acc
    result.avg_epoch_s = result.total_train_s / n_epochs
    return result


# ── SC-NeuroCore (ConvSNN) ───────────────────────────────────────────────


def bench_scneurocore_conv(train_loader, test_loader, n_epochs: int, device: str) -> BenchResult:
    from sc_neurocore.training import ConvSpikingNet
    from sc_neurocore.training.losses import spike_count_loss
    import sc_neurocore

    model = ConvSpikingNet(
        n_output=N_OUTPUT,
        beta=BETA,
        learn_beta=True,
        learn_threshold=True,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n_params = sum(p.numel() for p in model.parameters())
    criterion = spike_count_loss

    result = BenchResult(
        framework="sc-neurocore-conv",
        version=sc_neurocore.__version__,
        device=device,
        n_params=n_params,
    )

    t_total = time.perf_counter()
    for ep in range(1, n_epochs + 1):
        t0 = time.perf_counter()
        model.train()
        correct = total = 0
        total_loss = 0.0
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            # (batch, 1, 28, 28) -> (T, batch, 1, 28, 28)
            data = data.unsqueeze(0).expand(T, -1, -1, -1, -1)
            spk, _ = model(data)
            loss = criterion(spk, targets)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * targets.shape[0]
            correct += (spk.argmax(1) == targets).sum().item()
            total += targets.shape[0]
        tr_loss, tr_acc = total_loss / total, correct / total

        model.eval()
        correct = total = 0
        total_loss = 0.0
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                data = data.unsqueeze(0).expand(T, -1, -1, -1, -1)
                spk, _ = model(data)
                loss = criterion(spk, targets)
                total_loss += loss.item() * targets.shape[0]
                correct += (spk.argmax(1) == targets).sum().item()
                total += targets.shape[0]
        te_loss, te_acc = total_loss / total, correct / total

        dt = time.perf_counter() - t0
        result.epochs.append(EpochResult(ep, tr_loss, tr_acc, te_loss, te_acc, dt))
        print(f"  [sc-conv] Epoch {ep}/{n_epochs} | test {te_acc:.1%} | {dt:.1f}s")

    result.total_train_s = time.perf_counter() - t_total
    result.final_test_acc = result.epochs[-1].test_acc
    result.avg_epoch_s = result.total_train_s / n_epochs
    return result


# ── Norse ───────────────────────────────────────────────────────────────────


def bench_norse(train_loader, test_loader, n_epochs: int, device: str) -> BenchResult:
    try:
        import norse
        import norse.torch as ntorch
    except ImportError:
        return BenchResult("norse", "not installed", device, 0, error="ImportError")

    class NorseNet(nn.Module):
        def __init__(self):
            super().__init__()
            p = ntorch.LIFParameters(tau_mem_inv=1.0 / (1.0 - BETA))
            self.fc1 = nn.Linear(N_INPUT, N_HIDDEN)
            self.lif1 = ntorch.LIFCell(p)
            self.fc2 = nn.Linear(N_HIDDEN, N_HIDDEN)
            self.lif2 = ntorch.LIFCell(p)
            self.fc3 = nn.Linear(N_HIDDEN, N_OUTPUT)
            self.lif3 = ntorch.LIFCell(p)

        def forward(self, x):
            T_steps, batch, _ = x.shape
            s1 = s2 = s3 = None
            spike_sum = torch.zeros(batch, N_OUTPUT, device=x.device)
            for t in range(T_steps):
                z1, s1 = self.lif1(self.fc1(x[t]), s1)
                z2, s2 = self.lif2(self.fc2(z1), s2)
                z3, s3 = self.lif3(self.fc3(z2), s3)
                spike_sum = spike_sum + z3
            return spike_sum

    model = NorseNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n_params = sum(p.numel() for p in model.parameters())
    criterion = nn.CrossEntropyLoss()

    result = BenchResult("norse", norse.__version__, device, n_params)

    t_total = time.perf_counter()
    for ep in range(1, n_epochs + 1):
        t0 = time.perf_counter()
        model.train()
        correct = total = 0
        total_loss = 0.0
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            data = data.view(data.shape[0], -1).unsqueeze(0).expand(T, -1, -1)
            out = model(data)
            loss = criterion(out, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * targets.shape[0]
            correct += (out.argmax(1) == targets).sum().item()
            total += targets.shape[0]
        tr_loss, tr_acc = total_loss / total, correct / total

        model.eval()
        correct = total = 0
        total_loss = 0.0
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                data = data.view(data.shape[0], -1).unsqueeze(0).expand(T, -1, -1)
                out = model(data)
                loss = criterion(out, targets)
                total_loss += loss.item() * targets.shape[0]
                correct += (out.argmax(1) == targets).sum().item()
                total += targets.shape[0]
        te_loss, te_acc = total_loss / total, correct / total

        dt = time.perf_counter() - t0
        result.epochs.append(EpochResult(ep, tr_loss, tr_acc, te_loss, te_acc, dt))
        print(f"  [norse] Epoch {ep}/{n_epochs} | test {te_acc:.1%} | {dt:.1f}s")

    result.total_train_s = time.perf_counter() - t_total
    result.final_test_acc = result.epochs[-1].test_acc
    result.avg_epoch_s = result.total_train_s / n_epochs
    return result


# ── snnTorch ────────────────────────────────────────────────────────────────


def bench_snntorch(train_loader, test_loader, n_epochs: int, device: str) -> BenchResult:
    try:
        import snntorch
    except ImportError:
        return BenchResult("snntorch", "not installed", device, 0, error="ImportError")

    class SnnTorchNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(N_INPUT, N_HIDDEN)
            self.lif1 = snntorch.Leaky(beta=BETA)
            self.fc2 = nn.Linear(N_HIDDEN, N_HIDDEN)
            self.lif2 = snntorch.Leaky(beta=BETA)
            self.fc3 = nn.Linear(N_HIDDEN, N_OUTPUT)
            self.lif3 = snntorch.Leaky(beta=BETA)

        def forward(self, x):
            T_steps, batch, _ = x.shape
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
            mem3 = self.lif3.init_leaky()
            spike_sum = torch.zeros(batch, N_OUTPUT, device=x.device)
            for t in range(T_steps):
                cur1 = self.fc1(x[t])
                spk1, mem1 = self.lif1(cur1, mem1)
                cur2 = self.fc2(spk1)
                spk2, mem2 = self.lif2(cur2, mem2)
                cur3 = self.fc3(spk2)
                spk3, mem3 = self.lif3(cur3, mem3)
                spike_sum = spike_sum + spk3
            return spike_sum

    model = SnnTorchNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n_params = sum(p.numel() for p in model.parameters())
    criterion = nn.CrossEntropyLoss()

    result = BenchResult("snntorch", snntorch.__version__, device, n_params)

    t_total = time.perf_counter()
    for ep in range(1, n_epochs + 1):
        t0 = time.perf_counter()
        model.train()
        correct = total = 0
        total_loss = 0.0
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            data = data.view(data.shape[0], -1).unsqueeze(0).expand(T, -1, -1)
            out = model(data)
            loss = criterion(out, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * targets.shape[0]
            correct += (out.argmax(1) == targets).sum().item()
            total += targets.shape[0]
        tr_loss, tr_acc = total_loss / total, correct / total

        model.eval()
        correct = total = 0
        total_loss = 0.0
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                data = data.view(data.shape[0], -1).unsqueeze(0).expand(T, -1, -1)
                out = model(data)
                loss = criterion(out, targets)
                total_loss += loss.item() * targets.shape[0]
                correct += (out.argmax(1) == targets).sum().item()
                total += targets.shape[0]
        te_loss, te_acc = total_loss / total, correct / total

        dt = time.perf_counter() - t0
        result.epochs.append(EpochResult(ep, tr_loss, tr_acc, te_loss, te_acc, dt))
        print(f"  [snntorch] Epoch {ep}/{n_epochs} | test {te_acc:.1%} | {dt:.1f}s")

    result.total_train_s = time.perf_counter() - t_total
    result.final_test_acc = result.epochs[-1].test_acc
    result.avg_epoch_s = result.total_train_s / n_epochs
    return result


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--json", default=None)
    parser.add_argument(
        "--frameworks",
        nargs="+",
        default=[
            "sc-neurocore",
            "sc-neurocore-learnable",
            "sc-neurocore-conv",
            "norse",
            "snntorch",
        ],
    )
    args = parser.parse_args()

    print("=" * 60)
    print(" SNN Surrogate Gradient Training Benchmark")
    print(f" MNIST | 784→128→128→10 | T={T} | beta={BETA}")
    print(f" {args.epochs} epochs | batch={BATCH_SIZE} | Adam lr={LR}")
    print(f" Device: {args.device}")
    print(f" PyTorch: {torch.__version__}")
    print(f" Platform: {platform.platform()}")
    print("=" * 60)

    train_loader, test_loader = get_loaders(args.data_dir)
    print(f"MNIST loaded: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test\n")

    dispatch = {
        "sc-neurocore": bench_scneurocore,
        "sc-neurocore-learnable": bench_scneurocore_learnable,
        "sc-neurocore-conv": bench_scneurocore_conv,
        "norse": bench_norse,
        "snntorch": bench_snntorch,
    }

    results: Dict[str, BenchResult] = {}
    for fw in args.frameworks:
        print(f"--- {fw} ---")
        fn = dispatch.get(fw)
        if fn is None:
            print(f"  Unknown framework: {fw}")
            continue
        results[fw] = fn(train_loader, test_loader, args.epochs, args.device)
        r = results[fw]
        if r.error:
            print(f"  ERROR: {r.error}\n")
        else:
            print(
                f"  Done: {r.final_test_acc:.1%} accuracy, "
                f"{r.total_train_s:.1f}s total, "
                f"{r.avg_epoch_s:.1f}s/epoch, "
                f"{r.n_params} params\n"
            )

    # Summary table
    print("=" * 60)
    print(f" {'Framework':<16} {'Accuracy':>8} {'Total(s)':>9} {'Epoch(s)':>9} {'Params':>8}")
    print("-" * 60)
    for fw, r in results.items():
        if r.error:
            print(f" {fw:<16} {'ERROR':>8} {'-':>9} {'-':>9} {'-':>8}")
        else:
            print(
                f" {fw:<16} {r.final_test_acc:>7.1%} "
                f"{r.total_train_s:>9.1f} {r.avg_epoch_s:>9.1f} {r.n_params:>8}"
            )
    print("=" * 60)

    if args.json:
        out = {
            "meta": {
                "pytorch": torch.__version__,
                "platform": platform.platform(),
                "device": args.device,
                "epochs": args.epochs,
                "architecture": f"{N_INPUT}→{N_HIDDEN}→{N_HIDDEN}→{N_OUTPUT}",
                "timesteps": T,
                "beta": BETA,
                "batch_size": BATCH_SIZE,
                "lr": LR,
            },
            "results": {k: asdict(v) for k, v in results.items()},
        }
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to {args.json}")


if __name__ == "__main__":
    main()
