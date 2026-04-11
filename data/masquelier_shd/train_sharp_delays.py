#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Retrain SHD with progressive SIG sharpening for FPGA
"""Retrain best SHD model (axonal delays, QAT sp90) with progressive
Gaussian sharpening so delays converge to hardware-compatible integer
values without accuracy collapse.

Strategy: 3-phase training
  Phase 1 (epochs 0-49):   Fine-tune from checkpoint, SIG=15 (original)
  Phase 2 (epochs 50-99):  Progressive SIG: 15 → 1 (cosine schedule)
  Phase 3 (epochs 100-149): Fine-tune with SIG=0.5 (near-sharp)

The key insight: DCLS Gaussian blur (SIG=15) is integral to accuracy.
Sharp delays (SIG→0) drop test accuracy from 80.4% to 58.6%.
Progressive sharpening lets the network adapt weights to compensate.

Run from neuromorphic_training-main/ directory:
    python3 ../train_sharp_delays.py
"""
import csv
import json
import math
import os
import sys
import time

# Ensure training code is on path regardless of cwd
_script_dir = os.path.dirname(os.path.abspath(__file__))
_training_dir = os.path.join(_script_dir, "neuromorphic_training-main")
if os.path.isdir(_training_dir):
    # Local: script is in masquelier_shd/, code is in neuromorphic_training-main/
    sys.path.insert(0, _training_dir)
    os.chdir(_training_dir)
elif os.path.isfile(os.path.join(_script_dir, "configs", "config_SHD.py")):
    # Cloud: flat structure, configs/src/exp alongside script
    sys.path.insert(0, _script_dir)
    os.chdir(_script_dir)

import numpy as np
import torch

os.environ["WANDB_MODE"] = "disabled"

from configs.config_SHD import Config
from spikingjelly.activation_based import functional
from src.datasets import load_dataset
from src.modules import dcls_module
from src.SHD.snn import SNN_axonal_feedforward_delays
from src.utils import seed_everything


def get_sig_schedule(epoch: int, total_epochs: int = 150) -> float:
    """3-phase SIG schedule."""
    if epoch < 50:
        return 15.0  # Phase 1: original
    elif epoch < 100:
        # Phase 2: cosine anneal from 15 → 1
        progress = (epoch - 50) / 50
        return 1.0 + (15.0 - 1.0) * 0.5 * (1.0 + math.cos(math.pi * progress))
    else:
        # Phase 3: fine-tune at near-sharp
        return 0.5


def set_sig(model: torch.nn.Module, sig: float) -> None:
    """Set SIG value on all DCLS layers."""
    for layer in model.modules():
        if isinstance(layer, dcls_module) and hasattr(layer, "SIG"):
            layer.SIG.data.fill_(sig)


def train_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    config: Config,
) -> tuple[float, float]:
    """Train one epoch. Returns (loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, label, *_ in loader:
        x = x.to(device).float().permute(1, 0, 2)
        label = label.to(device)

        functional.reset_net(model)
        out = model(x)
        logits = out.sum(0)

        loss = criterion(logits, label)

        # Spike penalty
        if config.penalize_spike:
            for layer in model.layers:
                if hasattr(layer, "spikes") and layer.spikes is not None:
                    loss = loss + config.spike_penalty * layer.spikes.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Clamp delays
        model.clamp_delays()

        pred = logits.argmax(1)
        correct += (pred == label).sum().item()
        total += label.size(0)
        total_loss += loss.item() * label.size(0)

    return total_loss / total, 100.0 * correct / total


def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate. Returns (loss, accuracy)."""
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, label, *_ in loader:
            x = x.to(device).float().permute(1, 0, 2)
            label = label.to(device)
            functional.reset_net(model)
            out = model(x)
            logits = out.sum(0)
            loss = criterion(logits, label)
            pred = logits.argmax(1)
            correct += (pred == label).sum().item()
            total += label.size(0)
            total_loss += loss.item() * label.size(0)

    return total_loss / total, 100.0 * correct / total


if __name__ == "__main__":
    config = Config()
    config.hidden_layers = [128, 128]
    seed_everything(config.seed, is_cuda=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Output directory
    out_dir = os.path.join(
        "exp", "SHD", "SNN_axonal_feedforward_delays", "sharp_delays_retrain"
    )
    os.makedirs(out_dir, exist_ok=True)

    # Load dataset
    train_loader, valid_loader, test_loader = load_dataset(config)
    print(f"Train: {len(train_loader.dataset)}, Valid: {len(valid_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # Load pretrained checkpoint
    ckpt_path = "exp/SHD/SNN_axonal_feedforward_delays/quantized_sparsity_90/best.pth"
    model = SNN_axonal_feedforward_delays(config).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["net"])
    print(f"Loaded checkpoint: {ckpt['acc']:.2f}% (epoch {ckpt['epoch']})")

    # Round delays to integers (starting point for sharpening)
    model.round_pos()

    # NOTE: Do NOT call apply_sparsity_mask() — it generates a NEW random mask
    # that destroys the checkpoint's learned sparsity pattern (85% → 2% accuracy).
    # The checkpoint already has 90% sparsity baked in (zero weights stay zero).
    # We preserve sparsity by freezing zero weights via gradient hooks.
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.weight.requires_grad:
            mask = (m.weight.data != 0).float()
            m.weight.register_hook(lambda grad, mask=mask: grad * mask)

    # Optimiser: separate LR for weights and delay positions
    weight_params = []
    delay_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if ".P" in name:
            delay_params.append(p)
        else:
            weight_params.append(p)

    # NOTE: LR 1e-3 destroyed the model from epoch 0 (85% → 4.5%).
    # Fine-tuning a converged checkpoint needs much lower LR.
    # Phase 1: freeze weights, only train delays at 1e-4
    # Phase 2+3: unfreeze weights at 1e-5, delays at 1e-4
    optimizer = torch.optim.AdamW(
        [
            {"params": weight_params, "lr": 1e-5},
            {"params": delay_params, "lr": 1e-4},
        ],
        weight_decay=config.weight_decay,
    )

    # Freeze weights during Phase 1 — only delays adapt to rounding
    for p in weight_params:
        p.requires_grad = False

    criterion = torch.nn.CrossEntropyLoss()

    # Training log
    log_path = os.path.join(out_dir, "training_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "sig", "train_loss", "train_acc", "val_loss", "val_acc", "test_loss", "test_acc", "time_s"])

    best_val_acc = 0.0
    total_epochs = 150

    print(f"\n{'Epoch':>5} {'SIG':>6} {'Train':>8} {'Val':>8} {'Test':>8} {'Time':>6}")
    print("-" * 50)

    for epoch in range(total_epochs):
        sig = get_sig_schedule(epoch, total_epochs)
        set_sig(model, sig)

        # Unfreeze weights at Phase 2 start
        if epoch == 50:
            for p in weight_params:
                p.requires_grad = True
            print("  >>> Phase 2: weights unfrozen, LR=1e-5")

        t0 = time.perf_counter()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, config)
        val_loss, val_acc = evaluate(model, valid_loader, device)
        elapsed = time.perf_counter() - t0

        # Test every 10 epochs or at phase boundaries
        if epoch % 10 == 0 or epoch in [49, 50, 99, 100, 149]:
            test_loss, test_acc = evaluate(model, test_loader, device)
        else:
            test_loss, test_acc = -1, -1

        print(f"{epoch:>5} {sig:>6.1f} {train_acc:>7.1f}% {val_acc:>7.1f}% {test_acc:>7.1f}% {elapsed:>5.0f}s")

        # Save log
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{sig:.2f}", f"{train_loss:.4f}", f"{train_acc:.2f}",
                                     f"{val_loss:.4f}", f"{val_acc:.2f}", f"{test_loss:.4f}", f"{test_acc:.2f}", f"{elapsed:.1f}"])

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"net": model.state_dict(), "acc": val_acc, "epoch": epoch, "sig": sig},
                       os.path.join(out_dir, "best.pth"))

        # Save last
        torch.save({"net": model.state_dict(), "acc": val_acc, "epoch": epoch, "sig": sig},
                   os.path.join(out_dir, "last.pth"))

    # Final evaluation with sharp delays
    print("\n=== Final Evaluation (SIG=0.1, sharp delays) ===")
    set_sig(model, 0.1)
    model.round_pos()
    _, sharp_test = evaluate(model, test_loader, device)
    print(f"Test accuracy with sharp delays: {sharp_test:.1f}%")

    # Load best checkpoint and test sharp
    best_ckpt = torch.load(os.path.join(out_dir, "best.pth"), map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["net"])
    set_sig(model, 0.1)
    model.round_pos()
    _, best_sharp_test = evaluate(model, test_loader, device)
    print(f"Best checkpoint (epoch {best_ckpt['epoch']}, SIG={best_ckpt['sig']:.1f}) with sharp delays: {best_sharp_test:.1f}%")

    # Save config
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({
            "base_checkpoint": ckpt_path,
            "base_accuracy": ckpt["acc"],
            "strategy": "progressive_sig_sharpening",
            "phases": "SIG=15 (0-49), SIG=15→1 cosine (50-99), SIG=0.5 (100-149)",
            "final_sharp_test": sharp_test,
            "best_sharp_test": best_sharp_test,
            "best_val_acc": best_val_acc,
            "total_epochs": total_epochs,
        }, f, indent=2)

    print(f"\nResults saved to {out_dir}/")
    print(f"Original test accuracy (SIG=15): 80.4%")
    print(f"Original sharp test (SIG=0.1): 58.6%")
    print(f"Retrained sharp test (SIG=0.1): {best_sharp_test:.1f}%")
