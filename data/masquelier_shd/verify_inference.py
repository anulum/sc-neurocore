#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Masquelier SHD inference verifier
"""Verify SHD inference accuracy for Masquelier/Queant models.

Standalone script — no SC-NeuroCore dependency, just torch + spikingjelly + DCLS.
Run from neuromorphic_training-main/ directory.
"""

import sys
import os
import json
import torch

os.environ["WANDB_MODE"] = "disabled"

# Add model source
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "neuromorphic_training-main"))

from configs.config_SHD import Config
from src.SHD.snn import (
    SNN,
    SNN_synaptic_feedforward_delays,
    SNN_fixed_synaptic_feedforward_delays,
    SNN_axonal_feedforward_delays,
    SNN_fixed_axonal_feedforward_delays,
)
from src.datasets import load_dataset
from src.utils import seed_everything
from spikingjelly.activation_based import functional

MODEL_CLASSES = {
    "SNN": SNN,
    "SNN_synaptic_feedforward_delays": SNN_synaptic_feedforward_delays,
    "SNN_fixed_synaptic_feedforward_delays": SNN_fixed_synaptic_feedforward_delays,
    "SNN_axonal_feedforward_delays": SNN_axonal_feedforward_delays,
    "SNN_fixed_axonal_feedforward_delays": SNN_fixed_axonal_feedforward_delays,
}


def evaluate(model, loader, device, config):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in loader:
            x, label, *_ = batch
            if isinstance(x, torch.Tensor):
                x = x.to(device).float().permute(1, 0, 2)  # (B, T, N) -> (T, B, N)
            else:
                x = x.to(device)
            label = label.to(device)

            functional.reset_net(model)
            out = model(x)  # (T, B, 20)

            # Classification: sum membrane potential over time
            logits = out.sum(0)  # (B, 20)
            loss = criterion(logits, label)
            pred = logits.argmax(dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)
            total_loss += loss.item() * label.size(0)

    acc = 100.0 * correct / total
    avg_loss = total_loss / total
    return acc, avg_loss


if __name__ == "__main__":
    config = Config()
    seed_everything(config.seed, is_cuda=torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load dataset
    print("Loading SHD dataset...")
    train_loader, valid_loader, test_loader = load_dataset(config)
    print(
        f"  Train: {len(train_loader.dataset)}, Valid: {len(valid_loader.dataset)}, Test: {len(test_loader.dataset)}"
    )

    # Evaluate all models
    base = os.path.join(os.path.dirname(__file__), "neuromorphic_training-main", "exp", "SHD")
    results = []

    for arch_name in sorted(os.listdir(base)):
        arch_path = os.path.join(base, arch_name)
        if not os.path.isdir(arch_path):
            continue
        model_cls = MODEL_CLASSES.get(arch_name)
        if model_cls is None:
            continue

        for variant in sorted(os.listdir(arch_path)):
            ckpt_path = os.path.join(arch_path, variant, "best.pth")
            if not os.path.isfile(ckpt_path):
                continue

            # Adjust hidden_layers based on variant name
            # layer_64/layer_128 = single hidden layer; quantized_sparsity = two hidden [128,128]
            if "layer_64" in variant:
                config.hidden_layers = [64]
            elif "layer_128" in variant:
                config.hidden_layers = [128]
            else:
                config.hidden_layers = [128, 128]

            print(f"\n=== {arch_name}/{variant} (hidden={config.hidden_layers}) ===")
            model = model_cls(config).to(device)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            missing, unexpected = model.load_state_dict(ckpt["net"], strict=False)
            if missing:
                print(f"  WARNING: missing keys: {missing}")
            if unexpected:
                print(f"  WARNING: unexpected keys: {unexpected}")
            saved_acc = ckpt.get("acc", 0)
            saved_epoch = ckpt.get("epoch", -1)

            # Validation set (same split as training)
            val_acc, val_loss = evaluate(model, valid_loader, device, config)
            # Test set
            test_acc, test_loss = evaluate(model, test_loader, device, config)

            print(f"  Saved: {saved_acc:.2f}% (epoch {saved_epoch})")
            print(f"  Valid: {val_acc:.2f}% (loss {val_loss:.4f})")
            print(f"  Test:  {test_acc:.2f}% (loss {test_loss:.4f})")

            results.append(
                {
                    "architecture": arch_name,
                    "variant": variant,
                    "saved_acc": saved_acc,
                    "saved_epoch": saved_epoch,
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "test_acc": test_acc,
                    "test_loss": test_loss,
                }
            )

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), "inference_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to {out_path}")

    # Summary table
    print(f"\n{'Architecture':<45} {'Variant':<25} {'Saved':>6} {'Val':>6} {'Test':>6}")
    print("-" * 95)
    for r in sorted(results, key=lambda x: -x["test_acc"]):
        print(
            f"{r['architecture']:<45} {r['variant']:<25} {r['saved_acc']:>5.1f}% {r['val_acc']:>5.1f}% {r['test_acc']:>5.1f}%"
        )
