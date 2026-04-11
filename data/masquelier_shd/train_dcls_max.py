#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Train SHD with DCLS max (Hammouamri 2024) for FPGA integer delays
"""Train SHD model from scratch with DCLS 'max' version (triangular kernel
with cosine-annealed sigma) — Tim Masquelier's recommendation 2026-04-09.

Reference: Hammouamri et al. 2024, https://arxiv.org/abs/2306.00817

The 'max' version uses ((SIG - |X|).relu()).prod() — triangular kernel
with learnable bandwidth SIG. We schedule SIG from 15 → 0 over training:
  - Large SIG: wide tent → many neighbours contribute → smooth gradients
  - SIG → 0: narrow tent → effectively delta function → integer delays

Comparison with prior approaches:
  - vgauss sharpening (FAILED): fine-tuned converged checkpoint, weights couldn't adapt
  - v1 (SUCCESS, 72.5%): pure tent kernel with fixed width=1, no sharpening needed
  - max (THIS): from-scratch training with sigma annealing, expected > v1

Run: python3 train_dcls_max.py
"""

import csv
import json
import math
import os
import sys
import time

_script_dir = os.path.dirname(os.path.abspath(__file__))
_training_dir = os.path.join(_script_dir, "neuromorphic_training-main")
if os.path.isdir(_training_dir):
    sys.path.insert(0, _training_dir)
    os.chdir(_training_dir)
elif os.path.isfile(os.path.join(_script_dir, "configs", "config_SHD.py")):
    sys.path.insert(0, _script_dir)
    os.chdir(_script_dir)

import torch
import wandb

os.environ["WANDB_MODE"] = "disabled"
wandb.init(mode="disabled")

from configs.config_SHD import Config
from src.datasets import load_dataset
from src.modules import dcls_module
from src.SHD.snn import SNN_axonal_feedforward_delays
from src.SHD.trainer import test, init_optim_sche, count_parameters
from src.utils import seed_everything


def magnitude_prune(model: torch.nn.Module, sparsity_list: list):
    """Prune weights based on magnitude instead of random mask.

    Following Han et al. (2016), we select the largest (1-sparsity)
    fraction of weights per layer.
    """
    layers = []
    for m in model.modules():
        if isinstance(m, (torch.nn.Linear, dcls_module)) and m.weight.requires_grad:
            layers.append(m)

    for i, m in enumerate(layers):
        if i >= len(sparsity_list):
            break
        sparsity = sparsity_list[i]
        if sparsity <= 0:
            continue

        w_abs = m.weight.data.abs()
        threshold = torch.quantile(w_abs.view(-1), sparsity)
        mask = (w_abs >= threshold).float().to(m.weight.device)
        m.weight.data *= mask

        # Establish hook to freeze zero weights
        def make_hook(mask):
            return lambda grad: grad * mask

        m.weight.register_hook(make_hook(mask))


def integer_delay_penalty(model: torch.nn.Module) -> torch.Tensor:
    """Sum quadratic penalty (P - round(P))^2 over all DCLS delay parameters.

    Motivated by DelRec (2025) and Khalfaoui-Hassani et al. (ICLR 2024),
    this pushes learnable delays toward integer values to minimize rounding
    error for FPGA deployment.
    """
    penalty = 0.0
    for m in model.modules():
        if isinstance(m, dcls_module) and hasattr(m, "P"):
            penalty += torch.sum((m.P - m.P.round()) ** 2)
    return penalty


SIG_INIT = 15.0  # initial sigma — matches siginit in original config
SIG_FINAL = 0.23  # final sigma — narrow enough to behave as integer delay
# 0.23 is below 0.5 so rounding has at most 0.5/0.23 ratio of neighbour overlap


def get_sigma_schedule(epoch: int, total_epochs: int) -> float:
    """Cosine anneal SIG from SIG_INIT to SIG_FINAL across all epochs."""
    progress = epoch / max(1, total_epochs - 1)
    return SIG_FINAL + (SIG_INIT - SIG_FINAL) * 0.5 * (1.0 + math.cos(math.pi * progress))


def train_with_regulariser(train_loader, model, optimizer, epoch, device, config, lambda_delay=0.0):
    """Custom training loop that adds the integer delay regulariser."""
    from src.SHD.trainer import (
        reset_states,
        calc_loss_SHD,
        calc_metric_SHD,
        get_spike_cost,
        progress_bar,
    )
    import torch.nn.functional as F

    train_loss = 0
    accuracy = 0
    penalize_spikes = getattr(config, "penalize_spike", False)
    model.train()
    for batch_idx, (inputs, targets, _) in enumerate(train_loader):
        targets = F.one_hot(targets, config.output_size).float()
        inputs = inputs.permute(1, 0, 2).float().to(device)
        targets = targets.to(device)
        reset_states(model=model)
        outputs = model(inputs)
        loss = calc_loss_SHD(outputs, targets)
        if penalize_spikes:
            loss += config.spike_penalty * get_spike_cost(model)
        if lambda_delay > 0:
            loss += lambda_delay * integer_delay_penalty(model)
        train_loss += loss.item()
        accuracy += calc_metric_SHD(outputs, targets)
        for opt in optimizer:
            opt.zero_grad()
        loss.backward()
        for opt in optimizer:
            opt.step()
        if hasattr(model, "clamp_delays"):
            model.clamp_delays()
        progress_bar(
            batch_idx,
            len(train_loader),
            "Loss: %.3f | Acc: %.3f%%"
            % (train_loss / (batch_idx + 1), 100 * accuracy / (batch_idx + 1)),
        )
    return 100.0 * accuracy / len(train_loader), train_loss / len(train_loader)


def set_sigma(model: torch.nn.Module, sigma: float) -> None:
    """Set SIG on all DCLS layers (max version requires SIG initialization)."""
    for m in model.modules():
        if isinstance(m, dcls_module) and hasattr(m, "SIG"):
            m.SIG.data.fill_(sigma)


if __name__ == "__main__":
    config = Config()
    config.lambda_delay = 0.01  # Default penalty weight for integer delay regulariser
    config.hidden_layers = [128, 128]

    # === KEY CHANGE: DCLS max (triangular with scheduled SIG) ===
    config.DCLSversion = "max"
    print(f"DCLS version: {config.DCLSversion}")

    seed_everything(config.seed, is_cuda=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    out_dir = os.path.join("exp", "SHD", "SNN_axonal_feedforward_delays", "dcls_max")
    os.makedirs(out_dir, exist_ok=True)

    train_loader, valid_loader, test_loader = load_dataset(config)
    print(
        f"Train: {len(train_loader.dataset)}, Valid: {len(valid_loader.dataset)}, "
        f"Test: {len(test_loader.dataset)}"
    )

    model = SNN_axonal_feedforward_delays(config).to(device)
    magnitude_prune(model, config.weight_sparsity_mask)

    # Initialize SIG on all DCLS layers (snn.py only does this for 'gauss' version)
    set_sigma(model, SIG_INIT)
    for m in model.modules():
        if isinstance(m, dcls_module) and hasattr(m, "SIG"):
            m.SIG.requires_grad = False  # we schedule it manually
            print(
                f"  DCLS layer SIG initialised to {m.SIG.data.mean().item():.2f}, "
                f"requires_grad=False"
            )

    optimizer, scheduler = init_optim_sche(model, config)
    count_parameters(model)

    log_path = os.path.join(out_dir, "training_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "sigma",
                "train_acc",
                "train_loss",
                "val_acc",
                "val_loss",
                "test_acc",
                "test_loss",
                "lr",
                "lr_pos",
                "time_s",
            ]
        )

    best_val_acc = 0.0

    print(
        f"\n{'Epoch':>5} {'SIG':>6} {'Train':>8} {'Val':>8} {'Test':>8} "
        f"{'LR':>10} {'LR_pos':>10} {'Time':>6}"
    )
    print("-" * 75)

    for epoch in range(config.epochs):
        sigma = get_sigma_schedule(epoch, config.epochs)
        set_sigma(model, sigma)

        t0 = time.perf_counter()

        train_acc, train_loss = train_with_regulariser(
            train_loader,
            model,
            optimizer,
            epoch,
            device,
            config,
            lambda_delay=getattr(config, "lambda_delay", 0.01),
        )
        val_acc, val_loss = test(valid_loader, model, epoch, device, config)

        for sc in scheduler:
            sc.step()

        elapsed = time.perf_counter() - t0
        lr = optimizer[0].param_groups[0]["lr"]
        lr_pos = optimizer[1].param_groups[0]["lr"]

        if epoch % 10 == 0 or epoch in [config.epochs - 1]:
            test_acc, test_loss = test(test_loader, model, epoch, device, config)
        else:
            test_acc, test_loss = -1.0, -1.0

        print(
            f"{epoch:>5} {sigma:>6.2f} {train_acc:>7.1f}% {val_acc:>7.1f}% "
            f"{test_acc:>7.1f}% {lr:>10.2e} {lr_pos:>10.2e} {elapsed:>5.0f}s"
        )

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    epoch,
                    f"{sigma:.4f}",
                    f"{train_acc:.2f}",
                    f"{train_loss:.4f}",
                    f"{val_acc:.2f}",
                    f"{val_loss:.4f}",
                    f"{test_acc:.2f}",
                    f"{test_loss:.4f}",
                    f"{lr:.2e}",
                    f"{lr_pos:.2e}",
                    f"{elapsed:.1f}",
                ]
            )

        state = {"net": model.state_dict(), "acc": val_acc, "epoch": epoch, "sigma": sigma}
        torch.save(state, os.path.join(out_dir, "last.pth"))

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(state, os.path.join(out_dir, "best.pth"))

    # === Final evaluation ===
    # We evaluate TWO checkpoints:
    #   1. BEST (highest val_acc) at its NATIVE sigma — accuracy ceiling
    #   2. LAST (epoch 149) at SIG_FINAL — FPGA-deployable model
    # Forcing SIG_FINAL on the best checkpoint is wrong because best may have
    # been trained at a higher sigma (e.g. epoch 82, sigma=6.45) and a much
    # narrower kernel breaks weight optimisation. v9 (job 2055391165397598208)
    # hit this bug — best.pth was at sigma=6.45 and forced eval at 0.23 gave
    # 71.0% instead of the correct LAST-checkpoint result of 75.2%.
    print("\n=== Final Evaluation ===")

    def _eval_checkpoint(label, ckpt_path, force_sigma, do_round):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["net"])
        native_sigma = ckpt.get("sigma", float("nan"))
        epoch = ckpt.get("epoch", -1)
        val = ckpt.get("acc", float("nan"))
        eval_sigma = native_sigma if force_sigma is None else force_sigma
        set_sigma(model, eval_sigma)

        test_before, _ = test(test_loader, model, 0, device, config)
        print(f"\n[{label}] epoch={epoch}, native_sigma={native_sigma:.4f}, val_acc={val:.2f}%")
        print(f"[{label}] Test (sigma={eval_sigma:.4f}, BEFORE round): {test_before:.2f}%")

        for i, m in enumerate(model.modules()):
            if isinstance(m, dcls_module):
                pos = m.P.detach().cpu()
                frac = (pos - pos.round()).abs().mean().item()
                print(f"  DCLS layer {i}: mean |fractional part| = {frac:.4f}")

        test_after = float("nan")
        if do_round:
            model.round_pos()
            test_after, _ = test(test_loader, model, 0, device, config)
            print(f"[{label}] Test (sigma={eval_sigma:.4f}, AFTER round):  {test_after:.2f}%")
            print(f"[{label}] ROUNDING DROP: {test_before - test_after:.2f}%")

        return test_before, test_after, epoch, native_sigma

    best_path = os.path.join(out_dir, "best.pth")
    last_path = os.path.join(out_dir, "last.pth")

    # 1. BEST at NATIVE sigma — accuracy ceiling (NOT FPGA-deployable if sigma > 0.5)
    best_before, _, best_epoch, best_native_sigma = _eval_checkpoint(
        "BEST @ native sigma", best_path, force_sigma=None, do_round=False
    )

    # 2. LAST at SIG_FINAL — FPGA-deployable model
    last_before, last_after, last_epoch, last_native_sigma = _eval_checkpoint(
        "LAST @ SIG_FINAL", last_path, force_sigma=SIG_FINAL, do_round=True
    )

    # Save the FPGA-deployable rounded checkpoint (from LAST, not BEST)
    torch.save(
        {
            "net": model.state_dict(),
            "acc": last_after,
            "epoch": last_epoch,
            "dcls_version": "max",
            "sigma_init": SIG_INIT,
            "sigma_final": SIG_FINAL,
            "rounded": True,
            "source": "last.pth",
        },
        os.path.join(out_dir, "best_rounded.pth"),
    )

    # Use the LAST results as the primary headline numbers
    test_before = last_before
    test_after = last_after
    drop = test_before - test_after

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(
            {
                "dcls_version": "max",
                "reference": "Hammouamri 2024 arxiv 2306.00817",
                "sigma_schedule": "cosine",
                "sigma_init": SIG_INIT,
                "sigma_final": SIG_FINAL,
                "best_val_acc": best_val_acc,
                "best_native_sigma": best_native_sigma,
                "best_test_at_native_sigma": best_before,
                "last_epoch": last_epoch,
                "last_test_at_sig_final_before_round": test_before,
                "last_test_at_sig_final_after_round": test_after,
                "rounding_drop": drop,
                "fpga_deployable_test_acc": test_after,
                "comparison": {
                    "vgauss_original": 80.4,
                    "vgauss_rounded": 58.6,
                    "v1_test": 72.5,
                    "max_best_at_native_sigma": best_before,
                    "max_last_before_rounding": test_before,
                    "max_last_after_rounding": test_after,
                },
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to {out_dir}/")
    print("\n=== FINAL COMPARISON ===")
    print("vgauss original (SIG=15):  80.4% test")
    print("vgauss rounded (SIG=0.1):  58.6% test (-21.8%)")
    print("v1 (tent, fixed):          72.5% test (0% rounding drop)")
    print(f"max BEST @ native sigma={best_native_sigma:.2f}:  {best_before:.1f}% test (NOT FPGA)")
    print(
        f"max LAST @ sigma={SIG_FINAL}:  {test_before:.1f}% before / {test_after:.1f}% after round (FPGA-deployable)"
    )
