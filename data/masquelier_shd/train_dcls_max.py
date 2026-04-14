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

L1 sparsity mode (Tim Masquelier suggestion #3, 2026-04-09):
  Instead of random mask at init, train dense with L1 weight regularisation,
  then magnitude-prune and fine-tune. Set SHD_L1_WEIGHT > 0 to enable.
  See: Han et al. (2015) "Learning both Weights and Connections".

Run: python3 train_dcls_max.py
Env:
  SHD_LAMBDA_DELAY    — integer-delay regulariser weight (default 0.01)
  SHD_L1_WEIGHT       — L1 weight regularisation coefficient (default 0.0 = off)
  SHD_PRUNE_SPARSITY  — target sparsity for magnitude pruning (default 0.9)
  SHD_PRUNE_EPSILON   — abs threshold for epsilon pruning (default 0.01)
  SHD_FINETUNE_EPOCHS — epochs after pruning (default 20)
  SHD_EPOCHS          — total main-phase epochs (default = config.epochs)
  SHD_SIGMA_INIT      — cosine schedule start (default 15.0)
  SHD_SIGMA_FINAL     — cosine schedule end (default 0.23)
  SHD_OUTPUT_SUBDIR   — output subdirectory name

Tim Masquelier corrections (email [22/22], 2026-04-13):
  1. Delays are rounded inplace after each epoch (like Alexandre's code)
  2. Best checkpoint selected by FPGA-relevant val accuracy (sigma=0, rounded delays)
  3. Pruning uses epsilon threshold instead of fixed percentage
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


def round_delays_inplace(model: torch.nn.Module) -> None:
    """Round all DCLS delay parameters to nearest integer, inplace.

    Tim Masquelier email [22/22], 2026-04-13: Alexandre's training code
    rounds delays inplace after each epoch. This is why the integer-delay
    regulariser had no effect — delays were already integer after every step.
    """
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, dcls_module) and hasattr(m, "P"):
                m.P.data.round_()


def fpga_val_accuracy(
    model: torch.nn.Module, valid_loader, device, config, epoch: int,
) -> float:
    """Evaluate validation accuracy with rounded integer delays.

    Tim Masquelier email [22/22], 2026-04-13: "it makes no sense to select
    the best checkpoint based on the validation accuracy computed with
    vgauss or vmax with SIG>0 and non-integer delays. After each epoch,
    you need to temporarily switch to v1 (or vmax SIG=0) and round the
    delays just to estimate the validation accuracy."

    This function saves delay state, rounds delays, evaluates, and restores.
    """
    # Save original delay values
    saved_delays = {}
    for name, m in model.named_modules():
        if isinstance(m, dcls_module) and hasattr(m, "P"):
            saved_delays[name] = m.P.data.clone()
            m.P.data.round_()

    # Save and set sigma to 0 (v1-equivalent evaluation)
    saved_sigmas = {}
    for name, m in model.named_modules():
        if isinstance(m, dcls_module) and hasattr(m, "SIG"):
            saved_sigmas[name] = m.SIG.data.clone()
            m.SIG.data.fill_(0.0)

    val_acc, _ = test(valid_loader, model, epoch, device, config)

    # Restore original delays and sigmas
    for name, m in model.named_modules():
        if isinstance(m, dcls_module) and hasattr(m, "P"):
            if name in saved_delays:
                m.P.data.copy_(saved_delays[name])
        if isinstance(m, dcls_module) and hasattr(m, "SIG"):
            if name in saved_sigmas:
                m.SIG.data.copy_(saved_sigmas[name])

    return val_acc


def epsilon_prune(model: torch.nn.Module, epsilon: float = 0.01):
    """Prune weights with absolute value below epsilon.

    Tim Masquelier email [22/22], 2026-04-13: "instead of discarding a
    fixed % of the weights, I would discard those that are inferior to
    some epsilon (in absolute value). And it's a good idea to do so
    iteratively, until reaching 90% sparsity."
    """
    total_params = 0
    pruned_params = 0
    for m in model.modules():
        if isinstance(m, (torch.nn.Linear, dcls_module)) and m.weight.requires_grad:
            mask = m.weight.data.abs() >= epsilon
            m.weight.data *= mask.float()
            total_params += m.weight.numel()
            pruned_params += (~mask).sum().item()

            def make_hook(mask_t):
                return lambda grad: grad * mask_t.float()
            m.weight.register_hook(make_hook(mask))

    sparsity = pruned_params / max(1, total_params)
    print(f"  epsilon_prune(eps={epsilon}): {pruned_params}/{total_params} "
          f"pruned ({sparsity:.1%} sparsity)")
    return sparsity


def integer_delay_penalty(model: torch.nn.Module) -> torch.Tensor:
    """Sum quadratic penalty (P - round(P))^2 over all DCLS delay parameters.

    Motivated by DelRec (2025) and Khalfaoui-Hassani et al. (ICLR 2024),
    this pushes learnable delays toward integer values to minimise rounding
    error for FPGA deployment.
    """
    penalty = 0.0
    for m in model.modules():
        if isinstance(m, dcls_module) and hasattr(m, "P"):
            penalty += torch.sum((m.P - m.P.round()) ** 2)
    return penalty


SIG_INIT = float(os.environ.get("SHD_SIGMA_INIT", "15.0"))
SIG_FINAL = float(os.environ.get("SHD_SIGMA_FINAL", "0.0"))  # Tim: sigma must end at 0
# 0.23 is below 0.5 so rounding has at most 0.5/0.23 ratio of neighbour overlap


def get_sigma_schedule(epoch: int, total_epochs: int) -> float:
    """Cosine anneal SIG from SIG_INIT to SIG_FINAL across all epochs."""
    progress = epoch / max(1, total_epochs - 1)
    return SIG_FINAL + (SIG_INIT - SIG_FINAL) * 0.5 * (1.0 + math.cos(math.pi * progress))


def l1_weight_penalty(model: torch.nn.Module) -> torch.Tensor:
    """Sum of absolute weight values across all learnable Linear/DCLS layers.

    Encourages weight sparsity during training — the network learns which
    connections are unimportant and drives them toward zero.
    Reference: Han et al. (2015) "Learning both Weights and Connections".
    """
    penalty = torch.tensor(0.0, device=next(model.parameters()).device)
    for m in model.modules():
        if isinstance(m, (torch.nn.Linear, dcls_module)) and m.weight.requires_grad:
            penalty = penalty + m.weight.abs().sum()
    return penalty


def train_with_regulariser(
    train_loader, model, optimizer, epoch, device, config,
    lambda_delay=0.0, l1_weight=0.0,
):
    """Custom training loop with integer delay regulariser and L1 weight penalty."""
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
        if l1_weight > 0:
            loss += l1_weight * l1_weight_penalty(model)
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
    # Sweep parameters from environment (defaults match the published
    # 75.21% baseline so legacy invocations behave unchanged):
    #   SHD_LAMBDA_DELAY  — integer-delay regulariser weight (default 0.01)
    #   SHD_EPOCHS        — total epochs override (default = config.epochs)
    #   SHD_OUTPUT_SUBDIR — per-run output subdirectory under
    #                       exp/SHD/SNN_axonal_feedforward_delays/
    config.lambda_delay = float(os.environ.get("SHD_LAMBDA_DELAY", "0.01"))
    config.l1_weight = float(os.environ.get("SHD_L1_WEIGHT", "0.0"))
    config.prune_sparsity = float(os.environ.get("SHD_PRUNE_SPARSITY", "0.9"))
    config.finetune_epochs = int(os.environ.get("SHD_FINETUNE_EPOCHS", "20"))
    if os.environ.get("SHD_EPOCHS"):
        config.epochs = int(os.environ["SHD_EPOCHS"])
    out_subdir = os.environ.get("SHD_OUTPUT_SUBDIR", "dcls_max")
    config.hidden_layers = [128, 128]

    l1_mode = config.l1_weight > 0

    print(f"SHD_LAMBDA_DELAY   = {config.lambda_delay}")
    print(f"SHD_L1_WEIGHT      = {config.l1_weight}")
    print(f"SHD_PRUNE_SPARSITY = {config.prune_sparsity}")
    print(f"SHD_FINETUNE_EPOCHS= {config.finetune_epochs}")
    print(f"SHD_EPOCHS         = {config.epochs}")
    print(f"SHD_OUTPUT_SUBDIR  = {out_subdir}")
    print(f"L1 sparsity mode   = {l1_mode}")

    # === KEY CHANGE: DCLS max (triangular with scheduled SIG) ===
    config.DCLSversion = "max"
    print(f"DCLS version: {config.DCLSversion}")

    seed_everything(config.seed, is_cuda=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    out_dir = os.path.join("exp", "SHD", "SNN_axonal_feedforward_delays", out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    train_loader, valid_loader, test_loader = load_dataset(config)
    print(
        f"Train: {len(train_loader.dataset)}, Valid: {len(valid_loader.dataset)}, "
        f"Test: {len(test_loader.dataset)}"
    )

    model = SNN_axonal_feedforward_delays(config).to(device)
    if l1_mode:
        # L1 mode: train DENSE first — no mask at init. Pruning happens after training.
        print("L1 sparsity mode: skipping initial mask, training dense weights")
    else:
        magnitude_prune(model, config.weight_sparsity_mask)

    # Initialise SIG on all DCLS layers (snn.py only does this for 'gauss' version)
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
            l1_weight=getattr(config, "l1_weight", 0.0),
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

        # Tim [22/22]: round delays inplace after each epoch (like Alexandre's code)
        round_delays_inplace(model)

        # Tim [22/22]: evaluate with rounded delays at sigma=0 for FPGA-relevant accuracy
        fpga_val = fpga_val_accuracy(model, valid_loader, device, config, epoch)

        state = {
            "net": model.state_dict(), "acc": val_acc, "fpga_val_acc": fpga_val,
            "epoch": epoch, "sigma": sigma,
        }
        torch.save(state, os.path.join(out_dir, "last.pth"))

        if fpga_val >= best_val_acc:
            best_val_acc = fpga_val
            torch.save(state, os.path.join(out_dir, "best.pth"))

    # === L1 POST-TRAINING: magnitude prune + fine-tune ===
    # Tim Masquelier suggestion #3 (2026-04-09): instead of random mask at init,
    # train dense with L1 → prune by magnitude → fine-tune surviving weights.
    # Reference: Han et al. (2015) "Learning both Weights and Connections".
    if l1_mode:
        print(f"\n=== L1 Post-Training: Pruning at {config.prune_sparsity:.0%} sparsity ===")

        # Report weight distribution before pruning
        for name, m in model.named_modules():
            if isinstance(m, (torch.nn.Linear, dcls_module)) and m.weight.requires_grad:
                w = m.weight.data
                nz = (w != 0).sum().item()
                total = w.numel()
                near_zero = (w.abs() < 0.01).sum().item()
                print(f"  {name}: {nz}/{total} non-zero, {near_zero} near-zero (<0.01)")

        # Magnitude prune: keep top (1-sparsity) fraction per layer
        sparsity_list = [0.0]  # first layer (input→hidden1): no pruning
        for _ in range(len(config.weight_sparsity_mask) - 1):
            sparsity_list.append(config.prune_sparsity)
        magnitude_prune(model, sparsity_list)

        # Report after pruning
        total_params = sum(
            m.weight.numel()
            for m in model.modules()
            if isinstance(m, (torch.nn.Linear, dcls_module)) and m.weight.requires_grad
        )
        nonzero_params = sum(
            (m.weight.data != 0).sum().item()
            for m in model.modules()
            if isinstance(m, (torch.nn.Linear, dcls_module)) and m.weight.requires_grad
        )
        print(f"After pruning: {nonzero_params}/{total_params} non-zero ({100*nonzero_params/total_params:.1f}%)")

        # Evaluate immediately after pruning (before fine-tune)
        prune_test, _ = test(test_loader, model, 0, device, config)
        print(f"Test accuracy after pruning (before fine-tune): {prune_test:.2f}%")

        # Fine-tune: train with L1 disabled, pruned weights stay zero via hooks
        print(f"\n=== L1 Fine-Tune Phase: {config.finetune_epochs} epochs ===")
        # Reset optimiser for fine-tune (fresh momentum)
        ft_optimizer, ft_scheduler = init_optim_sche(model, config)

        for ft_epoch in range(config.finetune_epochs):
            # Sigma stays at SIG_FINAL during fine-tune
            set_sigma(model, SIG_FINAL)
            t0 = time.perf_counter()

            ft_train_acc, ft_train_loss = train_with_regulariser(
                train_loader, model, ft_optimizer, ft_epoch, device, config,
                lambda_delay=getattr(config, "lambda_delay", 0.01),
                l1_weight=0.0,  # L1 off during fine-tune
            )
            ft_val_acc, ft_val_loss = test(valid_loader, model, ft_epoch, device, config)

            for sc in ft_scheduler:
                sc.step()

            elapsed = time.perf_counter() - t0

            if ft_epoch % 5 == 0 or ft_epoch == config.finetune_epochs - 1:
                ft_test_acc, _ = test(test_loader, model, ft_epoch, device, config)
            else:
                ft_test_acc = -1.0

            print(
                f"  FT {ft_epoch:>3} {ft_train_acc:>7.1f}% {ft_val_acc:>7.1f}% "
                f"{ft_test_acc:>7.1f}% {elapsed:>5.0f}s"
            )

            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    f"ft_{ft_epoch}", f"{SIG_FINAL:.4f}",
                    f"{ft_train_acc:.2f}", f"{ft_train_loss:.4f}",
                    f"{ft_val_acc:.2f}", f"{ft_val_loss:.4f}",
                    f"{ft_test_acc:.2f}", "0.0",
                    f"{ft_optimizer[0].param_groups[0]['lr']:.2e}",
                    f"{ft_optimizer[1].param_groups[0]['lr']:.2e}",
                    f"{elapsed:.1f}",
                ])

            state = {
                "net": model.state_dict(), "acc": ft_val_acc,
                "epoch": config.epochs + ft_epoch, "sigma": SIG_FINAL,
                "l1_mode": True, "prune_sparsity": config.prune_sparsity,
                "finetune_epoch": ft_epoch,
            }
            torch.save(state, os.path.join(out_dir, "last.pth"))

            if ft_val_acc >= best_val_acc:
                best_val_acc = ft_val_acc
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

    config_data = {
        "dcls_version": "max",
        "reference": "Hammouamri 2024 arxiv 2306.00817",
        "sigma_schedule": "cosine",
        "sigma_init": SIG_INIT,
        "sigma_final": SIG_FINAL,
        "lambda_delay": getattr(config, "lambda_delay", 0.01),
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
    }
    if l1_mode:
        config_data["l1_sparsity"] = {
            "l1_weight": config.l1_weight,
            "prune_sparsity": config.prune_sparsity,
            "finetune_epochs": config.finetune_epochs,
            "method": "Han et al. 2015 — L1 + magnitude prune + fine-tune",
        }

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(
            config_data,
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
