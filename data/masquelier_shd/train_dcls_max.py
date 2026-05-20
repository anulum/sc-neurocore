#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
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
  SHD_PRUNE_EPSILONS  — comma-separated iterative epsilon schedule
  SHD_PRUNE_PROTOCOL  — one_shot or iterative_finetune (default one_shot)
  SHD_FINETUNE_EPOCHS — epochs after pruning (default 20)
  SHD_PRUNE_STEP_FINETUNE_EPOCHS — recovery epochs after each iterative step
  SHD_PRUNE_MAX_DEPLOYABLE_DROP — stop iterative pruning beyond this pp drop
  SHD_EPOCHS          — total main-phase epochs (default = config.epochs)
  SHD_SEED            — deterministic seed override (default = config.seed)
  SHD_SIGMA_INIT      — cosine schedule start (default 15.0)
  SHD_SIGMA_FINAL     — cosine schedule end (default 0.0)
  SHD_ROUND_EACH_EPOCH — round train delays after every epoch (default 0)
  SHD_HIDDEN_LAYERS   — comma-separated hidden-layer widths (default 128,128)
  SHD_OUTPUT_SUBDIR   — output subdirectory name

Tim Masquelier corrections (email [22/22], 2026-04-13):
  1. Best checkpoint selected by FPGA-relevant val accuracy (sigma=0, rounded delays)
  2. Optional inplace rounding after each epoch for Alexandre-style experiments
  3. Pruning uses epsilon threshold instead of fixed percentage
"""

import csv
import json
import math
import os
import sys
import time
from typing import Callable, Optional

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

from spikingjelly.activation_based import neuron, surrogate
from configs.config_SHD import Config
from src.datasets import SHD_dataloaders
from src.modules import dcls_module
from src.neurons import Vmin_LIFNode
from src.SHD.snn import SNN_axonal_feedforward_delays
from src.SHD.trainer import test, init_optim_sche, count_parameters
from src.utils import seed_everything


class CompatibleLIFNode(neuron.LIFNode):
    """Standard SpikingJelly LIFNode with Vmin_LIF-compatible constructor."""

    def __init__(
        self,
        tau: float = 2.0,
        decay_input: bool = True,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = 0.0,
        v_inf: Optional[float] = None,
        beta_v_inf: Optional[float] = None,
        surrogate_function: Callable = surrogate.Sigmoid(),
        detach_reset: bool = False,
        step_mode="s",
        backend="torch",
        store_v_seq: bool = False,
    ):
        del v_inf, beta_v_inf
        super().__init__(
            tau,
            decay_input,
            v_threshold,
            v_reset,
            surrogate_function,
            detach_reset,
            step_mode,
            backend,
            store_v_seq,
        )


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
    model: torch.nn.Module,
    valid_loader,
    device,
    config,
    epoch: int,
) -> float:
    """Evaluate validation accuracy with rounded integer delays.

    Tim Masquelier email [22/22], 2026-04-13: "it makes no sense to select
    the best checkpoint based on the validation accuracy computed with
    vgauss or vmax with SIG>0 and non-integer delays. After each epoch,
    you need to temporarily switch to v1 (or vmax SIG=0) and round the
    delays just to estimate the validation accuracy."

    The embedded trainer.test() mutates models by calling round_pos() before
    every evaluation. This wrapper saves both delay and sigma state, evaluates
    the deployable rounded-delay path, and restores the training state unless
    the caller separately opts into in-place rounding.
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
    print(
        f"  epsilon_prune(eps={epsilon}): {pruned_params}/{total_params} "
        f"pruned ({sparsity:.1%} sparsity)"
    )
    return sparsity


def parse_epsilon_schedule(text: str | None) -> list[float]:
    """Parse a positive, strictly increasing epsilon schedule."""
    if text is None or not text.strip():
        return []
    values: list[float] = []
    for raw in text.split(","):
        item = raw.strip()
        if not item:
            continue
        value = float(item)
        if value <= 0.0:
            raise ValueError("SHD_PRUNE_EPSILONS entries must be positive")
        if values and value <= values[-1]:
            raise ValueError("SHD_PRUNE_EPSILONS must be strictly increasing")
        values.append(value)
    return values


def parse_hidden_layers(text: str | None, default: list[int] | None = None) -> list[int]:
    """Parse positive hidden-layer widths from a comma-separated override."""
    fallback = list(default or [128, 128])
    if text is None or not text.strip():
        return fallback
    values: list[int] = []
    for raw in text.split(","):
        item = raw.strip()
        if not item:
            continue
        try:
            value = int(item)
        except ValueError as exc:
            raise ValueError("SHD_HIDDEN_LAYERS entries must be positive integers") from exc
        if value <= 0:
            raise ValueError("SHD_HIDDEN_LAYERS entries must be positive integers")
        values.append(value)
    if not values:
        raise ValueError("SHD_HIDDEN_LAYERS must contain at least one width")
    return values


def weight_sparsity(model: torch.nn.Module) -> dict[str, float]:
    """Count current sparsity across all trainable Linear/DCLS weights."""
    total_params = 0
    nonzero_params = 0
    for m in model.modules():
        if isinstance(m, (torch.nn.Linear, dcls_module)) and m.weight.requires_grad:
            total_params += m.weight.numel()
            nonzero_params += (m.weight.data != 0).sum().item()
    sparsity = 1.0 - (nonzero_params / max(1, total_params))
    return {
        "total_weights": total_params,
        "nonzero_weights": nonzero_params,
        "sparsity": sparsity,
    }


def iterative_epsilon_prune_to_target(
    model: torch.nn.Module,
    epsilon: float,
    target_sparsity: float,
    growth: float = 1.25,
    max_steps: int = 20,
) -> dict:
    """Raise an absolute-value threshold until target sparsity is reached.

    Tim Masquelier's April 13 guidance was to prune weights below an epsilon
    threshold iteratively instead of dropping a fixed percentage in one shot.
    This routine records the exact threshold path so the pruning rule remains
    reproducible and auditable.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0 for iterative epsilon pruning")
    if not 0.0 < target_sparsity < 1.0:
        raise ValueError("target_sparsity must be in (0, 1)")
    if growth <= 1.0:
        raise ValueError("growth must be > 1")

    history = []
    current_epsilon = epsilon
    sparsity = 0.0
    for step in range(max_steps):
        sparsity = epsilon_prune(model, current_epsilon)
        history.append({"step": step, "epsilon": current_epsilon, "sparsity": sparsity})
        if sparsity >= target_sparsity:
            break
        current_epsilon *= growth

    return {
        "initial_epsilon": epsilon,
        "final_epsilon": current_epsilon,
        "target_sparsity": target_sparsity,
        "achieved_sparsity": sparsity,
        "growth": growth,
        "max_steps": max_steps,
        "history": history,
    }


def iterative_epsilon_schedule(
    initial_epsilon: float,
    target_sparsity: float,
    growth: float,
    max_steps: int,
    explicit_schedule: list[float],
) -> list[float]:
    """Return the epsilon schedule for gradual prune/fine-tune recovery."""
    if explicit_schedule:
        return explicit_schedule
    if initial_epsilon <= 0.0:
        raise ValueError("initial epsilon must be positive")
    if not 0.0 < target_sparsity < 1.0:
        raise ValueError("target_sparsity must be in (0, 1)")
    if growth <= 1.0:
        raise ValueError("growth must be > 1")
    values = []
    epsilon = initial_epsilon
    for _ in range(max_steps):
        values.append(epsilon)
        epsilon *= growth
    return values


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
# Historical comparison runs used 0.23; deployable selection always scores sigma=0.
ROUND_EACH_EPOCH = os.environ.get("SHD_ROUND_EACH_EPOCH", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


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
    train_loader,
    model,
    optimizer,
    epoch,
    device,
    config,
    lambda_delay=0.0,
    l1_weight=0.0,
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


def fine_tune_pruned_model(
    *,
    train_loader,
    valid_loader,
    test_loader,
    model,
    optimizer,
    scheduler,
    start_epoch: int,
    epochs: int,
    device,
    config,
    log_path: str,
    phase_label: str,
    best_val_acc: float,
    out_dir: str,
    state_extra: dict,
) -> tuple[float, dict | None]:
    """Fine-tune a pruned model and select by deployable validation accuracy."""
    phase_best_val_acc = -float("inf")
    phase_best_state: dict | None = None
    for ft_epoch in range(epochs):
        global_epoch = start_epoch + ft_epoch
        set_sigma(model, SIG_FINAL)
        t0 = time.perf_counter()

        ft_train_acc, ft_train_loss = train_with_regulariser(
            train_loader,
            model,
            optimizer,
            global_epoch,
            device,
            config,
            lambda_delay=getattr(config, "lambda_delay", 0.01),
            l1_weight=0.0,
        )
        ft_fpga_val = fpga_val_accuracy(model, valid_loader, device, config, global_epoch)

        saved_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        ft_val_acc, ft_val_loss = test(valid_loader, model, global_epoch, device, config)
        model.load_state_dict(saved_state)

        for sc in scheduler:
            sc.step()

        elapsed = time.perf_counter() - t0

        if ft_epoch % 5 == 0 or ft_epoch == epochs - 1:
            ft_test_acc, _ = test(test_loader, model, global_epoch, device, config)
            model.load_state_dict(saved_state)
        else:
            ft_test_acc = -1.0

        print(
            f"  {phase_label} {ft_epoch:>3} {ft_train_acc:>7.1f}% "
            f"{ft_val_acc:>7.1f}%/{ft_fpga_val:>7.1f}% "
            f"{ft_test_acc:>7.1f}% {elapsed:>5.0f}s"
        )

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    f"{phase_label}_{ft_epoch}",
                    f"{SIG_FINAL:.4f}",
                    f"{ft_train_acc:.2f}",
                    f"{ft_train_loss:.4f}",
                    f"{ft_val_acc:.2f}",
                    f"{ft_fpga_val:.2f}",
                    f"{ft_val_loss:.4f}",
                    f"{ft_test_acc:.2f}",
                    "0.0",
                    f"{optimizer[0].param_groups[0]['lr']:.2e}",
                    f"{optimizer[1].param_groups[0]['lr']:.2e}",
                    f"{elapsed:.1f}",
                ]
            )

        state = {
            "net": model.state_dict(),
            "acc": ft_val_acc,
            "fpga_val_acc": ft_fpga_val,
            "epoch": global_epoch,
            "sigma": SIG_FINAL,
            **state_extra,
        }
        torch.save(state, os.path.join(out_dir, "last.pth"))

        if ft_fpga_val >= phase_best_val_acc:
            phase_best_val_acc = ft_fpga_val
            phase_best_state = state

        if ft_fpga_val >= best_val_acc:
            best_val_acc = ft_fpga_val
            torch.save(state, os.path.join(out_dir, "best.pth"))

    return best_val_acc, phase_best_state


if __name__ == "__main__":
    config = Config()
    # Sweep parameters from environment:
    #   SHD_LAMBDA_DELAY     — integer-delay regulariser weight (default 0.01)
    #   SHD_EPOCHS           — total epochs override (default = config.epochs)
    #   SHD_SEED             — deterministic seed override
    #   SHD_ROUND_EACH_EPOCH — opt into in-place delay rounding experiments
    #   SHD_OUTPUT_SUBDIR    — per-run output subdirectory under
    #                          exp/SHD/SNN_axonal_feedforward_delays/
    config.lambda_delay = float(os.environ.get("SHD_LAMBDA_DELAY", "0.01"))
    config.l1_weight = float(os.environ.get("SHD_L1_WEIGHT", "0.0"))
    config.prune_sparsity = float(os.environ.get("SHD_PRUNE_SPARSITY", "0.9"))
    config.prune_epsilon = float(os.environ.get("SHD_PRUNE_EPSILON", "0.01"))
    config.prune_epsilon_growth = float(os.environ.get("SHD_PRUNE_EPSILON_GROWTH", "1.25"))
    config.prune_method = os.environ.get("SHD_PRUNE_METHOD", "magnitude").strip().lower()
    config.prune_protocol = os.environ.get("SHD_PRUNE_PROTOCOL", "one_shot").strip().lower()
    config.finetune_epochs = int(os.environ.get("SHD_FINETUNE_EPOCHS", "20"))
    config.prune_epsilon_schedule = parse_epsilon_schedule(os.environ.get("SHD_PRUNE_EPSILONS"))
    config.prune_step_finetune_epochs = int(
        os.environ.get("SHD_PRUNE_STEP_FINETUNE_EPOCHS", str(config.finetune_epochs))
    )
    config.prune_max_deployable_drop = float(os.environ.get("SHD_PRUNE_MAX_DEPLOYABLE_DROP", "1.0"))
    config.prune_max_steps = int(os.environ.get("SHD_PRUNE_MAX_STEPS", "20"))
    if os.environ.get("SHD_SEED"):
        config.seed = int(os.environ["SHD_SEED"])
    if os.environ.get("SHD_EPOCHS"):
        config.epochs = int(os.environ["SHD_EPOCHS"])
    out_subdir = os.environ.get("SHD_OUTPUT_SUBDIR", "dcls_max")
    config.hidden_layers = parse_hidden_layers(os.environ.get("SHD_HIDDEN_LAYERS"), [128, 128])

    neuron_module_name = os.environ.get("SHD_NEURON_MODULE", "vmin_lif").strip().lower()
    if neuron_module_name in {"vmin", "vmin_lif", "vmin_lifnode"}:
        config.neuron_module = Vmin_LIFNode
        neuron_module_name = "vmin_lif"
    elif neuron_module_name in {"lif", "standard_lif", "lifnode"}:
        config.neuron_module = CompatibleLIFNode
        neuron_module_name = "standard_lif"
    else:
        raise ValueError("SHD_NEURON_MODULE must be one of: vmin_lif, standard_lif")

    l1_mode = config.l1_weight > 0

    print(f"SHD_LAMBDA_DELAY   = {config.lambda_delay}")
    print(f"SHD_L1_WEIGHT      = {config.l1_weight}")
    print(f"SHD_PRUNE_SPARSITY = {config.prune_sparsity}")
    print(f"SHD_PRUNE_METHOD   = {config.prune_method}")
    print(f"SHD_PRUNE_PROTOCOL = {config.prune_protocol}")
    print(f"SHD_PRUNE_EPSILON  = {config.prune_epsilon}")
    print(f"SHD_PRUNE_EPSILONS = {config.prune_epsilon_schedule}")
    print(f"SHD_PRUNE_EPSILON_GROWTH = {config.prune_epsilon_growth}")
    print(f"SHD_FINETUNE_EPOCHS= {config.finetune_epochs}")
    print(f"SHD_PRUNE_STEP_FINETUNE_EPOCHS = {config.prune_step_finetune_epochs}")
    print(f"SHD_PRUNE_MAX_DEPLOYABLE_DROP = {config.prune_max_deployable_drop}")
    print(f"SHD_PRUNE_MAX_STEPS = {config.prune_max_steps}")
    print(f"SHD_EPOCHS         = {config.epochs}")
    print(f"SHD_SEED           = {config.seed}")
    print(f"SHD_HIDDEN_LAYERS  = {config.hidden_layers}")
    print(f"SHD_NEURON_MODULE  = {neuron_module_name}")
    print(f"SHD_SIGMA_INIT     = {SIG_INIT}")
    print(f"SHD_SIGMA_FINAL    = {SIG_FINAL}")
    print(f"SHD_ROUND_EACH_EPOCH = {ROUND_EACH_EPOCH}")
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

    if config.dataset != "SHD":
        raise ValueError(f"Dataset {config.dataset} is not supported.")
    train_loader, valid_loader, test_loader = SHD_dataloaders(config)
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
                "fpga_val_acc",
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
        f"\n{'Epoch':>5} {'SIG':>6} {'Train':>8} {'Val/FPG':>17} {'Test':>8} "
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
        # Tim [22/22]: checkpoint selection is based on the deployable path:
        # rounded delays and sigma=0. fpga_val_accuracy() restores the train
        # state because trainer.test() rounds model delays in place.
        fpga_val = fpga_val_accuracy(model, valid_loader, device, config, epoch)

        if ROUND_EACH_EPOCH:
            round_delays_inplace(model)

        saved_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        val_acc, val_loss = test(valid_loader, model, epoch, device, config)
        if not ROUND_EACH_EPOCH:
            model.load_state_dict(saved_state)

        for sc in scheduler:
            sc.step()

        elapsed = time.perf_counter() - t0
        lr = optimizer[0].param_groups[0]["lr"]
        lr_pos = optimizer[1].param_groups[0]["lr"]

        if epoch % 10 == 0 or epoch in [config.epochs - 1]:
            test_acc, test_loss = test(test_loader, model, epoch, device, config)
            if not ROUND_EACH_EPOCH:
                model.load_state_dict(saved_state)
        else:
            test_acc, test_loss = -1.0, -1.0

        print(
            f"{epoch:>5} {sigma:>6.2f} {train_acc:>7.1f}% "
            f"{val_acc:>7.1f}%/{fpga_val:>7.1f}% "
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
                    f"{fpga_val:.2f}",
                    f"{val_loss:.4f}",
                    f"{test_acc:.2f}",
                    f"{test_loss:.4f}",
                    f"{lr:.2e}",
                    f"{lr_pos:.2e}",
                    f"{elapsed:.1f}",
                ]
            )

        state = {
            "net": model.state_dict(),
            "acc": val_acc,
            "fpga_val_acc": fpga_val,
            "epoch": epoch,
            "sigma": sigma,
            "round_each_epoch": ROUND_EACH_EPOCH,
        }
        torch.save(state, os.path.join(out_dir, "last.pth"))

        if fpga_val >= best_val_acc:
            best_val_acc = fpga_val
            torch.save(state, os.path.join(out_dir, "best.pth"))

    prune_summary = None
    iterative_prune_history: list[dict] = []
    best_sparse_state: dict | None = None
    best_sparse_record: dict | None = None

    # === L1 POST-TRAINING: prune + fine-tune ===
    # Tim Masquelier suggestion #3 (2026-04-13): prune by absolute epsilon
    # iteratively, fine-tune after each threshold, and select by deployable
    # rounded-delay validation rather than by native training validation.
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

        dense_best_val_acc = best_val_acc
        if config.prune_method == "epsilon" and config.prune_protocol == "iterative_finetune":
            schedule = iterative_epsilon_schedule(
                initial_epsilon=config.prune_epsilon,
                target_sparsity=config.prune_sparsity,
                growth=config.prune_epsilon_growth,
                max_steps=config.prune_max_steps,
                explicit_schedule=config.prune_epsilon_schedule,
            )
            print("iterative epsilon schedule:", schedule)
            best_sparse_val_acc = -float("inf")
            for step, epsilon in enumerate(schedule):
                print(f"\n=== Iterative prune step {step}: epsilon={epsilon} ===")
                achieved_sparsity = epsilon_prune(model, epsilon)
                sparsity_stats = weight_sparsity(model)
                step_val = fpga_val_accuracy(
                    model, valid_loader, device, config, config.epochs + step
                )
                step_test, _ = test(test_loader, model, config.epochs + step, device, config)
                print(
                    f"  immediate deployable val={step_val:.2f}% "
                    f"test={step_test:.2f}% sparsity={achieved_sparsity:.1%}"
                )

                ft_optimizer, ft_scheduler = init_optim_sche(model, config)
                previous_global_best = best_val_acc
                state_extra = {
                    "l1_mode": True,
                    "prune_protocol": "iterative_finetune",
                    "prune_step": step,
                    "prune_epsilon": epsilon,
                    "prune_target_sparsity": config.prune_sparsity,
                    "prune_sparsity": sparsity_stats["sparsity"],
                }
                best_val_acc, step_best_state = fine_tune_pruned_model(
                    train_loader=train_loader,
                    valid_loader=valid_loader,
                    test_loader=test_loader,
                    model=model,
                    optimizer=ft_optimizer,
                    scheduler=ft_scheduler,
                    start_epoch=config.epochs + step * config.prune_step_finetune_epochs,
                    epochs=config.prune_step_finetune_epochs,
                    device=device,
                    config=config,
                    log_path=log_path,
                    phase_label=f"prune{step}",
                    best_val_acc=best_val_acc,
                    out_dir=out_dir,
                    state_extra=state_extra,
                )
                step_best_val = (
                    step_best_state["fpga_val_acc"] if step_best_state is not None else step_val
                )
                if step_best_state is not None and step_best_val >= best_sparse_val_acc:
                    best_sparse_val_acc = step_best_val
                    best_sparse_state = step_best_state
                    best_sparse_record = {
                        "step": step,
                        "epsilon": epsilon,
                        "fpga_val_acc": step_best_val,
                        **weight_sparsity(model),
                    }
                    torch.save(
                        best_sparse_state,
                        os.path.join(out_dir, "best_sparse.pth"),
                    )

                deployable_drop = dense_best_val_acc - step_best_val
                record = {
                    "step": step,
                    "epsilon": epsilon,
                    "immediate_fpga_val_acc": step_val,
                    "immediate_test_acc": step_test,
                    "best_step_fpga_val_acc": step_best_val,
                    "deployable_drop_vs_dense_best": deployable_drop,
                    "global_best_before_step": previous_global_best,
                    "global_best_after_step": best_val_acc,
                    **weight_sparsity(model),
                }
                iterative_prune_history.append(record)
                print("iterative prune record:", record)
                if record["sparsity"] >= config.prune_sparsity:
                    print("target sparsity reached; stopping iterative pruning")
                    break
                if deployable_drop > config.prune_max_deployable_drop:
                    print(
                        "deployable-validation drop exceeded configured limit; "
                        "stopping iterative pruning"
                    )
                    break

            prune_summary = {
                "method": "iterative_epsilon_prune_finetune",
                "schedule": schedule,
                "step_finetune_epochs": config.prune_step_finetune_epochs,
                "max_deployable_drop": config.prune_max_deployable_drop,
                "dense_best_fpga_val_acc": dense_best_val_acc,
                "best_sparse_record": best_sparse_record,
                "history": iterative_prune_history,
            }
            print("iterative epsilon prune summary:", prune_summary)
        elif config.prune_method == "epsilon":
            prune_summary = iterative_epsilon_prune_to_target(
                model,
                epsilon=config.prune_epsilon,
                target_sparsity=config.prune_sparsity,
                growth=config.prune_epsilon_growth,
            )
            print("epsilon prune summary:", prune_summary)
        elif config.prune_method == "magnitude":
            # Magnitude prune: keep top (1-sparsity) fraction per layer
            sparsity_list = [0.0]  # first layer (input->hidden1): no pruning
            for _ in range(len(config.weight_sparsity_mask) - 1):
                sparsity_list.append(config.prune_sparsity)
            magnitude_prune(model, sparsity_list)
        else:
            raise ValueError("SHD_PRUNE_METHOD must be 'magnitude' or 'epsilon'")

        if config.prune_protocol != "iterative_finetune":
            sparsity_stats = weight_sparsity(model)
            print(
                "After pruning: "
                f"{sparsity_stats['nonzero_weights']}/"
                f"{sparsity_stats['total_weights']} non-zero "
                f"({100 * (1.0 - sparsity_stats['sparsity']):.1f}%)"
            )

            # Evaluate immediately after pruning (before fine-tune)
            prune_test, _ = test(test_loader, model, 0, device, config)
            print(f"Test accuracy after pruning (before fine-tune): {prune_test:.2f}%")

            # Fine-tune: train with L1 disabled, pruned weights stay zero via hooks
            print(f"\n=== L1 Fine-Tune Phase: {config.finetune_epochs} epochs ===")
            ft_optimizer, ft_scheduler = init_optim_sche(model, config)
            best_val_acc, best_sparse_state = fine_tune_pruned_model(
                train_loader=train_loader,
                valid_loader=valid_loader,
                test_loader=test_loader,
                model=model,
                optimizer=ft_optimizer,
                scheduler=ft_scheduler,
                start_epoch=config.epochs,
                epochs=config.finetune_epochs,
                device=device,
                config=config,
                log_path=log_path,
                phase_label="ft",
                best_val_acc=best_val_acc,
                out_dir=out_dir,
                state_extra={
                    "l1_mode": True,
                    "prune_protocol": config.prune_protocol,
                    "prune_sparsity": config.prune_sparsity,
                },
            )
            if best_sparse_state is not None:
                torch.save(best_sparse_state, os.path.join(out_dir, "best_sparse.pth"))

    # === Final evaluation ===
    # We evaluate THREE views:
    #   1. BEST at native sigma — accuracy ceiling only
    #   2. BEST at SIG_FINAL with rounded delays — deployable selected model
    #   3. LAST at SIG_FINAL with rounded delays — historical comparison
    # Older runs used LAST as the deployable model because best.pth had been
    # selected by native validation. best.pth is now selected by fpga_val_acc,
    # so the deployable headline must come from BEST under rounded/SIG_FINAL
    # evaluation.
    print("\n=== Final Evaluation ===")

    def _eval_checkpoint(label, ckpt_path, force_sigma, do_round):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
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

    # 2. BEST at SIG_FINAL — FPGA-deployable selected model
    best_deploy_before, best_deploy_after, best_deploy_epoch, _ = _eval_checkpoint(
        "BEST @ SIG_FINAL", best_path, force_sigma=SIG_FINAL, do_round=True
    )

    # Save the FPGA-deployable rounded checkpoint from the selected BEST model.
    torch.save(
        {
            "net": model.state_dict(),
            "acc": best_deploy_after,
            "epoch": best_deploy_epoch,
            "dcls_version": "max",
            "sigma_init": SIG_INIT,
            "sigma_final": SIG_FINAL,
            "rounded": True,
            "source": "best.pth",
        },
        os.path.join(out_dir, "best_rounded.pth"),
    )

    # 3. LAST at SIG_FINAL — historical comparison
    last_before, last_after, last_epoch, last_native_sigma = _eval_checkpoint(
        "LAST @ SIG_FINAL", last_path, force_sigma=SIG_FINAL, do_round=True
    )

    # Use the deployable BEST results as the primary headline numbers.
    test_before = best_deploy_before
    test_after = best_deploy_after
    drop = test_before - test_after

    config_data = {
        "dcls_version": "max",
        "reference": "Hammouamri 2024 arxiv 2306.00817",
        "neuron_module": neuron_module_name,
        "sigma_schedule": "cosine",
        "sigma_init": SIG_INIT,
        "sigma_final": SIG_FINAL,
        "seed": config.seed,
        "round_each_epoch": ROUND_EACH_EPOCH,
        "lambda_delay": getattr(config, "lambda_delay", 0.01),
        "best_val_acc": best_val_acc,
        "best_fpga_deployable_epoch": best_deploy_epoch,
        "best_fpga_deployable_before_round": best_deploy_before,
        "best_fpga_deployable_after_round": best_deploy_after,
        "best_native_sigma": best_native_sigma,
        "best_test_at_native_sigma": best_before,
        "last_epoch": last_epoch,
        "last_test_at_sig_final_before_round": last_before,
        "last_test_at_sig_final_after_round": last_after,
        "rounding_drop": drop,
        "fpga_deployable_test_acc": test_after,
        "comparison": {
            "vgauss_original": 80.4,
            "vgauss_rounded": 58.6,
            "v1_test": 72.5,
            "max_best_at_native_sigma": best_before,
            "max_best_after_rounding": best_deploy_after,
            "max_last_before_rounding": last_before,
            "max_last_after_rounding": last_after,
        },
    }
    if l1_mode:
        config_data["l1_sparsity"] = {
            "l1_weight": config.l1_weight,
            "prune_sparsity": config.prune_sparsity,
            "finetune_epochs": config.finetune_epochs,
            "prune_method": config.prune_method,
            "prune_protocol": config.prune_protocol,
            "prune_epsilon": config.prune_epsilon,
            "prune_epsilon_schedule": config.prune_epsilon_schedule,
            "prune_epsilon_growth": config.prune_epsilon_growth,
            "prune_step_finetune_epochs": config.prune_step_finetune_epochs,
            "prune_max_deployable_drop": config.prune_max_deployable_drop,
            "epsilon_prune_summary": prune_summary,
            "iterative_prune_history": iterative_prune_history,
            "best_sparse_record": best_sparse_record,
            "method": "L1 + prune + fine-tune",
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
        f"max BEST @ sigma={SIG_FINAL}:  {best_deploy_before:.1f}% before / {best_deploy_after:.1f}% after round (FPGA-selected)"
    )
    print(
        f"max LAST @ sigma={SIG_FINAL}:  {last_before:.1f}% before / {last_after:.1f}% after round (historical comparison)"
    )
