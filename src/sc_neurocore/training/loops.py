# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Training and evaluation loops for SNN models

"""Training and evaluation loops for SNN models."""

from __future__ import annotations

from typing import Any, Callable, Tuple

import torch
from torch.utils.data import DataLoader

from .losses import spike_count_loss


def _device_usable(device: torch.device) -> bool:
    """Return whether a Torch device can execute a minimal tensor operation."""
    try:
        probe = torch.empty(1, device=device)
        probe.fill_(1)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return bool(probe.cpu().item() == 1)
    except (AssertionError, RuntimeError):
        return False


def auto_device() -> torch.device:
    """Select best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        cuda_device = torch.device("cuda")
        if _device_usable(cuda_device):
            return cuda_device
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        if _device_usable(mps_device):
            return mps_device
    return torch.device("cpu")


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    n_timesteps: int,
    loss_fn: Callable[..., torch.Tensor] = spike_count_loss,
    device: str | torch.device = "cpu",
    max_grad_norm: float | None = None,
    flatten_input: bool = True,
) -> Tuple[float, float]:
    """One training epoch. Returns (avg_loss, accuracy).

    Parameters
    ----------
    flatten_input : bool
        If True (default), flatten data to (batch, features) for feedforward SNNs.
        Set to False for convolutional models that need spatial dimensions.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        if flatten_input:
            data = data.view(data.shape[0], -1)
        # Prepend time dimension: (batch, ...) -> (T, batch, ...)
        data = data.unsqueeze(0).expand(n_timesteps, *data.shape)

        spike_counts, _ = model(data)
        loss = loss_fn(spike_counts, targets)

        optimizer.zero_grad()
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item() * targets.shape[0]
        correct += (spike_counts.argmax(dim=1) == targets).sum().item()
        total += targets.shape[0]

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    n_timesteps: int,
    loss_fn: Callable[..., torch.Tensor] = spike_count_loss,
    device: str = "cpu",
    flatten_input: bool = True,
) -> Tuple[float, float]:
    """Evaluate model. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        if flatten_input:
            data = data.view(data.shape[0], -1)
        data = data.unsqueeze(0).expand(n_timesteps, *data.shape)

        spike_counts, _ = model(data)
        loss = loss_fn(spike_counts, targets)

        total_loss += loss.item() * targets.shape[0]
        correct += (spike_counts.argmax(dim=1) == targets).sum().item()
        total += targets.shape[0]

    return total_loss / total, correct / total
