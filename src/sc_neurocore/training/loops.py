# SPDX-License-Identifier: AGPL-3.0-or-later
"""Training and evaluation loops for SNN models."""

from __future__ import annotations

from typing import Callable, Tuple

import torch
from torch.utils.data import DataLoader

from .losses import spike_count_loss


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    n_timesteps: int,
    loss_fn: Callable = spike_count_loss,
    device: str = "cpu",
) -> Tuple[float, float]:
    """One training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        data = data.view(data.shape[0], -1)
        # (batch, features) -> (T, batch, features)
        data = data.unsqueeze(0).expand(n_timesteps, -1, -1)

        spike_counts, _ = model(data)
        loss = loss_fn(spike_counts, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * targets.shape[0]
        correct += (spike_counts.argmax(dim=1) == targets).sum().item()
        total += targets.shape[0]

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    n_timesteps: int,
    loss_fn: Callable = spike_count_loss,
    device: str = "cpu",
) -> Tuple[float, float]:
    """Evaluate model. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        data = data.view(data.shape[0], -1)
        data = data.unsqueeze(0).expand(n_timesteps, -1, -1)

        spike_counts, _ = model(data)
        loss = loss_fn(spike_counts, targets)

        total_loss += loss.item() * targets.shape[0]
        correct += (spike_counts.argmax(dim=1) == targets).sum().item()
        total += targets.shape[0]

    return total_loss / total, correct / total
