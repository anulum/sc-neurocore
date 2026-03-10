# SPDX-License-Identifier: AGPL-3.0-or-later
"""Loss functions for SNN training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def spike_count_loss(spike_counts: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy on spike counts. Bohte 2011."""
    return F.cross_entropy(spike_counts, targets)


def membrane_loss(membrane_acc: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy on accumulated membrane potential."""
    return F.cross_entropy(membrane_acc, targets)


def spike_rate_loss(
    spike_counts: torch.Tensor,
    targets: torch.Tensor,
    n_timesteps: int,
    target_rate: float = 0.8,
) -> torch.Tensor:
    """MSE between output spike rates and one-hot target pattern."""
    rates = spike_counts / n_timesteps
    n_classes = rates.shape[1]
    bg_rate = (1.0 - target_rate) / max(n_classes - 1, 1)
    target_rates = torch.full_like(rates, bg_rate)
    target_rates.scatter_(1, targets.unsqueeze(1), target_rate)
    return F.mse_loss(rates, target_rates)


def spike_l1_loss(spike_counts: torch.Tensor, n_timesteps: int) -> torch.Tensor:
    """L1 penalty on mean spike rate. Encourages sparse firing."""
    return (spike_counts / n_timesteps).abs().mean()


def spike_l2_loss(spike_counts: torch.Tensor, n_timesteps: int) -> torch.Tensor:
    """L2 penalty on mean spike rate. Penalizes high-firing neurons."""
    return ((spike_counts / n_timesteps) ** 2).mean()
