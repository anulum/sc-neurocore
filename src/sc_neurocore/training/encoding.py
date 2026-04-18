# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Spike encoding: convert continuous values to spike trains

"""Spike encoding: convert continuous values to spike trains.

All encoders return tensors of shape (T, *input_shape) with values in {0, 1}.
"""

from __future__ import annotations

import torch


def rate_encode(x: torch.Tensor, n_timesteps: int) -> torch.Tensor:
    """Poisson rate coding. Higher values spike more often.

    x: values in [0, 1], shape (*batch). Returns (T, *batch).
    """
    x = x.clamp(0.0, 1.0)
    return (torch.rand(n_timesteps, *x.shape, device=x.device) < x.unsqueeze(0)).float()


def latency_encode(x: torch.Tensor, n_timesteps: int, tau: float = 5.0) -> torch.Tensor:
    """Time-to-first-spike latency coding. Stronger input → earlier spike.

    x: values in [0, 1], shape (*batch). Returns (T, *batch).
    """
    x = x.clamp(1e-6, 1.0)
    spike_time = (tau * (1.0 - x)).long().clamp(0, n_timesteps - 1)
    spikes = torch.zeros(n_timesteps, *x.shape, device=x.device)
    timesteps = torch.arange(n_timesteps, device=x.device)
    for t in range(n_timesteps):
        spikes[t] = (spike_time == t).float()
    return spikes


def delta_encode(x: torch.Tensor, threshold: float = 0.1) -> torch.Tensor:
    """Delta coding: spike on temporal change exceeding threshold.

    x: shape (T, *batch). Returns same shape with spikes where |dx| > threshold.
    """
    dx = torch.zeros_like(x)
    dx[1:] = x[1:] - x[:-1]
    return (dx.abs() > threshold).float()
