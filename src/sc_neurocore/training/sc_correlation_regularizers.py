# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Stochastic-computing correlation regularizers

"""Correlation regularizers for stochastic-computing bitstream training."""

from __future__ import annotations

import torch


def _require_stream_bank(streams: torch.Tensor) -> None:
    if streams.ndim != 2:
        raise ValueError("streams must be a 2D tensor shaped (n_streams, bitstream_length)")
    if streams.shape[0] < 2:
        raise ValueError("streams must contain at least two streams")
    if streams.shape[1] < 2:
        raise ValueError("streams bitstream_length must be at least two")
    if not bool(torch.isfinite(streams.detach()).all()):
        raise ValueError("streams must contain only finite values")


def correlation_matrix(streams: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Return Pearson correlation matrix across bitstream rows."""

    _require_stream_bank(streams)
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    data = streams.to(dtype=torch.float32)
    centered = data - data.mean(dim=1, keepdim=True)
    norm = torch.linalg.vector_norm(centered, ord=2, dim=1, keepdim=True).clamp_min(eps)
    normalized = centered / norm
    corr: torch.Tensor = normalized @ normalized.T
    corr = corr.clamp(min=-1.0, max=1.0)
    indices = torch.arange(corr.shape[0], device=corr.device)
    corr[indices, indices] = 1.0
    return corr


def pairwise_correlation_penalty(
    streams: torch.Tensor,
    *,
    threshold: float,
    weight: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Penalize off-diagonal stream correlations above ``threshold``."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in the closed interval [0, 1]")
    if weight < 0.0:
        raise ValueError("weight must be nonnegative")

    corr = correlation_matrix(streams, eps=eps)
    mask = ~torch.eye(corr.shape[0], dtype=torch.bool, device=corr.device)
    excess = torch.relu(corr[mask].abs() - threshold)
    return weight * excess.square().mean()


def correlation_penalty(
    observed: torch.Tensor,
    *,
    target: float,
    weight: float = 1.0,
) -> torch.Tensor:
    """Differentiable mean-square penalty toward a target correlation."""

    if not -1.0 <= target <= 1.0:
        raise ValueError("target must be in the closed interval [-1, 1]")
    if weight < 0.0:
        raise ValueError("weight must be nonnegative")
    if observed.numel() == 0:
        raise ValueError("observed must not be empty")
    if not bool(torch.isfinite(observed.detach()).all()):
        raise ValueError("observed must contain only finite values")

    return weight * (observed - target).square().mean()
