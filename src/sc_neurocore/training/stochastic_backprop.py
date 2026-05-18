# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Stochastic-computing backpropagation objective helpers

"""Objective helpers for backpropagation through stochastic-computing paths."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .sc_correlation_regularizers import correlation_penalty, pairwise_correlation_penalty
from .sc_estimators import DifferentiableSCConfig, relaxed_sc_multiply


@dataclass(frozen=True, slots=True)
class SCTrainingObjectiveConfig:
    """Weights and targets for SC-aware training objective components."""

    length_cost_weight: float = 0.0
    correlation_cost_weight: float = 0.0
    variance_cost_weight: float = 0.0
    resource_cost_weight: float = 0.0
    correlation_threshold: float = 0.2
    correlation_target: float = 0.0

    def __post_init__(self) -> None:
        for field_name in (
            "length_cost_weight",
            "correlation_cost_weight",
            "variance_cost_weight",
            "resource_cost_weight",
        ):
            if getattr(self, field_name) < 0.0:
                raise ValueError(f"{field_name} must be nonnegative")
        if not 0.0 <= self.correlation_threshold <= 1.0:
            raise ValueError("correlation_threshold must be in the closed interval [0, 1]")
        if not -1.0 <= self.correlation_target <= 1.0:
            raise ValueError("correlation_target must be in the closed interval [-1, 1]")


@dataclass(frozen=True, slots=True)
class SCResourceProxy:
    """Hardware proxy values used in SC-aware objective shaping."""

    lut: float = 0.0
    power_mw: float = 0.0
    latency_cycles: float = 0.0

    def __post_init__(self) -> None:
        for field_name in ("lut", "power_mw", "latency_cycles"):
            if getattr(self, field_name) < 0.0:
                raise ValueError(f"{field_name} must be nonnegative")

    def normalized_cost(self) -> float:
        """Return a dimensionless bounded-scale resource proxy."""

        return self.lut / 100_000.0 + self.power_mw / 10_000.0 + self.latency_cycles / 1_000_000.0


@dataclass(frozen=True, slots=True)
class SCObjectiveBreakdown:
    """Named scalar components of an SC-aware training objective."""

    task_loss: torch.Tensor
    length_cost: torch.Tensor
    correlation_cost: torch.Tensor
    variance_cost: torch.Tensor
    resource_cost: torch.Tensor
    total: torch.Tensor


def _scalar_like(reference: torch.Tensor, value: float) -> torch.Tensor:
    return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)


def relaxed_sc_linear(
    input_value: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    sc_config: DifferentiableSCConfig,
) -> torch.Tensor:
    """Linear layer whose multiply-accumulate path uses relaxed SC products."""

    if input_value.ndim < 1:
        raise ValueError("input_value must have at least one dimension")
    if weight.ndim != 2:
        raise ValueError("weight must be a 2D tensor shaped (out_features, in_features)")
    if input_value.shape[-1] != weight.shape[-1]:
        raise ValueError("input_value last dimension must match weight in_features")
    if bias is not None and bias.shape != (weight.shape[0],):
        raise ValueError("bias must be shaped (out_features,)")

    products = relaxed_sc_multiply(
        input_value.unsqueeze(-2),
        weight,
        sc_config,
    ).value
    output = products.sum(dim=-1)
    if bias is not None:
        output = output + bias
    return output


def stochastic_training_objective(
    task_loss: torch.Tensor,
    *,
    sc_config: DifferentiableSCConfig,
    streams: torch.Tensor | None = None,
    observed_correlation: torch.Tensor | None = None,
    resource_proxy: SCResourceProxy | None = None,
    objective_config: SCTrainingObjectiveConfig | None = None,
) -> SCObjectiveBreakdown:
    """Compose task loss with SC length, correlation, variance, and resource costs."""

    if task_loss.ndim != 0:
        raise ValueError("task_loss must be a scalar tensor")
    if not bool(torch.isfinite(task_loss.detach())):
        raise ValueError("task_loss must be finite")

    cfg = objective_config or SCTrainingObjectiveConfig()
    length_cost = _scalar_like(
        task_loss, cfg.length_cost_weight / float(sc_config.bitstream_length)
    )

    correlation_cost = _scalar_like(task_loss, 0.0)
    if streams is not None:
        correlation_cost = (
            correlation_cost
            + cfg.correlation_cost_weight
            * pairwise_correlation_penalty(
                streams,
                threshold=cfg.correlation_threshold,
            ).to(dtype=task_loss.dtype, device=task_loss.device)
        )
    if observed_correlation is not None:
        correlation_cost = correlation_cost + cfg.correlation_cost_weight * correlation_penalty(
            observed_correlation,
            target=cfg.correlation_target,
        ).to(dtype=task_loss.dtype, device=task_loss.device)

    variance_cost = _scalar_like(task_loss, 0.0)
    if streams is not None:
        stream_variance = streams.to(dtype=task_loss.dtype, device=task_loss.device).var(
            dim=1,
            unbiased=False,
        )
        variance_cost = (
            cfg.variance_cost_weight * stream_variance.mean() / float(sc_config.bitstream_length)
        )

    proxy = resource_proxy or SCResourceProxy()
    resource_cost = _scalar_like(task_loss, cfg.resource_cost_weight * proxy.normalized_cost())

    total = task_loss + length_cost + correlation_cost + variance_cost + resource_cost
    return SCObjectiveBreakdown(
        task_loss=task_loss,
        length_cost=length_cost,
        correlation_cost=correlation_cost,
        variance_cost=variance_cost,
        resource_cost=resource_cost,
        total=total,
    )
