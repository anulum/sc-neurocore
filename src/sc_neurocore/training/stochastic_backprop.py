# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Stochastic-computing backpropagation objective helpers

"""Objective helpers for backpropagation through stochastic-computing paths."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .sc_correlation_regularizers import correlation_penalty, pairwise_correlation_penalty
from .sc_estimators import SC_ENCODINGS, DifferentiableSCConfig, SCEncoding, relaxed_sc_multiply


@dataclass(frozen=True, slots=True)
class SCTrainingObjectiveConfig:
    """Weights and targets for SC-aware training objective components."""

    length_cost_weight: float = 0.0
    correlation_cost_weight: float = 0.0
    variance_cost_weight: float = 0.0
    resource_cost_weight: float = 0.0
    encoding_cost_weight: float = 0.0
    correlation_threshold: float = 0.2
    correlation_target: float = 0.0

    def __post_init__(self) -> None:
        for field_name in (
            "length_cost_weight",
            "correlation_cost_weight",
            "variance_cost_weight",
            "resource_cost_weight",
            "encoding_cost_weight",
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


@dataclass(frozen=True, slots=True)
class SCBackpropDesignSpace:
    """Discrete SC architecture choices exposed to differentiable joint training."""

    bitstream_lengths: tuple[int, ...] = (64, 128, 256)
    encodings: tuple[SCEncoding, ...] = ("bipolar",)
    min_correlation: float = -0.25
    max_correlation: float = 0.25

    def __post_init__(self) -> None:
        if not self.bitstream_lengths:
            raise ValueError("bitstream_lengths must not be empty")
        previous_length = 0
        for length in self.bitstream_lengths:
            if not isinstance(length, int) or length <= 0:
                raise ValueError("bitstream_lengths must contain positive integers")
            if length <= previous_length:
                raise ValueError("bitstream_lengths must be strictly increasing")
            previous_length = length
        if not self.encodings:
            raise ValueError("encodings must not be empty")
        for encoding in self.encodings:
            if encoding not in SC_ENCODINGS:
                raise ValueError(f"encodings must be selected from {SC_ENCODINGS}")
        if not -1.0 <= float(self.min_correlation) <= 1.0:
            raise ValueError("correlation bounds must stay in the closed interval [-1, 1]")
        if not -1.0 <= float(self.max_correlation) <= 1.0:
            raise ValueError("correlation bounds must stay in the closed interval [-1, 1]")
        if float(self.min_correlation) > float(self.max_correlation):
            raise ValueError("min_correlation must not exceed max_correlation")


@dataclass(frozen=True, slots=True)
class SCBackpropJointReport:
    """Joint stochastic-backpropagation result with exportable SC design metadata."""

    prediction: torch.Tensor
    breakdown: SCObjectiveBreakdown
    selected_sc_config: DifferentiableSCConfig
    expected_bitstream_length: torch.Tensor
    length_probabilities: torch.Tensor
    encoding_probabilities: torch.Tensor
    selected_encoding: SCEncoding
    selected_bitstream_length: int
    selected_correlation: torch.Tensor


def _scalar_like(reference: torch.Tensor, value: float) -> torch.Tensor:
    return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)


def _tensor_like(reference: torch.Tensor, values: tuple[float | int, ...]) -> torch.Tensor:
    return torch.as_tensor(values, dtype=reference.dtype, device=reference.device)


def _require_tensor_domain(name: str, value: torch.Tensor, lower: float, upper: float) -> None:
    detached = value.detach()
    if detached.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    if not bool(torch.isfinite(detached).all()):
        raise ValueError(f"{name} must contain only finite values")
    if bool((detached < lower).any() or (detached > upper).any()):
        raise ValueError(f"{name} must be in the closed interval [{lower:g}, {upper:g}]")


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


def _bernoulli_product_expectation_with_correlation_tensor(
    p_input: torch.Tensor,
    p_weight: torch.Tensor,
    correlation: torch.Tensor,
) -> torch.Tensor:
    independent = p_input * p_weight
    eps = torch.finfo(independent.dtype).eps
    variance_scale = torch.sqrt(
        torch.clamp(
            p_input * (1.0 - p_input) * p_weight * (1.0 - p_weight),
            min=eps,
        )
    )
    return torch.clamp(independent + correlation * variance_scale, min=0.0, max=1.0)


def _relaxed_sc_linear_with_correlation_tensor(
    input_value: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    sc_config: DifferentiableSCConfig,
    correlation: torch.Tensor,
) -> torch.Tensor:
    if sc_config.encoding == "unipolar":
        _require_tensor_domain("unipolar input_value", input_value, 0.0, 1.0)
        _require_tensor_domain("unipolar weight", weight, 0.0, 1.0)
        p_input = input_value.unsqueeze(-2)
        p_weight = weight
        products = _bernoulli_product_expectation_with_correlation_tensor(
            p_input,
            p_weight,
            correlation,
        )
    else:
        _require_tensor_domain("bipolar input_value", input_value, -1.0, 1.0)
        _require_tensor_domain("bipolar weight", weight, -1.0, 1.0)
        p_input = (input_value.unsqueeze(-2) + 1.0) * 0.5
        p_weight = (weight + 1.0) * 0.5
        unipolar_product = _bernoulli_product_expectation_with_correlation_tensor(
            p_input,
            p_weight,
            correlation,
        )
        products = 4.0 * unipolar_product - 2.0 * p_input - 2.0 * p_weight + 1.0

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


def stochastic_backprop_joint_objective(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    length_logits: torch.Tensor,
    encoding_logits: torch.Tensor,
    correlation_logit: torch.Tensor,
    base_config: DifferentiableSCConfig,
    design_space: SCBackpropDesignSpace | None = None,
    resource_proxy: SCResourceProxy | None = None,
    objective_config: SCTrainingObjectiveConfig | None = None,
) -> SCBackpropJointReport:
    """Train a relaxed SC layer jointly over weights and SC architecture variables.

    The selected config remains discrete and exportable, while length, encoding,
    and correlation costs are computed from differentiable proxies so one
    objective can update model parameters and stochastic-computing design
    variables in the same backward pass.
    """

    if targets.shape[:-1] != inputs.shape[:-1]:
        raise ValueError("targets batch dimensions must match inputs batch dimensions")
    if weight.ndim != 2:
        raise ValueError("weight must be a 2D tensor shaped (out_features, in_features)")
    if inputs.shape[-1] != weight.shape[-1]:
        raise ValueError("inputs last dimension must match weight in_features")
    if targets.shape[-1] != weight.shape[0]:
        raise ValueError("targets last dimension must match weight out_features")
    if bias is not None and bias.shape != (weight.shape[0],):
        raise ValueError("bias must be shaped (out_features,)")
    if length_logits.ndim != 1:
        raise ValueError("length_logits must be a 1D tensor")
    if encoding_logits.ndim != 1:
        raise ValueError("encoding_logits must be a 1D tensor")
    if correlation_logit.ndim > 0 and correlation_logit.numel() != 1:
        raise ValueError("correlation_logit must be scalar")

    cfg = objective_config or SCTrainingObjectiveConfig()
    space = design_space or SCBackpropDesignSpace(encodings=(base_config.encoding,))
    if length_logits.shape[0] != len(space.bitstream_lengths):
        raise ValueError("length_logits size must match design_space.bitstream_lengths")
    if encoding_logits.shape[0] != len(space.encodings):
        raise ValueError("encoding_logits size must match design_space.encodings")

    reference = weight
    length_probs = torch.softmax(length_logits.to(dtype=reference.dtype, device=reference.device), dim=0)
    length_values = _tensor_like(reference, space.bitstream_lengths)
    expected_length = (length_probs * length_values).sum()
    selected_length_index = int(torch.argmax(length_probs.detach()).item())
    selected_length = space.bitstream_lengths[selected_length_index]

    encoding_probs = torch.softmax(
        encoding_logits.to(dtype=reference.dtype, device=reference.device),
        dim=0,
    )
    encoding_indices = torch.arange(
        len(space.encodings),
        dtype=reference.dtype,
        device=reference.device,
    )
    selected_encoding_index = int(torch.argmax(encoding_probs.detach()).item())
    selected_encoding = space.encodings[selected_encoding_index]

    correlation_unit = torch.sigmoid(correlation_logit.to(dtype=reference.dtype, device=reference.device))
    min_correlation = _scalar_like(reference, float(space.min_correlation))
    max_correlation = _scalar_like(reference, float(space.max_correlation))
    selected_correlation = min_correlation + correlation_unit.reshape(()) * (
        max_correlation - min_correlation
    )

    selected_config = DifferentiableSCConfig(
        bitstream_length=selected_length,
        encoding=selected_encoding,
        generator=base_config.generator,
        estimator=base_config.estimator,
        input_seed=base_config.input_seed,
        weight_seed=base_config.weight_seed,
        correlation=float(selected_correlation.detach().cpu().item()),
        lfsr_polynomial=base_config.lfsr_polynomial,
    )
    bounded_weight = weight.sigmoid() if selected_encoding == "unipolar" else weight.tanh()
    prediction = _relaxed_sc_linear_with_correlation_tensor(
        inputs,
        bounded_weight,
        bias,
        selected_config,
        selected_correlation,
    )
    task_loss = F.mse_loss(prediction, targets)
    base_breakdown = stochastic_training_objective(
        task_loss,
        sc_config=selected_config,
        resource_proxy=resource_proxy,
        objective_config=cfg,
    )

    differentiable_length_cost = cfg.length_cost_weight / expected_length.clamp_min(
        torch.finfo(expected_length.dtype).eps
    )
    differentiable_correlation_cost = cfg.correlation_cost_weight * correlation_penalty(
        selected_correlation,
        target=cfg.correlation_target,
    ).to(dtype=reference.dtype, device=reference.device)
    encoding_cost = cfg.encoding_cost_weight * (encoding_probs * encoding_indices).sum()
    total = (
        task_loss
        + differentiable_length_cost
        + differentiable_correlation_cost
        + encoding_cost
        + base_breakdown.resource_cost
    )

    breakdown = SCObjectiveBreakdown(
        task_loss=task_loss,
        length_cost=differentiable_length_cost,
        correlation_cost=differentiable_correlation_cost,
        variance_cost=encoding_cost,
        resource_cost=base_breakdown.resource_cost,
        total=total,
    )
    return SCBackpropJointReport(
        prediction=prediction,
        breakdown=breakdown,
        selected_sc_config=selected_config,
        expected_bitstream_length=expected_length,
        length_probabilities=length_probs,
        encoding_probabilities=encoding_probs,
        selected_encoding=selected_encoding,
        selected_bitstream_length=selected_length,
        selected_correlation=selected_correlation,
    )
