# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Stochastic backpropagation benchmark evidence

"""Deterministic benchmark evidence for SC-aware stochastic backpropagation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from sc_neurocore.training.sc_estimators import (
    DifferentiableSCConfig,
    relaxed_sc_multiply,
    sampled_sc_multiply,
)
from sc_neurocore.training.stochastic_backprop import (
    SCResourceProxy,
    SCTrainingObjectiveConfig,
    relaxed_sc_linear,
    stochastic_training_objective,
)

STOCHASTIC_BACKPROP_BENCHMARK_SCHEMA_VERSION = "sc-neurocore.stochastic-backprop-benchmark.v1"


def build_stochastic_backprop_benchmark(
    *,
    bitstream_length: int = 256,
    steps: int = 32,
    learning_rate: float = 0.4,
) -> dict[str, Any]:
    """Return deterministic evidence for SC-aware backpropagation loss reduction."""

    if bitstream_length <= 0:
        raise ValueError("bitstream_length must be a positive integer")
    if steps <= 0:
        raise ValueError("steps must be a positive integer")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")

    torch.manual_seed(20260518)
    sc_config = DifferentiableSCConfig(
        bitstream_length=bitstream_length,
        encoding="bipolar",
        generator="sobol",
        estimator="pathwise_relaxation",
        input_seed=101,
        weight_seed=211,
        correlation=0.0,
    )
    objective_config = SCTrainingObjectiveConfig(
        length_cost_weight=0.01,
        correlation_cost_weight=0.01,
        variance_cost_weight=0.01,
        resource_cost_weight=0.01,
        correlation_threshold=0.2,
    )
    resource_proxy = SCResourceProxy(
        lut=512.0, power_mw=1.5, latency_cycles=float(bitstream_length)
    )

    inputs = torch.tensor(
        [
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    target_weight = torch.tensor([[0.55, -0.35]], dtype=torch.float32)
    target_bias = torch.tensor([0.05], dtype=torch.float32)
    targets = inputs @ target_weight.T + target_bias

    raw_weight = torch.tensor([[0.05, 0.05]], dtype=torch.float32, requires_grad=True)
    raw_bias = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)
    optimiser = torch.optim.SGD([raw_weight, raw_bias], lr=learning_rate)

    initial_weight = torch.tanh(raw_weight.detach()).clone()
    initial_bias = torch.tanh(raw_bias.detach()).clone()
    initial_loss = _task_loss(inputs, targets, initial_weight, initial_bias, sc_config)
    best_loss = initial_loss

    for _ in range(steps):
        optimiser.zero_grad(set_to_none=True)
        weight = torch.tanh(raw_weight)
        bias = torch.tanh(raw_bias)
        prediction = relaxed_sc_linear(inputs, weight, bias, sc_config)
        task_loss = F.mse_loss(prediction, targets)
        sampled = sampled_sc_multiply(
            inputs[:, :1], weight[:, :1].expand_as(inputs[:, :1]), sc_config
        )
        objective = stochastic_training_objective(
            task_loss,
            sc_config=sc_config,
            streams=sampled.product_bits.reshape(-1, sc_config.bitstream_length),
            observed_correlation=sampled.input_statistics.correlation.flatten(),
            resource_proxy=resource_proxy,
            objective_config=objective_config,
        )
        total_for_backward: Any = objective.total
        total_for_backward.backward()
        optimiser.step()
        best_loss = min(best_loss, float(task_loss.detach().item()))

    trained_weight = torch.tanh(raw_weight.detach())
    trained_bias = torch.tanh(raw_bias.detach())
    final_prediction = relaxed_sc_linear(inputs, trained_weight, trained_bias, sc_config)
    final_task_loss = F.mse_loss(final_prediction, targets)
    final_sampled = sampled_sc_multiply(
        inputs[:, :1],
        trained_weight[:, :1].expand_as(inputs[:, :1]),
        sc_config,
    )
    final_relaxed = relaxed_sc_multiply(
        inputs[:, :1], trained_weight[:, :1].expand_as(inputs[:, :1]), sc_config
    )
    final_objective = stochastic_training_objective(
        final_task_loss,
        sc_config=sc_config,
        streams=final_sampled.product_bits.reshape(-1, sc_config.bitstream_length),
        observed_correlation=final_sampled.input_statistics.correlation.flatten(),
        resource_proxy=resource_proxy,
        objective_config=objective_config,
    )

    return {
        "schema_version": STOCHASTIC_BACKPROP_BENCHMARK_SCHEMA_VERSION,
        "evidence_class": "deterministic_training_simulation",
        "hardware_measurement_claimed": False,
        "sc_config": {
            "bitstream_length": sc_config.bitstream_length,
            "encoding": sc_config.encoding,
            "generator": sc_config.generator,
            "estimator": sc_config.estimator,
            "input_seed": sc_config.input_seed,
            "weight_seed": sc_config.weight_seed,
            "correlation": sc_config.correlation,
        },
        "training": {
            "steps": steps,
            "learning_rate": learning_rate,
        },
        "loss": {
            "initial": initial_loss,
            "final": float(final_task_loss.detach().item()),
            "best": best_loss,
        },
        "initial_parameters": {
            "weight": _round_nested(initial_weight.tolist()),
            "bias": _round_nested(initial_bias.tolist()),
        },
        "trained_parameters": {
            "weight": _round_nested(trained_weight.tolist()),
            "bias": _round_nested(trained_bias.tolist()),
        },
        "objective_terms": {
            "length_cost": float(final_objective.length_cost.detach().item()),
            "correlation_cost": float(final_objective.correlation_cost.detach().item()),
            "variance_cost": float(final_objective.variance_cost.detach().item()),
            "resource_cost": float(final_objective.resource_cost.detach().item()),
            "total": float(final_objective.total.detach().item()),
        },
        "stream_evidence": {
            "sampled_product_mae": float(
                (final_sampled.value - final_relaxed.value).abs().mean().item()
            ),
            "input_rate": _round_nested(final_sampled.input_statistics.rate.tolist()),
            "weight_rate": _round_nested(final_sampled.weight_statistics.rate.tolist()),
            "input_max_abs_correlation": float(
                final_sampled.input_statistics.max_abs_off_diagonal_correlation.item()
            ),
            "weight_max_abs_correlation": float(
                final_sampled.weight_statistics.max_abs_off_diagonal_correlation.item()
            ),
        },
    }


def write_stochastic_backprop_benchmark(
    path: str | Path,
    *,
    bitstream_length: int = 256,
    steps: int = 32,
    learning_rate: float = 0.4,
) -> Path:
    """Write a canonical stochastic backpropagation benchmark report."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = build_stochastic_backprop_benchmark(
        bitstream_length=bitstream_length,
        steps=steps,
        learning_rate=learning_rate,
    )
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _task_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    sc_config: DifferentiableSCConfig,
) -> float:
    prediction = relaxed_sc_linear(inputs, weight, bias, sc_config)
    return float(F.mse_loss(prediction, targets).detach().item())


def _round_nested(value: Any) -> Any:
    if isinstance(value, list):
        return [_round_nested(item) for item in value]
    return round(float(value), 8)
