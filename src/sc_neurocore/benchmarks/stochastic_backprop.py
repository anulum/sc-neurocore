# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Stochastic backpropagation benchmark evidence

"""Deterministic benchmark evidence for SC-aware stochastic backpropagation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F

from sc_neurocore.training.sc_estimators import (
    DifferentiableSCConfig,
    relaxed_sc_multiply,
    sampled_sc_multiply,
)
from sc_neurocore.training.stochastic_backprop import (
    SCBackpropDesignSpace,
    SCBackpropJointReport,
    SCResourceProxy,
    SCTrainingObjectiveConfig,
    relaxed_sc_linear,
    stochastic_backprop_joint_objective,
    stochastic_training_objective,
)

STOCHASTIC_BACKPROP_BENCHMARK_SCHEMA_VERSION = "sc-neurocore.stochastic-backprop-benchmark.v1"
STOCHASTIC_BACKPROP_ESTIMATOR_REGRESSION_SCHEMA_VERSION = (
    "sc-neurocore.stochastic-backprop-estimator-regression.v1"
)
STOCHASTIC_BACKPROP_EVIDENCE_BOUNDARY = "local_simulation_and_executable_hdl_parity"


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
        encoding_cost_weight=0.01,
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
    length_options = _design_length_options(bitstream_length)
    design_space = SCBackpropDesignSpace(
        bitstream_lengths=length_options,
        encodings=("bipolar", "unipolar"),
        min_correlation=-0.15,
        max_correlation=0.15,
    )
    length_logits = torch.tensor([0.0, 2.0, -1.0], dtype=torch.float32, requires_grad=True)
    encoding_logits = torch.tensor([2.0, -1.0], dtype=torch.float32, requires_grad=True)
    correlation_logit = torch.tensor(0.1, dtype=torch.float32, requires_grad=True)
    optimiser = torch.optim.SGD(
        [raw_weight, raw_bias, length_logits, encoding_logits, correlation_logit],
        lr=learning_rate,
    )

    initial_weight = torch.tanh(raw_weight.detach()).clone()
    initial_bias = torch.tanh(raw_bias.detach()).clone()
    initial_report = stochastic_backprop_joint_objective(
        inputs,
        targets,
        weight=raw_weight,
        bias=raw_bias,
        length_logits=length_logits,
        encoding_logits=encoding_logits,
        correlation_logit=correlation_logit,
        base_config=sc_config,
        design_space=design_space,
        resource_proxy=resource_proxy,
        objective_config=objective_config,
    )
    initial_loss = float(initial_report.breakdown.task_loss.detach().item())
    best_loss = initial_loss

    for _ in range(steps):
        optimiser.zero_grad(set_to_none=True)
        joint_objective = stochastic_backprop_joint_objective(
            inputs,
            targets,
            weight=raw_weight,
            bias=raw_bias,
            length_logits=length_logits,
            encoding_logits=encoding_logits,
            correlation_logit=correlation_logit,
            base_config=sc_config,
            design_space=design_space,
            resource_proxy=resource_proxy,
            objective_config=objective_config,
        )
        weight = torch.tanh(raw_weight)
        sampled = sampled_sc_multiply(
            inputs[:, :1],
            weight[:, :1].expand_as(inputs[:, :1]),
            joint_objective.selected_sc_config,
        )
        objective = stochastic_training_objective(
            joint_objective.breakdown.task_loss,
            sc_config=joint_objective.selected_sc_config,
            streams=sampled.product_bits.reshape(
                -1,
                joint_objective.selected_sc_config.bitstream_length,
            ),
            observed_correlation=sampled.input_statistics.correlation.flatten(),
            resource_proxy=resource_proxy,
            objective_config=objective_config,
        )
        total_for_backward: Any = joint_objective.breakdown.total + (
            objective.correlation_cost + objective.variance_cost
        )
        total_for_backward.backward()
        optimiser.step()
        with torch.no_grad():
            raw_weight.clamp_(-3.0, 3.0)
            raw_bias.clamp_(-3.0, 3.0)
            length_logits.clamp_(-8.0, 8.0)
            encoding_logits.clamp_(-8.0, 8.0)
            correlation_logit.clamp_(-8.0, 8.0)
        best_loss = min(best_loss, float(joint_objective.breakdown.task_loss.detach().item()))

    trained_weight = torch.tanh(raw_weight.detach())
    trained_bias = torch.tanh(raw_bias.detach())
    final_joint_report = stochastic_backprop_joint_objective(
        inputs,
        targets,
        weight=raw_weight,
        bias=raw_bias,
        length_logits=length_logits,
        encoding_logits=encoding_logits,
        correlation_logit=correlation_logit,
        base_config=sc_config,
        design_space=design_space,
        resource_proxy=resource_proxy,
        objective_config=objective_config,
    )
    sc_config = final_joint_report.selected_sc_config
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
        "evidence_boundary": STOCHASTIC_BACKPROP_EVIDENCE_BOUNDARY,
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
        "joint_design": {
            "enabled": True,
            "design_space": {
                "bitstream_lengths": list(design_space.bitstream_lengths),
                "encodings": list(design_space.encodings),
                "min_correlation": design_space.min_correlation,
                "max_correlation": design_space.max_correlation,
            },
            "initial": _joint_design_snapshot(initial_report),
            "final": _joint_design_snapshot(final_joint_report),
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
        "estimator_variance": _estimator_variance_evidence(
            bitstream_length=sc_config.bitstream_length
        ),
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


def build_stochastic_backprop_estimator_regression_manifest(
    *,
    bitstream_lengths: tuple[int, ...] = (64, 128, 256),
    sample_count: int = 32,
) -> dict[str, Any]:
    """Return seeded estimator-family variance evidence across bitstream lengths."""

    _validate_bitstream_length_grid(bitstream_lengths)
    if sample_count < 2:
        raise ValueError("sample_count must be at least two")

    results = [
        _estimator_variance_evidence(
            bitstream_length=bitstream_length,
            sample_count=sample_count,
        )
        for bitstream_length in bitstream_lengths
    ]
    score_variances = [row["estimators"]["score_function"]["variance"] for row in results]
    pathwise_variances = [row["estimators"]["pathwise_relaxation"]["variance"] for row in results]
    acceptance = {
        "score_function_longest_variance_below_shortest": (
            score_variances[0] > score_variances[-1]
        ),
        "pathwise_variance_zero": all(variance == 0.0 for variance in pathwise_variances),
        "all_variances_finite_nonnegative": all(
            _all_estimator_variances_are_finite_nonnegative(row) for row in results
        ),
    }
    return {
        "SPDX-License-Identifier": "AGPL-3.0-or-later",
        "schema_version": STOCHASTIC_BACKPROP_ESTIMATOR_REGRESSION_SCHEMA_VERSION,
        "evidence_class": "deterministic_estimator_regression",
        "evidence_boundary": STOCHASTIC_BACKPROP_EVIDENCE_BOUNDARY,
        "hardware_measurement_claimed": False,
        "status": "pass" if all(acceptance.values()) else "fail",
        "bitstream_lengths": list(bitstream_lengths),
        "sample_count": sample_count,
        "estimators": [
            "pathwise_relaxation",
            "straight_through",
            "score_function",
        ],
        "acceptance": acceptance,
        "results": results,
    }


def write_stochastic_backprop_estimator_regression_manifest(
    path: str | Path,
    *,
    bitstream_lengths: tuple[int, ...] = (64, 128, 256),
    sample_count: int = 32,
) -> Path:
    """Write seeded estimator-family regression evidence to canonical JSON."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = build_stochastic_backprop_estimator_regression_manifest(
        bitstream_lengths=bitstream_lengths,
        sample_count=sample_count,
    )
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _design_length_options(bitstream_length: int) -> tuple[int, int, int]:
    low = max(2, bitstream_length // 2)
    if low >= bitstream_length:
        low = bitstream_length
    high = bitstream_length * 2
    options = tuple(sorted({low, bitstream_length, high}))
    if len(options) == 3:
        return options
    return (bitstream_length, bitstream_length + 1, bitstream_length + 2)


def _validate_bitstream_length_grid(bitstream_lengths: tuple[int, ...]) -> None:
    if len(bitstream_lengths) < 2:
        raise ValueError("bitstream_lengths must contain at least two entries")
    previous_length = 0
    for bitstream_length in bitstream_lengths:
        if not isinstance(bitstream_length, int) or bitstream_length <= 0:
            raise ValueError("bitstream_lengths must contain positive integers")
        if bitstream_length <= previous_length:
            raise ValueError("bitstream_lengths must be strictly increasing")
        previous_length = bitstream_length


def _joint_design_snapshot(report: SCBackpropJointReport) -> dict[str, Any]:
    return {
        "selected_bitstream_length": report.selected_bitstream_length,
        "expected_bitstream_length": round(
            float(report.expected_bitstream_length.detach().item()),
            8,
        ),
        "length_probabilities": _round_nested(report.length_probabilities.detach().cpu().tolist()),
        "selected_encoding": report.selected_encoding,
        "encoding_probabilities": _round_nested(
            report.encoding_probabilities.detach().cpu().tolist()
        ),
        "correlation": float(report.selected_sc_config.correlation),
    }


def _estimator_variance_evidence(
    *, bitstream_length: int, sample_count: int = 32
) -> dict[str, Any]:
    if sample_count < 2:
        raise ValueError("sample_count must be at least two")

    input_probability = torch.tensor(0.65, dtype=torch.float32)
    raw_weight = torch.tensor(0.2, dtype=torch.float32, requires_grad=True)
    weight_probability = torch.sigmoid(raw_weight)
    target = torch.tensor(0.32, dtype=torch.float32)
    expected_product = input_probability * weight_probability
    reference_loss = (expected_product - target).square()
    cast(Any, reference_loss).backward()
    raw_weight_grad = raw_weight.grad
    if raw_weight_grad is None:
        raise RuntimeError("reference gradient was not populated")
    pathwise_gradient = float(raw_weight_grad.detach().item())

    pathwise_estimates = [pathwise_gradient for _ in range(sample_count)]
    straight_through_estimates: list[float] = []
    score_function_estimates: list[float] = []
    for sample_index in range(sample_count):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(20260520 + sample_index)
        input_bits = (
            torch.rand(bitstream_length, generator=generator, dtype=torch.float32)
            < input_probability
        ).to(dtype=torch.float32)
        weight_bits = (
            torch.rand(bitstream_length, generator=generator, dtype=torch.float32)
            < weight_probability.detach()
        ).to(dtype=torch.float32)
        sampled_product = (input_bits * weight_bits).mean()
        probability_gradient = weight_probability.detach() * (1.0 - weight_probability.detach())
        straight_through_gradient = (
            2.0 * (sampled_product - target) * input_probability * probability_gradient
        )
        score_gradient = (
            (sampled_product - target).square()
            * (
                (weight_bits - weight_probability.detach())
                / (weight_probability.detach() * (1.0 - weight_probability.detach()))
            ).sum()
            * probability_gradient
        )
        straight_through_estimates.append(float(straight_through_gradient.item()))
        score_function_estimates.append(float(score_gradient.item()))

    return {
        "sample_count": sample_count,
        "bitstream_length": bitstream_length,
        "reference": {
            "estimator": "pathwise_relaxation",
            "gradient": round(pathwise_gradient, 12),
            "loss": round(float(reference_loss.detach().item()), 12),
        },
        "estimators": {
            "pathwise_relaxation": _gradient_estimator_stats(
                pathwise_estimates,
                reference_gradient=pathwise_gradient,
                assumption="differentiable relaxed Bernoulli expectation",
            ),
            "straight_through": _gradient_estimator_stats(
                straight_through_estimates,
                reference_gradient=pathwise_gradient,
                assumption="sampled forward pass with relaxed backward proxy",
            ),
            "score_function": _gradient_estimator_stats(
                score_function_estimates,
                reference_gradient=pathwise_gradient,
                assumption="seeded REINFORCE likelihood-ratio estimate without baseline",
            ),
        },
    }


def _gradient_estimator_stats(
    estimates: list[float],
    *,
    reference_gradient: float,
    assumption: str,
) -> dict[str, Any]:
    values = torch.tensor(estimates, dtype=torch.float64)
    mean = float(values.mean().item())
    variance = float(values.var(unbiased=False).item())
    return {
        "assumption": assumption,
        "mean": round(mean, 12),
        "variance": round(variance, 12),
        "absolute_bias": round(abs(mean - reference_gradient), 12),
        "min": round(float(values.min().item()), 12),
        "max": round(float(values.max().item()), 12),
    }


def _all_estimator_variances_are_finite_nonnegative(row: dict[str, Any]) -> bool:
    for estimator in row["estimators"].values():
        variance = estimator["variance"]
        if not isinstance(variance, int | float):
            return False
        if not math.isfinite(float(variance)) or variance < 0.0:
            return False
    return True


def _round_nested(value: Any) -> Any:
    if isinstance(value, list):
        return [_round_nested(item) for item in value]
    return round(float(value), 8)
