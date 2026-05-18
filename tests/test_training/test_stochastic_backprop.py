# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Tests for stochastic-computing backpropagation objective

"""Tests for stochastic-computing backpropagation objective helpers."""

import pytest

torch = pytest.importorskip("torch")

from sc_neurocore.training.sc_estimators import DifferentiableSCConfig
from sc_neurocore.training.stochastic_backprop import (
    SCResourceProxy,
    SCTrainingObjectiveConfig,
    relaxed_sc_linear,
    stochastic_training_objective,
)


def _sc_config(**overrides):
    values = {
        "bitstream_length": 128,
        "encoding": "bipolar",
        "generator": "sobol",
        "estimator": "pathwise_relaxation",
        "input_seed": 3,
        "weight_seed": 11,
        "correlation": 0.0,
    }
    values.update(overrides)
    return DifferentiableSCConfig(**values)


def test_stochastic_objective_combines_weighted_costs_and_backpropagates():
    model_loss = torch.tensor(0.5, requires_grad=True)
    observed_correlation = torch.tensor([0.4, -0.3], requires_grad=True)
    streams = torch.tensor(
        [
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
        ]
    )
    objective_config = SCTrainingObjectiveConfig(
        length_cost_weight=2.0,
        correlation_cost_weight=0.5,
        variance_cost_weight=0.25,
        resource_cost_weight=0.1,
        correlation_threshold=0.1,
    )
    resource_proxy = SCResourceProxy(lut=100.0, power_mw=10.0, latency_cycles=4.0)

    breakdown = stochastic_training_objective(
        task_loss=model_loss,
        sc_config=_sc_config(),
        streams=streams,
        observed_correlation=observed_correlation,
        resource_proxy=resource_proxy,
        objective_config=objective_config,
    )

    assert breakdown.total.item() > model_loss.item()
    assert breakdown.length_cost.item() == pytest.approx(2.0 / 128.0)
    assert breakdown.correlation_cost.item() > 0.0

    breakdown.total.backward()
    assert model_loss.grad.item() == pytest.approx(1.0)
    assert observed_correlation.grad is not None


def test_relaxed_sc_linear_matches_linear_for_independent_bipolar_path():
    cfg = _sc_config(correlation=0.0)
    inputs = torch.tensor([[0.25, -0.5], [0.75, 0.5]])
    weights = torch.tensor([[0.4, -0.2], [-0.3, 0.6]])
    bias = torch.tensor([0.1, -0.2])

    out = relaxed_sc_linear(inputs, weights, bias, cfg)

    assert torch.allclose(out, inputs @ weights.T + bias)


def test_relaxed_sc_linear_backpropagates_through_weights():
    cfg = _sc_config(correlation=0.0)
    inputs = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
    weights = torch.tensor([[0.1, -0.2]], requires_grad=True)
    target = torch.tensor([[0.8], [-0.8]])

    prediction = relaxed_sc_linear(inputs, weights, None, cfg)
    loss = torch.nn.functional.mse_loss(prediction, target)
    loss.backward()

    assert weights.grad is not None
    assert weights.grad.abs().sum().item() > 0.0


@pytest.mark.parametrize(
    "field",
    [
        "length_cost_weight",
        "correlation_cost_weight",
        "variance_cost_weight",
        "resource_cost_weight",
    ],
)
def test_objective_config_rejects_negative_weights(field):
    with pytest.raises(ValueError, match=field):
        SCTrainingObjectiveConfig(**{field: -0.1})


def test_resource_proxy_rejects_negative_components():
    with pytest.raises(ValueError, match="lut"):
        SCResourceProxy(lut=-1.0, power_mw=0.0, latency_cycles=0.0)


def test_stochastic_objective_rejects_non_scalar_task_loss():
    with pytest.raises(ValueError, match="task_loss"):
        stochastic_training_objective(torch.ones(2), sc_config=_sc_config())
