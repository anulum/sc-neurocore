# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for stochastic-computing backpropagation objective

"""Tests for stochastic-computing backpropagation objective helpers."""

import pytest

torch = pytest.importorskip("torch")

import sc_neurocore.training as training
from sc_neurocore.training.sc_estimators import DifferentiableSCConfig
from sc_neurocore.training.stochastic_backprop import (
    SCBackpropDesignSpace,
    SCBackpropJointReport,
    SCResourceProxy,
    SCTrainingObjectiveConfig,
    relaxed_sc_linear,
    stochastic_backprop_joint_objective,
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


def test_joint_objective_backpropagates_through_model_and_sc_design_variables():
    inputs = torch.tensor([[0.2, -0.4], [0.8, 0.1]], dtype=torch.float32)
    targets = torch.tensor([[0.3], [0.6]], dtype=torch.float32)
    weights = torch.tensor([[0.25, -0.35]], dtype=torch.float32, requires_grad=True)
    bias = torch.tensor([0.05], dtype=torch.float32, requires_grad=True)
    length_logits = torch.tensor([0.6, -0.2, 0.1], dtype=torch.float32, requires_grad=True)
    encoding_logits = torch.tensor([0.8, -0.3], dtype=torch.float32, requires_grad=True)
    correlation_logit = torch.tensor(0.2, dtype=torch.float32, requires_grad=True)
    design_space = SCBackpropDesignSpace(
        bitstream_lengths=(64, 128, 256),
        encodings=("bipolar", "unipolar"),
        min_correlation=-0.25,
        max_correlation=0.25,
    )
    objective_config = SCTrainingObjectiveConfig(
        length_cost_weight=1.5,
        encoding_cost_weight=0.3,
    )

    report = stochastic_backprop_joint_objective(
        inputs,
        targets,
        weight=weights,
        bias=bias,
        length_logits=length_logits,
        encoding_logits=encoding_logits,
        correlation_logit=correlation_logit,
        base_config=_sc_config(),
        design_space=design_space,
        objective_config=objective_config,
    )

    assert isinstance(report, SCBackpropJointReport)
    assert report.prediction.shape == targets.shape
    assert report.selected_bitstream_length in design_space.bitstream_lengths
    assert report.selected_encoding in design_space.encodings
    assert report.selected_sc_config.bitstream_length == report.selected_bitstream_length
    assert report.selected_sc_config.encoding == report.selected_encoding
    assert report.expected_bitstream_length.item() > 0.0
    assert torch.isfinite(report.breakdown.total)

    report.breakdown.total.backward()

    for tensor in (weights, bias, length_logits, encoding_logits, correlation_logit):
        assert tensor.grad is not None
        assert tensor.grad.abs().sum().item() > 0.0


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"bitstream_lengths": ()}, "bitstream_lengths"),
        ({"bitstream_lengths": (128, 64)}, "strictly increasing"),
        ({"bitstream_lengths": (64, 64)}, "strictly increasing"),
        ({"encodings": ()}, "encodings"),
        ({"encodings": ("ternary",)}, "encodings"),
        ({"min_correlation": -1.5}, "correlation"),
        ({"min_correlation": 0.4, "max_correlation": 0.2}, "correlation"),
    ],
)
def test_joint_design_space_rejects_invalid_contracts(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SCBackpropDesignSpace(**kwargs)


def test_joint_objective_rejects_logit_shape_mismatches():
    with pytest.raises(ValueError, match="length_logits"):
        stochastic_backprop_joint_objective(
            torch.ones(2, 2),
            torch.ones(2, 1),
            weight=torch.ones(1, 2, requires_grad=True),
            bias=None,
            length_logits=torch.ones(2, requires_grad=True),
            encoding_logits=torch.ones(1, requires_grad=True),
            correlation_logit=torch.tensor(0.0, requires_grad=True),
            base_config=_sc_config(),
            design_space=SCBackpropDesignSpace(bitstream_lengths=(64, 128, 256)),
        )


def test_joint_backprop_api_is_exported_from_training_package():
    assert training.SCBackpropDesignSpace is SCBackpropDesignSpace
    assert training.SCBackpropJointReport is SCBackpropJointReport
    assert training.stochastic_backprop_joint_objective is stochastic_backprop_joint_objective
