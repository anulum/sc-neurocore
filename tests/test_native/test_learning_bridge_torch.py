# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Torch learning-layer tests

"""Tests for Torch parameters, input safety, quantisation, and state."""

from __future__ import annotations

import numpy as np
import pytest
import warnings

torch = pytest.importorskip("torch")

from sc_neurocore._native.learning_torch import TorchRuleLayer
from sc_neurocore._native.learning_torch_precision import (
    normalise_bit_spec,
    normalise_clip,
    quantise_tensor,
)
from sc_neurocore._native.learning_torch_support import rule_parameters, validate_input
from sc_neurocore._native.learning_validation import (
    RULE_BCM,
    RULE_ELIGENT,
    RULE_REWARD_STDP,
    RULE_STDP,
)


@pytest.mark.parametrize("rule_type", [RULE_STDP, RULE_REWARD_STDP, RULE_BCM, RULE_ELIGENT])
def test_rule_parameter_defaults_are_finite(rule_type: int) -> None:
    parameters = rule_parameters(rule_type, 0.01, 0.012, {})
    assert len(parameters) == 5
    assert np.all(np.isfinite(parameters))
    assert parameters[2] > 0.0 and parameters[3] > 0.0 and parameters[4] > 0.0


def test_rule_parameter_overrides_and_common_tau() -> None:
    common = rule_parameters(RULE_STDP, 0.1, 0.2, {"tau": 4.0, "param_a_minus": 0.3})
    split = rule_parameters(
        RULE_REWARD_STDP,
        0.1,
        0.2,
        {"tau_plus": 5.0, "tau_minus": 6.0, "tau_e": 7.0},
    )
    assert common == pytest.approx([0.1, 0.3, 4.0, 4.0, 1.0])
    assert split == pytest.approx([0.1, 0.05, 5.0, 6.0, 7.0])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"param_a_minus": -1.0},
        {"tau": 0.0},
        {"tau_plus": float("nan")},
        {"tau_minus": -1.0},
        {"tau_e": 0.0},
    ],
)
def test_rule_parameter_overrides_reject_invalid_values(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        rule_parameters(RULE_STDP, 0.1, 0.2, kwargs)


def test_validate_input_moves_dtype_and_checks_shape() -> None:
    values = torch.tensor([0, 1, 0], dtype=torch.int64)
    result = validate_input(
        values,
        name="spikes",
        count=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
        probability=True,
    )
    assert result.dtype == torch.float32
    with pytest.raises(ValueError, match="shape"):
        validate_input(
            values.reshape(1, 3),
            name="spikes",
            count=3,
            device=torch.device("cpu"),
            dtype=torch.float32,
            probability=True,
        )


@pytest.mark.parametrize("values", [[0.0, float("nan")], [-0.1, 0.0], [0.0, 1.1]])
def test_validate_input_rejects_unsafe_values(values: list[float]) -> None:
    with pytest.raises(ValueError):
        validate_input(
            torch.tensor(values),
            name="spikes",
            count=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
            probability=True,
        )


def test_validate_input_allows_unbounded_finite_rewards() -> None:
    result = validate_input(
        torch.tensor([-2.0, 3.0]),
        name="rewards",
        count=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        probability=False,
    )
    assert result.tolist() == [-2.0, 3.0]


@pytest.mark.parametrize(
    "spec",
    [
        4,
        [2, 3, 4],
        np.array([2, 3, 4], dtype=np.int32),
        np.array([2.0, 3.0, 4.0]),
        torch.tensor([2, 3, 4]),
        torch.tensor([2.0, 3.0, 4.0]),
    ],
)
def test_bit_specs_accept_integral_scalar_and_vectors(spec: object) -> None:
    result = normalise_bit_spec(spec, count=3, device=torch.device("cpu"), field="weight_bits")
    assert result is not None
    assert result.dtype == torch.int64 and result.numel() == 3


def test_bit_spec_none_disables_quantisation() -> None:
    assert normalise_bit_spec(None, count=3, device=torch.device("cpu"), field="bits") is None


@pytest.mark.parametrize(
    "spec",
    [
        True,
        [2, True, 4],
        np.array([True, False]),
        np.array(["2", "3"]),
        np.array([2.5, 3.0]),
        np.array([np.nan, 3.0]),
        torch.tensor([True, False]),
        torch.tensor([2.5, 3.0]),
        torch.tensor([float("inf"), 3.0]),
    ],
)
def test_bit_specs_reject_non_integral_values(spec: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        normalise_bit_spec(spec, count=2, device=torch.device("cpu"), field="bits")


@pytest.mark.parametrize("spec", [[2], [2, 3], 1, 32])
def test_bit_specs_enforce_shape_and_bounds(spec: object) -> None:
    expected_error = ValueError
    if spec == [2]:
        result = normalise_bit_spec(spec, count=3, device=torch.device("cpu"), field="bits")
        assert result is not None and result.tolist() == [2, 2, 2]
        return
    with pytest.raises(expected_error):
        normalise_bit_spec(spec, count=3, device=torch.device("cpu"), field="bits")


def test_clip_and_quantisation_helpers() -> None:
    assert normalise_clip(1, field="clip") == 1.0
    for value in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            normalise_clip(value, field="clip")
    values = torch.tensor([-2.0, 0.37, 2.0])
    assert quantise_tensor(values, None, 1.0) is values
    bits = torch.tensor([3, 3, 3])
    quantised = quantise_tensor(values, bits, 1.0)
    assert torch.all(quantised >= -1.0) and torch.all(quantised <= 1.0)


def test_torch_layer_initialises_all_state_and_precision() -> None:
    layer = TorchRuleLayer(
        3,
        rule_type=RULE_ELIGENT,
        weight=0.4,
        mixed_precision_bits=4,
        trace_bits=[3, 4, 5],
    )
    assert layer.weights.tolist() == pytest.approx([0.4] * 3)
    assert layer.theta_m.tolist() == pytest.approx([1.0] * 3)
    assert layer.act_avg.tolist() == pytest.approx([0.4] * 3)
    assert layer._weight_bits.tolist() == [4, 4, 4]
    assert layer._trace_bits.tolist() == [3, 4, 5]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"count": 0},
        {"rule_type": 9},
        {"weight": -1.0},
        {"autograd": 1},
        {"count": 3, "unexpected": 1},
    ],
)
def test_torch_layer_rejects_invalid_constructor_options(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        TorchRuleLayer(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("rule_type", [RULE_STDP, RULE_REWARD_STDP, RULE_BCM, RULE_ELIGENT])
def test_non_autograd_forward_and_reset(rule_type: int) -> None:
    layer = TorchRuleLayer(3, rule_type=rule_type, autograd=False, weight_bits=4)
    pre = torch.tensor([1.0, 0.0, 1.0])
    post = torch.tensor([0.0, 1.0, 1.0])
    rewards = torch.tensor([0.2, -0.1, 0.3])
    result = layer.forward(pre, post, rewards, dt=1.0)
    assert result is layer.weights and torch.all(torch.isfinite(result))
    layer.reset()
    if rule_type == RULE_STDP:
        assert torch.count_nonzero(layer.pre_trace) == 0
    elif rule_type == RULE_REWARD_STDP:
        assert torch.count_nonzero(layer.eligibility) == 0
    elif rule_type == RULE_BCM:
        assert torch.all(layer.theta_m == 0.5)
    else:
        assert torch.count_nonzero(layer.eligibility) == 0


def test_forward_warning_validation_numpy_step_and_state() -> None:
    layer = TorchRuleLayer(3, rule_type=RULE_REWARD_STDP)
    with pytest.warns(UserWarning, match="expects 'rewards'"):
        layer.forward(torch.ones(3), torch.zeros(3), rewards=None)
    layer.step(np.ones(3), np.zeros(3), np.array([0.1, 0.2, 0.3]))
    state = layer.get_state_dict()
    clone = TorchRuleLayer(3, rule_type=RULE_REWARD_STDP)
    clone.load_state_dict(state)
    assert clone.get_weights() == pytest.approx(layer.get_weights())
    assert clone.get_weights() is not layer.get_weights()


def test_forward_without_reward_is_quiet_for_unsupervised_rule() -> None:
    layer = TorchRuleLayer(3, rule_type=RULE_STDP)
    with warnings.catch_warnings(record=True) as captured:
        layer.forward(torch.ones(3), torch.zeros(3), rewards=None)
    assert not captured


def test_forward_rejects_shape_values_and_timestep() -> None:
    layer = TorchRuleLayer(3)
    with pytest.raises(ValueError, match="shape"):
        layer.forward(torch.ones(2), torch.ones(3))
    with pytest.raises(ValueError, match="values in"):
        layer.forward(torch.ones(3) * 2, torch.ones(3))
    with pytest.raises(ValueError, match="finite"):
        layer.forward(torch.ones(3), torch.ones(3), torch.tensor([0.0, np.nan, 0.0]))
    with pytest.raises(ValueError, match="dt"):
        layer.forward(torch.ones(3), torch.ones(3), dt=0.0)
