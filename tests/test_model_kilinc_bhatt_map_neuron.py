# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Kilinc-Bhatt experimental map contracts

"""Exact source-model tests for the experimental adaptive-threshold map."""

from __future__ import annotations

import math

import pytest

from sc_neurocore.neurons.models import KilincBhattMapNeuron as PublicKilincBhattMapNeuron
from sc_neurocore.neurons.models.kilinc_bhatt_map_neuron import KilincBhattMapNeuron


def _state(neuron: KilincBhattMapNeuron) -> tuple[float, float]:
    return neuron.x, neuron.theta


def test_public_registry_and_defaults() -> None:
    neuron = KilincBhattMapNeuron()
    assert PublicKilincBhattMapNeuron is KilincBhattMapNeuron
    assert _state(neuron) == (0.0, 0.0)
    assert (neuron.k, neuron.beta, neuron.gamma) == (1.5, 0.95, 0.3)
    assert (neuron.theta_spike, neuron.x_threshold) == (0.8, 0.8)


def test_source_recurrence_anchor_and_upward_crossing_events() -> None:
    neuron = KilincBhattMapNeuron()
    expected = (
        (1.75, 0.0, 1),
        (0.7486334232083991, 0.3, 0),
        (1.5375899090935132, 0.285, 1),
        (0.9524735955148995, 0.57075, 0),
    )
    for x_expected, theta_expected, event_expected in expected:
        event = neuron.step(1.0)
        assert neuron.x == pytest.approx(x_expected, abs=1e-15)
        assert neuron.theta == pytest.approx(theta_expected, abs=1e-15)
        assert event == event_expected


def test_stable_sigmoid_handles_extreme_finite_arguments() -> None:
    assert KilincBhattMapNeuron._sigmoid(1.0) > 0.5
    assert KilincBhattMapNeuron._sigmoid(-1.0) < 0.5
    assert KilincBhattMapNeuron._sigmoid(sys_float_max()) == 1.0
    assert KilincBhattMapNeuron._sigmoid(-sys_float_max()) == 0.0


def sys_float_max() -> float:
    return float.fromhex("0x1.fffffffffffffp+1023")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("x", math.nan),
        ("theta", math.inf),
        ("k", math.nan),
        ("beta", math.inf),
        ("gamma", math.nan),
        ("theta_spike", math.inf),
        ("x_threshold", math.nan),
        ("x", -5.1),
        ("theta", 5.1),
        ("k", -0.1),
        ("k", 5.1),
        ("beta", -0.1),
        ("beta", 1.1),
        ("gamma", -0.1),
        ("gamma", 2.1),
        ("theta_spike", -0.1),
        ("theta_spike", 2.1),
        ("x_threshold", -0.1),
        ("x_threshold", 2.1),
    ],
)
def test_constructor_rejects_invalid_scientific_configuration(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        KilincBhattMapNeuron(**{field: value})


@pytest.mark.parametrize("current", (math.nan, math.inf, -math.inf))
def test_non_finite_drive_is_rejected_before_mutation(current: float) -> None:
    neuron = KilincBhattMapNeuron()
    before = _state(neuron)
    with pytest.raises(ValueError, match="current"):
        neuron.step(current)
    assert _state(neuron) == before


def test_corrupted_runtime_configuration_is_rejected_before_mutation() -> None:
    neuron = KilincBhattMapNeuron()
    neuron.beta = 1.1
    before = _state(neuron)
    with pytest.raises(ValueError, match="beta"):
        neuron.step(1.0)
    assert _state(neuron) == before


def test_non_finite_candidate_is_rejected_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    neuron = KilincBhattMapNeuron()
    before = _state(neuron)
    monkeypatch.setattr(KilincBhattMapNeuron, "_sigmoid", staticmethod(lambda _z: math.inf))
    with pytest.raises(FloatingPointError, match="candidate"):
        neuron.step(1.0)
    assert _state(neuron) == before


def test_finite_extreme_drive_uses_documented_state_clamp() -> None:
    neuron = KilincBhattMapNeuron()
    assert neuron.step(1.0e308) == 1
    assert _state(neuron) == (5.0, 0.0)


def test_reset_preserves_parameters_and_clears_only_dynamic_state() -> None:
    neuron = KilincBhattMapNeuron(k=2.0, beta=0.8, gamma=0.5)
    neuron.step(1.0)
    neuron.reset()
    assert _state(neuron) == (0.0, 0.0)
    assert (neuron.k, neuron.beta, neuron.gamma) == (2.0, 0.8, 0.5)
