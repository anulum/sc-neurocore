# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - GammaMotorNeuron behavioural tests

from __future__ import annotations

import math

import pytest

from sc_neurocore.neurons.models.gamma_motor_neuron import GammaMotorNeuron


def _snapshot(neuron: GammaMotorNeuron) -> tuple[float, float]:
    return (neuron.v, neuron.adapt)


def _spikes(neuron: GammaMotorNeuron, drive: float, steps: int) -> int:
    return sum(neuron.step(drive) for _ in range(steps))


def test_subthreshold_drive_converges_to_continuous_relaxation() -> None:
    neuron = GammaMotorNeuron(v=-65.0, adapt=0.0, dt=0.5, tau=8.0, tau_adapt=100.0)

    spike = neuron.step(4.0)

    expected_v = -61.0 + (-65.0 + 61.0) * math.exp(-0.5 / 8.0)
    expected_adapt = 0.3 * (expected_v + 65.0) * (1.0 - math.exp(-0.5 / 100.0))
    assert spike == 0
    assert neuron.v == pytest.approx(expected_v)
    assert neuron.adapt == pytest.approx(expected_adapt)


def test_negative_drive_is_clamped_to_passive_leak() -> None:
    neuron = GammaMotorNeuron(v=-60.0, adapt=0.0)

    neuron.step(-10.0)

    assert -65.0 < neuron.v < -60.0
    assert neuron.adapt > 0.0


def test_dynamic_subtype_fires_at_least_as_often_as_static_subtype() -> None:
    dynamic = GammaMotorNeuron()
    static = GammaMotorNeuron.static_type()

    dynamic_spikes = _spikes(dynamic, 20.0, 2_000)
    static_spikes = _spikes(static, 20.0, 2_000)

    assert dynamic_spikes > 0
    assert static_spikes > 0
    assert dynamic_spikes >= static_spikes


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tau": 0.0},
        {"tau_adapt": 0.0},
        {"dt": 0.0},
        {"gain": -1.0},
        {"v_reset": -40.0},
        {"v": math.inf},
    ],
)
def test_invalid_physical_configuration_is_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        GammaMotorNeuron(**kwargs)


def test_non_finite_drive_does_not_mutate_state() -> None:
    neuron = GammaMotorNeuron()
    before = _snapshot(neuron)

    with pytest.raises(ValueError):
        neuron.step(math.nan)

    assert _snapshot(neuron) == before


def test_corrupted_runtime_state_does_not_mutate_state() -> None:
    neuron = GammaMotorNeuron()
    neuron.tau = 0.0
    before = _snapshot(neuron)

    with pytest.raises(ValueError):
        neuron.step(20.0)

    assert _snapshot(neuron) == before
