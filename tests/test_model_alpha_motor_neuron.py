# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - model behavioural tests

import math

import pytest

from sc_neurocore.neurons.models.alpha_motor_neuron import AlphaMotorNeuron


def snapshot(neuron: AlphaMotorNeuron) -> tuple[float, ...]:
    return (neuron.v, neuron.h, neuron.n, neuron.m_pic, neuron.h_pic, neuron.ca, neuron.ca_buf)


def test_default_state_and_step_preserve_physiological_bounds() -> None:
    neuron = AlphaMotorNeuron()

    spike = neuron.step(0.0)

    assert spike in (0, 1)
    assert math.isfinite(neuron.v)
    assert 0.0 <= neuron.h <= 1.0
    assert 0.0 <= neuron.n <= 1.0
    assert 0.0 <= neuron.m_pic <= 1.0
    assert 0.0 <= neuron.h_pic <= 1.0
    assert neuron.ca >= 0.0
    assert neuron.ca_buf >= 0.0


def test_current_depolarizes_passive_alpha_motor_membrane() -> None:
    neuron = AlphaMotorNeuron(g_na=0.0, g_k=0.0, g_pic=0.0, g_ahp=0.0, g_l=0.0)
    v0 = neuron.v

    spike = neuron.step(5.0)

    assert spike == 0
    assert neuron.v > v0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dt": 0.0},
        {"c_m": 0.0},
        {"tau_ca": 0.0},
        {"h": 1.2},
        {"ca": -1e-6},
        {"g_na": -1.0},
        {"buf_ratio": 1.2},
        {"v": math.inf},
    ],
)
def test_invalid_physical_configuration_is_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        AlphaMotorNeuron(**kwargs)


def test_non_finite_current_does_not_mutate_state() -> None:
    neuron = AlphaMotorNeuron()
    before = snapshot(neuron)

    with pytest.raises(ValueError):
        neuron.step(math.nan)

    assert snapshot(neuron) == before


def test_corrupted_runtime_gate_does_not_mutate_state() -> None:
    neuron = AlphaMotorNeuron()
    neuron.h = -0.1
    before = snapshot(neuron)

    with pytest.raises(ValueError):
        neuron.step(1.0)

    assert snapshot(neuron) == before


def test_unstable_candidate_does_not_mutate_state() -> None:
    neuron = AlphaMotorNeuron()
    neuron.g_na = 1e308
    before = snapshot(neuron)

    with pytest.raises(ValueError):
        neuron.step(1.0)

    assert snapshot(neuron) == before


def test_spike_flag_reports_upward_threshold_crossing_without_reset() -> None:
    neuron = AlphaMotorNeuron(
        v=-20.1,
        g_na=0.0,
        g_k=0.0,
        g_pic=0.0,
        g_ahp=0.0,
        g_l=0.0,
        c_m=1.0,
    )

    assert neuron.step(20.0) == 1
    assert neuron.v > neuron.v_threshold
