# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - DCNNeuron behavioural tests

from __future__ import annotations

import math

import pytest

from sc_neurocore.neurons.models.dcn_neuron import DCNNeuron


def _snapshot(neuron: DCNNeuron) -> tuple[float, ...]:
    return (neuron.v, neuron.h, neuron.n, neuron.p, neuron.s, neuron.r, neuron.ca)


def test_default_step_preserves_physical_state_bounds() -> None:
    neuron = DCNNeuron()

    spike = neuron.step(0.0)

    assert spike in (0, 1)
    assert -100.0 <= neuron.v <= 60.0
    assert 0.0 <= neuron.h <= 1.0
    assert 0.0 <= neuron.n <= 1.0
    assert 0.0 <= neuron.p <= 1.0
    assert 0.0 <= neuron.s <= 1.0
    assert 0.0 <= neuron.r <= 1.0
    assert neuron.ca >= 0.0


def test_seven_current_conductance_surface_is_present() -> None:
    neuron = DCNNeuron()

    assert neuron.g_na > 0.0
    assert neuron.g_nap > 0.0
    assert neuron.g_k > 0.0
    assert neuron.g_t > 0.0
    assert neuron.g_ahp > 0.0
    assert neuron.g_h > 0.0
    assert neuron.g_l > 0.0


def test_t_type_deinactivation_increases_during_hyperpolarisation() -> None:
    neuron = DCNNeuron()
    before = neuron.s

    for _ in range(500):
        neuron.step(-5.0)

    assert neuron.s >= before


def test_ih_depolarises_from_hyperpolarised_state() -> None:
    with_ih = DCNNeuron(v=-80.0)
    without_ih = DCNNeuron(v=-80.0, g_h=0.0)

    for _ in range(400):
        with_ih.step(0.0)
        without_ih.step(0.0)

    assert with_ih.v > without_ih.v


@pytest.mark.parametrize(
    "kwargs",
    [
        {"h": -0.1},
        {"n": 1.1},
        {"ca": -1e-6},
        {"g_na": -1.0},
        {"c_m": 0.0},
        {"phi": 0.0},
        {"tau_ca": 0.0},
        {"kd_ahp": 0.0},
        {"dt": 0.0},
        {"gain": -1.0},
        {"_sub_steps": 0},
        {"v": math.inf},
    ],
)
def test_invalid_physical_configuration_is_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        DCNNeuron(**kwargs)


def test_non_finite_current_does_not_mutate_state() -> None:
    neuron = DCNNeuron()
    before = _snapshot(neuron)

    with pytest.raises(ValueError):
        neuron.step(math.nan)

    assert _snapshot(neuron) == before


def test_unstable_candidate_does_not_mutate_state() -> None:
    neuron = DCNNeuron()
    neuron.g_na = math.inf
    before = _snapshot(neuron)

    with pytest.raises(ValueError):
        neuron.step(5.0)

    assert _snapshot(neuron) == before
