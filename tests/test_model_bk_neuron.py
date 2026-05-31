# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - model behavioural tests

import math

import pytest

from sc_neurocore.neurons.models.bk_neuron import BKNeuron


def snapshot(neuron: BKNeuron) -> tuple[float, float, float, float]:
    return neuron.v, neuron.h, neuron.n, neuron.ca


def test_default_step_preserves_finite_gates_and_calcium() -> None:
    neuron = BKNeuron()

    spike = neuron.step(0.0)

    assert spike in (0, 1)
    assert math.isfinite(neuron.v)
    assert 0.0 <= neuron.h <= 1.0
    assert 0.0 <= neuron.n <= 1.0
    assert neuron.ca >= 0.0


def test_bk_current_hyperpolarizes_relative_to_no_bk_current() -> None:
    with_bk = BKNeuron(g_na=0.0, g_k=0.0, g_bk=3.0, g_l=0.0, ca=1.0)
    without_bk = BKNeuron(g_na=0.0, g_k=0.0, g_bk=0.0, g_l=0.0, ca=1.0)

    with_bk.step(20.0)
    without_bk.step(20.0)

    assert with_bk.v < without_bk.v


def test_spike_increments_calcium_and_keeps_voltage_subthreshold() -> None:
    neuron = BKNeuron(g_na=0.0, g_k=0.0, g_bk=0.0, g_l=0.0, c_m=1.0, v=-20.1)

    assert neuron.step(200.0) == 1
    assert neuron.v < neuron.v_threshold
    assert neuron.ca > 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dt": 0.0},
        {"c_m": 0.0},
        {"tau_ca": 0.0},
        {"h": -0.1},
        {"n": 1.1},
        {"ca": -0.1},
        {"g_bk": -1.0},
        {"gain": math.inf},
        {"_sub_steps": 0},
    ],
)
def test_invalid_physical_configuration_is_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        BKNeuron(**kwargs)


def test_non_finite_current_does_not_mutate_state() -> None:
    neuron = BKNeuron()
    before = snapshot(neuron)

    with pytest.raises(ValueError):
        neuron.step(math.nan)

    assert snapshot(neuron) == before


def test_corrupted_runtime_calcium_does_not_mutate_state() -> None:
    neuron = BKNeuron()
    neuron.ca = math.inf
    before = snapshot(neuron)

    with pytest.raises(ValueError):
        neuron.step(1.0)

    assert snapshot(neuron) == before


def test_unstable_input_drive_does_not_mutate_state() -> None:
    neuron = BKNeuron()
    neuron.gain = 1e308
    before = snapshot(neuron)

    with pytest.raises(ValueError):
        neuron.step(1e308)

    assert snapshot(neuron) == before


def test_candidate_outside_safety_bounds_does_not_mutate_state() -> None:
    neuron = BKNeuron(g_na=0.0, g_k=0.0, g_bk=0.0, g_l=0.0, c_m=1e-12, v_threshold=1e30)
    before = snapshot(neuron)

    with pytest.raises(ValueError):
        neuron.step(1.0)

    assert snapshot(neuron) == before
