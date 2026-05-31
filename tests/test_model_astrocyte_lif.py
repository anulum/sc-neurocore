# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - model behavioural tests

import math

import pytest

from sc_neurocore.neurons.models.astrocyte_lif import AstrocyteLIFNeuron


def snapshot(neuron: AstrocyteLIFNeuron) -> tuple[float, float]:
    return neuron.v, neuron.ca


def test_presynaptic_spike_raises_calcium_by_delta_from_rest() -> None:
    neuron = AstrocyteLIFNeuron()

    spike = neuron.step_with_pre(0.0, True)

    assert spike == 0
    assert neuron.ca == pytest.approx(neuron.ca_delta)
    assert math.isfinite(neuron.v)


def test_gliotransmitter_feedback_depolarizes_membrane_when_calcium_crosses_threshold() -> None:
    with_feedback = AstrocyteLIFNeuron(ca=0.6, g_glio=2.0)
    without_feedback = AstrocyteLIFNeuron(ca=0.6, g_glio=0.0)

    with_feedback.step(0.0)
    without_feedback.step(0.0)

    assert with_feedback.v > without_feedback.v


def test_threshold_crossing_resets_voltage_and_reports_spike() -> None:
    neuron = AstrocyteLIFNeuron(tau_m=1.0, v=-50.1)

    assert neuron.step(200.0) == 1
    assert neuron.v == neuron.v_reset


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tau_m": 0.0},
        {"tau_ca": 0.0},
        {"dt": 0.0},
        {"theta": -70.0},
        {"ca_delta": -0.1},
        {"ca_thresh": -0.1},
        {"g_glio": -1.0},
        {"v": math.inf},
        {"ca": math.nan},
    ],
)
def test_invalid_physical_configuration_is_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        AstrocyteLIFNeuron(**kwargs)


def test_non_finite_current_does_not_mutate_state() -> None:
    neuron = AstrocyteLIFNeuron()
    before = snapshot(neuron)

    with pytest.raises(ValueError):
        neuron.step(math.nan)

    assert snapshot(neuron) == before


def test_non_boolean_presynaptic_flag_does_not_mutate_state() -> None:
    neuron = AstrocyteLIFNeuron()
    before = snapshot(neuron)

    with pytest.raises(TypeError):
        neuron.step_with_pre(0.0, 1)  # type: ignore[arg-type]

    assert snapshot(neuron) == before


def test_corrupted_runtime_calcium_does_not_mutate_voltage() -> None:
    neuron = AstrocyteLIFNeuron()
    neuron.ca = math.inf
    before = snapshot(neuron)

    with pytest.raises(ValueError):
        neuron.step(1.0)

    assert snapshot(neuron) == before


def test_non_finite_candidate_does_not_mutate_state() -> None:
    neuron = AstrocyteLIFNeuron(ca=1.0)
    neuron.g_glio = 1e309
    before = snapshot(neuron)

    with pytest.raises(ValueError):
        neuron.step(1.0)

    assert snapshot(neuron) == before
