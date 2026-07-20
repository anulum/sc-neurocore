# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for homeostatic LIF neuron

import pytest

from sc_neurocore.neurons.homeostatic_lif import (
    THRESHOLD_CEILING_MULT,
    THRESHOLD_FLOOR,
    HomeostaticLIFNeuron,
)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("target_rate", -0.1),
        ("target_rate", 1.1),
        ("target_rate", float("nan")),
        ("adaptation_rate", -0.001),
        ("adaptation_rate", float("inf")),
        ("rate_trace", -0.1),
        ("rate_trace", 1.1),
        ("trace_decay", -0.1),
        ("trace_decay", 1.1),
        ("trace_decay", float("nan")),
    ],
)
def test_invalid_homeostatic_lif_parameters_fail_closed(field, value):
    kwargs = {
        "target_rate": 0.1,
        "adaptation_rate": 0.01,
        "rate_trace": 0.0,
        "trace_decay": 0.99,
        "noise_std": 0.0,
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match=field):
        HomeostaticLIFNeuron(**kwargs)


def test_repeated_spiking_raises_threshold_until_ceiling():
    neuron = HomeostaticLIFNeuron(
        target_rate=0.0,
        adaptation_rate=1.0,
        trace_decay=0.0,
        noise_std=0.0,
    )
    ceiling = neuron.initial_threshold * THRESHOLD_CEILING_MULT

    for _ in range(100):
        neuron.step(10.0)

    assert neuron.v_threshold == pytest.approx(ceiling)


def test_repeated_silence_lowers_threshold_until_floor():
    neuron = HomeostaticLIFNeuron(
        target_rate=1.0,
        adaptation_rate=1.0,
        trace_decay=0.0,
        noise_std=0.0,
    )

    for _ in range(100):
        neuron.step(0.0)

    assert neuron.v_threshold == pytest.approx(THRESHOLD_FLOOR)


def test_rate_trace_is_exponential_spike_average():
    neuron = HomeostaticLIFNeuron(
        target_rate=0.0,
        adaptation_rate=0.0,
        trace_decay=0.5,
        noise_std=0.0,
    )

    neuron.step(10.0)
    assert neuron.rate_trace == pytest.approx(0.5)
    neuron.step(0.0)
    assert neuron.rate_trace == pytest.approx(0.25)


def test_homeostatic_lif_get_state_reports_threshold_and_trace() -> None:
    """The public state includes the adapted threshold and rate trace."""
    neuron = HomeostaticLIFNeuron(target_rate=0.1, noise_std=0.0)
    neuron.step(1.0)

    state = neuron.get_state()

    assert state["threshold"] == pytest.approx(neuron.v_threshold)
    assert state["rate_trace"] == pytest.approx(neuron.rate_trace)
