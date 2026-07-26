# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hindmarsh-Rose dynamics tests

"""Repeated-step, state, bursting, and finite-trace dynamics contracts."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.hindmarsh_rose import HindmarshRoseNeuron


def test_simulate_matches_repeated_step() -> None:
    trace_a, spikes_a = HindmarshRoseNeuron().simulate(500, 3.0, backend="python")
    manual = []
    spikes_b = 0
    stepper = HindmarshRoseNeuron()
    for _ in range(500):
        spikes_b += stepper.step(3.0)
        manual.append(stepper.x)
    np.testing.assert_array_equal(trace_a, np.asarray(manual, dtype=np.float64))
    assert spikes_a == spikes_b


def test_final_state_advances_instance() -> None:
    neuron = HindmarshRoseNeuron()
    _trace, _spikes = neuron.simulate(500, 3.0, backend="python")
    manual = HindmarshRoseNeuron()
    for _ in range(500):
        manual.step(3.0)
    assert (neuron.x, neuron.y, neuron.z) == (manual.x, manual.y, manual.z)


def test_bursting_under_drive() -> None:
    _trace, spikes = HindmarshRoseNeuron().simulate(20000, 3.0, backend="python")
    assert spikes > 10


def test_subthreshold_silent() -> None:
    _trace, spikes = HindmarshRoseNeuron().simulate(20000, 0.0, backend="python")
    assert spikes == 0


def test_trace_is_finite() -> None:
    trace, _spikes = HindmarshRoseNeuron().simulate(60000, 3.0, backend="python")
    assert np.all(np.isfinite(trace))
