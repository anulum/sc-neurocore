# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — FitzHugh-Nagumo dynamics tests

"""Repeated-step, state, firing, silence, and finite-trace dynamics contracts."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.fitzhugh_nagumo import FitzHughNagumoNeuron


def test_simulate_matches_repeated_step() -> None:
    trace_a, spikes_a = FitzHughNagumoNeuron().simulate(500, 0.5, backend="python")
    manual = []
    spikes_b = 0
    stepper = FitzHughNagumoNeuron()
    for _ in range(500):
        spikes_b += stepper.step(0.5)
        manual.append(stepper.v)
    np.testing.assert_array_equal(trace_a, np.asarray(manual, dtype=np.float64))
    assert spikes_a == spikes_b


def test_final_state_advances_instance() -> None:
    neuron = FitzHughNagumoNeuron()
    _trace, _spikes = neuron.simulate(500, 0.5, backend="python")
    manual = FitzHughNagumoNeuron()
    for _ in range(500):
        manual.step(0.5)
    assert neuron.v == manual.v and neuron.w == manual.w


def test_tonic_firing_under_drive() -> None:
    _trace, spikes = FitzHughNagumoNeuron().simulate(20000, 0.5, backend="python")
    assert spikes > 5


def test_rest_at_zero_drive_eventually_silent() -> None:
    # Without drive the neuron relaxes to the stable fixed point: no sustained firing.
    _trace, spikes = FitzHughNagumoNeuron().simulate(20000, 0.0, backend="python")
    assert spikes == 0


def test_trace_is_finite() -> None:
    trace, _spikes = FitzHughNagumoNeuron().simulate(50000, 0.5, backend="python")
    assert np.all(np.isfinite(trace))
