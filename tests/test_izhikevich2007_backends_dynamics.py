# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Izhikevich-2007 dynamics tests

"""Repeated-step, state, firing, silence, and reset dynamics contracts."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.izhikevich2007 import Izhikevich2007Neuron


def test_simulate_matches_repeated_step() -> None:
    trace_a, spikes_a = Izhikevich2007Neuron().simulate(500, 300.0, backend="python")
    manual = []
    spikes_b = 0
    stepper = Izhikevich2007Neuron()
    for _ in range(500):
        spikes_b += stepper.step(300.0)
        manual.append(stepper.v)
    np.testing.assert_array_equal(trace_a, np.asarray(manual, dtype=np.float64))
    assert spikes_a == spikes_b


def test_final_state_advances_instance() -> None:
    neuron = Izhikevich2007Neuron()
    _trace, _spikes = neuron.simulate(500, 300.0, backend="python")
    manual = Izhikevich2007Neuron()
    for _ in range(500):
        manual.step(300.0)
    assert (neuron.v, neuron.u) == (manual.v, manual.u)


def test_tonic_firing_under_drive() -> None:
    _trace, spikes = Izhikevich2007Neuron().simulate(20000, 300.0, backend="python")
    assert spikes > 10


def test_subthreshold_silent() -> None:
    _trace, spikes = Izhikevich2007Neuron().simulate(20000, 0.0, backend="python")
    assert spikes == 0


def test_trace_resets_below_vpeak() -> None:
    # Every recorded sample is at or below the peak (the reset fires on >= vpeak).
    trace, spikes = Izhikevich2007Neuron().simulate(20000, 300.0, backend="python")
    assert spikes > 0
    assert np.all(trace <= Izhikevich2007Neuron().vpeak)
