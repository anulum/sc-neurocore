# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Focused Rulkov backend contracts

"""Focused cross-backend Rulkov map contracts."""

from .rulkov_map_backends_support import *


def test_simulate_matches_repeated_step() -> None:
    # The N-step path must equal calling step() N times (same state evolution
    # and the same upward-crossing spike count).
    trace_a, spikes_a = RulkovMapNeuron().simulate(300, 1.0, backend="python")
    manual = []
    spikes_b = 0
    stepper = RulkovMapNeuron()
    for _ in range(300):
        spikes_b += stepper.step(1.0)
        manual.append(stepper.x)
    np.testing.assert_array_equal(trace_a, np.asarray(manual, dtype=np.float64))
    assert spikes_a == spikes_b


def test_final_state_advances_instance() -> None:
    neuron = RulkovMapNeuron()
    _trace, _spikes = neuron.simulate(500, 1.0, backend="python")
    manual = RulkovMapNeuron()
    for _ in range(500):
        manual.step(1.0)
    assert neuron.x == manual.x and neuron.y == manual.y


def test_spiking_produces_upward_crossings() -> None:
    _trace, spikes = RulkovMapNeuron().simulate(50000, 0.5, backend="python")
    assert spikes > 10


def test_silent_at_zero_current() -> None:
    _trace, spikes = RulkovMapNeuron().simulate(50000, 0.0, backend="python")
    assert spikes == 0


def test_long_run_is_finite_and_bounded() -> None:
    trace, _spikes = RulkovMapNeuron().simulate(100_000, 0.5, backend="python")
    assert np.all(np.isfinite(trace))
    assert trace.min() >= -3.0 and trace.max() < 10.0
