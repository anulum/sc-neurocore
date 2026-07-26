# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Ibarz-Tanaka reproducibility tests

"""Trace digest, repeated-step, and population-path reproducibility."""

from __future__ import annotations

import hashlib

import numpy as np

from sc_neurocore.network.population import Population
from sc_neurocore.neurons.models.ibarz_tanaka_map import IbarzTanakaMapNeuron


def test_reproducibility_hash_is_stable() -> None:
    """The descriptor's driven fast-state trace digest is exact."""
    trace, events = IbarzTanakaMapNeuron().simulate(1000, 0.2, backend="python")
    digest = hashlib.sha256(trace.tobytes()).hexdigest()
    assert events == 33
    assert digest == "68000d6955ffcaedffa3a851f70e8f118156312ab224638defb408ae0b3002ed"


def test_batch_matches_repeated_source_steps() -> None:
    """The batch dispatcher and public step surface commit the same recurrence."""
    batch = IbarzTanakaMapNeuron()
    trace, events = batch.simulate(300, 0.2, backend="python")
    stepper = IbarzTanakaMapNeuron()
    expected_trace = []
    expected_events = 0
    for _step in range(300):
        expected_events += stepper.step(0.2)
        expected_trace.append(stepper.v)
    np.testing.assert_array_equal(trace, np.asarray(expected_trace, dtype=np.float64))
    assert events == expected_events
    assert (batch.v, batch.u) == (stepper.v, stepper.u)


def test_population_path_observes_the_fast_state_and_events() -> None:
    """The standard population surface consumes the renamed v state correctly."""
    population = Population(IbarzTanakaMapNeuron, n=4, label="ibarz-tanaka")
    events = 0
    current = np.full(4, 0.2, dtype=np.float64)
    for _step in range(1000):
        events += int(population.step_all(current).sum())
    assert events == 4 * 33
    np.testing.assert_array_equal(
        population.voltages,
        np.asarray([neuron.v for neuron in population.neurons]),
    )
