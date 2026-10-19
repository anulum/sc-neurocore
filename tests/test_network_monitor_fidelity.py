# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — monitors record what the network did, across runs too

"""Monitors report the run that happened, and a second run continues the first.

Reproducers of RIGOR review 266: a rate bin divided by a duration it did not
cover; a mapping-backed EquationNeuron was recorded as zeros; a second
``Network.run`` restarted the step index and gave a negative inter-spike
interval; the correlogram of a populated pair lost every nonzero lag; neurons
built from one explicit initial state shared one mapping; and ``record=[]``
crashed after the snapshot time was appended. Each case runs real populations
through the public ``Network``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from sc_neurocore.network import (
    Network,
    Population,
    Projection,
    RateMonitor,
    SpikeMonitor,
    StateMonitor,
)
from sc_neurocore.neurons.equation_builder import EquationNeuron

COUNTER = {"equations": {"v": "v + 1"}, "dt": 1.0, "method": "map"}


def _counter(initial: float = 0.0, **extra: Any) -> Population:
    """Return a one-neuron population whose state counts steps from ``initial``."""
    return Population(EquationNeuron, 1, params={**COUNTER, "state": {"v": initial}, **extra})


def _always_spiking(n: int = 1) -> Population:
    """Return neurons that spike on every step (threshold v >= 0 with v -> 0)."""
    params = {
        "equations": {"v": "0.0"},
        "dt": 1.0,
        "method": "map",
        "state": {"v": 0.0},
        "threshold": "v >= 0.0",
        "reset": {"v": "0.0"},
    }
    return Population(EquationNeuron, n, params=params)


@pytest.mark.parametrize(
    ("bin_ms", "dt", "rate"),
    [(10, 0.003, 1000.0 / 3.0 * 1.0), (1, 0.010, 100.0), (10, 0.001, 1000.0)],
)
def test_a_rate_divides_by_the_time_its_bin_covered(bin_ms: int, dt: float, rate: float) -> None:
    """NEURO-MON-RATE-01: bin 10 ms at dt 3 ms covers 9 ms (333 Hz, was 300 Hz).

    A neuron spiking every step fires at 1/dt; bin 1 ms at dt 10 ms reported 1000 Hz
    for a 100 Hz neuron. The divisible bin is the control.
    """
    population = _always_spiking()
    monitor = RateMonitor(population, bin_ms=bin_ms)
    Network(population, monitor).run(dt * 20, dt=dt, backend="python")
    assert monitor.rate.size > 0
    np.testing.assert_allclose(monitor.rate, rate, rtol=1e-12)


def test_a_mapping_backed_state_is_recorded_not_zeroed() -> None:
    """NEURO-MON-MAPPING-01: v = v + 1 from 2 was recorded as 0, 0, 0."""
    population = _counter(2.0)
    monitor = StateMonitor(population)
    Network(population, monitor).run(3.0, dt=1.0, backend="python")
    assert monitor.traces["v"][:, 0].tolist() == [3.0, 4.0, 5.0]
    assert population.get_states()["v"].tolist() == [5.0]
    assert population.voltages.tolist() == [5.0]


def test_a_second_run_continues_the_step_index() -> None:
    """NEURO-MON-CLOCK-01: two 4-step runs recorded 0..3 twice and an ISI of -3."""
    whole, split = _always_spiking(), _always_spiking()
    whole_spikes, split_spikes = SpikeMonitor(whole), SpikeMonitor(split)
    whole_states, split_states = StateMonitor(whole), StateMonitor(split)
    whole_rate, split_rate = RateMonitor(whole, bin_ms=4), RateMonitor(split, bin_ms=4)
    Network(whole, whole_spikes, whole_states, whole_rate).run(0.008, dt=0.001)
    network = Network(split, split_spikes, split_states, split_rate)
    network.run(0.004, dt=0.001)
    network.run(0.004, dt=0.001)
    assert split_spikes.spike_times.tolist() == whole_spikes.spike_times.tolist() == list(range(8))
    assert split_states.t.tolist() == whole_states.t.tolist()
    assert split_rate.t.tolist() == whole_rate.t.tolist() == [3, 7]
    assert split_spikes.isi(0).tolist() == [1] * 7


def test_a_second_run_delivers_the_last_spikes_of_the_first() -> None:
    """A spike on the last step of one run reaches its target on the first step of the next."""
    results = []
    for runs in ([0.004], [0.003, 0.001]):
        source = _always_spiking()
        target = Population(
            EquationNeuron,
            1,
            params={"equations": {"v": "v + I"}, "dt": 1.0, "method": "map", "state": {"v": 0.0}},
        )
        monitor = StateMonitor(target)
        projection = Projection(source, target, 1.0, topology="all_to_all")
        network = Network(source, target, projection, monitor)
        for duration in runs:
            network.run(duration, dt=0.001, backend="python")
        results.append(monitor.traces["v"][:, 0].tolist())
    assert results[1] == results[0] == [0.0, 1.0, 2.0, 3.0]


def test_continuing_at_another_dt_is_refused() -> None:
    """A step index has one dt; a later run at another dt would mean other times."""
    network = Network(_always_spiking())
    network.run(0.004, dt=0.001)
    with pytest.raises(ValueError, match="dt=0.002"):
        network.run(0.004, dt=0.002)


def test_a_populated_correlogram_keeps_every_requested_lag() -> None:
    """NEURO-MON-LAG-01: max_lag 3 returned only lag 0 for an alternating pair."""
    from sc_neurocore.analysis.spike_stats import cross_correlation

    population = Population(EquationNeuron, 2, params={**COUNTER, "state": {"v": 0.0}})
    monitor = SpikeMonitor(population)
    for t in range(8):
        monitor.record(np.array([t % 2, 1 - t % 2]), t)
    correlation, lags = monitor.cross_correlation(0, 1, max_lag=3)
    assert lags.tolist() == [-3, -2, -1, 0, 1, 2, 3]
    trains = monitor.spike_trains
    bins = [np.zeros(8, dtype=np.int8) for _ in range(2)]
    for i in (0, 1):
        bins[i][trains[i]] = 1
    expected, _ = cross_correlation(bins[0], bins[1], max_lag_ms=3.0, dt=0.001)
    np.testing.assert_allclose(correlation, expected)
    assert np.count_nonzero(correlation[lags != 0]) == 6


def test_neurons_built_from_one_initial_state_are_independent() -> None:
    """NEURO-MON-ALIAS-01: one step made both neurons 2 and changed the caller's mapping."""
    initial = {"v": 0.0}
    population = Population(EquationNeuron, 2, params={**COUNTER, "state": initial})
    population.step_all(np.zeros(2))
    assert population.get_states()["v"].tolist() == [1.0, 1.0]
    assert initial == {"v": 0.0}


def test_an_empty_record_selection_records_no_neuron() -> None:
    """NEURO-MON-SUBSET-01: record=[] raised IndexError after appending the time."""
    population = _counter()
    monitor = StateMonitor(population, record=[])
    Network(population, monitor).run(2.0, dt=1.0, backend="python")
    assert monitor.t.tolist() == [0, 1]
    assert monitor.traces["v"].shape == (2, 0)


def test_a_record_index_outside_the_population_is_refused_before_a_run() -> None:
    """An index the population does not have is refused at construction."""
    with pytest.raises(IndexError, match=r"\[0, 1\)"):
        StateMonitor(_counter(), record=[1])


def test_spikes_or_delayed_current_in_flight_keep_a_run_off_the_native_runner() -> None:
    """The native runner starts with no spikes and empty delay buffers, so it may not continue."""
    source = _always_spiking()
    target = _counter()
    network = Network(source, target, Projection(source, target, 1.0, delay=2.0))
    network.run(0.003, dt=0.001, backend="python")
    reasons = network._rust_incompatibilities()
    assert any("last step still to deliver" in reason for reason in reasons)
    assert any("delayed current" in reason for reason in reasons)
