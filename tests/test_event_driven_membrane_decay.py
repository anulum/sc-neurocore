# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Event-driven simulator membrane decay and input checks

"""Hold the event-driven simulator to the exact leaky integrate-and-fire closed form.

A neuron leaks from its last update, whether or not it has spiked; an input inside the refractory period
is discarded while the membrane keeps relaxing from the spike; an event later than the run's duration
waits for the next run; the clock-driven estimate counts a partial millisecond; and every input that
would make the result meaningless is refused.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from sc_neurocore.event_driven import EventDrivenSimulator

TAU = 20.0


def _isolated(**overrides: Any) -> EventDrivenSimulator:
    """Return two unconnected neurons for current-pulse experiments, with ``overrides`` applied."""
    parameters: dict[str, Any] = {
        "n_neurons": 2,
        "connectivity": [],
        "threshold": 1.0,
        "tau_mem": TAU,
        "refractory": 2.0,
    }
    parameters.update(overrides)
    return EventDrivenSimulator(**parameters)


def _relax(v: float, v_rest: float, dt: float) -> float:
    """Return the closed-form leaky membrane after ``dt`` ms without input."""
    return v_rest + (v - v_rest) * math.exp(-dt / TAU)


def test_a_neuron_leaks_before_its_first_spike() -> None:
    """Two 0.6 pulses ten time constants apart decay in between and never reach threshold 1.0."""
    sim = _isolated()
    sim.inject_current([(0.0, 0, 0.6), (10 * TAU, 0, 0.6)])
    spikes, stats = sim.run(10 * TAU + 1.0)
    assert spikes == []
    assert stats.total_events_processed == 2
    assert sim.membrane_potential[0] == pytest.approx(0.6 * math.exp(-10.0) + 0.6, abs=1e-12)


def test_decay_after_a_spike_runs_from_the_last_update() -> None:
    """After a spike, each input decays the membrane from the previous input, not from the spike."""
    v_rest, v_reset = 0.1, -0.5
    sim = _isolated(v_rest=v_rest, v_reset=v_reset)
    sim.inject_current([(0.0, 0, 1.5), (5.0, 0, 0.3), (10.0, 0, 0.3), (15.0, 0, 0.3)])
    spikes, _ = sim.run(20.0)
    assert spikes == [(0.0, 0)]
    expected = _relax(v_reset, v_rest, 5.0) + 0.3
    expected = _relax(expected, v_rest, 5.0) + 0.3
    expected = _relax(expected, v_rest, 5.0) + 0.3
    assert sim.membrane_potential[0] == pytest.approx(expected, abs=1e-12)


def test_simultaneous_inputs_add_without_decay() -> None:
    """Two pulses at the same time on one neuron sum exactly; the first relaxes from rest."""
    sim = _isolated(v_rest=0.05)
    sim.inject_current([(3.0, 0, 0.4), (3.0, 0, 0.3)])
    spikes, _ = sim.run(5.0)
    assert spikes == []
    assert sim.membrane_potential[0] == pytest.approx(0.05 + 0.4 + 0.3, abs=1e-12)


@pytest.mark.parametrize(
    ("connectivity", "member"),
    [
        ([(0, 0, 1.0, 0.0)], 0),
        ([(0, 1, 1.0, 0.0), (1, 0, 1.0, 0.0)], 0),
        ([(0, 1, 1.0, 0.0), (1, 2, 1.0, 0.0), (2, 1, 1.0, 0.0)], 1),
    ],
    ids=["self-loop", "pair", "cycle-behind-a-chain"],
)
def test_a_zero_delay_cycle_without_refractory_is_refused(
    connectivity: list[tuple[int, int, float, float]], member: int
) -> None:
    """With refractory 0, zero-delay connections that form a cycle could fire forever at one instant."""
    with pytest.raises(ValueError, match=f"cycle through neuron {member}"):
        EventDrivenSimulator(n_neurons=3, connectivity=connectivity, refractory=0.0)


def test_cycles_with_a_delay_or_a_refractory_period_still_run() -> None:
    """A delayed cycle, zero-delay chains and diamonds, and a guarded self-loop are accepted."""
    diamond = [(0, 1, 1.0, 0.0), (0, 2, 1.0, 0.0), (1, 3, 1.0, 0.0), (2, 3, 1.0, 0.0)]
    assert EventDrivenSimulator(n_neurons=4, connectivity=diamond, refractory=0.0).n_neurons == 4
    delayed = EventDrivenSimulator(
        n_neurons=2, connectivity=[(0, 1, 1.0, 0.0), (1, 0, 1.0, 1.0)], refractory=0.0
    )
    delayed.inject_current([(0.0, 0, 1.0)])
    spikes, _ = delayed.run(2.0)
    assert spikes == [(0.0, 0), (0.0, 1), (1.0, 0), (1.0, 1), (2.0, 0), (2.0, 1)]
    chain = EventDrivenSimulator(
        n_neurons=3, connectivity=[(0, 1, 1.0, 0.0), (1, 2, 1.0, 0.0)], refractory=0.0
    )
    chain.inject_current([(0.0, 0, 1.0)])
    assert chain.run(1.0)[0] == [(0.0, 0), (0.0, 1), (0.0, 2)]
    guarded = EventDrivenSimulator(n_neurons=1, connectivity=[(0, 0, 1.0, 0.0)], refractory=0.5)
    guarded.inject_current([(0.0, 0, 1.0)])
    spikes, stats = guarded.run(1.0)
    assert (spikes, stats.total_events_processed) == ([(0.0, 0)], 2)


def test_an_input_inside_the_refractory_period_is_discarded() -> None:
    """An input 1 ms after a spike is counted and dropped; the membrane keeps relaxing from the spike."""
    v_rest, v_reset = 0.1, -0.5
    sim = _isolated(v_rest=v_rest, v_reset=v_reset)
    sim.inject_current([(0.0, 0, 1.5), (1.0, 0, 0.9), (4.0, 0, 0.4)])
    spikes, stats = sim.run(5.0)
    assert spikes == [(0.0, 0)]
    assert stats.total_events_processed == 3
    assert sim.membrane_potential[0] == pytest.approx(_relax(v_reset, v_rest, 4.0) + 0.4, abs=1e-12)


def test_an_event_after_the_duration_waits_for_the_next_run() -> None:
    """A synaptic event due at 50 ms is kept by a 10 ms run and delivered by the next, longer run."""
    sim = EventDrivenSimulator(n_neurons=2, connectivity=[(0, 1, 2.0, 50.0)], threshold=1.0)
    sim.inject_spikes([(0.0, 0)])
    first, first_stats = sim.run(10.0)
    assert (first, first_stats.total_events_processed) == ([], 0)
    second, _ = sim.run(60.0)
    assert second == [(50.0, 1)]


def test_the_clock_driven_estimate_counts_a_partial_millisecond() -> None:
    """A 0.5 ms run of four neurons costs a clock-driven simulator one step of four updates."""
    sim = _isolated(n_neurons=4)
    sim.inject_current([(0.25, 0, 2.0)])
    _, stats = sim.run(0.5)
    assert stats.total_events_processed == 1
    assert stats.speedup_vs_clockdriven == pytest.approx(4.0)


def test_events_cannot_be_injected_before_the_current_time() -> None:
    """Once an event at 5 ms is processed, inputs at 3 ms are refused; inputs at 5 ms are accepted."""
    sim = _isolated()
    sim.inject_current([(5.0, 0, 0.2)])
    sim.run(10.0)
    with pytest.raises(ValueError, match="before the simulator's current time"):
        sim.inject_current([(3.0, 0, 0.2)])
    with pytest.raises(ValueError, match="before the simulator's current time"):
        sim.inject_spikes([(3.0, 0)])
    sim.inject_current([(5.0, 1, 0.2)])
    sim.reset()
    sim.inject_current([(0.0, 0, 0.2)])


def test_the_membrane_reader_returns_a_copy() -> None:
    """Writing into the returned array leaves the simulator's membrane untouched."""
    sim = _isolated(v_rest=0.25)
    view = sim.membrane_potential
    view[:] = 9.0
    assert np.array_equal(sim.membrane_potential, np.full(2, 0.25))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"n_neurons": 0}, "n_neurons"),
        ({"n_neurons": True}, "n_neurons"),
        ({"n_neurons": 2.0}, "n_neurons"),
        ({"tau_mem": 0.0}, "tau_mem"),
        ({"tau_mem": -1.0}, "tau_mem"),
        ({"tau_mem": math.nan}, "tau_mem"),
        ({"refractory": -1.0}, "refractory"),
        ({"threshold": math.inf}, "threshold"),
        ({"v_rest": math.nan}, "v_rest"),
        ({"v_reset": -math.inf}, "v_reset"),
        ({"connectivity": [(2, 0, 1.0, 0.0)]}, "source"),
        ({"connectivity": [(0, -1, 1.0, 0.0)]}, "target"),
        ({"connectivity": [(0, 1, math.nan, 0.0)]}, "weight"),
        ({"connectivity": [(0, 1, 1.0, -1.0)]}, "delay"),
        ({"connectivity": [(0, 1, 1.0, math.inf)]}, "delay"),
    ],
)
def test_the_constructor_refuses_meaningless_parameters(
    overrides: dict[str, Any], message: str
) -> None:
    """Non-positive sizes or time constants, non-finite values and unusable synapses are refused."""
    with pytest.raises(ValueError, match=message):
        _isolated(**overrides)


def test_injection_and_run_refuse_meaningless_inputs() -> None:
    """Unknown neurons, non-finite times or currents and negative or non-finite durations are refused."""
    sim = _isolated()
    with pytest.raises(ValueError, match="neuron"):
        sim.inject_spikes([(0.0, 2)])
    with pytest.raises(ValueError, match="time"):
        sim.inject_spikes([(math.nan, 0)])
    with pytest.raises(ValueError, match="neuron"):
        sim.inject_current([(0.0, -1, 0.5)])
    with pytest.raises(ValueError, match="current"):
        sim.inject_current([(0.0, 0, math.inf)])
    with pytest.raises(ValueError, match="duration"):
        sim.run(-1.0)
    with pytest.raises(ValueError, match="duration"):
        sim.run(math.nan)
