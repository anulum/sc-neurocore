# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Execution trace recorder for spike-level debugging

"""Record full SNN execution trace for post-hoc analysis.

Captures per-neuron per-timestep: voltage, spike, input current.
Enables temporal debugging: find where spikes diverge, trace causal
chains through synaptic connections, compare two runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ExecutionTrace:
    """Complete execution trace of an SNN run.

    Attributes
    ----------
    n_neurons : int
        Total neurons across all populations.
    n_steps : int
        Number of simulation timesteps.
    spikes : ndarray of shape (n_steps, n_neurons)
        Binary spike matrix.
    voltages : ndarray of shape (n_steps, n_neurons)
        Membrane voltages.
    currents : ndarray of shape (n_steps, n_neurons)
        Input currents.
    population_labels : list of str
        Population names.
    population_ranges : list of (start, end)
        Neuron index ranges per population.
    """

    n_neurons: int
    n_steps: int
    spikes: np.ndarray[Any, Any]
    voltages: np.ndarray[Any, Any]
    currents: np.ndarray[Any, Any]
    population_labels: list[str] = field(default_factory=list)
    population_ranges: list[tuple[int, int]] = field(default_factory=list)

    @property
    def spike_count(self) -> int:
        """Total spikes in the trace."""
        return int(self.spikes.sum())

    @property
    def firing_rates(self) -> np.ndarray[Any, Any]:
        """Per-neuron firing rate (spikes per step)."""
        rates: np.ndarray[Any, Any] = self.spikes.mean(axis=0)
        return rates

    def neuron_trace(self, neuron_id: int) -> dict[str, np.ndarray[Any, Any]]:
        """Extract full trace for one neuron."""
        return {
            "spikes": self.spikes[:, neuron_id],
            "voltages": self.voltages[:, neuron_id],
            "currents": self.currents[:, neuron_id],
            "spike_times": np.where(self.spikes[:, neuron_id] > 0)[0],
        }

    def spike_times(self, neuron_id: int) -> np.ndarray[Any, Any]:
        """Timesteps when a neuron spiked."""
        return np.where(self.spikes[:, neuron_id] > 0)[0]

    def population_spikes(self, pop_label: str) -> np.ndarray[Any, Any]:
        """Spike matrix for one population."""
        for label, (start, end) in zip(self.population_labels, self.population_ranges):
            if label == pop_label:
                return self.spikes[:, start:end]
        raise ValueError(f"Population '{pop_label}' not found")


class SpikeTracer:
    """Records execution trace during SNN simulation.

    Wraps a Network and intercepts step_all to record spikes,
    voltages, and currents at every timestep.

    Usage
    -----
    >>> tracer = SpikeTracer(network)
    >>> trace = tracer.run(duration=0.1, dt=0.001)
    >>> divergence = find_divergence(trace, expected_spikes)
    """

    def __init__(self, network):  # type: ignore[no-untyped-def]
        self.network = network

    def run(self, duration: float, dt: float = 0.001, seed: int = 42) -> ExecutionTrace:
        """Run the network and record full execution trace."""

        np.random.seed(seed)
        n_steps = int(round(duration / dt))

        # Map populations to global neuron indices
        pop_labels = []
        pop_ranges = []
        total_neurons = 0
        for pop in self.network.populations:
            start = total_neurons
            total_neurons += pop.n
            pop_ranges.append((start, start + pop.n))
            pop_labels.append(pop.label)

        # Allocate trace arrays
        all_spikes = np.zeros((n_steps, total_neurons), dtype=np.int8)
        all_voltages = np.zeros((n_steps, total_neurons), dtype=np.float64)
        all_currents = np.zeros((n_steps, total_neurons), dtype=np.float64)

        # Run simulation step by step
        pop_to_currents = {id(p): np.zeros(p.n, dtype=np.float64) for p in self.network.populations}
        last_spikes = {id(p): np.zeros(p.n, dtype=np.int8) for p in self.network.populations}

        for t in range(n_steps):
            for pid in pop_to_currents:
                pop_to_currents[pid][:] = 0.0

            self.network._apply_stimuli(pop_to_currents, t, dt)
            self.network._apply_projections(pop_to_currents, last_spikes)

            for pop, (start, end) in zip(self.network.populations, pop_ranges):
                pid = id(pop)
                currents = pop_to_currents[pid]
                spikes = pop.step_all(currents)
                last_spikes[pid] = spikes

                all_spikes[t, start:end] = spikes
                all_voltages[t, start:end] = pop.voltages
                all_currents[t, start:end] = currents

                # Record to monitors
                self.network._record(pop, spikes, t, dt)

            self.network._update_plasticity(last_spikes)

        return ExecutionTrace(
            n_neurons=total_neurons,
            n_steps=n_steps,
            spikes=all_spikes,
            voltages=all_voltages,
            currents=all_currents,
            population_labels=pop_labels,
            population_ranges=pop_ranges,
        )
