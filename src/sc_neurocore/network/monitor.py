# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Monitors: spike, state, and rate recording

"""Monitors: spike, state, and rate recording during simulation."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .population import Population


class SpikeMonitor:
    """Records (neuron_idx, timestep) pairs from a population."""

    def __init__(self, population: Population, label: str | None = None) -> None:
        self.population = population
        self.label = label or f"spikes_{population.label}"
        self._neuron_ids: list[int] = []
        self._timesteps: list[int] = []

    def record(self, spikes: np.ndarray[Any, Any], t_step: int) -> None:
        """Store spike events for this timestep (from binary spike vector)."""
        idx = np.nonzero(spikes)[0]
        for i in idx:
            self._neuron_ids.append(int(i))
            self._timesteps.append(t_step)

    def record_event(self, neuron_id: int, t_step: int) -> None:
        """Store a single spike event directly (from Rust backend)."""
        self._neuron_ids.append(neuron_id)
        self._timesteps.append(t_step)

    @property
    def spike_times(self) -> np.ndarray[Any, Any]:
        """All spike timesteps as 1-D array."""
        return np.array(self._timesteps, dtype=np.int64)

    @property
    def spike_trains(self) -> dict[int, np.ndarray[Any, Any]]:
        """Per-neuron spike timestep arrays."""
        trains: dict[int, list[int]] = {}
        for nid, ts in zip(self._neuron_ids, self._timesteps):
            trains.setdefault(nid, []).append(ts)
        return {k: np.array(v, dtype=np.int64) for k, v in trains.items()}

    @property
    def count(self) -> int:
        """Total number of spikes recorded."""
        return len(self._neuron_ids)

    def raster_data(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Return (timesteps, neuron_ids) arrays for raster plots."""
        return (
            np.array(self._timesteps, dtype=np.int64),
            np.array(self._neuron_ids, dtype=np.int64),
        )

    def firing_rates(self, n_steps: int, dt: float = 0.001) -> np.ndarray[Any, Any]:
        """Mean firing rate (Hz) per neuron over the simulation."""
        duration = n_steps * dt
        rates = np.zeros(self.population.n, dtype=np.float64)
        if duration <= 0:
            return rates
        for nid in self._neuron_ids:
            rates[nid] += 1.0
        rates /= duration
        return rates

    def isi(self, neuron: int) -> np.ndarray[Any, Any]:
        """Inter-spike intervals (timestep units) for a single neuron."""
        trains = self.spike_trains
        ts = trains.get(neuron, np.array([], dtype=np.int64))
        if ts.size < 2:
            return np.array([], dtype=np.int64)
        return np.diff(ts)

    def cross_correlation(
        self, i: int, j: int, max_lag: int = 50
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Cross-correlogram between neurons i and j."""
        from sc_neurocore.analysis.spike_stats import cross_correlation as _cc

        trains = self.spike_trains
        ts_i = trains.get(i, np.array([], dtype=np.int64))
        ts_j = trains.get(j, np.array([], dtype=np.int64))
        lags = np.arange(-max_lag, max_lag + 1)
        if ts_i.size == 0 or ts_j.size == 0:
            return np.zeros(len(lags)), lags
        max_t = max(ts_i.max(), ts_j.max()) + 1
        bin_i = np.zeros(max_t, dtype=np.int8)
        bin_j = np.zeros(max_t, dtype=np.int8)
        bin_i[ts_i] = 1
        bin_j[ts_j] = 1
        # One sample per timestep: the delegate takes dt in seconds and the lag in
        # milliseconds, so a 1 ms sample makes max_lag_ms count timesteps. dt=1.0
        # (one second per sample) collapsed every lag below 1000 steps to zero.
        correlation, _lags_ms = _cc(bin_i, bin_j, max_lag_ms=float(max_lag), dt=0.001)
        return correlation, lags


class StateMonitor:
    """Records state variable traces from a population."""

    def __init__(
        self,
        population: Population,
        variables: list[str] | None = None,
        record: list[int] | None = None,
    ) -> None:
        self.population = population
        self.variables = variables or ["v"]
        if record is not None:
            # An integer index array, empty included; refused before any run.
            indices = np.asarray(record, dtype=np.int64).reshape(-1)
            if indices.size and (indices.min() < 0 or indices.max() >= population.n):
                raise IndexError(
                    f"record indices must lie in [0, {population.n}); got {list(record)}"
                )
            record = [int(i) for i in indices]
        self.record = record
        self._data: dict[str, list[np.ndarray[Any, Any]]] = {v: [] for v in self.variables}
        self._t: list[int] = []

    def snapshot(self, t_step: int) -> None:
        """Capture current state variables."""
        self._t.append(t_step)
        states = self.population.get_states()
        for v in self.variables:
            arr = states.get(v, np.zeros(self.population.n))
            if self.record is not None:
                arr = arr[np.asarray(self.record, dtype=np.int64)]
            self._data[v].append(arr.copy())

    @property
    def traces(self) -> dict[str, np.ndarray[Any, Any]]:
        """Variable traces as {name: (n_steps, n_neurons)} arrays."""
        return {k: np.array(v) if v else np.empty((0, 0)) for k, v in self._data.items()}

    @property
    def t(self) -> np.ndarray[Any, Any]:
        """Return the recorded snapshot timestep array."""
        return np.array(self._t, dtype=np.int64)


class RateMonitor:
    """Population firing rate in time bins."""

    def __init__(self, population: Population, bin_ms: int = 10) -> None:
        self.population = population
        self.bin_ms = bin_ms
        self._spike_counts: list[int] = []
        self._bin_edges: list[int] = []
        self._exposures: list[float] = []
        self._current_count = 0
        self._steps_in_bin = 0
        self._current_exposure = 0.0

    def record(self, spikes: np.ndarray[Any, Any], t_step: int, dt: float = 0.001) -> None:
        """Accumulate spikes; flush when a bin completes.

        A bin holds the whole number of steps nearest below ``bin_ms`` (at least
        one); its rate divides by the time those steps actually cover, so a
        10 ms bin at dt 3 ms is a 9 ms bin and a 1 ms bin at dt 10 ms a 10 ms one.
        """
        self._current_count += int(spikes.sum())
        self._steps_in_bin += 1
        self._current_exposure += dt
        # The small epsilon keeps 0.003 / 0.001 = 2.9999999999999996 at 3 steps.
        steps_per_bin = max(1, int(self.bin_ms / 1000.0 / dt + 1e-9))
        if self._steps_in_bin >= steps_per_bin:
            self._spike_counts.append(self._current_count)
            self._bin_edges.append(t_step)
            self._exposures.append(self._current_exposure)
            self._current_count = 0
            self._steps_in_bin = 0
            self._current_exposure = 0.0

    @property
    def rate(self) -> np.ndarray[Any, Any]:
        """Firing rate (Hz) per bin: spikes per neuron over the time the bin covered."""
        if not self._spike_counts:
            return np.array([], dtype=np.float64)
        counts = np.array(self._spike_counts, dtype=np.float64)
        exposure = np.array(self._exposures, dtype=np.float64)
        return counts / (exposure * self.population.n)

    @property
    def t(self) -> np.ndarray[Any, Any]:
        """Bin edge timestep array."""
        return np.array(self._bin_edges, dtype=np.int64)
