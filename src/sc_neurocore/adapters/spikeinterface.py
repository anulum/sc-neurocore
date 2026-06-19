# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SpikeInterface / Neo adapter for experimental data

"""SpikeInterface/Neo adapter: import experimental spike data into SC-NeuroCore.

Converts between SpikeInterface sorting results (or raw spike trains)
and SC-NeuroCore's internal representations (Population spike arrays,
TensorStream, bitstream encoding).

Without SpikeInterface installed, provides pure-NumPy conversion
functions that accept the same data format (unit_ids, spike_times).

    from sc_neurocore.adapters.spikeinterface import (
        spike_trains_to_bitstreams,
        spike_trains_to_population_input,
        from_sorting,  # requires spikeinterface
    )
"""

from __future__ import annotations

from typing import Any

import numpy as np


def spike_trains_to_bitstreams(
    spike_times: dict[int, np.ndarray[Any, Any]],
    duration_ms: float,
    dt: float = 1.0,
) -> np.ndarray[Any, Any]:
    """Convert spike times to binary bitstream matrix.

    Parameters
    ----------
    spike_times : dict mapping unit_id → array of spike times (ms)
    duration_ms : float
        Total recording duration in ms.
    dt : float
        Time bin width in ms.

    Returns
    -------
    np.ndarray
        Shape (n_units, n_bins), dtype uint8, binary {0, 1}.
    """
    n_bins = int(np.ceil(duration_ms / dt))
    unit_ids = sorted(spike_times.keys())
    n_units = len(unit_ids)

    matrix = np.zeros((n_units, n_bins), dtype=np.uint8)
    for i, uid in enumerate(unit_ids):
        times = np.asarray(spike_times[uid], dtype=np.float64)
        bins = np.clip((times / dt).astype(int), 0, n_bins - 1)
        matrix[i, bins] = 1

    return matrix


def spike_trains_to_population_input(
    spike_times: dict[int, np.ndarray[Any, Any]],
    duration_ms: float,
    dt: float = 1.0,
) -> np.ndarray[Any, Any]:
    """Convert spike times to current input array for Population.step_all().

    Each spike becomes a current pulse of amplitude 1.0 at the spike time bin.

    Parameters
    ----------
    spike_times : dict mapping unit_id → array of spike times (ms)
    duration_ms : float
    dt : float

    Returns
    -------
    np.ndarray
        Shape (n_timesteps, n_units), suitable for time-stepped simulation.
    """
    bitstreams = spike_trains_to_bitstreams(spike_times, duration_ms, dt)
    return bitstreams.T.astype(np.float64)


def firing_rates_to_sc_probs(
    spike_times: dict[int, np.ndarray[Any, Any]],
    duration_ms: float,
    max_rate_hz: float = 100.0,
) -> np.ndarray[Any, Any]:
    """Convert firing rates to SC probabilities in [0, 1].

    Parameters
    ----------
    spike_times : dict mapping unit_id → array of spike times (ms)
    duration_ms : float
    max_rate_hz : float
        Rate corresponding to probability 1.0.

    Returns
    -------
    np.ndarray
        Shape (n_units,), probabilities in [0, 1].
    """
    unit_ids = sorted(spike_times.keys())
    probs = np.zeros(len(unit_ids))
    for i, uid in enumerate(unit_ids):
        n_spikes = len(spike_times[uid])
        rate_hz = n_spikes / (duration_ms / 1000.0)
        probs[i] = np.clip(rate_hz / max_rate_hz, 0.0, 1.0)
    return probs


def from_sorting(sorting: Any, dt: float = 1.0) -> np.ndarray[Any, Any]:  # pragma: no cover
    """Convert a SpikeInterface SortingExtractor to bitstream matrix.

    Parameters
    ----------
    sorting : spikeinterface.core.BaseSorting
        SpikeInterface sorting result.
    dt : float
        Time bin width in ms.

    Returns
    -------
    np.ndarray
        Shape (n_units, n_bins), dtype uint8.
    """
    unit_ids = sorting.get_unit_ids()
    fs = sorting.get_sampling_frequency()
    n_frames = sorting.get_total_samples()
    duration_ms = n_frames / fs * 1000.0

    spike_times = {}
    for uid in unit_ids:
        frames = sorting.get_unit_spike_train(uid)
        spike_times[int(uid)] = frames / fs * 1000.0  # convert to ms

    return spike_trains_to_bitstreams(spike_times, duration_ms, dt)
