# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Record spikes over time and compute basic statistics

"""Bitstream spike recorder for binary spike-train statistics.

The recorder stores a one-dimensional spike stream with one binary sample per
time step. It provides deterministic NumPy-backed summaries used by stochastic
core tests, layer recorders, examples, and API documentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class BitstreamSpikeRecorder:
    """Record a binary spike train and compute basic spike statistics.

    Parameters
    ----------
    dt_ms:
        Duration represented by one recorded sample in milliseconds. A value of
        ``0.0`` is accepted for legacy zero-duration dry runs and produces a
        firing rate of ``0.0``.
    spikes:
        Existing spike samples to seed the recorder with. Values must be binary
        integers where ``1`` means a spike occurred at that sample.
    """

    dt_ms: float = 1.0
    spikes: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate seeded recorder state."""
        if self.dt_ms < 0.0:
            raise ValueError("dt_ms must be non-negative.")
        for spike in self.spikes:
            self._validate_spike(spike)

    @staticmethod
    def _validate_spike(spike: int) -> None:
        if spike not in (0, 1):
            raise ValueError("Spike must be 0 or 1.")

    def record(self, spike: int) -> None:
        """Append one binary spike sample.

        Parameters
        ----------
        spike:
            Binary sample to record. ``1`` represents a spike and ``0``
            represents silence for the current sample.

        Raises
        ------
        ValueError
            If ``spike`` is not ``0`` or ``1``.
        """
        self._validate_spike(spike)
        self.spikes.append(spike)

    def reset(self) -> None:
        """Remove all recorded spike samples while preserving ``dt_ms``."""
        self.spikes.clear()

    def as_array(self) -> np.ndarray[Any, Any]:
        """Return the recorded spike train as a NumPy ``uint8`` array."""
        return np.array(self.spikes, dtype=np.uint8)

    def total_spikes(self) -> int:
        """Return the number of recorded spike samples equal to ``1``."""
        return int(np.sum(self.as_array()))

    def firing_rate_hz(self) -> float:
        """Return the mean firing rate in hertz.

        The rate is computed as ``total_spikes / duration_seconds``. Empty
        recordings and legacy ``dt_ms == 0.0`` dry runs return ``0.0``.
        """
        spikes = self.as_array()
        sample_count = spikes.size
        if sample_count == 0:
            return 0.0
        duration_ms = sample_count * self.dt_ms
        if duration_ms == 0:
            return 0.0
        return float(self.total_spikes() / (duration_ms / 1000.0))

    def isi_histogram(
        self,
        bins: int = 10,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Compute a histogram of inter-spike intervals in milliseconds.

        Parameters
        ----------
        bins:
            Number of histogram bins. Must be positive.

        Returns
        -------
        hist:
            Histogram counts.
        bin_edges:
            Bin edges (ms).

        Raises
        ------
        ValueError
            If ``bins`` is less than one.
        """
        if bins < 1:
            raise ValueError("bins must be positive.")
        spikes = self.as_array()
        spike_indices = np.where(spikes == 1)[0]

        if spike_indices.size < 2:
            return np.zeros(bins, dtype=int), np.linspace(0, 1, bins + 1)

        isi_steps = np.diff(spike_indices)
        isi_ms = isi_steps * self.dt_ms

        histogram_range = (float(np.min(isi_ms)), float(np.max(isi_ms)))
        hist, bin_edges = np.histogram(isi_ms, bins=bins, range=histogram_range)
        return hist, bin_edges
