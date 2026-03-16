# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations
from typing import Any
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np


@dataclass
class BitstreamSpikeRecorder:
    """
    Record spikes over time and compute basic statistics.

    Stores a 1D bitstream of spikes (0/1) and provides:
    - total spikes
    - firing rate (Hz) given dt (ms)
    - inter-spike interval (ISI) histogram
    """

    dt_ms: float = 1.0
    spikes: List[int] = field(default_factory=list)

    def record(self, spike: int) -> None:
        if spike not in (0, 1):
            raise ValueError("Spike must be 0 or 1.")
        self.spikes.append(spike)

    def reset(self) -> None:
        self.spikes.clear()

    def as_array(self) -> np.ndarray[Any, Any]:
        return np.array(self.spikes, dtype=np.uint8)

    def total_spikes(self) -> int:
        return int(np.sum(self.as_array()))

    def firing_rate_hz(self) -> float:
        spikes = self.as_array()
        T = spikes.size
        if T == 0:
            return 0.0
        duration_ms = T * self.dt_ms
        if duration_ms == 0:
            return 0.0
        return float(self.total_spikes() / (duration_ms / 1000.0))

    def isi_histogram(
        self,
        bins: int = 10,
    ) -> Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """
        Compute histogram of inter-spike intervals in ms.

        Returns
        -------
        hist : np.ndarray
            Histogram counts.
        bin_edges : np.ndarray
            Bin edges (ms).
        """
        spikes = self.as_array()
        spike_indices = np.where(spikes == 1)[0]

        if spike_indices.size < 2:
            return np.zeros(bins, dtype=int), np.linspace(0, 1, bins + 1)

        isi_steps = np.diff(spike_indices)
        isi_ms = isi_steps * self.dt_ms

        hist, bin_edges = np.histogram(isi_ms, bins=bins)
        return hist, bin_edges
