# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Multimodal spike train fusion

"""Fuse spike trains from multiple sensor modalities.

Real neuromorphic systems combine vision (DVS), audio (cochlea),
IMU, and other sensors. This module aligns, normalizes, and merges
spike trains from sensors with different time resolutions and rates.

Fusion modes:
  - concatenate: stack channels from all modalities
  - sum: element-wise OR (any-modality spike)
  - attention: learned cross-modal weighting
"""

from __future__ import annotations

from dataclasses import dataclass

from typing import Any
import numpy as np


@dataclass
class ModalityConfig:
    """Configuration for one sensor modality."""

    name: str
    n_channels: int
    dt_us: float
    max_rate_hz: float = 1000.0


class MultiModalFusion:
    """Fuse spike trains from multiple sensor modalities.

    Parameters
    ----------
    modalities : list of ModalityConfig
        Sensor modality definitions.
    output_dt_us : float
        Output time bin width in microseconds (common timebase).
    mode : str
        Fusion mode: 'concatenate', 'sum', or 'attention'.
    """

    def __init__(
        self,
        modalities: list[ModalityConfig],
        output_dt_us: float = 1000.0,
        mode: str = "concatenate",
    ):
        self.modalities = modalities
        self.output_dt_us = output_dt_us
        self.mode = mode

        if mode == "concatenate":
            self.n_output = sum(m.n_channels for m in modalities)
        elif mode == "sum":
            max_ch = max(m.n_channels for m in modalities)
            self.n_output = max_ch
        elif mode == "attention":
            self.n_output = sum(m.n_channels for m in modalities)
            n_mod = len(modalities)
            self.attention_weights = np.ones(n_mod) / n_mod
        else:
            raise ValueError(f"Unknown mode '{mode}'")

    def fuse(
        self, spike_trains: dict[str, np.ndarray[Any, Any]], duration_us: float
    ) -> np.ndarray[Any, Any]:
        """Fuse spike trains from all modalities into a unified output.

        Parameters
        ----------
        spike_trains : dict mapping modality name to spike matrix
            Each matrix has shape (n_bins_modality, n_channels_modality).
        duration_us : float
            Total duration in microseconds.

        Returns
        -------
        ndarray of shape (n_output_bins, n_output_channels)
        """
        n_output_bins = max(1, int(np.ceil(duration_us / self.output_dt_us)))

        resampled = []
        for mod in self.modalities:
            if mod.name not in spike_trains:
                resampled.append(np.zeros((n_output_bins, mod.n_channels), dtype=np.float64))
                continue

            spikes = spike_trains[mod.name]
            n_bins_in = spikes.shape[0]

            # Resample to output timebase
            if n_bins_in == n_output_bins:
                resampled.append(spikes.astype(np.float64))
            else:
                # Linear resampling via bin mapping
                out = np.zeros((n_output_bins, mod.n_channels), dtype=np.float64)
                ratio = n_bins_in / max(n_output_bins, 1)
                for t_out in range(n_output_bins):
                    t_in_start = int(t_out * ratio)
                    t_in_end = min(int((t_out + 1) * ratio), n_bins_in)
                    if t_in_start < t_in_end:
                        out[t_out] = spikes[t_in_start:t_in_end].max(axis=0)
                resampled.append(out)

            # Rate normalization: scale so max rate maps to 1.0
            r = resampled[-1]
            max_val = r.max()
            if max_val > 0:
                resampled[-1] = r / max_val

        if self.mode == "concatenate":
            return np.concatenate(resampled, axis=1)

        if self.mode == "sum":
            # Pad smaller modalities and combine
            max_ch = self.n_output
            padded = []
            for r in resampled:
                if r.shape[1] < max_ch:
                    pad = np.zeros((r.shape[0], max_ch - r.shape[1]))
                    padded.append(np.concatenate([r, pad], axis=1))
                else:
                    padded.append(r[:, :max_ch])
            return np.clip(sum(padded), 0, 1)

        if self.mode == "attention":
            weighted = []
            for i, r in enumerate(resampled):
                weighted.append(r * self.attention_weights[i])
            return np.concatenate(weighted, axis=1)

        raise ValueError(f"Unknown mode '{self.mode}'")  # pragma: no cover
