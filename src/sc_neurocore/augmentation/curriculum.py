# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Curriculum learning scheduler for SNN training

"""Curriculum learning: start easy, ramp to hard.

Schedules training difficulty along multiple axes:
- Sequence length: start short, increase over epochs
- Firing rate scale: start with amplified rates, decay to natural
- Noise level: start clean, add noise progressively
- Augmentation intensity: ramp up over training

Reference: Bengio et al. 2009 — "Curriculum Learning"
Applied to SNNs: no framework provides this as a built-in scheduler.
"""

from __future__ import annotations

from typing import Any

from dataclasses import dataclass

import numpy as np


@dataclass
class SpikeCurriculum:
    """Schedule training difficulty across epochs.

    Parameters
    ----------
    total_epochs : int
        Total training epochs.
    start_timesteps : int
        Initial sequence length.
    end_timesteps : int
        Final sequence length.
    start_rate_scale : float
        Initial firing rate multiplier (>1 = amplified = easier).
    end_rate_scale : float
        Final firing rate multiplier (1.0 = natural).
    start_noise : float
        Initial background noise rate.
    end_noise : float
        Final background noise rate.
    warmup_fraction : float
        Fraction of epochs for linear warmup (0.0-1.0).
    """

    total_epochs: int
    start_timesteps: int = 10
    end_timesteps: int = 100
    start_rate_scale: float = 2.0
    end_rate_scale: float = 1.0
    start_noise: float = 0.0
    end_noise: float = 0.05
    warmup_fraction: float = 0.3

    def _progress(self, epoch: int) -> float:
        """Compute curriculum progress in [0, 1]."""
        warmup_end = int(self.total_epochs * self.warmup_fraction)
        if warmup_end <= 0:
            return 1.0
        return min(1.0, epoch / warmup_end)

    def timesteps(self, epoch: int) -> int:
        """Sequence length for this epoch."""
        p = self._progress(epoch)
        return int(self.start_timesteps + p * (self.end_timesteps - self.start_timesteps))

    def rate_scale(self, epoch: int) -> float:
        """Return the firing-rate multiplier for the given epoch."""
        p = self._progress(epoch)
        return self.start_rate_scale + p * (self.end_rate_scale - self.start_rate_scale)

    def noise_rate(self, epoch: int) -> float:
        """Background noise rate for this epoch."""
        p = self._progress(epoch)
        return self.start_noise + p * (self.end_noise - self.start_noise)

    def apply_to_spikes(
        self, spikes: np.ndarray[Any, Any], epoch: int, seed: int = 0
    ) -> np.ndarray[Any, Any]:
        """Apply curriculum-scheduled transforms to a spike tensor.

        Parameters
        ----------
        spikes : ndarray of shape (T, n_neurons)
        epoch : int
        seed : int

        Returns
        -------
        ndarray
            Transformed spikes (possibly truncated/padded to scheduled T).
        """
        rng = np.random.RandomState(seed)
        T_target = self.timesteps(epoch)
        T_actual = spikes.shape[0]

        # Truncate or pad to scheduled length
        if T_actual > T_target:
            out = spikes[:T_target].copy()
        elif T_actual < T_target:
            pad = np.zeros((T_target - T_actual, spikes.shape[1]), dtype=spikes.dtype)
            out = np.concatenate([spikes, pad], axis=0)
        else:
            out = spikes.copy()

        out = out.astype(np.float64)

        # Rate scaling (probabilistic spike duplication or dropout)
        scale = self.rate_scale(epoch)
        if scale < 1.0:  # pragma: no cover
            mask = rng.random(out.shape) < scale
            out = out * mask
        elif scale > 1.0:
            extra = (rng.random(out.shape) < (scale - 1.0)).astype(np.float64)
            out = np.clip(out + extra * (1 - out), 0, 1)

        # Add noise
        noise = self.noise_rate(epoch)
        if noise > 0:  # pragma: no cover
            noise_spikes = (rng.random(out.shape) < noise).astype(np.float64)
            out = np.clip(out + noise_spikes, 0, 1)

        return out.astype(spikes.dtype)

    def schedule_summary(self) -> str:
        """Print the curriculum schedule."""
        lines = ["Epoch | T    | Rate Scale | Noise"]
        lines.append("-" * 40)
        for e in range(0, self.total_epochs, max(1, self.total_epochs // 10)):
            lines.append(
                f"{e:5d} | {self.timesteps(e):4d} | {self.rate_scale(e):10.2f} | {self.noise_rate(e):.4f}"
            )
        lines.append(
            f"{self.total_epochs:5d} | {self.timesteps(self.total_epochs):4d} | "
            f"{self.rate_scale(self.total_epochs):10.2f} | {self.noise_rate(self.total_epochs):.4f}"
        )
        return "\n".join(lines)
