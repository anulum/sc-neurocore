# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Spike-domain data augmentation

"""Spike-aware data augmentation that preserves temporal causality.

Standard image augmentations (flip, rotate, crop) don't apply to spike
trains. This module provides spike-native transforms: temporal jitter,
spike dropout, rate scaling, polarity flip, background noise injection,
and hot pixel simulation.

No SNN framework provides composable spike-domain augmentation.
"""

from __future__ import annotations

from typing import Any

from dataclasses import dataclass

import numpy as np


@dataclass
class SpikeAugment:
    """Composable spike-domain augmentation.

    Parameters
    ----------
    jitter_steps : int
        Max temporal jitter in timesteps (spikes shift +/- jitter).
    dropout_rate : float
        Probability of dropping each spike (0.0 = none, 1.0 = all).
    rate_scale : tuple of float
        (min_scale, max_scale) for random firing rate scaling.
    polarity_flip_prob : float
        Probability of flipping spike polarity (for DVS ON/OFF channels).
    bg_noise_rate : float
        Background noise spike probability per neuron per step.
    hot_pixel_prob : float
        Probability of a neuron becoming a hot pixel (fires every step).
    seed : int
        Random seed for reproducibility.
    """

    jitter_steps: int = 0
    dropout_rate: float = 0.0
    rate_scale: tuple[float, float] = (1.0, 1.0)
    polarity_flip_prob: float = 0.0
    bg_noise_rate: float = 0.0
    hot_pixel_prob: float = 0.0
    seed: int = 42

    def __call__(self, spikes: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Apply all augmentations to a spike tensor.

        Parameters
        ----------
        spikes : ndarray of shape (T, n_neurons)
            Binary spike matrix.

        Returns
        -------
        ndarray of same shape
            Augmented spike matrix.
        """
        rng = np.random.RandomState(self.seed)
        out = spikes.copy().astype(np.float64)

        if self.jitter_steps > 0:
            out = self._temporal_jitter(out, rng)

        if self.dropout_rate > 0:
            out = self._spike_dropout(out, rng)

        if self.rate_scale != (1.0, 1.0):
            out = self._rate_scaling(out, rng)

        if self.polarity_flip_prob > 0:
            out = self._polarity_flip(out, rng)

        if self.bg_noise_rate > 0:
            out = self._background_noise(out, rng)

        if self.hot_pixel_prob > 0:
            out = self._hot_pixel(out, rng)

        return np.clip(out, 0, 1).astype(spikes.dtype)

    def _temporal_jitter(
        self, spikes: np.ndarray[Any, Any], rng: np.random.RandomState
    ) -> np.ndarray[Any, Any]:
        T, N = spikes.shape
        result = np.zeros_like(spikes)
        for t in range(T):
            for n in range(N):
                if spikes[t, n] > 0:
                    shift = rng.randint(-self.jitter_steps, self.jitter_steps + 1)
                    new_t = max(0, min(T - 1, t + shift))
                    result[new_t, n] = 1.0
        return result

    def _spike_dropout(
        self, spikes: np.ndarray[Any, Any], rng: np.random.RandomState
    ) -> np.ndarray[Any, Any]:
        mask = rng.random(spikes.shape) > self.dropout_rate
        dropped: np.ndarray[Any, Any] = spikes * mask
        return dropped

    def _rate_scaling(
        self, spikes: np.ndarray[Any, Any], rng: np.random.RandomState
    ) -> np.ndarray[Any, Any]:
        lo, hi = self.rate_scale
        scale = rng.uniform(lo, hi)
        if scale >= 1.0:  # pragma: no cover
            return spikes
        # Probabilistically drop spikes to reduce rate
        keep_prob = scale
        mask = rng.random(spikes.shape) < keep_prob
        scaled: np.ndarray[Any, Any] = spikes * mask
        return scaled

    def _polarity_flip(
        self, spikes: np.ndarray[Any, Any], rng: np.random.RandomState
    ) -> np.ndarray[Any, Any]:
        T, N = spikes.shape
        if N % 2 != 0:
            return spikes
        result = spikes.copy()
        if rng.random() < self.polarity_flip_prob:
            half = N // 2
            result[:, :half], result[:, half:] = spikes[:, half:].copy(), spikes[:, :half].copy()
        return result

    def _background_noise(
        self, spikes: np.ndarray[Any, Any], rng: np.random.RandomState
    ) -> np.ndarray[Any, Any]:
        noise = (rng.random(spikes.shape) < self.bg_noise_rate).astype(np.float64)
        noisy: np.ndarray[Any, Any] = np.clip(spikes + noise, 0, 1)
        return noisy

    def _hot_pixel(
        self, spikes: np.ndarray[Any, Any], rng: np.random.RandomState
    ) -> np.ndarray[Any, Any]:
        T, N = spikes.shape
        hot_mask = rng.random(N) < self.hot_pixel_prob
        result = spikes.copy()
        result[:, hot_mask] = 1.0
        return result
