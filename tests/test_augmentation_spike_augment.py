# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikeAugment from former test_augmentation.py

"""Focused suite: TestSpikeAugment from former test_augmentation.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.augmentation import (
    SpikeAugment,
)
from tests.augmentation_support import (
    _make_spikes,
)


class TestSpikeAugment:
    """Augmenting spike rasters."""

    def test_identity(self) -> None:
        """No augmentation leaves the raster unchanged."""
        aug = SpikeAugment()
        spikes = _make_spikes()
        out = aug(spikes)
        np.testing.assert_array_equal(out, spikes)

    def test_temporal_jitter(self) -> None:
        """Jitter keeps the shape and some spikes."""
        aug = SpikeAugment(jitter_steps=2, seed=42)
        spikes = _make_spikes()
        out = aug(spikes)
        assert out.shape == spikes.shape
        # Total spike count may differ slightly due to collisions
        assert out.sum() > 0

    def test_spike_dropout(self) -> None:
        """Dropout removes spikes."""
        aug = SpikeAugment(dropout_rate=0.5, seed=42)
        spikes = _make_spikes(rate=0.5)
        out = aug(spikes)
        assert out.sum() < spikes.sum()

    def test_dropout_zero(self) -> None:
        """Zero dropout leaves the raster unchanged."""
        aug = SpikeAugment(dropout_rate=0.0)
        spikes = _make_spikes()
        out = aug(spikes)
        np.testing.assert_array_equal(out, spikes)

    def test_rate_scaling_down(self) -> None:
        """Scaling the rate down removes spikes."""
        aug = SpikeAugment(rate_scale=(0.3, 0.3), seed=42)
        spikes = _make_spikes(rate=0.5)
        out = aug(spikes)
        assert out.sum() < spikes.sum()

    def test_rate_scaling_identity(self) -> None:
        """A rate scale of 1 leaves the raster unchanged."""
        aug = SpikeAugment(rate_scale=(1.0, 1.0))
        spikes = _make_spikes()
        out = aug(spikes)
        np.testing.assert_array_equal(out, spikes)

    def test_polarity_flip(self) -> None:
        """A polarity flip swaps the two channel halves."""
        aug = SpikeAugment(polarity_flip_prob=1.0, seed=42)
        spikes = np.zeros((10, 8), dtype=np.int8)
        spikes[:, :4] = 1  # ON channels
        out = aug(spikes)
        # ON and OFF should be swapped
        assert out[:, 4:].sum() == spikes[:, :4].sum()
        assert out[:, :4].sum() == 0

    def test_polarity_flip_odd_channels(self) -> None:
        """An odd channel count is left unflipped."""
        aug = SpikeAugment(polarity_flip_prob=1.0, seed=42)
        spikes = _make_spikes(N=7)  # odd channels — no flip
        out = aug(spikes)
        np.testing.assert_array_equal(out, spikes)

    def test_background_noise(self) -> None:
        """Background noise adds spikes."""
        aug = SpikeAugment(bg_noise_rate=0.5, seed=42)
        spikes = np.zeros((20, 10), dtype=np.int8)
        out = aug(spikes)
        assert out.sum() > 0

    def test_hot_pixel(self) -> None:
        """A hot pixel fires on every step."""
        aug = SpikeAugment(hot_pixel_prob=0.5, seed=42)
        spikes = np.zeros((20, 10), dtype=np.int8)
        out = aug(spikes)
        # Some columns should be all-ones
        col_sums = out.sum(axis=0)
        assert any(s == 20 for s in col_sums)

    def test_combined(self) -> None:
        """Combined augmentations keep a binary raster."""
        aug = SpikeAugment(
            jitter_steps=1,
            dropout_rate=0.1,
            bg_noise_rate=0.01,
            seed=42,
        )
        spikes = _make_spikes()
        out = aug(spikes)
        assert out.shape == spikes.shape
        assert set(np.unique(out)).issubset({0, 1})

    def test_output_dtype_preserved(self) -> None:
        """Augmentation keeps the int8 dtype."""
        aug = SpikeAugment(dropout_rate=0.1, seed=42)
        spikes = _make_spikes().astype(np.int8)
        out = aug(spikes)
        assert out.dtype == np.int8
