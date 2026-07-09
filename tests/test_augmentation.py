# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

# Tests for sc_neurocore.augmentation (spike augment + curriculum)

from __future__ import annotations

import numpy as np

from sc_neurocore.augmentation import SpikeAugment, SpikeCurriculum


def _make_spikes(T=20, N=10, rate=0.2, seed=42):
    rng = np.random.RandomState(seed)
    return (rng.random((T, N)) < rate).astype(np.int8)


class TestSpikeAugment:
    def test_identity(self):
        aug = SpikeAugment()
        spikes = _make_spikes()
        out = aug(spikes)
        np.testing.assert_array_equal(out, spikes)

    def test_temporal_jitter(self):
        aug = SpikeAugment(jitter_steps=2, seed=42)
        spikes = _make_spikes()
        out = aug(spikes)
        assert out.shape == spikes.shape
        # Total spike count may differ slightly due to collisions
        assert out.sum() > 0

    def test_spike_dropout(self):
        aug = SpikeAugment(dropout_rate=0.5, seed=42)
        spikes = _make_spikes(rate=0.5)
        out = aug(spikes)
        assert out.sum() < spikes.sum()

    def test_dropout_zero(self):
        aug = SpikeAugment(dropout_rate=0.0)
        spikes = _make_spikes()
        out = aug(spikes)
        np.testing.assert_array_equal(out, spikes)

    def test_rate_scaling_down(self):
        aug = SpikeAugment(rate_scale=(0.3, 0.3), seed=42)
        spikes = _make_spikes(rate=0.5)
        out = aug(spikes)
        assert out.sum() < spikes.sum()

    def test_rate_scaling_identity(self):
        aug = SpikeAugment(rate_scale=(1.0, 1.0))
        spikes = _make_spikes()
        out = aug(spikes)
        np.testing.assert_array_equal(out, spikes)

    def test_polarity_flip(self):
        aug = SpikeAugment(polarity_flip_prob=1.0, seed=42)
        spikes = np.zeros((10, 8), dtype=np.int8)
        spikes[:, :4] = 1  # ON channels
        out = aug(spikes)
        # ON and OFF should be swapped
        assert out[:, 4:].sum() == spikes[:, :4].sum()
        assert out[:, :4].sum() == 0

    def test_polarity_flip_odd_channels(self):
        aug = SpikeAugment(polarity_flip_prob=1.0, seed=42)
        spikes = _make_spikes(N=7)  # odd channels — no flip
        out = aug(spikes)
        np.testing.assert_array_equal(out, spikes)

    def test_background_noise(self):
        aug = SpikeAugment(bg_noise_rate=0.5, seed=42)
        spikes = np.zeros((20, 10), dtype=np.int8)
        out = aug(spikes)
        assert out.sum() > 0

    def test_hot_pixel(self):
        aug = SpikeAugment(hot_pixel_prob=0.5, seed=42)
        spikes = np.zeros((20, 10), dtype=np.int8)
        out = aug(spikes)
        # Some columns should be all-ones
        col_sums = out.sum(axis=0)
        assert any(s == 20 for s in col_sums)

    def test_combined(self):
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

    def test_output_dtype_preserved(self):
        aug = SpikeAugment(dropout_rate=0.1, seed=42)
        spikes = _make_spikes().astype(np.int8)
        out = aug(spikes)
        assert out.dtype == np.int8


class TestSpikeCurriculum:
    def test_timesteps_progression(self):
        c = SpikeCurriculum(total_epochs=100, start_timesteps=10, end_timesteps=200)
        assert c.timesteps(0) == 10
        t_mid = c.timesteps(50)
        assert 10 < t_mid <= 200
        t_end = c.timesteps(100)
        assert t_end == 200

    def test_rate_scale_progression(self):
        c = SpikeCurriculum(
            total_epochs=100,
            start_rate_scale=3.0,
            end_rate_scale=1.0,
            warmup_fraction=0.5,
        )
        assert c.rate_scale(0) == 3.0
        assert c.rate_scale(100) == 1.0
        assert 1.0 < c.rate_scale(25) < 3.0

    def test_noise_progression(self):
        c = SpikeCurriculum(
            total_epochs=100,
            start_noise=0.0,
            end_noise=0.1,
            warmup_fraction=0.5,
        )
        assert c.noise_rate(0) == 0.0
        assert c.noise_rate(100) == 0.1

    def test_apply_to_spikes_truncate(self):
        c = SpikeCurriculum(total_epochs=100, start_timesteps=5, end_timesteps=50)
        spikes = _make_spikes(T=100)
        out = c.apply_to_spikes(spikes, epoch=0)
        assert out.shape[0] == 5

    def test_apply_to_spikes_pad(self):
        c = SpikeCurriculum(total_epochs=100, start_timesteps=50, end_timesteps=50)
        spikes = _make_spikes(T=10)
        out = c.apply_to_spikes(spikes, epoch=0)
        assert out.shape[0] == 50

    def test_apply_rate_amplification(self):
        c = SpikeCurriculum(
            total_epochs=100,
            start_timesteps=20,
            end_timesteps=20,
            start_rate_scale=2.0,
            end_rate_scale=1.0,
            warmup_fraction=0.5,
        )
        spikes = _make_spikes(T=20, rate=0.1)
        out = c.apply_to_spikes(spikes, epoch=0, seed=42)
        # Rate scale 2.0 should add extra spikes
        assert out.sum() >= spikes.sum()

    def test_schedule_summary(self):
        c = SpikeCurriculum(total_epochs=50, start_timesteps=10, end_timesteps=100)
        s = c.schedule_summary()
        assert "Epoch" in s
        assert "Rate Scale" in s

    def test_warmup_zero(self):
        c = SpikeCurriculum(total_epochs=100, warmup_fraction=0.0)
        assert c.timesteps(0) == c.end_timesteps
