# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikeCurriculum from former test_augmentation.py

"""Focused suite: TestSpikeCurriculum from former test_augmentation.py."""

from __future__ import annotations

from tests.augmentation_support import *  # noqa: F403


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
