# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWaveform from former test_spike_train_stats_extended.py

"""Focused suite: TestWaveform from former test_spike_train_stats_extended.py."""

from __future__ import annotations

from tests.spike_train_stats_extended_support import *  # noqa: F403

class TestWaveform:
    def test_waveform_width(self, waveform_fixture):
        w = waveform_width(waveform_fixture, dt=1.0 / 60)
        assert w > 0

    def test_waveform_amplitude(self, waveform_fixture):
        a = waveform_amplitude(waveform_fixture)
        assert a > 0

    def test_waveform_repolarization_slope(self, waveform_fixture):
        s = waveform_repolarization_slope(waveform_fixture, dt=1.0 / 60)
        assert s > 0

    def test_waveform_recovery_slope(self, waveform_fixture):
        s = waveform_recovery_slope(waveform_fixture, dt=1.0 / 60)
        assert np.isfinite(s)

    def test_waveform_halfwidth(self, waveform_fixture):
        hw = waveform_halfwidth(waveform_fixture, dt=1.0 / 60)
        assert hw > 0

    def test_waveform_pt_ratio(self, waveform_fixture):
        r = waveform_pt_ratio(waveform_fixture)
        assert r > 0
