# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWaveformAmplitude from former test_spike_stats_waveform.py

"""Focused suite: TestWaveformAmplitude from former test_spike_stats_waveform.py."""

from __future__ import annotations

from tests.spike_stats_waveform_support import *  # noqa: F403

class TestWaveformAmplitude:
    def test_typical(self):
        w = _typical_waveform()
        result = waveform_amplitude(w)
        assert result > 0

    def test_flat(self):
        result = waveform_amplitude(np.zeros(10))
        assert result == 0.0
