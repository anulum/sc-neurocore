# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRepolarizationSlope from former test_spike_stats_waveform.py

"""Focused suite: TestRepolarizationSlope from former test_spike_stats_waveform.py."""

from __future__ import annotations

from tests.spike_stats_waveform_support import *  # noqa: F403

class TestRepolarizationSlope:
    def test_typical(self):
        w = _typical_waveform()
        result = waveform_repolarization_slope(w)
        assert np.isfinite(result)

    def test_trough_at_end(self):
        # Trough at second-to-last
        w = np.array([1.0, 0.0, -1.0])
        result = waveform_repolarization_slope(w)
        assert np.isnan(result) or np.isfinite(result)
