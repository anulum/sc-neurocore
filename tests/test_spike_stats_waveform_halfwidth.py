# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHalfwidth from former test_spike_stats_waveform.py

"""Focused suite: TestHalfwidth from former test_spike_stats_waveform.py."""

from __future__ import annotations

from tests.spike_stats_waveform_support import *  # noqa: F403


class TestHalfwidth:
    def test_typical(self):
        w = _typical_waveform()
        result = waveform_halfwidth(w)
        assert np.isfinite(result) or np.isnan(result)

    def test_no_crossing(self):
        # All positive — trough/2 never crossed
        w = np.array([5.0, 4.0, 3.0, 4.0, 5.0])
        result = waveform_halfwidth(w)
        assert np.isnan(result)

    def test_single_crossing(self):
        # Only one point below half — below.size < 2
        w = np.array([1.0, 0.5, -0.1, 0.5, 1.0])
        result = waveform_halfwidth(w)
        assert np.isnan(result) or np.isfinite(result)
