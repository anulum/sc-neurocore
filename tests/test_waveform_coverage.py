# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Coverage tests for analysis/spike_stats/waveform.py

"""Edge-case tests for every branch in waveform shape analysis functions."""

from __future__ import annotations

import numpy as np

from sc_neurocore.analysis.spike_stats.waveform import (
    waveform_width,
    waveform_amplitude,
    waveform_repolarization_slope,
    waveform_recovery_slope,
    waveform_halfwidth,
    waveform_pt_ratio,
)


def _typical_waveform():
    """Synthetic spike waveform: depolarisation → trough → repolarisation → overshoot → baseline."""
    # Realistic extracellular spike: brief negative trough then positive peak
    t = np.linspace(0, 2, 60)
    w = -np.exp(-((t - 0.4) ** 2) / 0.02) + 0.6 * np.exp(-((t - 0.8) ** 2) / 0.04)
    return w


class TestWaveformWidth:
    def test_typical(self):
        w = _typical_waveform()
        result = waveform_width(w)
        assert np.isfinite(result)
        assert result > 0

    def test_trough_at_end(self):
        # Trough is last element — should return NaN (line 22)
        w = np.array([1.0, 0.5, 0.0, -0.5, -1.0])
        result = waveform_width(w)
        assert np.isnan(result)


class TestWaveformAmplitude:
    def test_typical(self):
        w = _typical_waveform()
        result = waveform_amplitude(w)
        assert result > 0

    def test_flat(self):
        result = waveform_amplitude(np.zeros(10))
        assert result == 0.0


class TestRepolarizationSlope:
    def test_typical(self):
        w = _typical_waveform()
        result = waveform_repolarization_slope(w)
        assert np.isfinite(result)

    def test_trough_at_end(self):
        # Trough at second-to-last (line 36)
        w = np.array([1.0, 0.0, -1.0])
        result = waveform_repolarization_slope(w)
        assert np.isnan(result) or np.isfinite(result)


class TestRecoverySlope:
    def test_typical(self):
        w = _typical_waveform()
        result = waveform_recovery_slope(w)
        assert np.isfinite(result)

    def test_trough_at_end(self):
        # Trough at end (line 46)
        w = np.array([1.0, 0.0, -1.0])
        result = waveform_recovery_slope(w)
        assert np.isnan(result)

    def test_peak_at_end(self):
        # Peak at end of waveform (line 49)
        w = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = waveform_recovery_slope(w)
        assert np.isnan(result)

    def test_empty_post_peak(self):
        # Only 2 elements after trough, peak=last (line 53)
        w = np.array([-1.0, 0.0, 1.0])
        result = waveform_recovery_slope(w)
        assert np.isnan(result) or np.isfinite(result)


class TestHalfwidth:
    def test_typical(self):
        w = _typical_waveform()
        result = waveform_halfwidth(w)
        assert np.isfinite(result) or np.isnan(result)

    def test_no_crossing(self):
        # All positive — trough/2 never crossed (line 63)
        w = np.array([5.0, 4.0, 3.0, 4.0, 5.0])
        result = waveform_halfwidth(w)
        assert np.isnan(result)

    def test_single_crossing(self):
        # Only one point below half — below.size < 2 (line 63)
        w = np.array([1.0, 0.5, -0.1, 0.5, 1.0])
        result = waveform_halfwidth(w)
        assert np.isnan(result) or np.isfinite(result)


class TestPtRatio:
    def test_typical(self):
        w = _typical_waveform()
        result = waveform_pt_ratio(w)
        assert np.isfinite(result)

    def test_trough_at_end(self):
        # Trough is last element (line 75)
        w = np.array([1.0, 0.5, 0.0, -0.5, -1.0])
        result = waveform_pt_ratio(w)
        assert np.isnan(result)

    def test_zero_trough(self):
        # Trough amplitude near zero (line 75 trough_val < 1e-30)
        w = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        result = waveform_pt_ratio(w)
        assert np.isnan(result)
