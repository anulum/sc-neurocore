# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWaveform from former test_spike_train_stats_extended.py

"""Focused suite: TestWaveform from former test_spike_train_stats_extended.py."""

from __future__ import annotations

from typing import Any

from sc_neurocore.analysis.spike_stats import waveform_amplitude
from sc_neurocore.analysis.spike_stats import waveform_halfwidth
from sc_neurocore.analysis.spike_stats import waveform_pt_ratio
from sc_neurocore.analysis.spike_stats import waveform_recovery_slope
from sc_neurocore.analysis.spike_stats import waveform_repolarization_slope
from sc_neurocore.analysis.spike_stats import waveform_width
import numpy as np
from tests.spike_train_stats_extended_support import waveform_fixture as waveform_fixture


class TestWaveform:
    """Shape measures of a synthetic extracellular waveform."""

    def test_waveform_width(self, waveform_fixture: np.ndarray[Any, Any]) -> None:
        """The trough-to-peak width is positive."""
        w = waveform_width(waveform_fixture, dt=1.0 / 60)
        assert w > 0

    def test_waveform_amplitude(self, waveform_fixture: np.ndarray[Any, Any]) -> None:
        """The amplitude of the waveform is positive."""
        a = waveform_amplitude(waveform_fixture)
        assert a > 0

    def test_waveform_repolarization_slope(self, waveform_fixture: np.ndarray[Any, Any]) -> None:
        """The repolarisation slope is positive."""
        s = waveform_repolarization_slope(waveform_fixture, dt=1.0 / 60)
        assert s > 0

    def test_waveform_recovery_slope(self, waveform_fixture: np.ndarray[Any, Any]) -> None:
        """The recovery slope of the waveform is finite."""
        s = waveform_recovery_slope(waveform_fixture, dt=1.0 / 60)
        assert np.isfinite(s)

    def test_waveform_halfwidth(self, waveform_fixture: np.ndarray[Any, Any]) -> None:
        """The half-width of the waveform is positive."""
        hw = waveform_halfwidth(waveform_fixture, dt=1.0 / 60)
        assert hw > 0

    def test_waveform_pt_ratio(self, waveform_fixture: np.ndarray[Any, Any]) -> None:
        """The peak-to-trough ratio is positive."""
        r = waveform_pt_ratio(waveform_fixture)
        assert r > 0
