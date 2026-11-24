# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPtRatio from former test_spike_stats_waveform.py

"""Focused suite: TestPtRatio from former test_spike_stats_waveform.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.analysis.spike_stats.waveform import (
    waveform_pt_ratio,
)
from tests.spike_stats_waveform_support import (
    _typical_waveform,
)


class TestPtRatio:
    """The peak-to-trough ratio of a spike waveform."""

    def test_typical(self) -> None:
        """The peak-to-trough ratio of the synthetic spike is finite."""
        w = _typical_waveform()
        result = waveform_pt_ratio(w)
        assert np.isfinite(result)

    def test_trough_at_end(self) -> None:
        """A trough at the last sample gives NaN."""
        # Trough is last element
        w = np.array([1.0, 0.5, 0.0, -0.5, -1.0])
        result = waveform_pt_ratio(w)
        assert np.isnan(result)

    def test_zero_trough(self) -> None:
        """A trough of zero gives NaN."""
        # Trough amplitude near zero (line 75 trough_val < 1e-30)
        w = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        result = waveform_pt_ratio(w)
        assert np.isnan(result)
