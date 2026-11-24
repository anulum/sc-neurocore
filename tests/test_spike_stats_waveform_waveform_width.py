# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWaveformWidth from former test_spike_stats_waveform.py

"""Focused suite: TestWaveformWidth from former test_spike_stats_waveform.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.analysis.spike_stats.waveform import (
    waveform_width,
)
from tests.spike_stats_waveform_support import (
    _typical_waveform,
)


class TestWaveformWidth:
    """The trough-to-peak width of a spike waveform."""

    def test_typical(self) -> None:
        """The width of the synthetic spike is finite and positive."""
        w = _typical_waveform()
        result = waveform_width(w)
        assert np.isfinite(result)
        assert result > 0

    def test_trough_at_end(self) -> None:
        """A trough at the last sample gives NaN."""
        # Trough is last element — should return NaN
        w = np.array([1.0, 0.5, 0.0, -0.5, -1.0])
        result = waveform_width(w)
        assert np.isnan(result)
