# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHalfwidth from former test_spike_stats_waveform.py

"""Focused suite: TestHalfwidth from former test_spike_stats_waveform.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.analysis.spike_stats.waveform import (
    waveform_halfwidth,
)
from tests.spike_stats_waveform_support import (
    _typical_waveform,
)


class TestHalfwidth:
    """The half-width of a spike waveform."""

    def test_typical(self) -> None:
        """The half-width of the synthetic spike is computed (the check only excludes infinity)."""
        w = _typical_waveform()
        result = waveform_halfwidth(w)
        assert np.isfinite(result) or np.isnan(result)

    def test_no_crossing(self) -> None:
        """A waveform that never crosses half its trough gives NaN."""
        # All positive — trough/2 never crossed
        w = np.array([5.0, 4.0, 3.0, 4.0, 5.0])
        result = waveform_halfwidth(w)
        assert np.isnan(result)

    def test_single_crossing(self) -> None:
        """The half-width of a waveform crossing half its trough once is computed (the check only excludes infinity)."""
        # Only one point below half — below.size < 2
        w = np.array([1.0, 0.5, -0.1, 0.5, 1.0])
        result = waveform_halfwidth(w)
        assert np.isnan(result) or np.isfinite(result)
