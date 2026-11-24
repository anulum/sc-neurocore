# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDriftMetric from former test_spike_stats_sorting_quality.py

"""Focused suite: TestDriftMetric from former test_spike_stats_sorting_quality.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.analysis.spike_stats.sorting_quality import (
    drift_metric,
)
from tests.spike_stats_sorting_quality_support import (
    _rng,
)


class TestDriftMetric:
    """The waveform drift metric."""

    def test_typical(self) -> None:
        """Waveforms that double in amplitude halfway give a positive drift."""
        rng = _rng()
        n = 100
        waveforms = rng.normal(0, 1, (n, 30))
        timestamps = np.arange(n, dtype=float)
        # Add drift
        waveforms[50:] *= 2
        result = drift_metric(waveforms, timestamps)
        assert result > 0

    def test_too_few(self) -> None:
        """Five waveforms give NaN."""
        waveforms = _rng().normal(0, 1, (5, 10))
        timestamps = np.arange(5, dtype=float)
        result = drift_metric(waveforms, timestamps)
        assert np.isnan(result)

    def test_no_drift(self) -> None:
        """Constant waveforms give a drift of zero."""
        waveforms = np.ones((20, 10))
        timestamps = np.arange(20, dtype=float)
        result = drift_metric(waveforms, timestamps)
        assert result == 0.0
