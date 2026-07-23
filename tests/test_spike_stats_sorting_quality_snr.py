# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSNR from former test_spike_stats_sorting_quality.py

"""Focused suite: TestSNR from former test_spike_stats_sorting_quality.py."""

from __future__ import annotations

from tests.spike_stats_sorting_quality_support import *  # noqa: F403

class TestSNR:
    def test_typical(self) -> None:
        rng = _rng()
        waveforms = rng.normal(0, 0.1, (50, 30))
        waveforms[:, 15] += 2.0  # add peak
        result = snr(waveforms)
        assert result > 1

    def test_too_few(self) -> None:
        result = snr(np.array([[1, 2, 3]]))
        assert np.isnan(result)

    def test_zero_noise(self) -> None:
        waveforms = np.ones((5, 10))
        result = snr(waveforms)
        assert result == float("inf") or np.isfinite(result)
