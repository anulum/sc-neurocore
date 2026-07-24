# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestExtractSpikeTimesBasic from former test_autofit.py

"""Focused suite: TestExtractSpikeTimesBasic from former test_autofit.py."""

from __future__ import annotations

from tests.autofit_support import *  # noqa: F403


class TestExtractSpikeTimesBasic:
    def test_no_spikes_subthreshold(self):
        v = np.array([0.0, 0.0, 0.0, 0.0])
        times = extract_spike_times(v, threshold=0.5)
        assert len(times) == 0

    def test_single_crossing(self):
        v = np.array([-1.0, -0.5, 0.5, 1.0, 0.5])
        times = extract_spike_times(v, threshold=0.0, dt=1.0)
        assert len(times) == 1
        assert times[0] == pytest.approx(1.0)

    def test_multiple_crossings(self):
        v = np.array([-1.0, 1.0, -1.0, 1.0, -1.0])
        times = extract_spike_times(v, threshold=0.0, dt=0.5)
        assert len(times) == 2

    def test_dt_scaling(self):
        v = np.array([-1.0, 1.0, -1.0])
        times_dt1 = extract_spike_times(v, threshold=0.0, dt=1.0)
        times_dt2 = extract_spike_times(v, threshold=0.0, dt=2.0)
        assert times_dt2[0] == 2.0 * times_dt1[0]
