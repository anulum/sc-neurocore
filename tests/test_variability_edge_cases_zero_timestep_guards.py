# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestZeroTimestepGuards from former test_variability_edge_cases.py

"""Focused suite: TestZeroTimestepGuards from former test_variability_edge_cases.py."""

from __future__ import annotations

from tests.variability_edge_cases_support import *  # noqa: F403

class TestZeroTimestepGuards:
    """A zero timestep collapses every inter-spike interval to zero, the only
    way to reach the mean/sum==0 guards that strictly-positive ISIs never hit."""

    @staticmethod
    def _three_spikes():
        train = np.zeros(50, dtype=np.int8)
        train[[5, 15, 30]] = 1
        return train

    def test_cv_isi_zero_mean_interval(self):
        assert np.isnan(cv_isi(self._three_spikes(), dt=0.0))

    def test_cv2_no_positive_sums(self):
        assert np.isnan(cv2(self._three_spikes(), dt=0.0))

    def test_local_variation_no_positive_sums(self):
        assert np.isnan(local_variation(self._three_spikes(), dt=0.0))

    def test_lvr_every_pair_sum_nonpositive(self):
        # Each consecutive ISI sum is zero, so the per-pair skip runs for every
        # pair and the contributing count stays zero -> NaN.
        assert np.isnan(lvr(self._three_spikes(), dt=0.0))
