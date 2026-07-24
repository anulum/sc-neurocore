# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTrimmedMean from former test_federated_sc.py

"""Focused suite: TestTrimmedMean from former test_federated_sc.py."""

from __future__ import annotations

from federated_sc_support import *  # noqa: F403


class TestTrimmedMean:
    def test_removes_extremes(self):
        vecs = [
            np.array([1.0, 1.0]),
            np.array([1.1, 0.9]),
            np.array([0.9, 1.1]),
            np.array([100.0, -100.0]),
            np.array([1.0, 1.0]),
        ]
        result = trimmed_mean(vecs, trim_fraction=0.2)
        assert abs(result[0] - 1.0) < 0.2
        assert abs(result[1] - 1.0) < 0.2

    def test_matches_mean_without_trimming(self):
        vecs = [np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0])]
        result = trimmed_mean(vecs, trim_fraction=0.0)
        np.testing.assert_array_almost_equal(result, np.array([3.0, 4.0]))

    def test_over_trimming_falls_back_to_full_mean(self):
        # With two clients the minimum trim of one from each end removes every
        # row, so the aggregator falls back to the untrimmed mean rather than
        # averaging an empty slice.
        vecs = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        result = trimmed_mean(vecs, trim_fraction=0.1)
        np.testing.assert_array_almost_equal(result, np.array([2.0, 3.0]))
