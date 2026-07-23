# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPairwiseCorrelation from former test_spike_train_stats.py

"""Focused suite: TestPairwiseCorrelation from former test_spike_train_stats.py."""

from __future__ import annotations

from tests.spike_train_stats_support import *  # noqa: F403

class TestPairwiseCorrelation:
    def test_self_correlation(self):
        train = _poisson_train(100.0, 1.0)
        mat = pairwise_correlation([train, train])
        np.testing.assert_allclose(mat[0, 1], 1.0, atol=1e-10)

    def test_shape(self):
        trains = [_poisson_train(50.0, 0.5, seed=i) for i in range(5)]
        mat = pairwise_correlation(trains)
        assert mat.shape == (5, 5)
