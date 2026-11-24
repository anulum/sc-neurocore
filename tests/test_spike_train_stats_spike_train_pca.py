# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikeTrainPCA from former test_spike_train_stats.py

"""Focused suite: TestSpikeTrainPCA from former test_spike_train_stats.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.analysis.spike_stats import (
    spike_train_pca,
)
from tests.spike_train_stats_support import (
    _poisson_train,
)


class TestSpikeTrainPCA:
    """PCA of a population of spike trains."""

    def test_shape(self) -> None:
        """Three components of eight trains give three projections and three non-negative variances."""
        trains = [_poisson_train(50.0 + i * 10, 0.5, seed=i) for i in range(8)]
        proj, var = spike_train_pca(trains, n_components=3)
        assert proj.shape[0] == 3
        assert var.size == 3
        assert np.all(var >= 0)
