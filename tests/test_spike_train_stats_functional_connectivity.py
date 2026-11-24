# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFunctionalConnectivity from former test_spike_train_stats.py

"""Focused suite: TestFunctionalConnectivity from former test_spike_train_stats.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.analysis.spike_stats import (
    functional_connectivity,
)
from tests.spike_train_stats_support import (
    _poisson_train,
)


class TestFunctionalConnectivity:
    """The functional-connectivity matrix of a population."""

    def test_symmetric(self) -> None:
        """Four trains give a symmetric 4x4 matrix with ones on the diagonal."""
        trains = [_poisson_train(50.0, 0.5, seed=i) for i in range(4)]
        mat = functional_connectivity(trains, max_lag_ms=10.0)
        np.testing.assert_allclose(mat, mat.T, atol=1e-12)
        assert mat.shape == (4, 4)
        np.testing.assert_allclose(np.diag(mat), 1.0)
