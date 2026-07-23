# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPopulationVectorDecode from former test_spike_train_stats.py

"""Focused suite: TestPopulationVectorDecode from former test_spike_train_stats.py."""

from __future__ import annotations

from tests.spike_train_stats_support import *  # noqa: F403

class TestPopulationVectorDecode:
    def test_shape(self):
        trains = [_poisson_train(50.0, 0.5, seed=i) for i in range(4)]
        dirs = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2])
        decoded = population_vector_decode(trains, dirs, window=50)
        assert decoded.size > 0

    def test_empty(self):
        assert population_vector_decode([], np.array([])).size == 0
