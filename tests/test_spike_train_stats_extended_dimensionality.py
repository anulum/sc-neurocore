# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDimensionality from former test_spike_train_stats_extended.py

"""Focused suite: TestDimensionality from former test_spike_train_stats_extended.py."""

from __future__ import annotations

from tests.spike_train_stats_extended_support import *  # noqa: F403

class TestDimensionality:
    def test_demixed_pca(self, population):
        conds = {
            0: population[:3],
            1: population[2:],
        }
        proj, explained = demixed_pca(conds, n_components=2)
        assert proj.ndim == 2
        assert explained.size == 2

    def test_factor_analysis(self, population):
        loadings, psi = factor_analysis(population, n_factors=2)
        assert loadings.shape == (5, 2)
        assert psi.size == 5
