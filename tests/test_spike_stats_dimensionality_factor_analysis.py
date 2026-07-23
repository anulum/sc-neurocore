# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFactorAnalysis from former test_spike_stats_dimensionality.py

"""Focused suite: TestFactorAnalysis from former test_spike_stats_dimensionality.py."""

from __future__ import annotations

from tests.spike_stats_dimensionality_support import *  # noqa: F403

class TestFactorAnalysis:
    def test_typical(self) -> None:
        loadings, psi = factor_analysis(_trains(5), n_factors=2)
        assert loadings.shape == (5, 2)
        assert psi.shape == (5,)
        assert np.all(psi > 0)

    def test_python_backend(self) -> None:
        loadings, psi = factor_analysis(_trains(5), n_factors=2, backend="python")
        assert loadings.shape == (5, 2)

    def test_empty_trains(self) -> None:
        loadings, psi = factor_analysis([])
        assert loadings.size == 0 and psi.size == 0

    def test_unknown_backend(self) -> None:
        with pytest.raises(ValueError, match="not available"):
            factor_analysis(_trains(5), backend="cuda")
