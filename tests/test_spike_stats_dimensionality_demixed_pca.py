# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDemixedPCA from former test_spike_stats_dimensionality.py

"""Focused suite: TestDemixedPCA from former test_spike_stats_dimensionality.py."""

from __future__ import annotations

from tests.spike_stats_dimensionality_support import *  # noqa: F403

class TestDemixedPCA:
    def test_typical(self) -> None:
        proj, expl = demixed_pca(_conditions(), n_components=2)
        assert proj.ndim == 2
        assert expl.size == 2

    def test_python_backend(self) -> None:
        proj, expl = demixed_pca(_conditions(), n_components=2, backend="python")
        assert proj.shape[1] == 2

    def test_insufficient_conditions(self) -> None:
        proj, expl = demixed_pca({0: [np.array([1, 0], dtype=np.int8)]})
        assert proj.size == 0 and expl.size == 0

    def test_empty_condition_skipped(self) -> None:
        # a condition with no neurons is skipped; two usable conditions remain
        conds = {0: _trains(3, 400), 1: [], 2: _trains(3, 400, seed=9)}
        proj, expl = demixed_pca(conds, n_components=2, bin_size=10)
        assert proj.size > 0 and expl.size == 2

    def test_only_empty_conditions(self) -> None:
        # fewer than two usable conditions -> empty result
        proj, expl = demixed_pca({0: _trains(3, 400), 1: []}, n_components=2)
        assert proj.size == 0 and expl.size == 0

    def test_unknown_backend(self) -> None:
        with pytest.raises(ValueError, match="not available"):
            demixed_pca(_conditions(), backend="cuda")
