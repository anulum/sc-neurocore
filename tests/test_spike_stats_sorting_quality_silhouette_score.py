# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSilhouetteScore from former test_spike_stats_sorting_quality.py

"""Focused suite: TestSilhouetteScore from former test_spike_stats_sorting_quality.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.analysis.spike_stats.sorting_quality import (
    silhouette_score,
)
from tests.spike_stats_sorting_quality_support import (
    _rng,
)


class TestSilhouetteScore:
    """The silhouette score of labelled features."""

    def test_typical(self) -> None:
        """Two separated classes give a score inside [-1, 1]."""
        rng = _rng()
        features = np.vstack([rng.normal(0, 1, (10, 2)), rng.normal(5, 1, (10, 2))])
        labels = np.array([0] * 10 + [1] * 10)
        result = silhouette_score(features, labels)
        assert -1 <= result <= 1

    def test_single_point(self) -> None:
        """One point gives a score of zero."""
        result = silhouette_score(np.array([[1, 2]]), np.array([0]))
        assert result == 0.0

    def test_single_class(self) -> None:
        """One class gives a score of zero."""
        features = _rng().normal(0, 1, (10, 2))
        labels = np.zeros(10, dtype=int)
        result = silhouette_score(features, labels)
        assert result == 0.0
