# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (dimensionality) from former test_spike_stats_toolkit_edge_contracts.py

"""Edge contracts of PCA and demixed PCA on empty and minimal inputs."""

from __future__ import annotations

import numpy as np
from sc_neurocore.analysis.spike_stats.dimensionality import (
    demixed_pca,
    spike_train_pca,
)


def test_pca_empty() -> None:
    """PCA of no trains gives an empty projection."""
    s, e = spike_train_pca([])
    assert s.size == 0


def test_pca_1d() -> None:
    """PCA of a single train gives one component."""
    s, e = spike_train_pca([np.array([1, 0, 1], dtype=np.int8)])
    assert s.shape[0] == 1


def test_demixed_insufficient() -> None:
    """Demixed PCA of a single condition gives an empty projection."""
    s, e = demixed_pca({0: [np.array([1, 0], dtype=np.int8)]})
    assert s.size == 0
