# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (causality) from former test_spike_stats_toolkit_edge_contracts.py

"""Edge contracts of the Granger-causality estimators on short and constant trains."""

from __future__ import annotations

import numpy as np
from sc_neurocore.analysis.spike_stats.causality import (
    conditional_granger_causality,
    pairwise_granger_causality,
    spectral_granger_causality,
)


def test_granger_short() -> None:
    """Pairwise Granger causality of trains shorter than the model order is zero."""
    assert pairwise_granger_causality(np.zeros(3), np.zeros(3), order=5) == 0.0


def test_granger_constant() -> None:
    """Pairwise Granger causality of constant trains is finite."""
    a = np.ones(100, dtype=np.int8)
    r = pairwise_granger_causality(a, a, order=2)
    assert np.isfinite(r)


def test_conditional_granger_short() -> None:
    """Conditional Granger causality of trains shorter than the model order is zero."""
    assert conditional_granger_causality(np.zeros(3), np.zeros(3), np.zeros(3), order=5) == 0.0


def test_conditional_granger_constant() -> None:
    """Conditional Granger causality of constant trains is finite."""
    a = np.ones(100, dtype=np.int8)
    r = conditional_granger_causality(a, a, a, order=2)
    assert np.isfinite(r)


def test_spectral_granger_singular() -> None:
    """Spectral Granger causality of silent, singular trains still returns a spectrum."""
    trains = [np.zeros(50, dtype=np.int8)] * 3
    r = spectral_granger_causality(trains, order=2)
    assert r.shape[0] > 0
