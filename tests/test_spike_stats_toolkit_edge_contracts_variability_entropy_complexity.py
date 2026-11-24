# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (entropy_complexity) from former test_spike_stats_toolkit_edge_contracts_variability.py

"""Edge contracts of the ISI, Lempel-Ziv, approximate, sample and permutation entropies."""

from __future__ import annotations

import numpy as np
from sc_neurocore.analysis.spike_stats.variability import (
    approximate_entropy,
    isi_entropy,
    lempel_ziv_complexity,
    permutation_entropy,
    sample_entropy,
)


def test_isi_entropy_silent() -> None:
    """The ISI entropy of a silent train is zero or NaN."""
    r = isi_entropy(np.zeros(10, dtype=np.int8))
    # Silent train → no ISIs → 0 entropy or nan
    assert r == 0.0 or np.isnan(r)


def test_lempel_ziv_silent() -> None:
    """The Lempel-Ziv complexity of an empty train is zero."""
    r = lempel_ziv_complexity(np.array([], dtype=np.int8))
    assert r == 0.0


def test_apen_high_dim() -> None:
    """Approximate entropy with an embedding longer than the train is zero or NaN."""
    r = approximate_entropy(np.zeros(5, dtype=np.int8), m=10)
    assert r == 0.0 or np.isnan(r)


def test_sampen_zero() -> None:
    """Sample entropy of a silent train is computed (the check only excludes infinity)."""
    r = sample_entropy(np.zeros(10, dtype=np.int8))
    assert np.isfinite(r) or np.isnan(r)


def test_perm_entropy_short() -> None:
    """Permutation entropy of a train shorter than the order is NaN."""
    assert np.isnan(permutation_entropy(np.zeros(3, dtype=np.int8), order=5))


def test_perm_entropy_few_patterns() -> None:
    """Permutation entropy with too few ordinal patterns is NaN."""
    assert np.isnan(permutation_entropy(np.zeros(5, dtype=np.int8), order=3, delay=3))
