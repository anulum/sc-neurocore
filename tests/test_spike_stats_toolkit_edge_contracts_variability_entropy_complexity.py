# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (entropy_complexity) from former test_spike_stats_toolkit_edge_contracts_variability.py

from __future__ import annotations

from tests.spike_stats_toolkit_edge_contracts_support import *  # noqa: F403


def test_isi_entropy_silent():
    r = isi_entropy(np.zeros(10, dtype=np.int8))
    # Silent train → no ISIs → 0 entropy or nan
    assert r == 0.0 or np.isnan(r)


def test_lempel_ziv_silent():
    r = lempel_ziv_complexity(np.array([], dtype=np.int8))
    assert r == 0.0


def test_apen_high_dim():
    r = approximate_entropy(np.zeros(5, dtype=np.int8), m=10)
    assert r == 0.0 or np.isnan(r)


def test_sampen_zero():
    r = sample_entropy(np.zeros(10, dtype=np.int8))
    assert np.isfinite(r) or np.isnan(r)


def test_perm_entropy_short():
    assert np.isnan(permutation_entropy(np.zeros(3, dtype=np.int8), order=5))


def test_perm_entropy_few_patterns():
    assert np.isnan(permutation_entropy(np.zeros(5, dtype=np.int8), order=3, delay=3))
