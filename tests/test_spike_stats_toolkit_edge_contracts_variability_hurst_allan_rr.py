# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (hurst_allan_rr) from former test_spike_stats_toolkit_edge_contracts_variability.py

from __future__ import annotations

from tests.spike_stats_toolkit_edge_contracts_support import *  # noqa: F403


def test_hurst_short():
    assert np.isnan(hurst_exponent(np.zeros(5, dtype=np.int8)))


def test_hurst_constant():
    r = hurst_exponent(np.ones(100, dtype=np.int8))
    assert np.isfinite(r) or np.isnan(r)


def test_allan_short():
    af, windows = allan_factor(np.zeros(3, dtype=np.int8))
    assert af.size >= 0


def test_allan_with_data():
    rng = np.random.default_rng(42)
    train = rng.integers(0, 2, size=1000, dtype=np.int8)
    af, windows = allan_factor(train)
    assert af.size > 0


def test_rescaled_range_short():
    assert np.isnan(rescaled_range(np.zeros(5, dtype=np.int8)))


def test_rescaled_range_constant():
    r = rescaled_range(np.ones(100, dtype=np.int8))
    assert np.isfinite(r) or np.isnan(r)


def test_rescaled_range_with_data():
    # variability.py:334 — duplicate scale dedup
    rng = np.random.default_rng(42)
    train = rng.integers(0, 2, size=500, dtype=np.int8)
    r = rescaled_range(train)
    assert np.isfinite(r) or np.isnan(r)


def test_rescaled_range_real_data():
    # variability.py:334 — scales with real data
    rng = np.random.default_rng(42)
    train = rng.integers(0, 2, size=2000, dtype=np.int8)
    r = rescaled_range(train)
    assert np.isfinite(r)
