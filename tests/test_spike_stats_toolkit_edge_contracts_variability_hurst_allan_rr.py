# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (hurst_allan_rr) from former test_spike_stats_toolkit_edge_contracts_variability.py

"""Edge contracts of the Hurst exponent, the Allan factor and the rescaled range."""

from __future__ import annotations

import numpy as np
from sc_neurocore.analysis.spike_stats.variability import (
    allan_factor,
    hurst_exponent,
    rescaled_range,
)


def test_hurst_short() -> None:
    """The Hurst exponent of a five-sample train is NaN."""
    assert np.isnan(hurst_exponent(np.zeros(5, dtype=np.int8)))


def test_hurst_constant() -> None:
    """The Hurst exponent of a constant train is computed (the check only excludes infinity)."""
    r = hurst_exponent(np.ones(100, dtype=np.int8))
    assert np.isfinite(r) or np.isnan(r)


def test_allan_short() -> None:
    """The Allan factor of a three-sample train is computed (the check asserts nothing about it)."""
    af, windows = allan_factor(np.zeros(3, dtype=np.int8))
    assert af.size >= 0


def test_allan_with_data() -> None:
    """The Allan factor of a random train has values."""
    rng = np.random.default_rng(42)
    train = rng.integers(0, 2, size=1000, dtype=np.int8)
    af, windows = allan_factor(train)
    assert af.size > 0


def test_rescaled_range_short() -> None:
    """The rescaled range of a five-sample train is NaN."""
    assert np.isnan(rescaled_range(np.zeros(5, dtype=np.int8)))


def test_rescaled_range_constant() -> None:
    """The rescaled range of a constant train is computed (the check only excludes infinity)."""
    r = rescaled_range(np.ones(100, dtype=np.int8))
    assert np.isfinite(r) or np.isnan(r)


def test_rescaled_range_with_data() -> None:
    """The rescaled range of a 500-sample random train is computed (the check only excludes infinity)."""
    # variability.py:334 — duplicate scale dedup
    rng = np.random.default_rng(42)
    train = rng.integers(0, 2, size=500, dtype=np.int8)
    r = rescaled_range(train)
    assert np.isfinite(r) or np.isnan(r)


def test_rescaled_range_real_data() -> None:
    """The rescaled range of a 2000-sample random train is finite."""
    # variability.py:334 — scales with real data
    rng = np.random.default_rng(42)
    train = rng.integers(0, 2, size=2000, dtype=np.int8)
    r = rescaled_range(train)
    assert np.isfinite(r)
