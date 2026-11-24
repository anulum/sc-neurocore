# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (patterns) from former test_spike_stats_toolkit_edge_contracts.py

"""Edge contracts of spike directionality and the cubic higher-order statistic."""

from __future__ import annotations

import numpy as np
from sc_neurocore.analysis.spike_stats.patterns import (
    cubic_higher_order,
    spike_directionality,
)


def test_directionality_empty() -> None:
    """Spike directionality with an empty first train is zero."""
    assert spike_directionality(np.array([]), np.array([0.1])) == 0.0


def test_directionality_zero() -> None:
    """Spike directionality of two empty trains is zero."""
    assert spike_directionality(np.array([]), np.array([])) == 0.0


def test_cubic_higher_order_short() -> None:
    """The cubic higher-order statistic of a short silent train still has entries."""
    r = cubic_higher_order(np.zeros(5, dtype=np.int8), max_lag=2)
    assert r.shape[0] > 0
