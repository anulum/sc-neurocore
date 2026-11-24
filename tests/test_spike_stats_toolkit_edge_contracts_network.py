# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (network) from former test_spike_stats_toolkit_edge_contracts.py

"""Unitary events and cell-assembly detection on random populations."""

from __future__ import annotations

import numpy as np
from sc_neurocore.analysis.spike_stats.network import (
    cell_assembly_detection,
    unitary_events,
)


def test_unitary_events_significant() -> None:
    """Unitary-event analysis of a random population returns a list."""
    rng = np.random.default_rng(42)
    trains = [rng.integers(0, 2, size=200, dtype=np.int8) for _ in range(10)]
    r = unitary_events(trains, bin_size=5, alpha=0.99)
    assert isinstance(r, list)


def test_cell_assembly() -> None:
    """Cell-assembly detection on a random population returns a list."""
    rng = np.random.default_rng(0)
    trains = [rng.integers(0, 2, size=500, dtype=np.int8) for _ in range(20)]
    r = cell_assembly_detection(trains, bin_size=10)
    assert isinstance(r, list)
