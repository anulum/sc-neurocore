# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (network) from former test_spike_stats_toolkit_edge_contracts.py

from __future__ import annotations

from tests.spike_stats_toolkit_edge_contracts_support import *  # noqa: F403


def test_unitary_events_significant():
    rng = np.random.default_rng(42)
    trains = [rng.integers(0, 2, size=200, dtype=np.int8) for _ in range(10)]
    r = unitary_events(trains, bin_size=5, alpha=0.99)
    assert isinstance(r, list)


def test_cell_assembly():
    rng = np.random.default_rng(0)
    trains = [rng.integers(0, 2, size=500, dtype=np.int8) for _ in range(20)]
    r = cell_assembly_detection(trains, bin_size=10)
    assert isinstance(r, list)
