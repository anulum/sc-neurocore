# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (point_process) from former test_spike_stats_toolkit_edge_contracts.py

from __future__ import annotations

from tests.spike_stats_toolkit_edge_contracts_support import *  # noqa: F403


def test_hazard_few():
    h, e = isi_hazard_function(np.array([1, 0], dtype=np.int8))
    assert h.size == 0


def test_survivor_few():
    s, e = isi_survivor_function(np.array([1, 0], dtype=np.int8))
    assert s.size == 0


def test_renewal_few():
    r, e = renewal_density(np.array([1, 0], dtype=np.int8))
    assert r.size == 0
