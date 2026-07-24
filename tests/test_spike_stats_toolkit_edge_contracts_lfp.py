# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (lfp) from former test_spike_stats_toolkit_edge_contracts.py

from __future__ import annotations

from tests.spike_stats_toolkit_edge_contracts_support import *  # noqa: F403


def test_plv_no_spikes():
    assert phase_locking_value(np.zeros(100, dtype=np.int8), np.sin(np.linspace(0, 10, 100))) == 0.0


def test_sfc_short():
    f, p = spike_field_coherence(np.array([1], dtype=np.int8), np.array([1.0]))
    assert f.size == 0
