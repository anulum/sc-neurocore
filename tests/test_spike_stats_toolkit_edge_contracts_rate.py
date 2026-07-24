# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (rate) from former test_spike_stats_toolkit_edge_contracts.py

from __future__ import annotations

from tests.spike_stats_toolkit_edge_contracts_support import *  # noqa: F403


def test_psth_zero_bins():
    r, t = psth([np.array([], dtype=np.int8)], bin_ms=100.0)
    assert r.size == 0


def test_psth_empty_trial():
    r, t = psth([np.zeros(200, dtype=np.int8), np.array([], dtype=np.int8)], bin_ms=10.0)
    assert r.shape[0] > 0
