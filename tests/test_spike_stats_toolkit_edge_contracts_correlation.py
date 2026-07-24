# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (correlation) from former test_spike_stats_toolkit_edge_contracts.py

from __future__ import annotations

from tests.spike_stats_toolkit_edge_contracts_support import *  # noqa: F403


def test_pairwise_empty():
    assert pairwise_correlation([]).size == 0


def test_coherence_short():
    cc, freqs = spike_train_coherence(np.array([1], dtype=np.int8), np.array([0], dtype=np.int8))
    assert cc.size == 0


def test_event_sync_empty():
    assert (
        event_synchronization(np.array([], dtype=np.int8), np.array([1, 0], dtype=np.int8)) == 0.0
    )


def test_sttc_full_contract():
    a = np.ones(100, dtype=np.int8)
    b = np.ones(100, dtype=np.int8)
    r = spike_time_tiling_coefficient(a, b, delta_ms=50.0)
    assert np.isfinite(r)


def test_autocorr_zero_var():
    assert autocorrelation_time(np.zeros(100, dtype=np.int8)) == 0.0


def test_coincidence_below_expected():
    assert coincidence_index(np.zeros(100, dtype=np.int8), np.zeros(100, dtype=np.int8)) == 0.0


def test_event_sync_empty_b():
    assert (
        event_synchronization(np.array([1, 0], dtype=np.int8), np.array([], dtype=np.int8)) == 0.0
    )
