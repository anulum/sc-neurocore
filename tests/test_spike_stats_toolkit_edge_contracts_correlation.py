# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (correlation) from former test_spike_stats_toolkit_edge_contracts.py

"""Edge contracts of the correlation, coherence and synchrony estimators."""

from __future__ import annotations

import numpy as np
from sc_neurocore.analysis.spike_stats.correlation import (
    autocorrelation_time,
    coincidence_index,
    event_synchronization,
    pairwise_correlation,
    spike_time_tiling_coefficient,
    spike_train_coherence,
)


def test_pairwise_empty() -> None:
    """The pairwise correlation of no trains is empty."""
    assert pairwise_correlation([]).size == 0


def test_coherence_short() -> None:
    """Coherence of one-sample trains is empty."""
    cc, freqs = spike_train_coherence(np.array([1], dtype=np.int8), np.array([0], dtype=np.int8))
    assert cc.size == 0


def test_event_sync_empty() -> None:
    """Event synchronisation with an empty first train is zero."""
    assert (
        event_synchronization(np.array([], dtype=np.int8), np.array([1, 0], dtype=np.int8)) == 0.0
    )


def test_sttc_full_contract() -> None:
    """The STTC of two always-firing trains is finite."""
    a = np.ones(100, dtype=np.int8)
    b = np.ones(100, dtype=np.int8)
    r = spike_time_tiling_coefficient(a, b, delta_ms=50.0)
    assert np.isfinite(r)


def test_autocorr_zero_var() -> None:
    """The autocorrelation time of a silent train is zero."""
    assert autocorrelation_time(np.zeros(100, dtype=np.int8)) == 0.0


def test_coincidence_below_expected() -> None:
    """The coincidence index of two silent trains is zero."""
    assert coincidence_index(np.zeros(100, dtype=np.int8), np.zeros(100, dtype=np.int8)) == 0.0


def test_event_sync_empty_b() -> None:
    """Event synchronisation with an empty second train is zero."""
    assert (
        event_synchronization(np.array([1, 0], dtype=np.int8), np.array([], dtype=np.int8)) == 0.0
    )
