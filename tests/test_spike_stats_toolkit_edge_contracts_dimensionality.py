# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (dimensionality) from former test_spike_stats_toolkit_edge_contracts.py

from __future__ import annotations

from tests.spike_stats_toolkit_edge_contracts_support import *  # noqa: F403

def test_pca_empty():
    s, e = spike_train_pca([])
    assert s.size == 0


def test_pca_1d():
    s, e = spike_train_pca([np.array([1, 0, 1], dtype=np.int8)])
    assert s.shape[0] == 1


def test_demixed_insufficient():
    s, e = demixed_pca({0: [np.array([1, 0], dtype=np.int8)]})
    assert s.size == 0


