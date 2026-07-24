# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (sorting_quality) from former test_spike_stats_toolkit_edge_contracts.py

from __future__ import annotations

from tests.spike_stats_toolkit_edge_contracts_support import *  # noqa: F403


def test_isolation_dist_small():
    rng = np.random.default_rng(42)
    r = isolation_distance(rng.standard_normal((5, 2)), rng.standard_normal((10, 2)))
    assert np.isfinite(r)


def test_amplitude_cutoff_symmetric():
    rng = np.random.default_rng(0)
    amps = rng.standard_normal(200)
    r = amplitude_cutoff(amps)
    assert np.isfinite(r)
