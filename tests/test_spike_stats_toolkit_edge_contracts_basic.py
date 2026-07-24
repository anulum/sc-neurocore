# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (basic) from former test_spike_stats_toolkit_edge_contracts.py

from __future__ import annotations

from tests.spike_stats_toolkit_edge_contracts_support import *  # noqa: F403

def test_firing_rate_zero_duration():
    assert firing_rate(np.array([]), dt=0.001) == 0.0


def test_bin_spike_train_small():
    r = bin_spike_train(np.array([1, 0, 1]), bin_size=10)
    assert r[0] == 2


