# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (basic) from former test_spike_stats_toolkit_edge_contracts.py

"""Edge contracts of the basic spike-statistics helpers: firing rate and binning."""

from __future__ import annotations

import numpy as np
from sc_neurocore.analysis.spike_stats.basic import (
    bin_spike_train,
    firing_rate,
)


def test_firing_rate_zero_duration() -> None:
    """An empty train has a firing rate of zero."""
    assert firing_rate(np.array([]), dt=0.001) == 0.0


def test_bin_spike_train_small() -> None:
    """A train shorter than one bin is binned into a single bin holding both spikes."""
    r = bin_spike_train(np.array([1, 0, 1]), bin_size=10)
    assert r[0] == 2
