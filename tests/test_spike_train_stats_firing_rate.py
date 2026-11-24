# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFiringRate from former test_spike_train_stats.py

"""Focused suite: TestFiringRate from former test_spike_train_stats.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.analysis.spike_stats import (
    firing_rate,
)
from tests.spike_train_stats_support import (
    _poisson_train,
)


class TestFiringRate:
    """The mean firing rate of a spike train."""

    def test_known_rate(self) -> None:
        """A 100 Hz Poisson train has a rate between 70 and 140 Hz."""
        train = _poisson_train(100.0, 1.0)
        rate = firing_rate(train, dt=0.001)
        assert 70 < rate < 140

    def test_empty(self) -> None:
        """A silent train has a rate of zero."""
        assert firing_rate(np.zeros(100, dtype=np.uint8)) == 0.0
