# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFirstSpikeLatency from former test_spike_train_stats.py

"""Focused suite: TestFirstSpikeLatency from former test_spike_train_stats.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.analysis.spike_stats import (
    first_spike_latency,
)


class TestFirstSpikeLatency:
    """The latency of the first spike."""

    def test_known(self) -> None:
        """A spike at step 42 has a latency of 42 ms."""
        train = np.zeros(100, dtype=np.uint8)
        train[42] = 1
        assert abs(first_spike_latency(train) - 0.042) < 1e-10

    def test_no_spike(self) -> None:
        """A silent train gives NaN."""
        assert np.isnan(first_spike_latency(np.zeros(100, dtype=np.uint8)))
