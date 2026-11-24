# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBinSpikeTrain from former test_spike_train_stats.py

"""Focused suite: TestBinSpikeTrain from former test_spike_train_stats.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.analysis.spike_stats import (
    bin_spike_train,
)


class TestBinSpikeTrain:
    """Binning a binary spike train."""

    def test_basic(self) -> None:
        """Ten steps in bins of five give the counts 2 and 3."""
        train = np.array([1, 0, 1, 0, 0, 1, 1, 1, 0, 0], dtype=np.uint8)
        binned = bin_spike_train(train, bin_size=5)
        assert binned.tolist() == [2, 3]
