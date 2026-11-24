# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSurrogateISIShuffle from former test_spike_train_stats.py

"""Focused suite: TestSurrogateISIShuffle from former test_spike_train_stats.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.analysis.spike_stats import (
    surrogate_isi_shuffle,
)
from tests.spike_train_stats_support import (
    _poisson_train,
)


class TestSurrogateISIShuffle:
    """ISI-shuffling surrogates."""

    def test_preserves_count(self) -> None:
        """ISI shuffling changes the spike count by at most one."""
        train = _poisson_train(100.0, 0.5)
        surr = surrogate_isi_shuffle(train, seed=1)
        assert abs(surr.sum() - train.sum()) <= 1

    def test_different_order(self) -> None:
        """ISI shuffling changes the spike pattern."""
        train = _poisson_train(100.0, 1.0)
        surr = surrogate_isi_shuffle(train, seed=7)
        assert not np.array_equal(train, surr)
