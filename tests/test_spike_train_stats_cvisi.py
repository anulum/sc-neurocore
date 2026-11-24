# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCVISI from former test_spike_train_stats.py

"""Focused suite: TestCVISI from former test_spike_train_stats.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.analysis.spike_stats import (
    cv_isi,
)
from tests.spike_train_stats_support import (
    _poisson_train,
)


class TestCVISI:
    """The coefficient of variation of the ISIs."""

    def test_regular_low_cv(self) -> None:
        """A regular train has an ISI CV below 0.05."""
        train = np.zeros(1000, dtype=np.uint8)
        train[10::20] = 1
        assert cv_isi(train) < 0.05

    def test_poisson_near_one(self) -> None:
        """A 50 Hz Poisson train has an ISI CV between 0.5 and 1.5."""
        train = _poisson_train(50.0, 5.0)
        cv = cv_isi(train)
        assert 0.5 < cv < 1.5

    def test_too_few_spikes(self) -> None:
        """A single spike gives NaN."""
        train = np.zeros(100, dtype=np.uint8)
        train[50] = 1
        assert np.isnan(cv_isi(train))
