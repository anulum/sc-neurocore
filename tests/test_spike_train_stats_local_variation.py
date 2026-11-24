# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLocalVariation from former test_spike_train_stats.py

"""Focused suite: TestLocalVariation from former test_spike_train_stats.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.analysis.spike_stats import (
    local_variation,
)
from tests.spike_train_stats_support import (
    _poisson_train,
)


class TestLocalVariation:
    """The local variation of the ISIs."""

    def test_regular_low(self) -> None:
        """A regular train has a local variation below 0.1."""
        train = np.zeros(1000, dtype=np.uint8)
        train[10::20] = 1
        lv = local_variation(train)
        assert lv < 0.1

    def test_poisson_near_one(self) -> None:
        """A 50 Hz Poisson train has a local variation between 0.5 and 1.5."""
        lv = local_variation(_poisson_train(50.0, 5.0))
        assert 0.5 < lv < 1.5
