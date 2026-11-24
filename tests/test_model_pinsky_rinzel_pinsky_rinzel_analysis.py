# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPinskyRinzelAnalysis from former test_model_pinsky_rinzel.py

"""Focused suite: TestPinskyRinzelAnalysis from former test_model_pinsky_rinzel.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.analysis.spike_stats.basic import (
    spike_count,
)
from sc_neurocore.neurons.models.pinsky_rinzel import (
    PinskyRinzelNeuron,
)


class TestPinskyRinzelAnalysis:
    """Spike analysis of the PinskyRinzelNeuron."""

    def test_spike_count_matches_train_sum(self) -> None:
        """Spike count matches train sum."""
        n = PinskyRinzelNeuron()
        train = np.array([float(n.step(5.0)) for _ in range(50000)])
        assert spike_count(train) >= 5
        assert spike_count(train) == int(train.sum())
