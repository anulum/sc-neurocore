# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPrescottAnalysis from former test_model_prescott.py

"""Focused suite: TestPrescottAnalysis from former test_model_prescott.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.analysis.spike_stats.basic import (
    spike_count,
)
from sc_neurocore.neurons.models.prescott import (
    PrescottNeuron,
)


class TestPrescottAnalysis:
    """Spike analysis of the PrescottNeuron."""

    def test_spike_count(self) -> None:
        """The spikes of a driven train are counted."""
        n = PrescottNeuron()
        train = np.array([float(n.step(50.0)) for _ in range(100000)])
        assert spike_count(train) >= 3

    def test_spike_count_consistency(self) -> None:
        """Spike count consistency."""
        n = PrescottNeuron()
        train = np.array([float(n.step(50.0)) for _ in range(100000)])
        assert spike_count(train) == int(train.sum())
