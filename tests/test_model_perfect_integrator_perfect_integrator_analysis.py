# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPerfectIntegratorAnalysis from former test_model_perfect_integrator.py

"""Focused suite: TestPerfectIntegratorAnalysis from former test_model_perfect_integrator.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.analysis.spike_stats.basic import (
    spike_count,
)
from sc_neurocore.neurons.models.perfect_integrator import (
    PerfectIntegratorNeuron,
)


class TestPerfectIntegratorAnalysis:
    """Spike analysis of the PerfectIntegratorNeuron."""

    def test_spike_count_matches_manual(self) -> None:
        """Spike count matches manual."""
        n = PerfectIntegratorNeuron()
        train = np.array([float(n.step(5.0)) for _ in range(500)])
        manual_count = int(train.sum())
        assert spike_count(train) == manual_count

    def test_spike_count_long_run(self) -> None:
        """Long run spike count matches analytical prediction."""
        n = PerfectIntegratorNeuron()
        I = 5.0
        steps = 10000
        train = np.array([float(n.step(I)) for _ in range(steps)])
        analytical = steps * I * n.dt / (n.c_m * n.v_threshold)
        assert abs(spike_count(train) - analytical) <= 1
