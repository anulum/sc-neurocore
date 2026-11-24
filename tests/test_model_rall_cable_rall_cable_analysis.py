# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRallCableAnalysis from former test_model_rall_cable.py

"""Focused suite: TestRallCableAnalysis from former test_model_rall_cable.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.analysis.spike_stats.basic import (
    firing_rate,
    isi,
    spike_count,
)
from sc_neurocore.neurons.models.rall_cable import (
    RallCableNeuron,
)


class TestRallCableAnalysis:
    """Spike analysis of the RallCableNeuron."""

    def test_spike_count(self) -> None:
        """The spikes of a driven train are counted."""
        n = RallCableNeuron(n_comp=2, g_ratio=5.0)
        train = np.array([float(n.step(500.0)) for _ in range(50_000)])
        assert spike_count(train) >= 10

    def test_analysis_isi(self) -> None:
        """The analysis gives finite inter-spike intervals."""
        n = RallCableNeuron(n_comp=2, g_ratio=5.0)
        train = np.array([float(n.step(500.0)) for _ in range(50_000)])
        intervals = isi(train, dt=0.0001)
        if intervals.size > 0:
            assert np.all(np.isfinite(intervals))
            assert np.all(intervals > 0)

    def test_analysis_firing_rate(self) -> None:
        """The analysis gives the firing rate of a driven run."""
        n = RallCableNeuron(n_comp=2, g_ratio=5.0)
        train = np.array([float(n.step(500.0)) for _ in range(50_000)])
        rate = firing_rate(train, dt=0.0001)
        assert rate > 0

    def test_analysis_cross_validation(self) -> None:
        """Analysis cross validation."""
        n = RallCableNeuron(n_comp=2, g_ratio=5.0)
        train = np.array([float(n.step(500.0)) for _ in range(50_000)])
        sc = spike_count(train)
        dt_sim = 0.0001
        duration = len(train) * dt_sim
        rate = firing_rate(train, dt=dt_sim)
        if sc > 0:
            expected = sc / duration
            assert abs(rate - expected) < expected * 0.1
