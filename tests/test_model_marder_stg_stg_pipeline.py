# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSTGPipeline from former test_model_marder_stg.py

"""Focused suite: TestSTGPipeline from former test_model_marder_stg.py."""

from __future__ import annotations

from tests.model_marder_stg_support import *  # noqa: F403


class TestSTGPipeline:
    def test_population(self):
        assert Population(MarderSTGNeuron, n=10, label="stg").n == 10

    def test_network_spikes(self):
        pop = Population(MarderSTGNeuron, n=10, label="stg")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=2.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis_spike_count(self):
        n = MarderSTGNeuron()
        train = np.array([float(n.step(0.0)) for _ in range(50_000)])
        assert spike_count(train) >= 10
        assert spike_count(train) == int(train.sum())

    def test_analysis_firing_rate(self):
        n = MarderSTGNeuron()
        train = np.array([float(n.step(0.0)) for _ in range(50_000)])
        intervals = isi(train, dt=0.00005)
        if intervals.size > 0:
            assert np.all(intervals > 0)
        assert firing_rate(train, dt=0.00005) > 0
