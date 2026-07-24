# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAdaptiveThresholdIFNetwork from former test_model_adaptive_threshold_if.py

"""Focused suite: TestAdaptiveThresholdIFNetwork from former test_model_adaptive_threshold_if.py."""

from __future__ import annotations

from tests.model_adaptive_threshold_if_support import *  # noqa: F403


class TestAdaptiveThresholdIFNetwork:
    """Model works in the full SC-NeuroCore network pipeline."""

    def test_population_creation(self) -> None:
        pop = Population(AdaptiveThresholdIFNeuron, n=10, label="atif")
        assert pop.n == 10
        assert pop.model_name == "AdaptiveThresholdIFNeuron"

    def test_network_produces_spikes(self) -> None:
        pop = Population(AdaptiveThresholdIFNeuron, n=20, label="atif")
        proj = Projection(pop, pop, weight=1.0, probability=0.2, seed=42)
        drive = PoissonInput(n=20, rate_hz=500.0, weight=100.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, proj, drive, mon)
        net.run(duration=0.5, dt=0.001, backend="python")
        assert mon.count > 0, "network produced zero spikes"

    def test_spike_trains_extractable(self) -> None:
        pop = Population(AdaptiveThresholdIFNeuron, n=10, label="atif")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=100.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=0.3, dt=0.001, backend="python")
        trains = mon.spike_trains
        assert isinstance(trains, dict)
        assert len(trains) > 0, "no spike trains recorded"
