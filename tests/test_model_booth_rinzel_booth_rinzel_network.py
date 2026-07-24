# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBoothRinzelNetwork from former test_model_booth_rinzel.py

"""Focused suite: TestBoothRinzelNetwork from former test_model_booth_rinzel.py."""

from __future__ import annotations

from tests.model_booth_rinzel_support import *  # noqa: F403


class TestBoothRinzelNetwork:
    def test_population(self):
        pop = Population(BoothRinzelNeuron, n=5, label="br")
        assert pop.n == 5
        assert pop.model_name == "BoothRinzelNeuron"

    def test_network_spikes(self):
        pop = Population(BoothRinzelNeuron, n=10, label="br")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=10.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0, "network produced zero spikes"

    def test_with_projection(self):
        pop = Population(BoothRinzelNeuron, n=10, label="br")
        proj = Projection(pop, pop, weight=0.5, probability=0.2, seed=42)
        drive = PoissonInput(n=10, rate_hz=500.0, weight=10.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, proj, drive, mon)
        net.run(duration=0.5, dt=0.001, backend="python")
        trains = mon.spike_trains
        assert isinstance(trains, dict)
