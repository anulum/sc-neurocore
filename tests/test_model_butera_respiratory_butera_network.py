# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestButeraNetwork from former test_model_butera_respiratory.py

"""Focused suite: TestButeraNetwork from former test_model_butera_respiratory.py."""

from __future__ import annotations

from tests.model_butera_respiratory_support import *  # noqa: F403

class TestButeraNetwork:
    def test_population(self):
        pop = Population(ButeraRespiratoryNeuron, n=5, label="butera")
        assert pop.n == 5

    def test_network_spikes(self):
        pop = Population(ButeraRespiratoryNeuron, n=10, label="butera")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=100.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_with_projection(self):
        pop = Population(ButeraRespiratoryNeuron, n=10, label="butera")
        proj = Projection(pop, pop, weight=5.0, probability=0.3, seed=42)
        drive = PoissonInput(n=10, rate_hz=500.0, weight=100.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, proj, drive, mon)
        net.run(duration=0.5, dt=0.001, backend="python")
        assert isinstance(mon.spike_trains, dict)
