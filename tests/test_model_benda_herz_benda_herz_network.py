# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBendaHerzNetwork from former test_model_benda_herz.py

"""Focused suite: TestBendaHerzNetwork from former test_model_benda_herz.py."""

from __future__ import annotations

from tests.model_benda_herz_support import *  # noqa: F403


class TestBendaHerzNetwork:
    def test_population(self):
        pop = Population(BendaHerzNeuron, n=10, label="bh")
        assert pop.n == 10
        assert pop.model_name == "BendaHerzNeuron"

    def test_network_produces_spikes(self):
        pop = Population(BendaHerzNeuron, n=20, label="bh")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon.count > 0, "network produced zero spikes"

    def test_with_projection(self):
        pop = Population(BendaHerzNeuron, n=20, label="bh")
        proj = Projection(pop, pop, weight=5.0, probability=0.2, seed=42)
        drive = PoissonInput(n=20, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, proj, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert isinstance(mon.spike_trains, dict)
