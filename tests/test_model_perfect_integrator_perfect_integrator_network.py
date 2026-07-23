# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPerfectIntegratorNetwork from former test_model_perfect_integrator.py

"""Focused suite: TestPerfectIntegratorNetwork from former test_model_perfect_integrator.py."""

from __future__ import annotations

from tests.model_perfect_integrator_support import *  # noqa: F403

class TestPerfectIntegratorNetwork:
    def test_population_construction(self):
        pop = Population(PerfectIntegratorNeuron, n=10, label="pi")
        assert pop.n == 10

    def test_network_produces_spikes(self):
        pop = Population(PerfectIntegratorNeuron, n=20, label="pi")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=5.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_two_populations_different_drive(self):
        """Stronger drive → more spikes across a population."""
        pop_weak = Population(PerfectIntegratorNeuron, n=10, label="weak")
        pop_high_drive = Population(PerfectIntegratorNeuron, n=10, label="high_drive")
        drive_weak = PoissonInput(n=10, rate_hz=100.0, weight=2.0, dt=0.001, seed=42)
        drive_high_drive = PoissonInput(n=10, rate_hz=500.0, weight=5.0, dt=0.001, seed=42)
        mon_weak = SpikeMonitor(pop_weak)
        mon_high_drive = SpikeMonitor(pop_high_drive)
        net_weak = Network(pop_weak, drive_weak, mon_weak)
        net_high_drive = Network(pop_high_drive, drive_high_drive, mon_high_drive)
        net_weak.run(duration=0.5, dt=0.001, backend="python")
        net_high_drive.run(duration=0.5, dt=0.001, backend="python")
        assert mon_high_drive.count > mon_weak.count
