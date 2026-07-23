# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPLIFNetwork from former test_model_plif.py

"""Focused suite: TestPLIFNetwork from former test_model_plif.py."""

from __future__ import annotations

from tests.model_plif_support import *  # noqa: F403

class TestPLIFNetwork:
    def test_population(self):
        pop = Population(ParametricLIFNeuron, n=10, label="plif")
        assert pop.n == 10

    def test_network_spikes(self):
        pop = Population(ParametricLIFNeuron, n=20, label="plif")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=1.5, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0
