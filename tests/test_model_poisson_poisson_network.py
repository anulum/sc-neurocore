# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPoissonNetwork from former test_model_poisson.py

"""Focused suite: TestPoissonNetwork from former test_model_poisson.py."""

from __future__ import annotations

from tests.model_poisson_support import *  # noqa: F403

class TestPoissonNetwork:
    def test_population(self) -> None:
        pop = Population(PoissonNeuron, n=20, label="poisson")
        assert pop.n == 20

    def test_network_spikes(self) -> None:
        pop = Population(PoissonNeuron, n=20, label="poisson")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=1.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        # PoissonNeuron ignores input (fires at its own rate)
        assert mon.count > 0
