# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Rulkov map network contracts

"""Focused suite: TestRulkovNetwork from former test_model_rulkov_map.py."""

from __future__ import annotations

from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.network import Network
from sc_neurocore.network.population import Population
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.neurons.models.rulkov_map import RulkovMapNeuron


class TestRulkovNetwork:
    def test_population(self) -> None:
        assert Population(RulkovMapNeuron, n=10, label="rulkov").n == 10

    def test_network_spikes(self) -> None:
        pop = Population(RulkovMapNeuron, n=10, label="rulkov")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=2.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0
