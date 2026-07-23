# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRulkovNetwork from former test_model_rulkov_map.py

"""Focused suite: TestRulkovNetwork from former test_model_rulkov_map.py."""

from __future__ import annotations

from tests.model_rulkov_map_support import *  # noqa: F403

class TestRulkovNetwork:
    def test_population(self):
        assert Population(RulkovMapNeuron, n=10, label="rulkov").n == 10

    def test_network_spikes(self):
        pop = Population(RulkovMapNeuron, n=10, label="rulkov")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=2.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0
