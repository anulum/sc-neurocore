# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSFANetwork from former test_model_sfa.py

"""Focused suite: TestSFANetwork from former test_model_sfa.py."""

from __future__ import annotations

from tests.model_sfa_support import *  # noqa: F403

class TestSFANetwork:
    def test_population(self):
        assert Population(SFANeuron, n=10, label="sfa").n == 10

    def test_network_spikes(self):
        pop = Population(SFANeuron, n=10, label="sfa")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon.count > 0
