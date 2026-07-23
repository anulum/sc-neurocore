# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPinskyRinzelNetwork from former test_model_pinsky_rinzel.py

"""Focused suite: TestPinskyRinzelNetwork from former test_model_pinsky_rinzel.py."""

from __future__ import annotations

from tests.model_pinsky_rinzel_support import *  # noqa: F403

class TestPinskyRinzelNetwork:
    def test_population(self):
        pop = Population(PinskyRinzelNeuron, n=5, label="pr")
        assert pop.n == 5

    def test_network_spikes(self):
        pop = Population(PinskyRinzelNeuron, n=5, label="pr")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=5.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon.count > 0
