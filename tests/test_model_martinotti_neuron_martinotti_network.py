# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMartinottiNetwork from former test_model_martinotti_neuron.py

"""Focused suite: TestMartinottiNetwork from former test_model_martinotti_neuron.py."""

from __future__ import annotations

from tests.model_martinotti_neuron_support import *  # noqa: F403

class TestMartinottiNetwork:
    def test_population_size(self):
        assert Population(MartinottiNeuron, n=8, label="martinotti").n == 8

    def test_population_drives_spikes(self):
        pop = Population(MartinottiNeuron, n=5, label="martinotti")
        drive = PoissonInput(n=5, rate_hz=600.0, weight=8.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon.count > 0
