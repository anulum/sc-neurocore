# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestVIPNetwork from former test_model_vip_neuron.py

"""Focused suite: TestVIPNetwork from former test_model_vip_neuron.py."""

from __future__ import annotations

from tests.model_vip_neuron_support import *  # noqa: F403


class TestVIPNetwork:
    def test_population_size(self):
        assert Population(VIPNeuron, n=8, label="vip").n == 8

    def test_population_drives_spikes(self):
        pop = Population(VIPNeuron, n=5, label="vip")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=4.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon.count > 0
