# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSSTNetwork from former test_model_sst_neuron.py

"""Focused suite: TestSSTNetwork from former test_model_sst_neuron.py."""

from __future__ import annotations

from tests.model_sst_neuron_support import *  # noqa: F403

class TestSSTNetwork:
    def test_population_size(self):
        assert Population(SSTNeuron, n=8, label="sst").n == 8

    def test_population_drives_spikes(self):
        pop = Population(SSTNeuron, n=5, label="sst")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=6.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon.count > 0
