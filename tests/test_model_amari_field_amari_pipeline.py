# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAmariPipeline from former test_model_amari_field.py

"""Focused suite: TestAmariPipeline from former test_model_amari_field.py."""

from __future__ import annotations

from tests.model_amari_field_support import *  # noqa: F403

class TestAmariPipeline:
    def test_population(self):
        assert Population(AmariNeuralField, n=3, label="amari").n == 3

    def test_network_runs(self):
        """Network accepts AmariNeuralField. step() gets float from PoissonInput,
        but the model's step() expects array — Population wraps it."""
        pop = Population(AmariNeuralField, n=3, label="amari")
        drive = PoissonInput(n=3, rate_hz=100.0, weight=1.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert isinstance(mon.count, int)
