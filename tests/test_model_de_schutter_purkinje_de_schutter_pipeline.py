# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDeSchutterPipeline from former test_model_de_schutter_purkinje.py

"""Focused suite: TestDeSchutterPipeline from former test_model_de_schutter_purkinje.py."""

from __future__ import annotations

from tests.model_de_schutter_purkinje_support import *  # noqa: F403

class TestDeSchutterPipeline:
    def test_population(self) -> None:
        assert Population(DeSchutterPurkinjeNeuron, n=3, label="dsp").n == 3

    def test_network_runs(self) -> None:
        pop = Population(DeSchutterPurkinjeNeuron, n=3, label="dsp")
        drive = PoissonInput(n=3, rate_hz=100.0, weight=100.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert isinstance(mon.count, int)
