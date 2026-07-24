# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSigmaDeltaNetwork from former test_model_sigma_delta.py

"""Focused suite: TestSigmaDeltaNetwork from former test_model_sigma_delta.py."""

from __future__ import annotations

from tests.model_sigma_delta_support import *  # noqa: F403


class TestSigmaDeltaNetwork:
    def test_population(self):
        pop = Population(SigmaDeltaNeuron, n=10, label="sd")
        assert pop.n == 10

    def test_network_with_drive(self):
        pop = Population(SigmaDeltaNeuron, n=10, label="sd")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=0.5, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0
