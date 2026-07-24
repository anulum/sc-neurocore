# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPoissonNeuron from former test_new_neurons.py

"""Focused suite: TestPoissonNeuron from former test_new_neurons.py."""

from __future__ import annotations

from tests.new_neurons_support import *  # noqa: F403


class TestPoissonNeuron:
    def test_fires_at_rate(self):
        from sc_neurocore.neurons.models import PoissonNeuron

        n = PoissonNeuron(rate_hz=1000.0, dt_ms=1.0)
        spikes = sum(n.step() for _ in range(10000))
        rate = spikes / 10.0  # 10 seconds
        assert 500 < rate < 1500

    def test_zero_rate(self):
        from sc_neurocore.neurons.models import PoissonNeuron

        n = PoissonNeuron(rate_hz=0.0)
        spikes = sum(n.step() for _ in range(1000))
        assert spikes == 0
