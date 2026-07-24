# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHodgkinHuxley from former test_new_neurons.py

"""Focused suite: TestHodgkinHuxley from former test_new_neurons.py."""

from __future__ import annotations

from tests.new_neurons_support import *  # noqa: F403


class TestHodgkinHuxley:
    def test_fires(self):
        from sc_neurocore.neurons.models import HodgkinHuxleyNeuron

        n = HodgkinHuxleyNeuron()
        spikes = sum(n.step(10.0) for _ in range(100))
        assert spikes > 0

    def test_no_fire_without_input(self):
        from sc_neurocore.neurons.models import HodgkinHuxleyNeuron

        n = HodgkinHuxleyNeuron()
        spikes = sum(n.step(0.0) for _ in range(50))
        assert spikes == 0

    def test_gating_variables_bounded(self):
        from sc_neurocore.neurons.models import HodgkinHuxleyNeuron

        n = HodgkinHuxleyNeuron()
        for _ in range(100):
            n.step(10.0)
        assert 0.0 <= n.m <= 1.0
        assert 0.0 <= n.h <= 1.0
        assert 0.0 <= n.n <= 1.0
