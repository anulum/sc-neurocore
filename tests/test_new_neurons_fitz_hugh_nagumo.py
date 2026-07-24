# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFitzHughNagumo from former test_new_neurons.py

"""Focused suite: TestFitzHughNagumo from former test_new_neurons.py."""

from __future__ import annotations

from tests.new_neurons_support import *  # noqa: F403


class TestFitzHughNagumo:
    def test_fires(self):
        from sc_neurocore.neurons.models import FitzHughNagumoNeuron

        n = FitzHughNagumoNeuron()
        spikes = sum(n.step(1.0) for _ in range(500))
        assert spikes > 0

    def test_relaxation_oscillation(self):
        from sc_neurocore.neurons.models import FitzHughNagumoNeuron

        n = FitzHughNagumoNeuron()
        for _ in range(200):
            n.step(0.5)
        assert n.w != -0.5, "recovery variable must evolve"
