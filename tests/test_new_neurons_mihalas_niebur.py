# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMihalasNiebur from former test_new_neurons.py

"""Focused suite: TestMihalasNiebur from former test_new_neurons.py."""

from __future__ import annotations

from tests.new_neurons_support import *  # noqa: F403

class TestMihalasNiebur:
    def test_fires(self):
        from sc_neurocore.neurons.models import MihalasNieburNeuron

        n = MihalasNieburNeuron()
        spikes = sum(n.step(5.0) for _ in range(200))
        assert spikes > 0

    def test_adaptation_currents(self):
        from sc_neurocore.neurons.models import MihalasNieburNeuron

        n = MihalasNieburNeuron(r1=1.0, r2=0.5)
        for _ in range(50):
            n.step(5.0)
        # After spikes, adaptation currents should be non-zero
        assert n.i1 != 0.0 or n.i2 != 0.0
