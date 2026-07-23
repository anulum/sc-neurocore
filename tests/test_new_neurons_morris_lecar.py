# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMorrisLecar from former test_new_neurons.py

"""Focused suite: TestMorrisLecar from former test_new_neurons.py."""

from __future__ import annotations

from tests.new_neurons_support import *  # noqa: F403

class TestMorrisLecar:
    def test_fires(self):
        from sc_neurocore.neurons.models import MorrisLecarNeuron

        n = MorrisLecarNeuron()
        spikes = sum(n.step(100.0) for _ in range(500))
        assert spikes > 0

    def test_calcium_activation(self):
        from sc_neurocore.neurons.models import MorrisLecarNeuron

        n = MorrisLecarNeuron()
        for _ in range(100):
            n.step(100.0)
        assert n.w > 0.0, "potassium activation must grow"
