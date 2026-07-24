# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGalvesLocherbach from former test_biophysical_neurons.py

"""Focused suite: TestGalvesLocherbach from former test_biophysical_neurons.py."""

from __future__ import annotations

from tests.biophysical_neurons_support import *  # noqa: F403


class TestGalvesLocherbach:
    def test_stochastic_firing(self):
        from sc_neurocore.neurons.models import GalvesLocherbachNeuron

        n = GalvesLocherbachNeuron()
        spikes = sum(n.step(2.0) for _ in range(1000))
        assert spikes > 0

    def test_no_fire_without_input(self):
        from sc_neurocore.neurons.models import GalvesLocherbachNeuron

        n = GalvesLocherbachNeuron(steepness=20.0, threshold_rate=5.0)
        spikes = sum(n.step(0.0) for _ in range(100))
        assert spikes == 0
