# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAlphaNeuron from former test_new_neurons.py

"""Focused suite: TestAlphaNeuron from former test_new_neurons.py."""

from __future__ import annotations

from tests.new_neurons_support import *  # noqa: F403


class TestAlphaNeuron:
    def test_fires_with_excitation(self):
        from sc_neurocore.neurons.models import AlphaNeuron

        n = AlphaNeuron()
        spikes = sum(n.step(5.0) for _ in range(200))
        assert spikes > 0

    def test_inhibition_blocks(self):
        from sc_neurocore.neurons.models import AlphaNeuron

        n = AlphaNeuron()
        spikes = sum(n.step(2.0, 10.0) for _ in range(200))
        assert spikes == 0, "strong inhibition should block firing"
