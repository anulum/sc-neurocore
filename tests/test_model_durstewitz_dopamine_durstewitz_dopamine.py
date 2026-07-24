# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDurstewitzDopamine from former test_model_durstewitz_dopamine.py

"""Focused suite: TestDurstewitzDopamine from former test_model_durstewitz_dopamine.py."""

from __future__ import annotations

from tests.model_durstewitz_dopamine_support import *  # noqa: F403


class TestDurstewitzDopamine:
    def test_fires(self):
        from sc_neurocore.neurons.models.durstewitz_dopamine import DurstewitzDopamineNeuron

        n = DurstewitzDopamineNeuron()
        assert sum(n.step(10.0) for _ in range(300)) > 0

    def test_d1_modulation(self):
        from sc_neurocore.neurons.models.durstewitz_dopamine import DurstewitzDopamineNeuron

        n = DurstewitzDopamineNeuron(d1_level=0.8)
        for _ in range(100):
            n.step(8.0)
        assert n.v != -65.0
