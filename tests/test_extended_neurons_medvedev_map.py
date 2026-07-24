# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMedvedevMap from former test_extended_neurons.py

"""Focused suite: TestMedvedevMap from former test_extended_neurons.py."""

from __future__ import annotations

from tests.extended_neurons_support import *  # noqa: F403


class TestMedvedevMap:
    def test_dynamics(self):
        n = MedvedevMapNeuron()
        for _ in range(50):
            n.step(0.1)
        assert n.u != 0.0
