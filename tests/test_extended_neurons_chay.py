# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestChay from former test_extended_neurons.py

"""Focused suite: TestChay from former test_extended_neurons.py."""

from __future__ import annotations

from tests.extended_neurons_support import *  # noqa: F403


class TestChay:
    def test_drive_changes_state_without_leaving_physical_bounds(self):
        rest = ChayNeuron()
        driven = ChayNeuron()
        for _ in range(500):
            rest.step(0.0)
            driven.step(5.0)
        assert driven.v > rest.v
        assert 0.0 <= driven.n <= 1.0
        assert driven.ca >= 0.0

    def test_calcium(self):
        n = ChayNeuron()
        for _ in range(200):
            n.step(5.0)
        assert n.ca != 0.1
