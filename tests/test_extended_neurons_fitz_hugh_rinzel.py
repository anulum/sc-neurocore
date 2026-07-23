# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFitzHughRinzel from former test_extended_neurons.py

"""Focused suite: TestFitzHughRinzel from former test_extended_neurons.py."""

from __future__ import annotations

from tests.extended_neurons_support import *  # noqa: F403

class TestFitzHughRinzel:
    def test_bursting(self):
        n = FitzHughRinzelNeuron()
        assert sum(n.step(1.0) for _ in range(2000)) > 0

    def test_slow_var(self):
        n = FitzHughRinzelNeuron()
        for _ in range(500):
            n.step(0.5)
        assert n.y != 0.0
