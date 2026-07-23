# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDestexhe from former test_extended_neurons.py

"""Focused suite: TestDestexhe from former test_extended_neurons.py."""

from __future__ import annotations

from tests.extended_neurons_support import *  # noqa: F403

class TestDestexhe:
    def test_fires(self):
        n = DestexheThalamicNeuron()
        assert sum(n.step(5.0) for _ in range(200)) > 0

    def test_t_current(self):
        n = DestexheThalamicNeuron()
        for _ in range(100):
            n.step(3.0)
        assert n.h_t != 1.0
