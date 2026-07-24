# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGLIF from former test_extended_neurons.py

"""Focused suite: TestGLIF from former test_extended_neurons.py."""

from __future__ import annotations

from tests.extended_neurons_support import *  # noqa: F403


class TestGLIF:
    def test_fires(self):
        n = GLIFNeuron()
        assert sum(n.step(30.0) for _ in range(200)) > 0

    def test_adaptation(self):
        n = GLIFNeuron()
        for _ in range(100):
            n.step(30.0)
        assert n.i_asc1 != 0.0 or n.i_asc2 != 0.0
