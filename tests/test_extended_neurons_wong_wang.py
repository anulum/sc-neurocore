# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWongWang from former test_extended_neurons.py

"""Focused suite: TestWongWang from former test_extended_neurons.py."""

from __future__ import annotations

from tests.extended_neurons_support import *  # noqa: F403

class TestWongWang:
    def test_decision(self):
        n = WongWangUnit()
        for _ in range(2000):
            n.step(0.02, 0.0)
        assert abs(n.s1 - n.s2) > 0.01
