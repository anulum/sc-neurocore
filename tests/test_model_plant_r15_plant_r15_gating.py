# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPlantR15Gating from former test_model_plant_r15.py

"""Focused suite: TestPlantR15Gating from former test_model_plant_r15.py."""

from __future__ import annotations

from tests.model_plant_r15_support import *  # noqa: F403

class TestPlantR15Gating:
    def test_gating_bounded(self):
        """m, h, n should stay approximately in [0, 1]."""
        n = PlantR15Neuron()
        for _ in range(50000):
            n.step(1.0)
        for name, val in [("m", n.m), ("h", n.h), ("n", n.n)]:
            assert -0.01 <= val <= 1.01, f"{name} = {val:.6f}"

    def test_gating_at_equilibrium(self):
        """At fixed point, gating variables should be stable."""
        n = PlantR15Neuron()
        for _ in range(50000):
            n.step(0.0)
        g1 = (n.m, n.h, n.n)
        for _ in range(10000):
            n.step(0.0)
        g2 = (n.m, n.h, n.n)
        for name, v1, v2 in zip(["m", "h", "n"], g1, g2):
            assert abs(v1 - v2) < 1e-4, f"{name} drifted: {v1:.6f} → {v2:.6f}"
