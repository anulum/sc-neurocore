# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPlantR15Stability from former test_model_plant_r15.py

"""Focused suite: TestPlantR15Stability from former test_model_plant_r15.py."""

from __future__ import annotations

from tests.model_plant_r15_support import *  # noqa: F403

class TestPlantR15Stability:
    def test_moderate_current_finite(self):
        """Moderate current (I≤10) keeps all state finite."""
        n = PlantR15Neuron()
        for _ in range(50000):
            n.step(10.0)
        for name, val in [("v", n.v), ("m", n.m), ("h", n.h), ("n", n.n), ("ca", n.ca)]:
            assert np.isfinite(val), f"{name} = {val}"

    def test_high_current_divergence(self):
        """Very high current (I≥100) may cause voltage divergence.

        This documents a numerical limitation — Euler integration with
        dt=0.05 and 5 sub-steps can't handle extreme drive.
        """
        n = PlantR15Neuron()
        for _ in range(100000):
            n.step(100.0)
        # At I=100, V may diverge far from biological range
        # We just document this — not a bug, just Euler limitation
        assert np.isfinite(n.v), "V is NaN/Inf — complete numerical failure"

    @pytest.mark.parametrize("dt", [0.02, 0.05, 0.1])
    def test_dt_stability(self, dt: float):
        """Model stays finite across time-step sizes."""
        n = PlantR15Neuron(dt=dt)
        for _ in range(50000):
            n.step(1.0)
        assert np.isfinite(n.v)
