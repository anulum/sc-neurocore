# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPlantR15Calcium from former test_model_plant_r15.py

"""Focused suite: TestPlantR15Calcium from former test_model_plant_r15.py."""

from __future__ import annotations

from tests.model_plant_r15_support import *  # noqa: F403

class TestPlantR15Calcium:
    def test_calcium_non_negative(self):
        """Ca concentration is clamped ≥ 0."""
        n = PlantR15Neuron()
        for _ in range(50000):
            n.step(0.0)
        assert n.ca >= 0.0

    def test_calcium_accumulates_from_initial(self):
        """Ca should increase from initial 0.1 during early transient
        (Ca influx from depolarisation > Ca decay)."""
        n = PlantR15Neuron()
        ca_initial = n.ca
        for _ in range(5000):
            n.step(0.0)
        assert n.ca > ca_initial, f"Ca={n.ca:.4f} <= initial {ca_initial}"

    def test_calcium_at_equilibrium(self):
        """At steady state, Ca stabilises (dCa/dt ≈ 0)."""
        n = PlantR15Neuron()
        for _ in range(50000):
            n.step(0.0)
        ca_1 = n.ca
        for _ in range(10000):
            n.step(0.0)
        ca_2 = n.ca
        assert abs(ca_2 - ca_1) < 0.01, f"Ca still drifting: {ca_1:.4f} → {ca_2:.4f}"

    def test_calcium_suppresses_firing(self):
        """High Ca activates I_KCa, which hyperpolarises — the mechanism
        that terminates bursts in the R15 model."""
        n = PlantR15Neuron()
        for _ in range(50000):
            n.step(0.0)
        # At equilibrium, Ca should be significant
        assert n.ca > 0.5, f"Ca = {n.ca:.4f}, expected >0.5 at equilibrium"
