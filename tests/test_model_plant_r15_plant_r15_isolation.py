# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPlantR15Isolation from former test_model_plant_r15.py

"""Focused suite: TestPlantR15Isolation from former test_model_plant_r15.py."""

from __future__ import annotations

from sc_neurocore.neurons.models.plant_r15 import (
    PlantR15Neuron,
)


class TestPlantR15Isolation:
    """The PlantR15Neuron in isolation."""

    def test_construction_defaults(self) -> None:
        """Construction defaults."""
        n = PlantR15Neuron()
        assert n.v == -50.0
        assert n.m == 0.05
        assert n.h == 0.6
        assert n.n == 0.3
        assert n.ca == 0.1
        assert n.dt == 0.05
        assert n.v_threshold == -10.0

    def test_step_returns_binary(self) -> None:
        """One step returns a binary spike."""
        n = PlantR15Neuron()
        assert n.step(0.0) in (0, 1)

    def test_five_state_variables_evolve(self) -> None:
        """All five state variables (V, m, h, n, Ca) should change."""
        n = PlantR15Neuron()
        initial = (n.v, n.m, n.h, n.n, n.ca)
        for _ in range(100):
            n.step(1.0)
        final = (n.v, n.m, n.h, n.n, n.ca)
        for i, (name, v0, v1) in enumerate(zip(["v", "m", "h", "n", "ca"], initial, final)):
            assert v0 != v1, f"{name} didn't evolve"

    def test_substep_integration(self) -> None:
        """Model uses 5 sub-steps per step() call for numerical stability."""
        n = PlantR15Neuron()
        v_before = n.v
        n.step(1.0)
        # With 5 sub-steps × dt=0.05, effective integration = 5 × 0.05 = 0.25 ms
        # Voltage should have changed
        assert n.v != v_before

    def test_reset_restores_initial(self) -> None:
        """Reset restores initial."""
        n = PlantR15Neuron()
        for _ in range(500):
            n.step(1.0)
        n.reset()
        assert n.v == -50.0
        assert n.m == 0.05
        assert n.h == 0.6
        assert n.n == 0.3
        assert n.ca == 0.1
