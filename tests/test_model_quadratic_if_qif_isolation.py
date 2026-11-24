# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestQIFIsolation from former test_model_quadratic_if.py

"""Focused suite: TestQIFIsolation from former test_model_quadratic_if.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.neurons.models.quadratic_if import (
    QuadraticIFNeuron,
)


class TestQIFIsolation:
    """The QuadraticIFNeuron in isolation."""

    def test_construction_defaults(self) -> None:
        """Construction defaults."""
        n = QuadraticIFNeuron()
        assert n.v == -1.0
        assert n.v_reset == -1.0
        assert n.v_peak == 1.0
        assert n.dt == 0.01

    def test_step_returns_binary(self) -> None:
        """One step returns a binary spike."""
        assert QuadraticIFNeuron().step(0.0) in (0, 1)

    def test_voltage_evolves(self) -> None:
        """One driven step changes the voltage."""
        n = QuadraticIFNeuron()
        v0 = n.v
        n.step(1.0)
        assert n.v != v0

    def test_state_finite(self) -> None:
        """The state stays finite under drive."""
        n = QuadraticIFNeuron()
        for _ in range(50000):
            n.step(1.0)
        assert np.isfinite(n.v)

    def test_reset(self) -> None:
        """A reset after drive leaves a valid state."""
        n = QuadraticIFNeuron()
        for _ in range(100):
            n.step(2.0)
        n.reset()
        assert n.v == n.v_reset
