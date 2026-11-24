# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTrueNorthIsolation from former test_model_truenorth.py

"""Focused suite: TestTrueNorthIsolation from former test_model_truenorth.py."""

from __future__ import annotations

from sc_neurocore.neurons.models.truenorth import (
    TrueNorthNeuron,
)


class TestTrueNorthIsolation:
    """The TrueNorthNeuron in isolation."""

    def test_construction_defaults(self) -> None:
        """Construction defaults."""
        n = TrueNorthNeuron()
        assert n.v == 0
        assert n.leak == 0
        assert n.threshold == 100
        assert n.v_reset == 0

    def test_step_returns_binary(self) -> None:
        """One step returns a binary spike."""
        assert TrueNorthNeuron().step(0) in (0, 1)

    def test_integer_types(self) -> None:
        """All state and params are integers (digital neuron)."""
        n = TrueNorthNeuron()
        assert isinstance(n.v, int)
        assert isinstance(n.threshold, int)
        assert isinstance(n.leak, int)

    def test_state_evolves(self) -> None:
        """The state evolves under drive."""
        n = TrueNorthNeuron()
        n.step(50)
        assert n.v == 50  # 0 + 50 - 0 = 50

    def test_reset(self) -> None:
        """A reset after drive leaves a valid state."""
        n = TrueNorthNeuron()
        for _ in range(20):
            n.step(50)
        n.reset()
        assert n.v == 0
