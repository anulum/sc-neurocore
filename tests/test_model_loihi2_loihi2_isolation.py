# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLoihi2Isolation from former test_model_loihi2.py

"""Focused suite: TestLoihi2Isolation from former test_model_loihi2.py."""

from __future__ import annotations

from sc_neurocore.neurons.models.loihi2 import (
    Loihi2Neuron,
)


class TestLoihi2Isolation:
    """The Loihi2Neuron in isolation."""

    def test_defaults(self) -> None:
        """The defaults match the published model."""
        n = Loihi2Neuron()
        assert n.s1 == 0 and n.s2 == 0 and n.s3 == 0
        assert n.s1_threshold == 1000 and n.w12 == 1

    def test_integer_state(self) -> None:
        """Both state registers stay integer-valued after a step."""
        n = Loihi2Neuron()
        n.step(100)
        assert isinstance(n.s1, int) and isinstance(n.s2, int)

    def test_step_returns_binary(self) -> None:
        """One step returns a binary spike."""
        assert Loihi2Neuron().step(100) in (0, 1)

    def test_reset(self) -> None:
        """A reset after drive leaves a valid state."""
        n = Loihi2Neuron()
        for _ in range(100):
            n.step(100)
        n.reset()
        assert n.s1 == 0 and n.s2 == 0 and n.s3 == 0

    def test_deterministic(self) -> None:
        """The same inputs give the same trace."""
        traces = []
        for _ in range(2):
            n = Loihi2Neuron()
            trace = [(n.step(100), n.s1) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]
