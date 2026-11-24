# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLTCIsolation from former test_model_ltc.py

"""Focused suite: TestLTCIsolation from former test_model_ltc.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.neurons.models.ltc import (
    LiquidTimeConstantNeuron,
)


class TestLTCIsolation:
    """The LiquidTimeConstantNeuron in isolation."""

    def test_defaults(self) -> None:
        """The defaults match the published model."""
        n = LiquidTimeConstantNeuron()
        assert n.x == 0.0 and n.tau_base == 10.0
        assert n.w_tau == -0.5 and n.v_threshold == 1.0

    def test_step_returns_binary(self) -> None:
        """One step returns a binary spike."""
        assert LiquidTimeConstantNeuron().step(0.0) in (0, 1)

    def test_state_finite(self) -> None:
        """The state stays finite under drive."""
        n = LiquidTimeConstantNeuron()
        for _ in range(100_000):
            n.step(5.0)
        assert np.isfinite(n.x)

    def test_reset(self) -> None:
        """A reset after drive leaves a valid state."""
        n = LiquidTimeConstantNeuron()
        for _ in range(100):
            n.step(5.0)
        n.reset()
        assert n.x == 0.0

    def test_deterministic(self) -> None:
        """The same inputs give the same trace."""
        traces = []
        for _ in range(2):
            n = LiquidTimeConstantNeuron()
            trace = [(n.step(5.0), n.x) for _ in range(500)]
            traces.append(trace)
        assert traces[0] == traces[1]
