# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCSIsolation from former test_model_connor_stevens.py

"""Focused suite: TestCSIsolation from former test_model_connor_stevens.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.neurons.models.connor_stevens import (
    ConnorStevensNeuron,
)


class TestCSIsolation:
    """The ConnorStevensNeuron in isolation."""

    def test_defaults(self) -> None:
        """The defaults match the published model."""
        n = ConnorStevensNeuron()
        assert n.v == -68.0 and n.m == 0.01 and n.h == 0.99
        assert n.n == 0.1 and n.a == 0.5 and n.b == 0.1
        assert n.g_a == 47.7  # A-type current conductance
        assert n.dt == 0.01 and n.v_threshold == 0.0

    def test_six_state_variables(self) -> None:
        """The neuron has its six state variables."""
        n = ConnorStevensNeuron()
        for attr in ["v", "m", "h", "n", "a", "b"]:
            assert hasattr(n, attr)

    def test_step_returns_binary(self) -> None:
        """One step returns a binary spike."""
        assert ConnorStevensNeuron().step(0.0) in (0, 1)

    def test_state_finite_long_run(self) -> None:
        """The state stays finite over a long run."""
        n = ConnorStevensNeuron()
        for _ in range(500):
            n.step(20.0)
        for attr in ["v", "m", "h", "n", "a", "b"]:
            assert np.isfinite(getattr(n, attr)), f"{attr} not finite"

    def test_reset_restores_defaults(self) -> None:
        """A reset restores the defaults."""
        n = ConnorStevensNeuron()
        for _ in range(200):
            n.step(20.0)
        n.reset()
        assert n.v == -68.0 and n.m == 0.01 and n.h == 0.99
        assert n.n == 0.1 and n.a == 0.5 and n.b == 0.1

    def test_deterministic(self) -> None:
        """The same inputs give the same trace."""
        traces = []
        for _ in range(2):
            n = ConnorStevensNeuron()
            trace = [(n.step(20.0), n.v) for _ in range(100)]
            traces.append(trace)
        assert traces[0] == traces[1]
