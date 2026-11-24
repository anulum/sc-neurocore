# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHayIsolation from former test_model_hay_l5.py

"""Focused suite: TestHayIsolation from former test_model_hay_l5.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.neurons.models.hay_l5 import (
    HayL5PyramidalNeuron,
)


class TestHayIsolation:
    """The HayL5PyramidalNeuron in isolation."""

    def test_defaults(self) -> None:
        """The defaults match the published model."""
        n = HayL5PyramidalNeuron()
        assert n.v_s == -75.0 and n.v_t == -75.0 and n.v_a == -75.0
        assert n.h_na == 0.9 and n.n_k == 0.1
        assert n.m_ca == 0.0 and n.h_ca == 1.0 and n.m_ih == 0.0
        assert n.ca_a == 0.0001
        assert n.dt == 0.025

    def test_nine_state_variables(self) -> None:
        """The model has nine state variables."""
        n = HayL5PyramidalNeuron()
        for attr in ["v_s", "h_na", "n_k", "v_t", "m_ca", "h_ca", "m_ih", "v_a", "ca_a"]:
            assert hasattr(n, attr)

    def test_step_returns_binary(self) -> None:
        """One step returns a binary spike."""
        assert HayL5PyramidalNeuron().step(0.0) in (0, 1)

    def test_dual_input(self) -> None:
        """step() accepts current_soma and optional current_tuft."""
        n = HayL5PyramidalNeuron()
        n.step(5.0, current_tuft=2.0)
        assert np.isfinite(n.v_s)

    def test_state_finite_long_run(self) -> None:
        """The state stays finite over a long run."""
        n = HayL5PyramidalNeuron()
        for _ in range(10_000):
            n.step(10.0)
        for attr in ["v_s", "v_t", "v_a", "h_na", "n_k", "m_ca", "h_ca", "m_ih", "ca_a"]:
            assert np.isfinite(getattr(n, attr)), f"{attr} not finite"

    def test_reset_restores_defaults(self) -> None:
        """A reset restores the defaults."""
        n = HayL5PyramidalNeuron()
        for _ in range(2000):
            n.step(10.0)
        n.reset()
        assert n.v_s == -75.0 and n.v_t == -75.0 and n.v_a == -75.0
        assert n.ca_a == 0.0001

    def test_deterministic(self) -> None:
        """The same inputs give the same trace."""
        traces = []
        for _ in range(2):
            n = HayL5PyramidalNeuron()
            trace = [(n.step(10.0), n.v_s) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]
