# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDestIsolation from former test_model_destexhe_thalamic.py

"""Focused suite: TestDestIsolation from former test_model_destexhe_thalamic.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.neurons.models.destexhe_thalamic import (
    DestexheThalamicNeuron,
)


class TestDestIsolation:
    """The DestexheThalamicNeuron in isolation."""

    def test_defaults(self) -> None:
        """The defaults match the published model."""
        n = DestexheThalamicNeuron()
        assert n.v == -65.0 and n.h_na == 0.6 and n.n_k == 0.3
        assert n.m_t == 0.0 and n.h_t == 1.0
        assert n.g_t == 2.0  # T-current conductance
        assert n.dt == 0.02 and n.v_threshold == -20.0

    def test_five_state_variables(self) -> None:
        """The model has its five state variables."""
        n = DestexheThalamicNeuron()
        for attr in ["v", "h_na", "n_k", "m_t", "h_t"]:
            assert hasattr(n, attr)

    def test_step_returns_binary(self) -> None:
        """One step returns a binary spike."""
        assert DestexheThalamicNeuron().step(0.0) in (0, 1)

    def test_state_finite_long_run(self) -> None:
        """The state stays finite over a long run."""
        n = DestexheThalamicNeuron()
        for _ in range(10_000):
            n.step(5.0)
        for attr in ["v", "h_na", "n_k", "m_t", "h_t"]:
            assert np.isfinite(getattr(n, attr)), f"{attr} not finite"

    def test_reset_restores_defaults(self) -> None:
        """A reset restores the defaults."""
        n = DestexheThalamicNeuron()
        for _ in range(2000):
            n.step(5.0)
        n.reset()
        assert n.v == -65.0 and n.h_na == 0.6 and n.n_k == 0.3
        assert n.m_t == 0.0 and n.h_t == 1.0

    def test_deterministic(self) -> None:
        """The same inputs give the same trace."""
        traces = []
        for _ in range(2):
            n = DestexheThalamicNeuron()
            trace = [(n.step(5.0), n.v) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]
