# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSTGIsolation from former test_model_marder_stg.py

"""Focused suite: TestSTGIsolation from former test_model_marder_stg.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.neurons.models.marder_stg import (
    MarderSTGNeuron,
)
from tests.model_marder_stg_support import (
    _GATES,
)


class TestSTGIsolation:
    """The MarderSTGNeuron in isolation."""

    def test_defaults(self) -> None:
        """The defaults match the published model."""
        n = MarderSTGNeuron()
        assert n.v == -60.0 and n.ca == 0.05
        assert n.cm == 1.0 and n.tau_ca == 20.0 and n.f_ca == 0.94
        assert n.dt == 0.05 and n.v_threshold == -20.0

    def test_thirteen_state_variables(self) -> None:
        """Thirteen state variables."""
        n = MarderSTGNeuron()
        for s in ("v", *_GATES, "ca"):
            assert hasattr(n, s), f"missing state: {s}"

    def test_inactivation_gates_start_open(self) -> None:
        """Inactivation gates start open."""
        n = MarderSTGNeuron()
        assert (n.h_na, n.h_cat, n.h_cas, n.h_a) == (1.0, 1.0, 1.0, 1.0)

    def test_step_returns_binary(self) -> None:
        """One step returns a binary spike."""
        assert MarderSTGNeuron().step(0.0) in (0, 1)

    def test_all_states_evolve(self) -> None:
        """Voltage, gates and calcium all change without drive."""
        n = MarderSTGNeuron()
        initial = {s: getattr(n, s) for s in ("v", "m_na", "h_na", "m_cat", "m_kca", "ca")}
        for _ in range(5000):
            n.step(0.0)
        assert all(getattr(n, s) != v0 for s, v0 in initial.items())

    def test_state_finite_long_run(self) -> None:
        """The state stays finite over a long run."""
        n = MarderSTGNeuron()
        for _ in range(100_000):
            n.step(0.0)
        for attr in ("v", *_GATES, "ca"):
            assert np.isfinite(getattr(n, attr)), f"{attr} not finite"

    def test_reset_restores_defaults(self) -> None:
        """A reset restores the defaults."""
        n = MarderSTGNeuron()
        for _ in range(5000):
            n.step(2.0)
        n.reset()
        assert n.v == -60.0 and n.ca == 0.05
        assert (n.m_na, n.h_na, n.m_kca) == (0.0, 1.0, 0.0)

    def test_deterministic(self) -> None:
        """The same inputs give the same trace."""

        def trace() -> list[tuple[int, float]]:
            """Run a fresh neuron for 2000 undriven steps and return its spikes and voltages."""
            n = MarderSTGNeuron()
            return [(n.step(0.0), n.v) for _ in range(2000)]

        assert trace() == trace()
