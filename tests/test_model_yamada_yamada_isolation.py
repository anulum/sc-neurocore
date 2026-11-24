# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestYamadaIsolation from former test_model_yamada.py

"""Focused suite: TestYamadaIsolation from former test_model_yamada.py."""

from __future__ import annotations

import numpy as np
import pytest
from sc_neurocore.neurons.models.yamada import (
    YamadaNeuron,
)
from tests.model_yamada_support import (
    _rk4_reference,
)


class TestYamadaIsolation:
    """The YamadaNeuron in isolation."""

    def test_construction_defaults(self) -> None:
        """Construction defaults."""
        n = YamadaNeuron()
        assert n.v == -60.0
        assert n.n == 0.1
        assert n.q == 0.0
        assert n.tau_q == 300.0
        assert n.dt == 0.05

    def test_step_returns_binary(self) -> None:
        """One step returns a binary spike."""
        assert YamadaNeuron().step(0.0) in (0, 1)

    def test_step_matches_independent_rk4_candidate(self) -> None:
        """Step matches independent RK4 candidate."""
        n = YamadaNeuron(v=-52.0, n=0.22, q=0.08, dt=0.025)
        expected = _rk4_reference(n, 18.0)

        spike = n.step(18.0)

        assert (n.v, n.n, n.q) == pytest.approx(expected, rel=1e-14, abs=1e-14)
        assert spike == int(expected[0] >= n.v_threshold and n.v_threshold > -52.0)

    def test_three_variables_evolve(self) -> None:
        """Three variables evolve."""
        n = YamadaNeuron()
        initial = (n.v, n.n, n.q)
        for _ in range(500):
            n.step(50.0)
        for name, v0, v1 in zip(["v", "n", "q"], initial, (n.v, n.n, n.q)):
            assert v0 != v1, f"{name} didn't evolve"

    def test_state_finite(self) -> None:
        """The state stays finite under drive."""
        n = YamadaNeuron()
        for _ in range(200000):
            n.step(50.0)
        assert all(np.isfinite(v) for v in [n.v, n.n, n.q])

    def test_reset(self) -> None:
        """A reset after drive leaves a valid state."""
        n = YamadaNeuron()
        for _ in range(1000):
            n.step(50.0)
        n.reset()
        assert n.v == -60.0 and n.n == 0.1 and n.q == 0.0
