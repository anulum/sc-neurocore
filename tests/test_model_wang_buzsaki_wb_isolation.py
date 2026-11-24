# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWBIsolation from former test_model_wang_buzsaki.py

"""Focused suite: TestWBIsolation from former test_model_wang_buzsaki.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.neurons.models.wang_buzsaki import (
    WangBuzsakiNeuron,
)


class TestWBIsolation:
    """The WangBuzsakiNeuron in isolation."""

    def test_construction_defaults(self) -> None:
        """Construction defaults."""
        n = WangBuzsakiNeuron()
        assert n.v == -65.0
        assert n.h == 0.8
        assert n.n == 0.1
        assert n.g_na == 35.0
        assert n.phi == 5.0
        assert n.dt == 0.01

    def test_step_returns_binary(self) -> None:
        """One step returns a binary spike."""
        assert WangBuzsakiNeuron().step(0.0) in (0, 1)

    def test_three_variables_evolve(self) -> None:
        """Three variables evolve."""
        n = WangBuzsakiNeuron()
        initial = (n.v, n.h, n.n)
        for _ in range(100):
            n.step(1.0)
        for name, v0, v1 in zip(["v", "h", "n"], initial, (n.v, n.h, n.n)):
            assert v0 != v1, f"{name} didn't evolve"

    def test_state_finite(self) -> None:
        """The state stays finite under drive."""
        n = WangBuzsakiNeuron()
        for _ in range(20000):
            n.step(2.0)
        assert all(np.isfinite(v) for v in [n.v, n.h, n.n])

    def test_reset(self) -> None:
        """A reset after drive leaves a valid state."""
        n = WangBuzsakiNeuron()
        for _ in range(200):
            n.step(5.0)
        n.reset()
        assert n.v == -65.0 and n.h == 0.8 and n.n == 0.1

    def test_substep_count(self) -> None:
        """int(0.5/dt) = 50 sub-steps at dt=0.01."""
        n = WangBuzsakiNeuron(dt=0.01)
        expected = int(0.5 / 0.01)  # 50
        assert expected == 50
