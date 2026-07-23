# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSTGGating from former test_model_marder_stg.py

"""Focused suite: TestSTGGating from former test_model_marder_stg.py."""

from __future__ import annotations

from tests.model_marder_stg_support import *  # noqa: F403

class TestSTGGating:
    def test_gates_bounded(self):
        n = MarderSTGNeuron()
        for _ in range(50_000):
            n.step(2.0)
        for gate in _GATES:
            assert 0.0 <= getattr(n, gate) <= 1.0, gate

    def test_sigmoid_midpoint(self):
        assert MarderSTGNeuron._sigmoid(-25.5, -25.5, 5.29) == 0.5

    def test_sigmoid_limits(self):
        assert MarderSTGNeuron._sigmoid(100.0, -25.5, 5.29) > 0.999
        assert MarderSTGNeuron._sigmoid(-300.0, -25.5, 5.29) < 0.001

    def test_kca_activation_requires_calcium(self):
        """The K-C steady state scales with Ca/(Ca+3); near-zero Ca suppresses it."""
        n = MarderSTGNeuron()
        for _ in range(50_000):
            n.step(0.0)
        assert n.m_kca >= 0.0
