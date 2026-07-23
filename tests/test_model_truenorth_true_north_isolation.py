# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTrueNorthIsolation from former test_model_truenorth.py

"""Focused suite: TestTrueNorthIsolation from former test_model_truenorth.py."""

from __future__ import annotations

from tests.model_truenorth_support import *  # noqa: F403

class TestTrueNorthIsolation:
    def test_construction_defaults(self):
        n = TrueNorthNeuron()
        assert n.v == 0
        assert n.leak == 0
        assert n.threshold == 100
        assert n.v_reset == 0

    def test_step_returns_binary(self):
        assert TrueNorthNeuron().step(0) in (0, 1)

    def test_integer_types(self):
        """All state and params are integers (digital neuron)."""
        n = TrueNorthNeuron()
        assert isinstance(n.v, int)
        assert isinstance(n.threshold, int)
        assert isinstance(n.leak, int)

    def test_state_evolves(self):
        n = TrueNorthNeuron()
        n.step(50)
        assert n.v == 50  # 0 + 50 - 0 = 50

    def test_reset(self):
        n = TrueNorthNeuron()
        for _ in range(20):
            n.step(50)
        n.reset()
        assert n.v == 0
