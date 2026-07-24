# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLoihiCUBAIsolation from former test_model_loihi_cuba.py

"""Focused suite: TestLoihiCUBAIsolation from former test_model_loihi_cuba.py."""

from __future__ import annotations

from tests.model_loihi_cuba_support import *  # noqa: F403


class TestLoihiCUBAIsolation:
    def test_defaults(self):
        n = LoihiCUBANeuron()
        assert n.v == 0 and n.u == 0
        assert n.v_threshold == 1000 and n.tau_v == 10

    def test_integer_state(self):
        n = LoihiCUBANeuron()
        n.step(100)
        assert isinstance(n.v, int) and isinstance(n.u, int)

    def test_step_returns_binary(self):
        assert LoihiCUBANeuron().step(100) in (0, 1)

    def test_reset(self):
        n = LoihiCUBANeuron()
        for _ in range(100):
            n.step(100)
        n.reset()
        assert n.v == 0 and n.u == 0

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = LoihiCUBANeuron()
            trace = [(n.step(100), n.v) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]
