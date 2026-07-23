# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLNMIsolation from former test_model_lnm.py

"""Focused suite: TestLNMIsolation from former test_model_lnm.py."""

from __future__ import annotations

from tests.model_lnm_support import *  # noqa: F403

class TestLNMIsolation:
    def test_defaults(self):
        n = LearnableNeuronModel()
        assert n.v == 0.0 and n.alpha == 0.9 and n.beta == 0.1
        assert n.gamma == 0.05 and n.v_threshold == 1.0

    def test_step_returns_binary(self):
        assert LearnableNeuronModel().step(0.0) in (0, 1)

    def test_state_finite(self):
        n = LearnableNeuronModel()
        for _ in range(100_000):
            n.step(5.0)
        assert np.isfinite(n.v)

    def test_reset(self):
        n = LearnableNeuronModel()
        for _ in range(100):
            n.step(5.0)
        n.reset()
        assert n.v == 0.0

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = LearnableNeuronModel()
            trace = [(n.step(5.0), n.v) for _ in range(500)]
            traces.append(trace)
        assert traces[0] == traces[1]
