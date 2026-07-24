# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSFAIsolation from former test_model_sfa.py

"""Focused suite: TestSFAIsolation from former test_model_sfa.py."""

from __future__ import annotations

from tests.model_sfa_support import *  # noqa: F403


class TestSFAIsolation:
    def test_construction_defaults(self):
        n = SFANeuron()
        assert n.v == -70.0
        assert n.g_sfa == 0.0
        assert n.tau_sfa == 200.0
        assert n.delta_g == 0.5
        assert n.e_k == -80.0
        assert n.dt == 1.0

    def test_step_returns_binary(self):
        assert SFANeuron().step(0.0) in (0, 1)

    def test_state_evolves(self):
        n = SFANeuron()
        v0 = n.v
        n.step(50.0)
        assert n.v != v0

    def test_state_finite_long_run(self):
        n = SFANeuron()
        for _ in range(50000):
            n.step(50.0)
        assert np.isfinite(n.v) and np.isfinite(n.g_sfa)

    def test_reset(self):
        n = SFANeuron()
        for _ in range(100):
            n.step(50.0)
        n.reset()
        assert n.v == n.v_rest
        assert n.g_sfa == 0.0
