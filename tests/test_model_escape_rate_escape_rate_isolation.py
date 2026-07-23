# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEscapeRateIsolation from former test_model_escape_rate.py

"""Focused suite: TestEscapeRateIsolation from former test_model_escape_rate.py."""

from __future__ import annotations

from tests.model_escape_rate_support import *  # noqa: F403

class TestEscapeRateIsolation:
    def test_construction_all_defaults(self):
        n = EscapeRateNeuron()
        assert n.v == -70.0 and n.v_rest == -70.0 and n.v_reset == -70.0
        assert n.v_threshold == -50.0 and n.tau_m == 10.0
        assert n.rho_0 == 0.001 and n.delta_u == 3.0 and n.resistance == 1.0

    def test_step_returns_binary(self):
        assert EscapeRateNeuron().step(0.0) in (0, 1)

    def test_state_evolves(self):
        n = EscapeRateNeuron()
        v0 = n.v
        n.step(30.0)
        assert n.v != v0

    def test_state_finite_long_run(self):
        n = EscapeRateNeuron()
        for _ in range(100000):
            n.step(40.0)
        assert np.isfinite(n.v)

    def test_reset(self):
        n = EscapeRateNeuron()
        for _ in range(100):
            n.step(50.0)
        n.reset()
        assert n.v == n.v_rest
