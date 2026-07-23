# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestStochasticIFIsolation from former test_model_stochastic_if.py

"""Focused suite: TestStochasticIFIsolation from former test_model_stochastic_if.py."""

from __future__ import annotations

from tests.model_stochastic_if_support import *  # noqa: F403

class TestStochasticIFIsolation:
    def test_construction_defaults(self):
        n = StochasticIFNeuron()
        assert n.v == -70.0
        assert n.sigma == 3.0
        assert n.tau_m == 20.0
        assert n.v_threshold == -50.0
        assert n.dt == 1.0

    def test_step_returns_binary(self):
        assert StochasticIFNeuron().step(0.0) in (0, 1)

    def test_v_evolves_with_noise(self):
        n = StochasticIFNeuron()
        v0 = n.v
        n.step(10.0)
        assert n.v != v0

    def test_state_finite_long_run(self):
        n = StochasticIFNeuron()
        for _ in range(100000):
            n.step(20.0)
        assert np.isfinite(n.v)

    def test_reset(self):
        n = StochasticIFNeuron()
        for _ in range(100):
            n.step(30.0)
        n.reset()
        assert n.v == n.v_rest
