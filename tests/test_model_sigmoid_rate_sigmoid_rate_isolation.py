# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSigmoidRateIsolation from former test_model_sigmoid_rate.py

"""Focused suite: TestSigmoidRateIsolation from former test_model_sigmoid_rate.py."""

from __future__ import annotations

from tests.model_sigmoid_rate_support import *  # noqa: F403


class TestSigmoidRateIsolation:
    def test_defaults(self):
        n = SigmoidRateNeuron()
        assert n.r == 0.0 and n.tau == 10.0 and n.beta == 1.0 and n.theta == 0.0

    def test_step_returns_float(self):
        assert isinstance(SigmoidRateNeuron().step(0.0), (float, np.floating))

    def test_r_evolves(self):
        n = SigmoidRateNeuron()
        n.step(5.0)
        assert n.r > 0.0

    def test_state_finite(self):
        n = SigmoidRateNeuron()
        for _ in range(100000):
            n.step(5.0)
        assert np.isfinite(n.r)

    def test_reset(self):
        n = SigmoidRateNeuron(tau=7.0, beta=2.5, theta=-0.4, dt=0.2)
        for _ in range(100):
            n.step(5.0)
        n.reset()
        assert n.r == 0.0
        assert (n.tau, n.beta, n.theta, n.dt) == (7.0, 2.5, -0.4, 0.2)

    def test_python_batch_matches_scalar_steps(self):
        scalar = SigmoidRateNeuron(r=0.25, tau=10.0, beta=2.0, theta=1.0, dt=0.5)
        expected = np.asarray([scalar.step(3.0) for _ in range(32)])
        batched = SigmoidRateNeuron(r=0.25, tau=10.0, beta=2.0, theta=1.0, dt=0.5)
        actual = batched.simulate(32, 3.0, backend="python")
        np.testing.assert_array_equal(actual, expected)
        assert batched.r == scalar.r
