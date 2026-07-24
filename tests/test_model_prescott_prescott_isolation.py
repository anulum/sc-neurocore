# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPrescottIsolation from former test_model_prescott.py

"""Focused suite: TestPrescottIsolation from former test_model_prescott.py."""

from __future__ import annotations

from tests.model_prescott_support import *  # noqa: F403


class TestPrescottIsolation:
    def test_construction_defaults(self):
        n = PrescottNeuron()
        assert n.v == -65.0
        assert n.w == 0.0
        assert n.beta_w == -21.0
        assert n.tau_w == 100.0
        assert n.dt == 0.1

    def test_step_returns_binary(self):
        assert PrescottNeuron().step(0.0) in (0, 1)

    def test_two_state_variables_evolve(self):
        n = PrescottNeuron()
        v0, w0 = n.v, n.w
        for _ in range(500):
            n.step(50.0)
        assert n.v != v0
        assert n.w != w0

    def test_step_uses_candidate_first_rk4(self):
        n = PrescottNeuron()
        expected_v, expected_w = _prescott_rk4_after_call(n, 50.0)
        euler_v, euler_w = _prescott_rhs(n, n.v, n.w, 50.0)
        euler_candidate = (n.v + n.dt * euler_v, n.w + n.dt * euler_w)
        spike = n.step(50.0)
        assert spike == 0
        assert n.v == pytest.approx(expected_v, abs=1e-12)
        assert n.w == pytest.approx(expected_w, abs=1e-15)
        assert (n.v, n.w) != pytest.approx(euler_candidate)

    def test_state_finite_long_run(self):
        n = PrescottNeuron()
        for _ in range(50000):
            n.step(50.0)
        assert np.isfinite(n.v) and np.isfinite(n.w)

    def test_reset(self):
        n = PrescottNeuron()
        for _ in range(1000):
            n.step(50.0)
        n.reset()
        assert n.v == -65.0 and n.w == 0.0
