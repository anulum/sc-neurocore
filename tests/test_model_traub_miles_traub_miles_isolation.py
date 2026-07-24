# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTraubMilesIsolation from former test_model_traub_miles.py

"""Focused suite: TestTraubMilesIsolation from former test_model_traub_miles.py."""

from __future__ import annotations

from tests.model_traub_miles_support import *  # noqa: F403


class TestTraubMilesIsolation:
    def test_construction_defaults(self):
        n = TraubMilesNeuron()
        assert n.v == -67.0
        assert n.g_na == 100.0
        assert n.g_k == 80.0
        assert n.dt == 0.01
        assert n.v_threshold == -20.0

    def test_step_returns_binary(self):
        assert TraubMilesNeuron().step(0.0) in (0, 1)

    def test_four_variables_evolve(self):
        n = TraubMilesNeuron()
        initial = (n.v, n.m, n.h, n.n)
        for _ in range(100):
            n.step(5.0)
        for name, v0, v1 in zip(["v", "m", "h", "n"], initial, (n.v, n.m, n.h, n.n)):
            assert v0 != v1, f"{name} didn't evolve"

    def test_state_finite_long_run(self):
        n = TraubMilesNeuron()
        for _ in range(50000):
            n.step(5.0)
        for var in [n.v, n.m, n.h, n.n]:
            assert np.isfinite(var)

    def test_reset(self):
        n = TraubMilesNeuron()
        for _ in range(500):
            n.step(5.0)
        n.reset()
        assert n.v == -67.0 and n.m == 0.05 and n.h == 0.6 and n.n == 0.3

    def test_ten_substeps(self):
        """Model uses 10 sub-steps: 10 × dt=0.01 = 0.1 ms per step()."""
        n = TraubMilesNeuron()
        v0 = n.v
        n.step(5.0)
        # With 10 sub-steps, V should have changed substantially
        assert abs(n.v - v0) > 0.01

    def test_step_uses_candidate_first_rk4_substeps(self):
        n = TraubMilesNeuron(v=-63.5, m=0.08, h=0.55, n=0.32)
        expected = _rk4_expected_after_call(n, 4.0)
        euler_candidate = (
            -65.66233161606698,
            0.0415454873682337,
            0.5626228886787493,
            0.30359624347230457,
        )

        spike = n.step(4.0)

        assert spike == 0
        assert (n.v, n.m, n.h, n.n) == pytest.approx(expected, abs=1e-14)
        assert n.v == pytest.approx(-65.6638958700765, abs=1e-14)
        assert n.m == pytest.approx(0.04237301812907925, abs=1e-14)
        assert n.h == pytest.approx(0.5626824931070477, abs=1e-14)
        assert n.n == pytest.approx(0.30356298261126924, abs=1e-14)
        assert abs(n.v - euler_candidate[0]) > 1e-3
        assert abs(n.m - euler_candidate[1]) > 5e-4
