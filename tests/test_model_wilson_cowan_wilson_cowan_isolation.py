# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWilsonCowanIsolation from former test_model_wilson_cowan.py

"""Focused suite: TestWilsonCowanIsolation from former test_model_wilson_cowan.py."""

from __future__ import annotations

from tests.model_wilson_cowan_support import *  # noqa: F403

class TestWilsonCowanIsolation:
    def test_defaults(self):
        n = WilsonCowanUnit()
        assert n.e == 0.1 and n.i == 0.05
        assert n.w_ee == 10.0 and n.w_ei == 6.0
        assert n.tau_e == 1.0 and n.tau_i == 2.0

    def test_step_returns_float(self):
        """Returns E rate (float), not binary spike."""
        n = WilsonCowanUnit()
        result = n.step(0.0)
        assert isinstance(result, float)

    def test_both_variables_evolve(self):
        n = WilsonCowanUnit()
        e0, i0 = n.e, n.i
        for _ in range(100):
            n.step(5.0)
        assert n.e != e0 and n.i != i0

    def test_state_finite(self):
        n = WilsonCowanUnit()
        for _ in range(100000):
            n.step(5.0)
        assert np.isfinite(n.e) and np.isfinite(n.i)

    def test_reset(self):
        n = WilsonCowanUnit(w_ee=12.0, tau_i=3.0, theta=3.5, dt=0.05)
        for _ in range(100):
            n.step(5.0)
        n.reset()
        assert n.e == 0.1 and n.i == 0.05
        assert (n.w_ee, n.tau_i, n.theta, n.dt) == (12.0, 3.0, 3.5, 0.05)

    @pytest.mark.parametrize("n_steps", [-1, 1.5, True, 1 << 31])
    def test_simulate_rejects_invalid_batch_length(self, n_steps: object):
        n = WilsonCowanUnit()
        before = (n.e, n.i)
        with pytest.raises(ValueError, match="n_steps"):
            n.simulate(n_steps)  # type: ignore[arg-type]
        assert (n.e, n.i) == before

    def test_simulate_rejects_non_finite_current_before_mutation(self):
        n = WilsonCowanUnit()
        before = (n.e, n.i)
        with pytest.raises(ValueError, match="current"):
            n.simulate(4, math.nan)
        assert (n.e, n.i) == before

    def test_python_batch_numerical_failure_is_atomic(self):
        n = WilsonCowanUnit(dt=1.0e308)
        before = (n.e, n.i)
        with pytest.raises((ValueError, FloatingPointError)):
            n.simulate(2, 1.5, backend="python")
        assert (n.e, n.i) == before
