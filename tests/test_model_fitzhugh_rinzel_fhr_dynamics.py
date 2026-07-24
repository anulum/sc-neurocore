# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFHRDynamics from former test_model_fitzhugh_rinzel.py

"""Focused suite: TestFHRDynamics from former test_model_fitzhugh_rinzel.py."""

from __future__ import annotations

from tests.model_fitzhugh_rinzel_support import *  # noqa: F403


class TestFHRDynamics:
    def test_derivative_formula(self):
        """The derivative matches the three-state FitzHugh-Rinzel ODE."""
        n = FitzHughRinzelNeuron(v=-1.0, w=0.2, y=0.1)
        assert n._derivatives(n.v, n.w, n.y, 0.5) == pytest.approx(_rhs(n.v, n.w, n.y, 0.5))

    def test_derivative_rejects_nonfinite_runtime_inputs(self):
        """The ODE primitive rejects corrupted nonfinite runtime values."""
        n = FitzHughRinzelNeuron()
        with pytest.raises(FloatingPointError, match="state and current must be finite"):
            n._derivatives(math.nan, n.w, n.y, 0.5)

    def test_step_matches_independent_rk4_reference(self):
        n = FitzHughRinzelNeuron(v=-1.0, w=0.2, y=0.1)
        expected = _rk4_reference(n.v, n.w, n.y, 0.5, n.dt)
        assert n.step(0.5) == 0
        assert (n.v, n.w, n.y) == pytest.approx(expected, abs=1.0e-15)

    @pytest.mark.parametrize(
        "current, expected", [(0.0, 0), (0.5, 25), (0.8, 28), (1.0, 28), (2.0, 1)]
    )
    def test_deterministic_current_regimes(self, current: float, expected: int):
        n = FitzHughRinzelNeuron()
        assert len(_run(n, current=current, steps=10000)) == expected
        assert np.isfinite(n.v) and np.isfinite(n.w) and np.isfinite(n.y)

    def test_v_bounded(self):
        n = FitzHughRinzelNeuron()
        vs = [n.v]
        for _ in range(10000):
            n.step(0.5)
            vs.append(n.v)
        assert min(vs) > -3 and max(vs) < 3

    def test_isi_regularity(self):
        n = FitzHughRinzelNeuron()
        spikes = _run(n, current=0.5, steps=10000)
        isis = np.diff(spikes[2:]).astype(float)
        cv = np.std(isis) / np.mean(isis)
        assert cv < 0.3
