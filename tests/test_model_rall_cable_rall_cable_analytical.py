# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRallCableAnalytical from former test_model_rall_cable.py

"""Focused suite: TestRallCableAnalytical from former test_model_rall_cable.py."""

from __future__ import annotations

from tests.model_rall_cable_support import *  # noqa: F403

class TestRallCableAnalytical:
    def test_cable_equation_one_step(self) -> None:
        """Implicit step solves the sealed passive cable tridiagonal system."""
        n = RallCableNeuron(n_comp=3)
        v0 = n.v.copy()
        I = 100.0
        alpha = n.dt / n.tau_m
        offdiag = -alpha * n.g_ratio
        matrix = np.array(
            [
                [1.0 + alpha + alpha * n.g_ratio, offdiag, 0.0],
                [offdiag, 1.0 + alpha + 2.0 * alpha * n.g_ratio, offdiag],
                [0.0, offdiag, 1.0 + alpha + alpha * n.g_ratio],
            ]
        )
        rhs = v0 - n.v_rest
        rhs[-1] += alpha * I
        expected = np.linalg.solve(matrix, rhs) + n.v_rest
        n.step(I)
        np.testing.assert_allclose(n.v, expected, atol=1e-10)

    def test_implicit_step_separates_from_forward_euler(self) -> None:
        n_implicit = RallCableNeuron(n_comp=3, dt=2.0, g_ratio=5.0)
        n_euler = RallCableNeuron(n_comp=3, dt=2.0, g_ratio=5.0)
        before = n_euler.v.copy()
        explicit = np.zeros(3)
        for i in range(3):
            leak = -(before[i] - n_euler.v_rest)
            left = before[i - 1] if i > 0 else before[i]
            right = before[i + 1] if i < 2 else before[i]
            axial = n_euler.g_ratio * (left - 2.0 * before[i] + right)
            inj = 200.0 if i == 2 else 0.0
            explicit[i] = before[i] + (leak + axial + inj) / n_euler.tau_m * n_euler.dt
        n_implicit.step(200.0)
        assert not np.allclose(n_implicit.v, explicit)

    def test_input_at_distal_end_only(self) -> None:
        """Current injected only at compartment N-1."""
        n = RallCableNeuron(n_comp=3)
        n.step(100.0)
        # Distal end (2) got input, others only leak/axial
        assert n.v[2] > n.v[0]

    def test_boundary_conditions(self) -> None:
        """Sealed ends: left of comp 0 = v[0], right of comp N-1 = v[N-1]."""
        n = RallCableNeuron(n_comp=3)
        v0 = n.v.copy()
        # At rest all equal → axial=0, only distal gets current
        n.step(100.0)
        # Comp 0: left=v[0] (sealed), so axial = g_ratio*(v[0]-2v[0]+v[1])
        # With all equal at rest: axial=0 → dv[0] = leak/tau_m = 0 (at rest)
        assert abs(n.v[0] - v0[0]) < 0.01

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"n_comp": 0},
            {"tau_m": 0.0},
            {"tau_m": float("nan")},
            {"g_ratio": -0.1},
            {"dt": 0.0},
        ],
    )
    def test_rejects_invalid_configuration(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            RallCableNeuron(**kwargs)

    def test_rejects_non_finite_current_without_mutation(self) -> None:
        n = RallCableNeuron(n_comp=3)
        before = n.v.copy()
        with pytest.raises(ValueError):
            n.step(float("nan"))
        np.testing.assert_allclose(n.v, before)

    def test_rejects_corrupt_state_without_mutation(self) -> None:
        n = RallCableNeuron(n_comp=3)
        n.v[1] = float("nan")
        before = n.v.copy()
        with pytest.raises(ValueError):
            n.step(1.0)
        assert np.isnan(n.v[1])
        assert np.array_equal(np.isnan(n.v), np.isnan(before))
