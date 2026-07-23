# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPLIFIsolation from former test_model_plif.py

"""Focused suite: TestPLIFIsolation from former test_model_plif.py."""

from __future__ import annotations

from tests.model_plif_support import *  # noqa: F403

class TestPLIFIsolation:
    def test_construction_defaults(self):
        n = ParametricLIFNeuron()
        assert n.v == 0.0
        assert n.a == 0.0
        assert n.threshold == 1.0
        assert n.dt == 1.0

    def test_step_returns_binary(self):
        assert ParametricLIFNeuron().step(0.0) in (0, 1)

    def test_alpha_is_sigmoid_of_a(self):
        """alpha = 1 / (1 + exp(-a)) — the learnable decay parameter."""
        for a_val in [-3.0, -1.0, 0.0, 1.0, 3.0]:
            n = ParametricLIFNeuron(a=a_val)
            expected = 1.0 / (1.0 + np.exp(-a_val))
            assert abs(n.alpha - expected) < 1e-12, (
                f"a={a_val}: alpha={n.alpha}, expected={expected}"
            )

    def test_alpha_at_zero(self):
        """a=0 → alpha=0.5 (symmetric midpoint)."""
        assert ParametricLIFNeuron(a=0.0).alpha == 0.5

    def test_alpha_monotonic_in_a(self):
        """alpha increases monotonically with a."""
        alphas = [ParametricLIFNeuron(a=a).alpha for a in [-2, -1, 0, 1, 2]]
        assert all(alphas[i] < alphas[i + 1] for i in range(len(alphas) - 1))

    def test_alpha_bounded_0_1(self):
        """Sigmoid output ∈ (0, 1) for moderate a values."""
        for a_val in [-10.0, -5.0, 5.0, 10.0]:
            alpha = ParametricLIFNeuron(a=a_val).alpha
            assert 0.0 < alpha < 1.0, f"a={a_val}: alpha={alpha}"

    def test_alpha_saturates_at_extreme_a(self):
        """At extreme a, sigmoid saturates in float64.

        exp(-100) ≈ 3.7e-44 (not zero), but exp(100) overflows → alpha=1.0.
        """
        assert ParametricLIFNeuron(a=100.0).alpha == 1.0
        assert ParametricLIFNeuron(a=-100.0).alpha < 1e-40

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"v": np.nan},
            {"v": np.inf},
            {"a": np.nan},
            {"a": np.inf},
            {"threshold": 0.0},
            {"threshold": np.nan},
            {"dt": 0.0},
            {"dt": np.inf},
        ],
    )
    def test_rejects_non_physical_configuration(self, kwargs):
        with pytest.raises(ValueError):
            ParametricLIFNeuron(**kwargs)

    @pytest.mark.parametrize("current", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_current_before_state_mutation(self, current: float):
        n = ParametricLIFNeuron(v=0.25)
        before = n.v
        with pytest.raises(ValueError, match="current"):
            n.step(current)
        assert n.v == before

    @pytest.mark.parametrize(
        "field",
        ["v", "a", "threshold", "dt"],
    )
    def test_rejects_corrupted_runtime_state_before_mutation(self, field: str):
        n = ParametricLIFNeuron(v=0.25)
        before = n.v
        setattr(n, field, np.nan)
        with pytest.raises(ValueError, match="runtime"):
            n.step(0.1)
        if field != "v":
            assert n.v == before

    def test_rejects_non_finite_voltage_candidate_before_mutation(self):
        n = ParametricLIFNeuron(v=1.0e308, a=1000.0, threshold=1.7e308)
        before = n.v
        with pytest.raises(ValueError, match="voltage candidate"):
            n.step(1.0e308)
        assert n.v == before

    def test_alpha_is_stable_for_large_negative_parameter(self):
        n = ParametricLIFNeuron(a=-1000.0)
        assert n.alpha == 0.0
