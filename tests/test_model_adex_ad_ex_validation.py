# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAdExValidation from former test_model_adex.py

"""Focused suite: TestAdExValidation from former test_model_adex.py."""

from __future__ import annotations

from tests.model_adex_support import *  # noqa: F403


class TestAdExValidation:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("v", np.nan),
            ("w", np.inf),
            ("v_rest", -np.inf),
            ("v_reset", np.nan),
            ("v_threshold", np.inf),
            ("v_rh", -np.inf),
            ("a", np.nan),
            ("b", np.inf),
        ],
    )
    def test_rejects_non_finite_state_or_voltage_parameters(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            AdExNeuron(**{field: value})

    @pytest.mark.parametrize("field", ["delta_t", "tau", "tau_w", "c_m", "dt"])
    @pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
    def test_rejects_non_positive_or_non_finite_scale_parameters(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            AdExNeuron(**{field: value})

    @pytest.mark.parametrize("integrator", ["baseline_euler", "rk4", "rosenbrock"])
    @pytest.mark.parametrize("current", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_current_before_state_mutation(
        self, integrator: str, current: float
    ):
        n = AdExNeuron(v=-60.0, w=3.0, integrator=integrator)
        before = (n.v, n.w)
        with pytest.raises(ValueError, match="current"):
            n.step(current)
        assert (n.v, n.w) == before

    @pytest.mark.parametrize("integrator", ["baseline_euler", "rk4", "rosenbrock"])
    def test_rejects_non_finite_runtime_state_before_update(self, integrator: str):
        n = AdExNeuron(v=-60.0, w=3.0, integrator=integrator)
        n.w = float("nan")
        with pytest.raises(ValueError, match="runtime adaptation state"):
            n.step(0.0)
        assert np.isnan(n.w)

    @pytest.mark.parametrize("integrator", ["baseline_euler", "rk4", "rosenbrock"])
    def test_rejects_non_finite_integrator_update_before_state_mutation(self, integrator: str):
        n = AdExNeuron(v=-60.0, w=3.0, dt=1.0e308, integrator=integrator)
        before = (n.v, n.w)
        with pytest.raises(ValueError, match="integrator update"):
            n.step(1.0e308)
        assert (n.v, n.w) == before

    def test_rejects_non_finite_spike_adaptation_before_mutation(self):
        n = AdExNeuron(v=-49.0, w=0.0, a=6.25e306, b=1.0e308, tau_w=1.0, dt=1.0)
        before = (n.v, n.w)
        with pytest.raises(ValueError, match="spike adaptation"):
            n.step(0.0)
        assert (n.v, n.w) == before
