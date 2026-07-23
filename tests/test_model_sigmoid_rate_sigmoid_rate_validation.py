# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSigmoidRateValidation from former test_model_sigmoid_rate.py

"""Focused suite: TestSigmoidRateValidation from former test_model_sigmoid_rate.py."""

from __future__ import annotations

from tests.model_sigmoid_rate_support import *  # noqa: F403

class TestSigmoidRateValidation:
    @pytest.mark.parametrize("field", ["r", "beta", "theta"])
    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_state_and_transfer_parameters(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            SigmoidRateNeuron(**{field: value})

    @pytest.mark.parametrize("r", [-1.0e-12, 1.0 + 1.0e-12])
    def test_rejects_initial_rate_outside_unit_interval(self, r: float):
        with pytest.raises(ValueError, match="r must be in \\[0, 1\\]"):
            SigmoidRateNeuron(r=r)

    @pytest.mark.parametrize("field", ["tau", "dt"])
    @pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
    def test_rejects_non_positive_or_non_finite_time_parameters(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            SigmoidRateNeuron(**{field: value})

    def test_accepts_large_timestep_exact_relaxation(self):
        n = SigmoidRateNeuron(tau=0.1, dt=0.2)
        assert 0.0 <= n.step(1.0) <= 1.0

    @pytest.mark.parametrize("current", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_current_before_rate_mutation(self, current: float):
        n = SigmoidRateNeuron(r=0.25)
        before = n.r
        with pytest.raises(ValueError, match="current"):
            n.step(current)
        assert n.r == before

    @pytest.mark.parametrize("field", ["r", "tau", "dt"])
    def test_rejects_corrupted_runtime_state_before_rate_mutation(self, field: str):
        n = SigmoidRateNeuron(r=0.25)
        before = n.r
        setattr(n, field, np.nan)
        with pytest.raises(ValueError, match="runtime"):
            n.step(1.0)
        if field != "r":
            assert n.r == before

    def test_rejects_runtime_rate_outside_unit_interval_before_mutation(self):
        n = SigmoidRateNeuron(r=0.25)
        n.r = 1.5
        with pytest.raises(ValueError, match="runtime rate state must be in \\[0, 1\\]"):
            n.step(1.0)
        assert n.r == 1.5

    @pytest.mark.parametrize("field", ["tau", "dt"])
    def test_rejects_non_positive_runtime_time_parameter_before_mutation(self, field: str):
        n = SigmoidRateNeuron(r=0.25)
        before = n.r
        setattr(n, field, 0.0)
        with pytest.raises(ValueError, match="runtime time constants"):
            n.step(1.0)
        assert n.r == before

    def test_stable_sigmoid_rejects_nonsaturating_nan_argument(self):
        with pytest.raises(ValueError, match="sigmoid argument"):
            SigmoidRateNeuron._stable_sigmoid(np.inf, 1.0, 1.0)

    def test_large_runtime_timestep_preserves_rate_interval(self):
        n = SigmoidRateNeuron(r=1.0, tau=1.0e-308, dt=1.0e308)
        before = n.r
        assert n.step(-1.0e308) == pytest.approx(0.0, abs=1e-300)
        assert 0.0 <= n.r <= before

    def test_extreme_finite_drive_saturates_without_overflow_warning(self):
        n = SigmoidRateNeuron(beta=1.0e308, theta=0.0)
        with np.errstate(over="raise", invalid="raise"):
            high = n.step(1.0e308)
            n.reset()
            low = n.step(-1.0e308)
        assert 0.0 < high <= 1.0
        assert 0.0 <= low < high

    @pytest.mark.parametrize("n_steps", [-1, 1.5, True])
    def test_rejects_invalid_batch_length_without_mutation(self, n_steps: object):
        n = SigmoidRateNeuron(r=0.25)
        with pytest.raises(ValueError, match="n_steps"):
            n.simulate(cast(int, n_steps), 3.0, backend="python")
        assert n.r == 0.25

    def test_rejects_unknown_backend_without_mutation(self):
        n = SigmoidRateNeuron(r=0.25)
        with pytest.raises(ValueError, match="backend must be"):
            n.simulate(1, 3.0, backend="cuda")
        assert n.r == 0.25
