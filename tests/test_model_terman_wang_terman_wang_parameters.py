# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTermanWangParameters from former test_model_terman_wang.py

"""Focused suite: TestTermanWangParameters from former test_model_terman_wang.py."""

from __future__ import annotations

from tests.model_terman_wang_support import *  # noqa: F403


class TestTermanWangParameters:
    @pytest.mark.parametrize("field", ["v", "w", "alpha", "rho", "v_peak"])
    def test_rejects_non_numeric_state_offsets_and_threshold(self, field: str):
        with pytest.raises(TypeError, match=field):
            TermanWangOscillator(**{field: object()})

    @pytest.mark.parametrize("field", ["beta", "epsilon", "dt"])
    def test_rejects_non_numeric_positive_parameters(self, field: str):
        with pytest.raises(TypeError, match=field):
            TermanWangOscillator(**{field: object()})

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("v", np.nan),
            ("w", np.inf),
            ("alpha", np.nan),
            ("beta", 0.0),
            ("epsilon", 0.0),
            ("dt", 0.0),
            ("v_peak", np.inf),
        ],
    )
    def test_rejects_invalid_numerical_configuration(self, field: str, value: float):
        with pytest.raises(ValueError):
            TermanWangOscillator(**{field: value})

    def test_rejects_non_finite_current_before_state_mutation(self):
        n = TermanWangOscillator()
        before = (n.v, n.w)
        with pytest.raises(FloatingPointError, match="current"):
            n.step(np.nan)
        assert (n.v, n.w) == before

    def test_rejects_non_numeric_current_before_state_mutation(self):
        n = TermanWangOscillator()
        before = (n.v, n.w)
        with pytest.raises(TypeError, match="current"):
            n.step(object())
        assert (n.v, n.w) == before

    def test_rejects_corrupted_runtime_scale_before_state_mutation(self):
        n = TermanWangOscillator()
        n.beta = 0.0
        before = (n.v, n.w)
        with pytest.raises(ValueError, match="beta"):
            n.step(1.0)
        assert (n.v, n.w) == before

    def test_rejects_corrupted_runtime_state_before_mutation(self):
        n = TermanWangOscillator()
        n.v = np.inf
        before = (n.v, n.w)
        with pytest.raises(FloatingPointError, match="v must be finite"):
            n.step(1.0)
        assert (n.v, n.w) == before

    def test_rejects_cubic_overflow_before_state_mutation(self):
        n = TermanWangOscillator(v=1.0e308, w=-0.5)
        before = (n.v, n.w)
        with pytest.raises(FloatingPointError, match="derivative"):
            n.step(1.0)
        assert (n.v, n.w) == before

    def test_derivative_rejects_nonfinite_runtime_inputs(self):
        n = TermanWangOscillator()
        with pytest.raises(FloatingPointError, match="state and current must be finite"):
            n._derivatives(np.nan, n.w, 1.0)

    def test_derivative_rejects_nonfinite_output(self):
        n = TermanWangOscillator()
        n.alpha = np.inf
        with pytest.raises(FloatingPointError, match="derivative"):
            n._derivatives(n.v, n.w, 1.0)

    def test_rejects_nonfinite_candidate_directly(self):
        with pytest.raises(FloatingPointError, match="candidate"):
            TermanWangOscillator._validate_candidate(np.nan, -0.5)

    def test_epsilon_controls_timescale(self):
        n_fast = TermanWangOscillator(epsilon=0.1)
        n_slow = TermanWangOscillator(epsilon=0.005)
        s_fast = len(_run(n_fast, current=1.0, steps=100000))
        s_slow = len(_run(n_slow, current=1.0, steps=100000))
        assert s_fast != s_slow

    @pytest.mark.parametrize("dt", [0.02, 0.05, 0.1])
    def test_dt_stability(self, dt: float):
        n = TermanWangOscillator(dt=dt)
        for _ in range(50000):
            n.step(1.0)
        assert np.isfinite(n.v)

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = TermanWangOscillator()
            trace = [(n.step(1.0), n.v, n.w) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]
