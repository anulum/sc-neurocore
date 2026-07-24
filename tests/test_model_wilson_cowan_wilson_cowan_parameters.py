# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWilsonCowanParameters from former test_model_wilson_cowan.py

"""Focused suite: TestWilsonCowanParameters from former test_model_wilson_cowan.py."""

from __future__ import annotations

from tests.model_wilson_cowan_support import *  # noqa: F403


class TestWilsonCowanParameters:
    def test_accepts_normalised_saturation_boundary(self):
        n = WilsonCowanUnit(e=1.0, i=1.0)

        n.step(2.0)

        assert math.isfinite(n.e) and math.isfinite(n.i)
        assert n.e <= 1.0 and n.i <= 1.0

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("e", np.nan),
            ("i", np.inf),
            ("e", -0.1),
            ("i", 1.1),
            ("w_ee", -1.0),
            ("w_ei", -1.0),
            ("w_ie", -1.0),
            ("w_ii", -1.0),
            ("tau_e", 0.0),
            ("tau_i", 0.0),
            ("a", 0.0),
            ("theta", np.inf),
            ("dt", 0.0),
        ],
    )
    def test_rejects_invalid_numerical_configuration(self, field: str, value: float):
        with pytest.raises((ValueError, FloatingPointError)):
            WilsonCowanUnit(**{field: value})

    def test_rejects_non_finite_input_before_state_mutation(self):
        n = WilsonCowanUnit()
        before = (n.e, n.i)
        with pytest.raises(ValueError, match="external input"):
            n.step(np.nan)
        assert (n.e, n.i) == before

    def test_rejects_corrupted_runtime_state_before_mutation(self):
        n = WilsonCowanUnit()
        n.e = 1.5
        before = (n.e, n.i)
        with pytest.raises(FloatingPointError, match="e rate"):
            n.step(5.0)
        assert (n.e, n.i) == before

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("w_ee", -1.0),
            ("w_ei", -1.0),
            ("w_ie", -1.0),
            ("w_ii", -1.0),
            ("tau_e", 0.0),
            ("tau_i", 0.0),
            ("a", 0.0),
            ("theta", math.inf),
            ("dt", 0.0),
        ],
    )
    def test_rejects_runtime_parameter_corruption_before_state_mutation(
        self, field: str, value: float
    ):
        n = WilsonCowanUnit()
        setattr(n, field, value)
        before = (n.e, n.i)

        with pytest.raises((ValueError, FloatingPointError)):
            n.step(5.0)

        assert (n.e, n.i) == before

    def test_sigmoid_saturates_for_extreme_finite_drive(self):
        n = WilsonCowanUnit()
        baseline = 1.0 / (1.0 + math.exp(n.a * n.theta))
        assert abs(n._sigmoid(1.0e308) - (1.0 - baseline)) < 1e-12
        assert abs(n._sigmoid(-1.0e308) + baseline) < 1e-12

    def test_sigmoid_rejects_non_finite_drive(self):
        n = WilsonCowanUnit()

        with pytest.raises(ValueError, match="sigmoid input"):
            n._sigmoid(math.nan)

    def test_rejects_non_finite_derivative_before_state_mutation(self):
        n = WilsonCowanUnit(tau_e=1.0e-320)
        before = (n.e, n.i)

        with pytest.raises(FloatingPointError, match="derivative"):
            n.step(0.0)

        assert (n.e, n.i) == before

    @pytest.mark.parametrize("dt", [0.05, 0.1, 0.2])
    def test_dt_stability(self, dt: float):
        n = WilsonCowanUnit(dt=dt)
        for _ in range(10000):
            n.step(5.0)
        assert np.isfinite(n.e)

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = WilsonCowanUnit()
            trace = [(n.step(5.0), n.e, n.i) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]
