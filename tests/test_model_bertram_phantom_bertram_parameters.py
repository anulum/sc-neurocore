# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBertramParameters from former test_model_bertram_phantom.py

"""Focused suite: TestBertramParameters from former test_model_bertram_phantom.py."""

from __future__ import annotations

from tests.model_bertram_phantom_support import *  # noqa: F403

class TestBertramParameters:
    @pytest.mark.parametrize("g_ca", [2.0, 3.6, 5.0])
    def test_g_ca_sweep(self, g_ca: float):
        """Ca conductance affects excitability."""
        n = BertramPhantomBurster(g_ca=g_ca)
        for _ in range(50_000):
            n.step(200.0)
        assert np.isfinite(n.v)

    @pytest.mark.parametrize("g_s1", [2.0, 4.0, 6.0])
    def test_g_s1_controls_slow_inhibition(self, g_s1: float):
        """Stronger I_s1 shifts the driven fixed point downward."""
        n = BertramPhantomBurster(g_s1=g_s1)
        _run(n, current=200.0, steps=50_000)
        expected_v = {
            2.0: -21.724807478400923,
            4.0: -25.102495948370837,
            6.0: -29.29083237241305,
        }[g_s1]
        assert n.v == pytest.approx(expected_v, rel=0.0, abs=1.0e-12)

    def test_g_s2_modulates_ultraslow(self):
        """Stronger I_s2 shifts the driven fixed point downward."""
        states = []
        for g_s2 in [2.0, 4.0, 6.0]:
            n = BertramPhantomBurster(g_s2=g_s2)
            _run(n, current=200.0, steps=50_000)
            states.append(n.v)
        assert states[0] > states[1] > states[2]

    @pytest.mark.parametrize("tau_s1", [10000.0, 20000.0, 50000.0])
    def test_tau_s1_sweep(self, tau_s1: float):
        """Slow timescale affects burst period."""
        n = BertramPhantomBurster(tau_s1=tau_s1)
        for _ in range(50_000):
            n.step(200.0)
        assert np.isfinite(n.v) and np.isfinite(n.s1)

    @pytest.mark.parametrize("dt", [0.1, 0.5, 1.0])
    def test_dt_stability(self, dt: float):
        """RK4 integration remains finite across configured step sizes."""
        n = BertramPhantomBurster(dt=dt)
        for _ in range(50_000):
            n.step(200.0)
        assert np.isfinite(n.v) and np.isfinite(n.s1) and np.isfinite(n.s2)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"v": True}, "v"),
            ({"v": object()}, "v"),
            ({"v": 300.0}, "v outside"),
            ({"s1": -0.1}, "s1"),
            ({"s2": 1.1}, "s2"),
            ({"g_ca": -1.0}, "g_ca"),
            ({"c_m": 0.0}, "c_m"),
            ({"s_s2": 0.0}, "s_s2"),
            ({"tau_s1": 0.0}, "tau_s1"),
            ({"dt": 0.0}, "dt"),
            ({"v_threshold": float("nan")}, "v_threshold"),
        ],
    )
    def test_rejects_invalid_physical_parameters(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            BertramPhantomBurster(**kwargs)

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("v", float("nan"), "v"),
            ("s1", 1.5, "s1"),
            ("s2", -0.5, "s2"),
        ],
    )
    def test_rejects_corrupted_runtime_state_before_mutation(self, field, value, match):
        n = BertramPhantomBurster()
        previous = (n.v, n.s1, n.s2)
        setattr(n, field, value)

        with pytest.raises(ValueError, match=match):
            n.step(200.0)

        if field == "v":
            assert math.isnan(n.v)
            assert n.s1 == previous[1]
            assert n.s2 == previous[2]
        elif field == "s1":
            assert n.v == previous[0]
            assert n.s1 == value
            assert n.s2 == previous[2]
        else:
            assert n.v == previous[0]
            assert n.s1 == previous[1]
            assert n.s2 == value

    def test_rejects_nonfinite_runtime_current_before_mutation(self):
        n = BertramPhantomBurster()
        previous = (n.v, n.s1, n.s2)

        with pytest.raises(ValueError, match="current"):
            n.step(float("nan"))

        assert (n.v, n.s1, n.s2) == previous

    def test_rejects_nonfinite_candidate_before_mutation(self):
        n = BertramPhantomBurster()
        previous = (n.v, n.s1, n.s2)

        with pytest.raises(ValueError, match="candidate"):
            n.step(1.0e308)

        assert (n.v, n.s1, n.s2) == previous

    @pytest.mark.parametrize(
        ("candidate", "match"),
        [
            ((float("nan"), 0.1, 0.1), "candidate"),
            ((-50.0, -0.1, 0.1), "s1"),
            ((-50.0, 0.1, 1.1), "s2"),
        ],
    )
    def test_candidate_validation_rejects_invalid_rk4_state(self, candidate, match):
        """Candidate validation enforces the physical RK4 state envelope."""
        n = BertramPhantomBurster()
        with pytest.raises(ValueError, match=match):
            n._validate_candidate(*candidate)
