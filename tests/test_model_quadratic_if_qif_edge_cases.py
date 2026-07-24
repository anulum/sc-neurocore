# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestQIFEdgeCases from former test_model_quadratic_if.py

"""Focused suite: TestQIFEdgeCases from former test_model_quadratic_if.py."""

from __future__ import annotations

from tests.model_quadratic_if_support import *  # noqa: F403


class TestQIFEdgeCases:
    def test_quadratic_divergence(self):
        """At I>0, the exact flow follows the quadratic positive feedback."""
        n = QuadraticIFNeuron()
        n.v = 0.5  # positive side
        expected, spiked = _exact_qif_candidate(n, 1.0)
        n.step(1.0)
        assert not spiked
        assert n.v == pytest.approx(expected, abs=1e-12)

    def test_exact_flow_separates_from_raw_euler(self):
        n = QuadraticIFNeuron(v=0.5, dt=0.1)
        exact, spiked = _exact_qif_candidate(n, 1.0)
        euler = _euler_candidate(n, 1.0)
        n.step(1.0)
        assert not spiked
        assert abs(exact - euler) > 1e-3
        assert n.v == pytest.approx(exact, abs=1e-12)

    def test_exact_flow_resets_on_within_step_peak_crossing(self):
        n = QuadraticIFNeuron(v=0.95, dt=0.5)
        spike = n.step(1.0)
        assert spike == 1
        assert n.v == n.v_reset

    def test_custom_peak(self):
        n = QuadraticIFNeuron(v_peak=0.5)
        # Lower peak → fires sooner
        s_low = len(_run(n, current=1.0, steps=10000))
        n2 = QuadraticIFNeuron(v_peak=2.0)
        s_high = len(_run(n2, current=1.0, steps=10000))
        assert s_low > s_high

    @pytest.mark.parametrize("dt", [0.005, 0.01, 0.02])
    def test_dt_stability(self, dt: float):
        n = QuadraticIFNeuron(dt=dt)
        for _ in range(50000):
            n.step(1.0)
        assert np.isfinite(n.v)

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = QuadraticIFNeuron()
            trace = [(n.step(1.5), n.v) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]
