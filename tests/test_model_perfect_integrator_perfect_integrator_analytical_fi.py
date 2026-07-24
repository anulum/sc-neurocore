# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPerfectIntegratorAnalyticalFI from former test_model_perfect_integrator.py

"""Focused suite: TestPerfectIntegratorAnalyticalFI from former test_model_perfect_integrator.py."""

from __future__ import annotations

from tests.model_perfect_integrator_support import *  # noqa: F403


class TestPerfectIntegratorAnalyticalFI:
    """f–I curve: firing rate = I·dt / (C·(θ - V_reset)) for constant input.

    This is the exact analytical result for a perfect integrator.
    We verify the simulation matches to within ±1 spike (quantisation).
    """

    @pytest.mark.parametrize("current", [2.0, 5.0, 10.0, 20.0, 50.0])
    def test_fi_curve_analytical(self, current: float):
        n = PerfectIntegratorNeuron()
        steps = 10000
        spikes = sum(n.step(current) for _ in range(steps))
        isi_analytical = _analytical_isi_steps(
            current,
            n.c_m,
            n.v_threshold,
            n.v_reset,
            n.dt,
        )
        # Max 1 spike per step (discrete time clamp)
        expected_spikes = min(steps, steps / isi_analytical)
        # Allow ±1 spike for boundary quantisation
        assert abs(spikes - expected_spikes) <= 1, (
            f"I={current}: got {spikes}, expected {expected_spikes:.1f}"
        )

    def test_fi_linearity(self):
        """f(2I) / f(I) ≈ 2 — perfect integrator has exactly linear f-I."""
        steps = 5000
        n1 = PerfectIntegratorNeuron()
        n2 = PerfectIntegratorNeuron()
        s1 = sum(n1.step(3.0) for _ in range(steps))
        s2 = sum(n2.step(6.0) for _ in range(steps))
        ratio = s2 / s1 if s1 > 0 else float("inf")
        assert 1.95 <= ratio <= 2.05, f"ratio {ratio} deviates from 2.0"

    def test_fi_threshold_dependence(self):
        """Doubling threshold halves the rate (same current)."""
        steps = 5000
        I = 5.0
        n1 = PerfectIntegratorNeuron(v_threshold=1.0)
        n2 = PerfectIntegratorNeuron(v_threshold=2.0)
        s1 = sum(n1.step(I) for _ in range(steps))
        s2 = sum(n2.step(I) for _ in range(steps))
        ratio = s1 / s2 if s2 > 0 else float("inf")
        assert 1.95 <= ratio <= 2.05

    def test_fi_capacitance_dependence(self):
        """Doubling C_m halves the rate."""
        steps = 5000
        I = 5.0
        n1 = PerfectIntegratorNeuron(c_m=1.0)
        n2 = PerfectIntegratorNeuron(c_m=2.0)
        s1 = sum(n1.step(I) for _ in range(steps))
        s2 = sum(n2.step(I) for _ in range(steps))
        ratio = s1 / s2 if s2 > 0 else float("inf")
        assert 1.95 <= ratio <= 2.05
