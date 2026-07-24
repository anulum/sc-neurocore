# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPerfectIntegratorParameterSweep from former test_model_perfect_integrator.py

"""Focused suite: TestPerfectIntegratorParameterSweep from former test_model_perfect_integrator.py."""

from __future__ import annotations

from tests.model_perfect_integrator_support import *  # noqa: F403


class TestPerfectIntegratorParameterSweep:
    """Systematic parameter sweeps verifying scaling laws."""

    @pytest.mark.parametrize("c_m", [0.5, 1.0, 2.0, 5.0])
    def test_rate_inversely_proportional_to_capacitance(self, c_m: float):
        """f ∝ 1/C — verify across parameter range."""
        n = PerfectIntegratorNeuron(c_m=c_m)
        I = 10.0
        steps = 5000
        spikes = sum(n.step(I) for _ in range(steps))
        isi = _analytical_isi_steps(I, c_m, n.v_threshold, n.v_reset, n.dt)
        expected = min(steps, steps / isi)
        assert abs(spikes - expected) <= 1

    @pytest.mark.parametrize("threshold", [0.5, 1.0, 2.0, 5.0])
    def test_rate_inversely_proportional_to_threshold(self, threshold: float):
        """f ∝ 1/θ."""
        n = PerfectIntegratorNeuron(v_threshold=threshold)
        I = 10.0
        steps = 5000
        spikes = sum(n.step(I) for _ in range(steps))
        isi = _analytical_isi_steps(I, n.c_m, threshold, n.v_reset, n.dt)
        expected = min(steps, steps / isi)
        assert abs(spikes - expected) <= 1
