# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBRFEquations from former test_model_balanced_resonate_and_fire.py

"""Focused suite: TestBRFEquations from former test_model_balanced_resonate_and_fire.py."""

from __future__ import annotations

from tests.model_balanced_resonate_and_fire_support import *  # noqa: F403

class TestBRFEquations:
    def test_construction_defaults_match_paper_algorithm(self) -> None:
        neuron = BalancedResonateAndFireNeuron()
        assert neuron.x == 0.0
        assert neuron.y == 0.0
        assert neuron.q == 0.0
        assert neuron.omega == 10.0
        assert neuron.b_offset == 1.0
        assert neuron.threshold == 1.0
        assert neuron.gamma == 0.9
        assert neuron.dt == 0.01

    def test_divergence_boundary_formula(self) -> None:
        omega = 10.0
        dt = 0.01
        expected = (-1.0 + math.sqrt(1.0 - (dt * omega) ** 2)) / dt
        assert sustain_oscillation_boundary(omega, dt) == pytest.approx(expected)
        assert BalancedResonateAndFireNeuron(omega=omega, dt=dt).p_omega == pytest.approx(expected)

    def test_one_step_matches_algorithm_1(self) -> None:
        neuron = BalancedResonateAndFireNeuron(
            x=0.2,
            y=-0.1,
            q=0.3,
            omega=12.0,
            b_offset=0.75,
            threshold=1.0,
            gamma=0.9,
            dt=0.01,
        )
        p_omega = sustain_oscillation_boundary(12.0, 0.01)
        b_t = p_omega - 0.75 - 0.3
        expected_x = 0.2 + 0.01 * (b_t * 0.2 - 12.0 * -0.1 + 2.0)
        expected_y = -0.1 + 0.01 * (12.0 * 0.2 + b_t * -0.1)
        expected_spike = int(expected_x >= 1.3)

        spike = neuron.step(2.0)

        assert spike == expected_spike
        assert neuron.x == pytest.approx(expected_x)
        assert neuron.y == pytest.approx(expected_y)
        assert neuron.q == pytest.approx(0.9 * 0.3 + expected_spike)

    def test_threshold_uses_real_part_not_radius(self) -> None:
        neuron = BalancedResonateAndFireNeuron(x=0.0, y=5.0)
        assert neuron.step(0.0) == 0
        assert neuron.q == 0.0
