# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLegitimateEquationsStillWork from former test_equation_builder_adversarial.py

"""Focused suite: TestLegitimateEquationsStillWork from former test_equation_builder_adversarial.py."""

from __future__ import annotations

from tests.equation_builder_adversarial_support import *  # noqa: F403

class TestLegitimateEquationsStillWork:
    """Ensure the hardening does not break legitimate neuron equations."""

    def test_lif_equation(self) -> None:
        neuron = from_equations(
            "dv/dt = -(v - E_L)/tau_m + I/C",
            threshold="v > -50",
            reset="v = -65",
            params={"E_L": -65.0, "tau_m": 10.0, "C": 1.0},
            init={"v": -65.0},
        )
        for _ in range(100):
            neuron.step(I=10.0)
        assert neuron.state["v"] != -65.0  # Should have integrated

    def test_fitzhugh_nagumo(self) -> None:
        neuron = EquationNeuron(
            equations={
                "v": "v - v**3 / 3 - w + I",
                "w": "0.08 * (v + 0.7 - 0.8 * w)",
            },
            state={"v": -1.0, "w": 0.0},
            dt=0.01,
        )
        for _ in range(1000):
            neuron.step(I=0.5)
        # Should have evolved from initial state
        assert neuron.state["v"] != -1.0

    def test_transcendental_functions(self) -> None:
        neuron = EquationNeuron(
            equations={"v": "-v + exp(-v) + tanh(v) + sin(v)"},
            state={"v": 1.0},
            dt=0.01,
        )
        for _ in range(100):
            neuron.step()
        assert neuron.state["v"] != 1.0

    def test_conditional_expression(self) -> None:
        neuron = EquationNeuron(
            equations={"v": "v + 1 if v < 10 else -v"},
            state={"v": 0.0},
            dt=0.1,
        )
        for _ in range(50):
            neuron.step()
        # Should not crash
        assert isinstance(neuron.state["v"], float)
