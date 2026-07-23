# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFromEquations from former test_equation_builder.py

"""Focused suite: TestFromEquations from former test_equation_builder.py."""

from __future__ import annotations

from tests.equation_builder_support import *  # noqa: F403

class TestFromEquations:
    def test_brian2_style_lif(self) -> None:
        from sc_neurocore.neurons.equation_builder import from_equations

        n = from_equations(
            "dv/dt = -(v - E_L) / tau_m + I / C",
            threshold="v > -50",
            reset="v = -65",
            params={"E_L": -65.0, "tau_m": 10.0, "C": 1.0},
            init={"v": -65.0},
            dt=1.0,
        )
        spikes = sum(n.step(I=30.0) for _ in range(200))
        assert spikes > 0

    def test_multi_equation(self) -> None:
        from sc_neurocore.neurons.equation_builder import from_equations

        n = from_equations(
            "dv/dt = -(v - v_rest) / tau + I",
            "dw/dt = (v - w) / tau_w",
            params={"v_rest": 0.0, "tau": 10.0, "tau_w": 100.0},
            init={"v": 0.0, "w": 0.0},
            dt=1.0,
        )
        for _ in range(100):
            n.step(I=5.0)
        assert n.state["v"] != 0.0
        assert n.state["w"] != 0.0

    def test_invalid_equation_raises(self) -> None:
        from sc_neurocore.neurons.equation_builder import from_equations

        with pytest.raises(ValueError, match="Cannot parse"):
            from_equations("v = I + 1")

    def test_adex_from_string(self) -> None:
        from sc_neurocore.neurons.equation_builder import from_equations

        n = from_equations(
            "dv/dt = (-(v - E_L) + delta_T * exp((v - v_T) / delta_T) - w + I) / tau_m",
            "dw/dt = (a * (v - E_L) - w) / tau_w",
            threshold="v > 0",
            reset="v = -68; w = w + b",
            params={
                "E_L": -70.6,
                "v_T": -50.4,
                "delta_T": 2.0,
                "tau_m": 9.4,
                "tau_w": 144.0,
                "a": 0.004,
                "b": 0.0805,
            },
            init={"v": -70.6, "w": 0.0},
            dt=0.1,
        )
        for _ in range(500):
            n.step(I=5.0)
        assert n.state["v"] != -70.6, "AdEx dynamics must evolve"

    def test_hodgkin_huxley_from_string(self) -> None:
        from sc_neurocore.neurons.equation_builder import from_equations

        n = from_equations(
            "dv/dt = (-g_L * (v - E_L) - g_Na * m**3 * h * (v - E_Na) - g_K * n**4 * (v - E_K) + I) / C",
            "dm/dt = 0.1 * (v + 40) / (1 - exp(-(v + 40) / 10)) * (1 - m) - 4 * exp(-(v + 65) / 18) * m",
            "dh/dt = 0.07 * exp(-(v + 65) / 20) * (1 - h) - 1 / (1 + exp(-(v + 35) / 10)) * h",
            "dn/dt = 0.01 * (v + 55) / (1 - exp(-(v + 55) / 10)) * (1 - n) - 0.125 * exp(-(v + 65) / 80) * n",
            threshold="v > 0",
            reset="",
            params={
                "C": 1.0,
                "g_L": 0.3,
                "g_Na": 120.0,
                "g_K": 36.0,
                "E_L": -54.4,
                "E_Na": 50.0,
                "E_K": -77.0,
            },
            init={"v": -65.0, "m": 0.05, "h": 0.6, "n": 0.32},
            dt=0.01,
        )
        spikes = sum(n.step(I=10.0) for _ in range(1000))
        assert spikes > 0

    def test_non_numeric_reset_expression(self) -> None:
        from sc_neurocore.neurons.equation_builder import from_equations

        n = from_equations(
            "dv/dt = (-v + I) / tau",
            threshold="v > 1.0",
            reset="v = v_rest",
            params={"tau": 10.0, "v_rest": -1.0},
            init={"v": 0.0},
            dt=0.1,
        )
        spikes = sum(n.step(I=5.0) for _ in range(200))
        assert spikes > 0

    def test_get_state(self) -> None:
        from sc_neurocore.neurons.equation_builder import from_equations

        n = from_equations("dv/dt = I", init={"v": 0.0}, dt=0.1)
        n.step(I=1.0)
        state = n.get_state()
        assert "v" in state
        assert state["v"] != 0.0

    def test_reject_syntax_error(self) -> None:
        from sc_neurocore.neurons.equation_builder import EquationNeuron

        with pytest.raises(ValueError, match="Invalid equation syntax"):
            EquationNeuron(
                equations={"v": "v +* I"},
                state={"v": 0.0},
                dt=1.0,
            )

    def test_reject_unsafe_ast_node(self) -> None:
        from sc_neurocore.neurons.equation_builder import EquationNeuron

        with pytest.raises(ValueError, match="Unsafe AST node"):
            EquationNeuron(
                equations={"v": "[x for x in range(10)]"},
                state={"v": 0.0},
                dt=1.0,
            )

    def test_reject_blocked_name(self) -> None:
        from sc_neurocore.neurons.equation_builder import EquationNeuron

        with pytest.raises(ValueError, match="Blocked function"):
            EquationNeuron(
                equations={"v": "__import__('os')"},
                state={"v": 0.0},
                dt=1.0,
            )

    def test_reject_blocked_attribute(self) -> None:
        from sc_neurocore.neurons.equation_builder import EquationNeuron

        with pytest.raises(ValueError, match="Blocked attribute|Dunder attribute"):
            EquationNeuron(
                equations={"v": "v.__class__"},
                state={"v": 0.0},
                dt=1.0,
            )

    def test_reset_with_empty_parts(self) -> None:
        from sc_neurocore.neurons.equation_builder import from_equations

        n = from_equations(
            "dv/dt = (-v + I) / tau",
            threshold="v > 1.0",
            reset="v = -1.0; ; ",
            params={"tau": 10.0},
            init={"v": 0.0},
            dt=0.1,
        )
        spikes = sum(n.step(I=5.0) for _ in range(200))
        assert spikes > 0

    def test_escape_rate_detection_replays_state_and_rng(self) -> None:
        from sc_neurocore.neurons.equation_builder import EquationNeuron

        neuron = EquationNeuron(
            equations={"v": "I"},
            parameters={"rate": 0.3},
            state={"v": 0.0},
            reset={"v": "0.0"},
            dt=0.25,
            detection="escape_rate",
            rate_expression="rate",
            rng_seed=0x1234,
        )
        first = [neuron.step(I=1.0) for _ in range(1024)]
        first_state = (dict(neuron.state), neuron.escape_rng_state)
        neuron.reset()
        replay = [neuron.step(I=1.0) for _ in range(1024)]
        assert replay == first
        assert (neuron.state, neuron.escape_rng_state) == first_state
        assert neuron.escape_rng_initial_seed == 0x1234

    def test_escape_rate_does_not_consume_global_numpy_rng_for_zero_noise(self) -> None:
        from sc_neurocore.neurons.equation_builder import EquationNeuron

        neuron = EquationNeuron(
            equations={"v": "0.0"},
            parameters={"rate": 0.3},
            state={"v": 0.0},
            dt=1.0,
            detection="escape_rate",
            rate_expression="rate",
            rng_seed=42,
        )
        np.random.seed(123)
        expected = np.random.random()
        np.random.seed(123)
        neuron.step(I=0.0)
        assert np.random.random() == expected

    def test_escape_rate_failure_rolls_back_integrated_state_and_rng(self) -> None:
        from sc_neurocore.neurons.equation_builder import EquationNeuron

        neuron = EquationNeuron(
            equations={"v": "I"},
            parameters={"zero": 0.0},
            state={"v": 2.0},
            dt=1.0,
            detection="escape_rate",
            rate_expression="1.0 / zero",
            rng_seed=42,
        )
        before = (dict(neuron.state), neuron.escape_rng_state)
        with pytest.raises(ZeroDivisionError):
            neuron.step(I=1.0)
        assert (neuron.state, neuron.escape_rng_state) == before

    def test_escape_rate_reset_failure_rolls_back_consumed_rng(self) -> None:
        from sc_neurocore.neurons.equation_builder import EquationNeuron

        neuron = EquationNeuron(
            equations={"v": "I"},
            parameters={"rate": 1.0e308, "zero": 0.0},
            state={"v": 2.0},
            reset={"v": "1.0 / zero"},
            dt=1.0,
            detection="escape_rate",
            rate_expression="rate",
            rng_seed=42,
        )
        before = (dict(neuron.state), neuron.escape_rng_state)
        with pytest.raises(ZeroDivisionError):
            neuron.step(I=1.0)
        assert (neuron.state, neuron.escape_rng_state) == before

    def test_escape_rate_configuration_rejects_ambiguous_thresholds(self) -> None:
        from sc_neurocore.neurons.equation_builder import EquationNeuron

        with pytest.raises(ValueError, match="requires rate_expression"):
            EquationNeuron(
                equations={"v": "0.0"},
                state={"v": 0.0},
                detection="escape_rate",
            )

        with pytest.raises(ValueError, match="cannot combine"):
            EquationNeuron(
                equations={"v": "0.0"},
                state={"v": 0.0},
                threshold="v > 1.0",
                detection="escape_rate",
                rate_expression="0.1",
            )
