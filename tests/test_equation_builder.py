# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for Equation Builder

from __future__ import annotations

import pytest


class TestEquationNeuron:
    def test_lif_from_dict(self) -> None:
        from sc_neurocore.neurons.equation_builder import EquationNeuron

        n = EquationNeuron(
            equations={"v": "-(v - v_rest) / tau + R * I"},
            parameters={"v_rest": -65.0, "tau": 10.0, "R": 1.0},
            state={"v": -65.0},
            threshold="v > -50",
            reset={"v": "v_rest"},
            dt=1.0,
        )
        spikes = sum(n.step(I=30.0) for _ in range(200))
        assert spikes > 0

    def test_fitzhugh_nagumo(self) -> None:
        from sc_neurocore.neurons.equation_builder import EquationNeuron

        n = EquationNeuron(
            equations={
                "v": "v - v**3 / 3 - w + I",
                "w": "epsilon * (v + a - b * w)",
            },
            parameters={"epsilon": 0.08, "a": 0.7, "b": 0.8},
            state={"v": -1.0, "w": -0.5},
            threshold="v > 1.0",
            reset={"v": "-1.0"},
            constants={"v_reset_val": -1.0},
            dt=0.1,
        )
        spikes = sum(n.step(I=0.5) for _ in range(2000))
        assert spikes > 0

    def test_step_fails_closed_on_non_finite_state(self) -> None:
        """A diverging integration raises ``FloatingPointError`` instead of NaN.

        A non-resetting cubic oscillator stepped far past its stability limit blows
        up; the runner must fail closed on the resulting non-finite state (matching
        the hand neuron models' ``_validate_candidate`` contract) rather than
        silently feeding ``inf``/``nan`` into the threshold decision.
        """
        from sc_neurocore.neurons.equation_builder import EquationNeuron

        n = EquationNeuron(
            equations={"v": "v - v * v * v / 3.0 - w + I", "w": "0.08 * (v + 0.7 - 0.8 * w)"},
            state={"v": -1.0, "w": -0.5},
            threshold="v >= 1.0",
            detection="crossing",
            method="rk4",
            dt=5.0,  # far past the stability limit — the cubic diverges
        )
        with pytest.raises(FloatingPointError, match="became non-finite"):
            for _ in range(100):
                n.step(I=0.5)

    def test_step_stays_finite_within_stability_limit(self) -> None:
        """At the production step size the same oscillator never trips the guard."""
        from sc_neurocore.neurons.equation_builder import EquationNeuron

        n = EquationNeuron(
            equations={"v": "v - v * v * v / 3.0 - w + I", "w": "0.08 * (v + 0.7 - 0.8 * w)"},
            state={"v": -1.0, "w": -0.5},
            threshold="v >= 1.0",
            detection="crossing",
            method="rk4",
            dt=0.1,
        )
        for _ in range(3000):
            n.step(I=0.5)  # must not raise

    def test_substeps_defaults_to_one(self) -> None:
        """Without an explicit ``substeps`` the neuron is a plain single-step integrator."""
        from sc_neurocore.neurons.equation_builder import EquationNeuron

        n = EquationNeuron(equations={"v": "I"}, state={"v": 0.0}, dt=1.0)
        assert n.substeps == 1

    def test_substeps_rejects_non_positive_and_bool(self) -> None:
        """``substeps`` must be a positive integer; ``0``, negatives and ``bool`` are rejected."""
        from sc_neurocore.neurons.equation_builder import EquationNeuron

        for bad in (0, -3, True, False, 2.0):
            with pytest.raises(ValueError, match="substeps must be a positive integer"):
                EquationNeuron(equations={"v": "I"}, state={"v": 0.0}, dt=1.0, substeps=bad)  # type: ignore[arg-type]

    def test_macrostep_advances_n_integration_substeps(self) -> None:
        """One macro ``step()`` advances the state by ``substeps`` inner integration steps."""
        from sc_neurocore.neurons.equation_builder import EquationNeuron

        n = EquationNeuron(equations={"v": "I"}, state={"v": 0.0}, dt=1.0, substeps=5)
        n.step(I=2.0)
        # Euler ``v += I*dt`` applied five times: 0 + 5 * (2 * 1) = 10.
        assert n.state["v"] == 10.0

    def test_macrostep_crossing_fires_once_per_macro_boundary(self) -> None:
        """A non-resetting ramp counts one spike per macro step, not per sub-step crossing.

        The membrane crosses the threshold in the middle of the first macro window (at the
        third of five sub-steps), but the rising-edge decision is taken only on the macro
        boundary, so the macro step registers exactly one spike and the next macro step —
        already above threshold — registers none.
        """
        from sc_neurocore.neurons.equation_builder import EquationNeuron

        n = EquationNeuron(
            equations={"v": "I"},
            state={"v": 0.0},
            threshold="v >= 2.5",
            detection="crossing",
            dt=1.0,
            substeps=5,
        )
        spikes = [n.step(I=1.0) for _ in range(2)]
        assert spikes == [1, 0]
        assert n.state["v"] == 10.0

    def test_izhikevich(self) -> None:
        from sc_neurocore.neurons.equation_builder import EquationNeuron

        n = EquationNeuron(
            equations={
                "v": "0.04 * v**2 + 5 * v + 140 - u + I",
                "u": "a * (b * v - u)",
            },
            parameters={"a": 0.02, "b": 0.2, "c": -65.0, "d": 8.0},
            state={"v": -65.0, "u": -14.0},
            threshold="v > 30",
            reset={"v": "c", "u": "u + d"},
            dt=1.0,
        )
        spikes = sum(n.step(I=10.0) for _ in range(200))
        assert spikes > 0

    def test_reset_clears_state(self) -> None:
        from sc_neurocore.neurons.equation_builder import EquationNeuron

        n = EquationNeuron(
            equations={"v": "I"},
            state={"v": 0.0},
            dt=1.0,
        )
        n.step(I=5.0)
        assert n.state["v"] != 0.0
        n.reset()
        assert n.state["v"] == 0.0

    def test_rk4_method(self) -> None:
        from sc_neurocore.neurons.equation_builder import EquationNeuron

        n = EquationNeuron(
            equations={"v": "-(v - v_rest) / tau + I"},
            parameters={"v_rest": 0.0, "tau": 10.0},
            state={"v": 0.0},
            dt=0.1,
            method="rk4",
        )
        for _ in range(100):
            n.step(I=1.0)
        assert n.state["v"] > 1.0, "should converge toward steady state"

    def test_gauss_seidel_method(self) -> None:
        """Gauss-Seidel advances state variables sequentially in declaration order.

        A later-declared variable reads the already-updated value of an earlier one within
        the same step, unlike the simultaneous ``euler`` mode. With ``da/dt = 1`` and
        ``db/dt = a`` at ``dt = 1`` from ``a = b = 0``: the sequential mode commits ``a = 1``
        first, so ``b`` integrates the new ``a`` to ``1``; simultaneous Euler reads the
        pre-step ``a = 0`` and leaves ``b = 0``. The contrast pins the ordering, not just the
        branch being executed.
        """
        from sc_neurocore.neurons.equation_builder import EquationNeuron

        sequential = EquationNeuron(
            equations={"a": "1", "b": "a"},
            state={"a": 0.0, "b": 0.0},
            dt=1.0,
            method="gauss_seidel",
        )
        sequential.step()
        assert sequential.state["a"] == 1.0
        assert sequential.state["b"] == 1.0  # reads the committed a = 1, not the pre-step a = 0

        simultaneous = EquationNeuron(
            equations={"a": "1", "b": "a"},
            state={"a": 0.0, "b": 0.0},
            dt=1.0,
            method="euler",
        )
        simultaneous.step()
        assert simultaneous.state["a"] == 1.0
        assert simultaneous.state["b"] == 0.0  # reads the pre-step a = 0

    def test_gauss_seidel_fails_closed_on_non_finite_state(self) -> None:
        """A diverging sequential integration raises rather than propagating a non-finite state."""
        from sc_neurocore.neurons.equation_builder import EquationNeuron

        n = EquationNeuron(
            equations={"v": "v * v + 1e6"},
            state={"v": 1e200},
            dt=1.0,
            method="gauss_seidel",
        )
        with pytest.raises(FloatingPointError, match="became non-finite"):
            n.step()

    def test_no_threshold(self) -> None:
        from sc_neurocore.neurons.equation_builder import EquationNeuron

        n = EquationNeuron(
            equations={"v": "I"},
            state={"v": 0.0},
            dt=1.0,
        )
        spike = n.step(I=100.0)
        assert spike == 0

    def test_math_functions(self) -> None:
        from sc_neurocore.neurons.equation_builder import EquationNeuron

        n = EquationNeuron(
            equations={"v": "exp(-v) + tanh(I) + sin(v)"},
            state={"v": 0.0},
            dt=0.01,
        )
        n.step(I=1.0)
        assert n.state["v"] != 0.0

    def test_repr(self) -> None:
        from sc_neurocore.neurons.equation_builder import EquationNeuron

        n = EquationNeuron(equations={"v": "I"}, state={"v": 0.0})
        assert "EquationNeuron" in repr(n)
        assert "dv/dt" in repr(n)


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
