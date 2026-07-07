# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Exponential-Euler golden integrator tests

"""Tests for the linearised exponential-Euler (Rush–Larsen) golden integrator.

The defining property is exactness on the gating form ``dx/dt = (x_inf - x)/tau``:
exponential Euler must reproduce the analytic relaxation to machine precision even
at a step where forward Euler drifts, and must reduce to forward Euler in the
zero-Jacobian limit. Faithfulness is enforced at construction — a model whose
dynamics cannot be differentiated with respect to their own variable cannot use
the scheme.
"""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.expression_derivative import ExpressionDifferentiationError
from sc_neurocore.neurons.equation_builder import EquationNeuron


def _exprel(z: float) -> float:
    return 1.0 if z == 0 else float(np.expm1(z) / z)


def test_gating_relaxation_is_exact() -> None:
    """dx/dt = (x_inf - x)/tau integrates to the analytic exponential relaxation."""
    x_inf, tau, x0, dt = 0.8, 5.0, 0.0, 2.0  # dt/tau = 0.4: forward Euler drifts
    neuron = EquationNeuron(
        equations={"x": "(x_inf - x)/tau"},
        parameters={"x_inf": x_inf, "tau": tau},
        state={"x": x0},
        dt=dt,
        method="exp_euler",
    )
    for step in range(1, 21):
        neuron.step()
        analytic = x_inf + (x0 - x_inf) * np.exp(-step * dt / tau)
        assert neuron.state["x"] == pytest.approx(analytic, abs=1e-12)


def test_exponential_euler_beats_forward_euler_at_stiff_step() -> None:
    """At a stiff step exponential Euler is exact where forward Euler is not."""
    # Few steps at a stiff dt: forward Euler's transient error is largest before
    # both schemes saturate at the fixed point. Each neuron keeps its own state
    # dict (the builder does not copy it), so they must not be shared.
    x_inf, tau, dt, steps = 1.0, 4.0, 3.0, 3
    params = {"x_inf": x_inf, "tau": tau}
    exp_neuron = EquationNeuron(
        equations={"x": "(x_inf - x)/tau"},
        parameters=params,
        state={"x": 0.0},
        dt=dt,
        method="exp_euler",
    )
    euler_neuron = EquationNeuron(
        equations={"x": "(x_inf - x)/tau"},
        parameters=params,
        state={"x": 0.0},
        dt=dt,
        method="euler",
    )
    for _ in range(steps):
        exp_neuron.step()
        euler_neuron.step()
    analytic = x_inf + (0.0 - x_inf) * np.exp(-steps * dt / tau)
    assert abs(exp_neuron.state["x"] - analytic) < 1e-12
    assert abs(euler_neuron.state["x"] - analytic) > 1e-2


def test_zero_jacobian_reduces_to_forward_euler() -> None:
    """A constant derivative (A = 0) integrates identically to forward Euler."""
    exp_neuron = EquationNeuron(
        equations={"v": "c"}, parameters={"c": 3.0}, state={"v": 0.0}, dt=0.1, method="exp_euler"
    )
    euler_neuron = EquationNeuron(
        equations={"v": "c"}, parameters={"c": 3.0}, state={"v": 0.0}, dt=0.1, method="euler"
    )
    for _ in range(100):
        exp_neuron.step()
        euler_neuron.step()
    assert exp_neuron.state["v"] == pytest.approx(euler_neuron.state["v"])


def test_coupled_variables_update_simultaneously() -> None:
    """All increments use the pre-step state (forward, not Gauss–Seidel, update)."""
    tau1, tau2, dt = 5.0, 5.0, 1.0
    neuron = EquationNeuron(
        equations={"x1": "(a1 - x1)/tau1", "x2": "(x1 - x2)/tau2"},
        parameters={"a1": 1.0, "tau1": tau1, "tau2": tau2},
        state={"x1": 0.2, "x2": 0.5},
        dt=dt,
        method="exp_euler",
    )
    x1_0, x2_0 = 0.2, 0.5
    inc1 = ((1.0 - x1_0) / tau1) * dt * _exprel((-1.0 / tau1) * dt)
    # x2's increment must use the *pre-step* x1 (0.2), not the updated x1.
    inc2 = ((x1_0 - x2_0) / tau2) * dt * _exprel((-1.0 / tau2) * dt)
    neuron.step()
    assert neuron.state["x1"] == pytest.approx(x1_0 + inc1)
    assert neuron.state["x2"] == pytest.approx(x2_0 + inc2)


def test_threshold_and_reset_fire_under_exp_euler() -> None:
    """A leaky integrator on exp_euler charges to threshold, spikes, and resets."""
    neuron = EquationNeuron(
        equations={"v": "(E_l - v)/tau + I/C"},
        parameters={"E_l": -65.0, "tau": 10.0, "C": 1.0},
        state={"v": -65.0},
        threshold="v > v_th",
        reset={"v": "v_reset"},
        constants={"v_th": -50.0, "v_reset": -65.0},
        dt=0.5,
        method="exp_euler",
    )
    # Moderate drive: the membrane takes several steps to reach threshold, so the
    # spike train is partial (multi-step accumulation), not saturated every step.
    spikes = sum(neuron.step(I=10.0) for _ in range(400))
    assert 0 < spikes < 400  # a non-trivial partial spike train, not saturated
    # After a reset step the membrane sits back near the reset potential.
    assert neuron.state["v"] <= -50.0


def test_unsupported_method_is_rejected() -> None:
    with pytest.raises(ValueError, match="method must be one of"):
        EquationNeuron(equations={"v": "-v"}, state={"v": 1.0}, method="rosenbrock")


def test_non_differentiable_dynamics_rejected_at_construction() -> None:
    """exp_euler needs ∂f/∂x; a kink in x is refused up front, not at run time."""
    with pytest.raises(ExpressionDifferentiationError):
        EquationNeuron(equations={"v": "abs(v)"}, state={"v": 1.0}, method="exp_euler")


def test_euler_and_rk4_models_have_no_jacobian() -> None:
    """The Jacobian is built only for exponential Euler."""
    euler = EquationNeuron(equations={"v": "-v"}, state={"v": 1.0}, method="euler")
    rk4 = EquationNeuron(equations={"v": "-v"}, state={"v": 1.0}, method="rk4")
    assert euler._compiled_jacobian == {}
    assert rk4._compiled_jacobian == {}
