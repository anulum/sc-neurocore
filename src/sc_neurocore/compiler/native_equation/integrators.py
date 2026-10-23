# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native fixed-point integration stages

"""Represent every maintained EquationNeuron integrator as explicit scalar dependencies."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from ...hdl_gen._ident import sanitize_ident
from ...neurons.equation_builder import EquationNeuron
from .arithmetic import EquationArithmetic
from .expressions import EquationExpressions
from .scheduling import EquationSchedule


def integrate(
    neuron: EquationNeuron,
    arithmetic: EquationArithmetic,
    states: Mapping[str, int],
    bindings: Mapping[str, int],
    commit: Callable[[int, int], int],
    externs: Mapping[str, int] | None = None,
    schedule: EquationSchedule | None = None,
) -> dict[str, int]:
    """Append one complete integration substep without writing registers.

    Parameters
    ----------
    neuron : EquationNeuron
        Existing prepared model. The method, declaration order, equations,
        symbolic diagonal Jacobian and dt all come from this subject.
    arithmetic : EquationArithmetic
        Destination graph and quantised word/rounding policy.
    states : mapping of str to int
        Current signed Q-word identifiers in equation declaration order.
    bindings : mapping of str to int
        Signed Q-word parameters, constants and external inputs.
    commit : callable
        Apply the configured state overflow policy to a value of a supplied
        physical width; return a Q-word identifier. Gauss-Seidel reads these
        already committed earlier candidates within the same substep.
    externs : mapping of str to int, optional
        Signed i64 RNG step, seed and stream bindings, unchanged within a step.
    schedule : EquationSchedule or None, optional
        Shared expression scheduling; Euler/Gauss-Seidel dt products also honour
        their maintained named pipeline points. RK4 scaling stays combinational.

    Returns
    -------
    dict of str to int
        All post-integration candidate words. Thresholds, resets, history and
        register stores belong to the surrounding timestep transaction.

    Raises
    ------
    ValueError
        If the method is outside the existing five-method vocabulary.

    Notes
    -----
    RK4 stage advances wrap at the Q word width before the next derivative;
    only the final accumulation calls commit. The dt/2 and dt/6 constants
    quantise independently as in the maintained Verilog integrator. Exponential
    Euler uses the model's actual symbolic Jacobian and shared exprel lowering.
    """
    a, b, q = arithmetic, arithmetic.builder, arithmetic.q
    dw, wide = q.data_width, 2 * q.data_width

    def evaluate(expression: str, state: Mapping[str, int], width: int) -> int:
        return EquationExpressions(a, {**bindings, **state}, externs, schedule).lower(
            expression, width
        )

    def scaled(value: int, scale: float, name: str | None = None) -> int:
        product = b.word_binary(
            "mul", b.word_resize(value, f"i{wide}"), a.literal(scale, wide), f"i{wide}"
        )
        if schedule is not None and name is not None:
            product = schedule.named_product(product, wide, name)
        return a.round_product(product, wide)

    def accumulate(name: str, increment: int) -> int:
        ty = f"i{dw + 1}"
        raw = b.word_binary(
            "add", b.word_resize(states[name], ty), b.word_resize(increment, ty), ty
        )
        return commit(raw, dw + 1)

    if neuron.method == "map":
        return {
            name: commit(evaluate(expression, states, wide), wide)
            for name, expression in neuron.equations.items()
        }
    if neuron.method in {"euler", "gauss_seidel"}:
        candidates: dict[str, int] = {}
        environment = dict(states)
        for name, expression in neuron.equations.items():
            safe = sanitize_ident(name, context="state variable")
            increment = scaled(
                evaluate(expression, environment, wide), neuron.dt, f"_dt_mul_{safe}"
            )
            candidates[name] = accumulate(name, increment)
            if neuron.method == "gauss_seidel":
                environment[name] = candidates[name]
        return candidates
    if neuron.method == "exp_euler":
        candidates = {}
        dt = repr(neuron.dt)
        for name, expression in neuron.equations.items():
            jacobian = neuron.jacobian_expressions[name]
            increment_expression = f"(({expression}) * {dt}) * exprel((({jacobian})) * {dt})"
            candidates[name] = accumulate(name, evaluate(increment_expression, states, dw))
        return candidates
    if neuron.method != "rk4":
        raise ValueError(f"unsupported native equation integrator: {neuron.method!r}")

    def derivatives(state: Mapping[str, int]) -> dict[str, int]:
        return {
            name: evaluate(expression, state, dw) for name, expression in neuron.equations.items()
        }

    def advance(k: Mapping[str, int], scale: float) -> dict[str, int]:
        return {
            name: b.word_binary("add", states[name], scaled(k[name], scale), f"i{dw}")
            for name in neuron.equations
        }

    k1 = derivatives(states)
    k2 = derivatives(advance(k1, neuron.dt / 2.0))
    k3 = derivatives(advance(k2, neuron.dt / 2.0))
    k4 = derivatives(advance(k3, neuron.dt))
    candidates = {}
    for name in neuron.equations:
        terms = [k1[name], k2[name], k2[name], k3[name], k3[name], k4[name]]
        weight_type = f"i{dw + 3}"
        total = b.word_resize(terms[0], weight_type)
        for term in terms[1:]:
            total = b.word_binary("add", total, b.word_resize(term, weight_type), weight_type)
        product_width = 2 * dw + 3
        ty = f"i{product_width}"
        product = b.word_binary(
            "mul", b.word_resize(total, ty), a.literal(neuron.dt / 6.0, product_width), ty
        )
        candidates[name] = accumulate(name, a.round_product(product, product_width))
    return candidates
