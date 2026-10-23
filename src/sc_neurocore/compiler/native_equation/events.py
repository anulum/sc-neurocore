# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native equation threshold and reset transactions

"""Lower pre-reset event detection, simultaneous resets and graded spike values."""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from ...neurons.equation_builder import EquationNeuron
from .arithmetic import EquationArithmetic
from .expressions import EquationExpressions


@dataclass(frozen=True)
class EquationEvent:
    """Hold explicit next-state and event SSA values for one timestep transaction.

    Attributes
    ----------
    states : dict of str to int
        Signed Q-word candidates after simultaneous conditional reset selection.
    spike : int
        Bool SSA identifier, false outside the logical timestep boundary.
    value : int or None
        Signed Q-word graded value, zero unless a spike fires; None for binary spikes.
    """

    states: dict[str, int]
    spike: int
    value: int | None


def _lfsr_trial(
    neuron: EquationNeuron, expression: str, e: EquationExpressions, enabled: int
) -> int:
    """Advance the maintained 16-bit LFSR eight times per enabled logical trial."""
    a, b, q = e.a, e.a.builder, e.a.q
    initial = neuron.stochastic_rng_initial_seed
    if initial is None:
        raise ValueError("stochastic equation threshold has no initial RNG seed")
    current = b.state(a.constant(initial or 0xACE1, 16), "i16")
    sample = current
    for _ in range(8):
        feedback = sample
        for bit in (2, 3, 5):
            shifted = b.word_binary("lshr", sample, a.constant(bit, 16), "i16")
            feedback = b.word_binary("xor", feedback, shifted, "i16")
        feedback = b.word_binary("and", feedback, a.constant(1, 16), "i16")
        feedback = b.word_binary("shl", feedback, a.constant(15, 16), "i16")
        sample = b.word_binary(
            "or", b.word_binary("lshr", sample, a.constant(1, 16), "i16"), feedback, "i16"
        )
    b.store(current, sample, enabled)
    dw = q.data_width
    probability = e.lower(expression)
    probability_wide = b.word_resize(probability, f"i{dw + 1}")
    one = a.constant(1 << q.fraction, dw + 1)
    at_one = b.word_compare("ge", probability_wide, one)
    positive = b.word_compare("gt", probability, a.constant(0, dw))
    clamped = b.word_select(at_one, b.word_resize(one, f"i{dw}"), probability, f"i{dw}")
    clamped = b.word_select(positive, clamped, a.constant(0, dw), f"i{dw}")
    width = dw + 16
    unsigned = b.word_bitcast(clamped, f"u{dw}")
    magnitude = b.word_resize(unsigned, f"i{width}")
    product = b.word_binary("mul", magnitude, a.constant(65535, width), f"i{width}")
    floor = b.word_binary("lshr", product, a.constant(q.fraction, width), f"i{width}")
    floor = b.word_resize(b.word_resize(floor, "u16"), "i18")
    interior = b.word_binary("add", floor, a.constant(1, 18), "i18")
    threshold = b.word_select(at_one, a.constant(65536, 18), interior, "i18")
    threshold = b.word_select(positive, threshold, a.constant(0, 18), "i18")
    sample = b.word_resize(b.word_bitcast(sample, "u16"), "i18")
    return b.word_compare("lt", sample, threshold)


def lower_events(
    neuron: EquationNeuron,
    arithmetic: EquationArithmetic,
    candidates: Mapping[str, int],
    previous: Mapping[str, int],
    bindings: Mapping[str, int],
    commit: Callable[[int, int], int],
    enabled: int,
    externs: Mapping[str, int] | None = None,
) -> EquationEvent:
    """Append one model's threshold/history/reset logic without writing its state words.

    Parameters
    ----------
    neuron : EquationNeuron
        Prepared source model, including intrinsic-lowered Philox thresholds.
    arithmetic : EquationArithmetic
        Destination graph and fixed-point policy.
    candidates : mapping of str to int
        Integrated signed Q words, before any reset executes.
    previous : mapping of str to int
        State at the preceding macro boundary, for the model's previous aliases.
    bindings : mapping of str to int
        Signed Q-word constants, parameters and input.
    commit : callable
        State overflow policy: raw SSA identifier and physical width to Q word.
    enabled : int
        Bool SSA identifier for logical timestep completion. Event-history and
        threshold RNG registers advance only on this boundary.
    externs : mapping of str to int, optional
        Counter-based RNG bindings, shared with all derivatives in this timestep.

    Returns
    -------
    EquationEvent
        Explicit selected state, Bool spike and optional graded value identifiers.

    Raises
    ------
    ValueError
        If stochastic model metadata is inconsistent with its detection mechanism.

    Notes
    -----
    Every reset and the graded expression reads the same integrated candidate
    and macro-previous environment. Reset dictionary order never creates a
    sequential update. Crossing history stores the pre-reset active condition.
    """
    a, b, q = arithmetic, arithmetic.builder, arithmetic.q
    environment = {
        **bindings,
        **candidates,
        **{f"{name}_prev": word for name, word in previous.items()},
    }
    e = EquationExpressions(a, environment, externs)
    false = b.constant_i64(0, "bool")
    active = false
    if getattr(neuron, "_stochastic_threshold_enabled", False):
        if getattr(neuron, "_poisson_enabled", False):
            expression = neuron.probability_expression
        else:
            expression = (
                None
                if neuron.rate_expression is None
                else f"1.0 - exp(-(({neuron.rate_expression}) * {neuron.dt!r}))"
            )
        if not expression:
            raise ValueError(
                "stochastic equation threshold requires a probability or escape-rate expression"
            )
        active = _lfsr_trial(neuron, expression, e, enabled)
    elif neuron.threshold_expr:
        active = e.condition(ast.parse(neuron.threshold_expr, mode="eval").body)
        if getattr(neuron, "_edge_detection", False):
            initial = b.constant_i64(int(neuron.initial_threshold_active()), "bool")
            history = b.state(initial, "bool")
            b.store(history, active, enabled)
            active = b.word_select(history, false, active, "bool")
    fired = b.word_select(enabled, active, false, "bool")
    states = dict(candidates)
    for name, expression in neuron.reset_rules.items():
        reset = commit(e.lower(expression, 2 * q.data_width), 2 * q.data_width)
        states[name] = b.word_select(fired, reset, candidates[name], f"i{q.data_width}")
    value = None
    if neuron.spike_value_expr:
        raw = commit(e.lower(neuron.spike_value_expr, 2 * q.data_width), 2 * q.data_width)
        value = b.word_select(fired, raw, a.constant(0, q.data_width), f"i{q.data_width}")
    return EquationEvent(states, fired, value)
