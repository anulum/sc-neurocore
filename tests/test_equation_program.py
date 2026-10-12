# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Equation programs return what the expression means, with no dynamic code

"""Run translated expressions and compare them with the arithmetic written out in Python."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from sc_neurocore.neurons.equation_namespace import build_eval_namespace
from sc_neurocore.neurons.equation_program import compile_expression
from sc_neurocore.neurons.equation_refusals import EquationRefusal
from sc_neurocore.neurons.equation_safety import ExpressionSafetyValidator

V, W, TAU = -61.5, 0.25, 12.0
NAMESPACE = build_eval_namespace()


def _bindings(**extra: Any) -> dict[str, Any]:
    """Return the numerical namespace with three scalar bindings and any extras."""
    return {**build_eval_namespace(), "v": V, "w": W, "tau": TAU, **extra}


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        # Every pairing of a name, a literal and a nested operand around one operator.
        ("v + w", V + W),
        ("v - 2.5", V - 2.5),
        ("2.5 * w", 2.5 * W),
        ("v / (w + 1)", V / (W + 1)),
        ("(v + 1) % tau", (V + 1) % TAU),
        ("(v + 1) // 3", (V + 1) // 3),
        ("(v + 1) ** (w + 1)", (V + 1) ** (W + 1)),
        ("2 ** 3", 8),
        ("-v + +w", -V + +W),
        ("not v", False),
        ("exp(w) + sqrt(tau)", NAMESPACE["exp"](W) + NAMESPACE["sqrt"](TAU)),
        ("max(v, w, tau)", TAU),
        ("clip(v, -60.0, 0.0)", np.clip(V, -60.0, 0.0)),
        ("pi * 2", math.pi * 2),
        ("v if w > 1 else tau", TAU),
        ("v if w < 1 else tau", V),
        ("abs(min(v.real, -60)) + w.imag", abs(min(V, -60))),
    ],
)
def test_arithmetic_and_calls_return_the_written_out_value(expression: str, expected: Any) -> None:
    """A program computes exactly what the same arithmetic computes in Python.

    Parameters
    ----------
    expression : str
        Authored expression over the bindings ``v``, ``w`` and ``tau``.
    expected : Any
        The same arithmetic evaluated by the interpreter in this file.
    """
    value = compile_expression(expression)(_bindings())
    assert value == expected and type(value) is type(expected)


def test_comparisons_and_choices_short_circuit_like_the_language() -> None:
    """Chains stop at the first failed comparison; ``and``/``or`` return an operand."""
    seen: list[str] = []

    def probe(label: float) -> float:
        seen.append(str(label))
        return label

    bindings = _bindings(probe=probe)
    for expression, expected in (
        ("v < w", True),
        ("v == v", True),
        ("v != v", False),
        ("v <= w <= tau", True),
        ("w >= v > tau", False),
        ("v and w", W),
        ("0 and w", 0),
        ("0 or w", W),
        ("v or w", V),
        ("0 or 0.0", 0.0),
    ):
        assert compile_expression(expression)(bindings) == expected, expression
    assert compile_expression("probe(3) < probe(2) < probe(1)")(bindings) is False
    assert seen == ["3.0", "2.0"] or seen == ["3", "2"]
    seen.clear()
    assert compile_expression("probe(1) < probe(2) < probe(3)")(bindings) is True
    assert len(seen) == 3
    seen.clear()
    assert compile_expression("probe(0) and probe(5)")(bindings) == 0
    assert compile_expression("probe(7) or probe(5)")(bindings) == 7
    assert len(seen) == 2


def test_sequences_indexing_and_slices_follow_the_language() -> None:
    """Tuples, lists, indices and all three slice parts are built from programs."""
    table = [10.0, 20.0, 30.0, 40.0]
    bindings = _bindings(table=table, grid={(1, 2): 5.0}, one=1)
    assert compile_expression("table[one] + table[-1]")(bindings) == 60.0
    assert compile_expression("table[one:3]")(bindings) == [20.0, 30.0]
    assert compile_expression("table[::2]")(bindings) == [10.0, 30.0]
    assert compile_expression("table[:one]")(bindings) == [10.0]
    assert compile_expression("grid[one, 2]")(bindings) == 5.0
    assert compile_expression("(v, w)")(bindings) == (V, W)
    assert compile_expression("[v, w][one]")(bindings) == W
    assert compile_expression("min([v, 0])")(bindings) == V


@pytest.mark.parametrize(
    "expression",
    [
        "missing",
        "missing + missing",
        "v + missing",
        "missing + 1",
        "1 + missing",
        "missing + (v + 1)",
        "(v + 1) + missing",
        "missing(v)",
        "missing(v, w)",
    ],
)
def test_an_unbound_name_is_a_name_error_wherever_it_stands(expression: str) -> None:
    """Every place that reads a name reports a missing binding as ``NameError``.

    Parameters
    ----------
    expression : str
        Expression reading the unbound name ``missing`` as an operand or a function.
    """
    with pytest.raises(NameError, match="name 'missing' is not defined") as refused:
        compile_expression(expression)(_bindings())
    assert refused.value.__cause__ is None


def test_a_key_error_of_the_expression_itself_is_not_renamed() -> None:
    """A missing dictionary key inside an operand stays a ``KeyError``."""
    bindings = _bindings(grid={1: 2.0})
    for expression in ("v + grid[7]", "grid[7] + v", "grid[7] + 1", "exp(grid[7])"):
        with pytest.raises(KeyError):
            compile_expression(expression)(bindings)


def test_value_failures_keep_their_categories() -> None:
    """Division, indexing, attribute and type failures surface as the language raises them."""
    bindings = _bindings(table=[1.0])
    for expression, category in (
        ("1 / (w - w)", ZeroDivisionError),
        ("table[3]", IndexError),
        ("table.real", AttributeError),
        ("table + 1", TypeError),
        ("min([])", ValueError),
    ):
        with pytest.raises(category):
            compile_expression(expression)(bindings)


def test_a_program_is_built_only_from_a_tree_the_gate_accepted() -> None:
    """Refused text never becomes a program, and the caller's own gate is the one applied."""
    for expression in ("__import__('os')", "v.tofile(w)", "v.dtype", "f(v)(w)", "lambda: 1", "v +"):
        with pytest.raises(EquationRefusal):
            compile_expression(expression)
    shallow = ExpressionSafetyValidator(max_depth=2)
    with pytest.raises(EquationRefusal, match="exceeds limit 2"):
        compile_expression("v + w + 1", shallow)
    assert compile_expression("v + w + 1", ExpressionSafetyValidator())(_bindings()) == V + W + 1


class _WideGate(ExpressionSafetyValidator):
    """A gate that parses without checking, standing in for a gate that grew too wide."""

    def checked_tree(self, expr: str) -> Any:
        """Return the parsed tree without applying any rule."""
        import ast

        return ast.parse(expr, mode="eval")


@pytest.mark.parametrize(
    ("expression", "reason"),
    [
        ("v.tofile(w)", "plain name"),
        ("f(v)(w)", "plain name"),
        ("clip(v, a_min=0)", "plain name"),
        ("v.dtype", "not part of the equation grammar"),
        ("v.__class__", "not part of the equation grammar"),
        ("lambda: v", "cannot be evaluated"),
        ("{v: w}", "cannot be evaluated"),
        ("[x for x in v]", "cannot be evaluated"),
        ("f'{v}'", "cannot be evaluated"),
        ("~v", "cannot be evaluated"),
        ("v << 2", "cannot be evaluated"),
        ("v is w", "cannot be evaluated"),
        ("'{0.__class__}'", "string constant"),
        ("'a' * 3", "string constant"),
        ("b'%d' % v", "string constant"),
        ("v + 'a'", "string constant"),
        ("(v + 1) + 'a'", "string constant"),
        ("v in w", "cannot be evaluated"),
    ],
)
def test_the_translator_refuses_what_a_wider_gate_would_let_through(
    expression: str, reason: str
) -> None:
    """The translator has no branch for anything outside the closed grammar.

    Parameters
    ----------
    expression : str
        Expression that only a gate without rules would pass on.
    reason : str
        Part of the translator's refusal.
    """
    with pytest.raises(EquationRefusal, match=reason):
        compile_expression(expression, _WideGate())


@pytest.mark.parametrize(
    "expression",
    [
        "[0.0][:] * 4",
        "4 * [0.0][:]",
        "(v, w)[0:2] * one",
        "([0.0] if 1 else [0.0]) * 4",
        "max([0.0], [0.0]) * 4",
        "(v, w)[0:1] + (v, w)[1:2]",
        "min([[0.0]]) + min([[1.0]])",
    ],
)
def test_a_sequence_the_expression_built_cannot_be_repeated_however_it_is_wrapped(
    expression: str,
) -> None:
    """A slice, a choice or a call around a literal sequence does not carry it past the bound.

    These are the forms of the outside review of 2f778166f: a rule on the
    shape of the text accepted each of them, and each repeated or joined a
    sequence.

    Parameters
    ----------
    expression : str
        Expression whose ``*`` or ``+`` receives a tuple or list at run time.
    """
    program = compile_expression(expression)
    with pytest.raises(EquationRefusal, match="cannot be repeated or concatenated"):
        program(_bindings(one=1))


@pytest.mark.parametrize(
    "expression",
    [
        "((2 ** 64) * 1) ** 64",
        "(-(2 ** 64)) ** 64",
        "abs(2 ** 64) ** 64",
        "((2 ** 64) ** 64) ** 2",
        "(((2 ** 60) + 0) ** 60) ** 60",
    ],
)
def test_an_integer_power_of_a_power_is_bounded_however_it_is_wrapped(expression: str) -> None:
    """A product, a sign or a call around the inner power does not carry it past the bit bound.

    Parameters
    ----------
    expression : str
        Expression whose integer result would be longer than the bound.
    """
    program = compile_expression(expression)
    with pytest.raises(EquationRefusal, match="integer power exceeds"):
        program(_bindings())


def test_the_value_bounds_leave_ordinary_arithmetic_alone() -> None:
    """Elements of a sequence, small integer towers and float towers compute as before."""
    bindings = _bindings(a=3.0, b=2.0, table=[1.0, 2.0])
    for expression, expected in (
        ("(v, w)[0] * 2", V * 2),
        ("min([v, 0]) * 2 + max([w, 0])", V * 2 + W),
        ("(2 ** 3) ** 2", 64),
        ("(2 ** 64) ** 2", 2**128),
        ("(a ** 2) ** 3", 729.0),
        ("(a ** b) ** 2", 81.0),
        ("(2 ** 3) ** -1", 0.125),
        ("(a ** 2) ** 64 > 0", True),
        ("table * 2", [1.0, 2.0, 1.0, 2.0]),
    ):
        assert compile_expression(expression)(bindings) == expected, expression
