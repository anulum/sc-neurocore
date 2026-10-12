# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Equation programs: validated expressions run without dynamic code execution

"""Turn a validated equation expression into a program of plain function calls.

An authored expression is parsed and checked by
:class:`~sc_neurocore.neurons.equation_safety.ExpressionSafetyValidator`, and the
checked tree is then translated node by node into nested closures. Nothing is
handed to the interpreter's ``compile`` or ``eval``: an expression can do only
what the translators in this module do. Those are arithmetic, comparison,
boolean choice, reading a name from the supplied bindings, calling a function
that the bindings hold under a plain name, reading one of the attributes the
validator lists, indexing, and building a tuple or a list.

Each translator applies the operator that the interpreter itself would apply,
in the same order and with the same short-circuit rules, so a program returns
what evaluating the expression returned and raises the same exception
categories. A name that the bindings do not hold raises :class:`NameError`.

The translators do not rely on the gate alone: a call whose target is not a
plain name, a keyword argument, an attribute outside the gate's list and a
string or bytes constant are refused here as well, so a weaker gate cannot
widen what a program does.

What a program can reach is bounded by its bindings, not only by its grammar.
The author of an expression is untrusted; the code that builds the bindings is
trusted. An expression can call every function the bindings hold, with
positional arguments it builds, and can index and read ``real``/``imag`` of
every bound value. The gate lets nothing but the literal ``None`` reach the
output-array position of the array functions of the equation namespace; a
function a caller binds under its own name is called as written. Bindings are
a plain ``dict``: a mapping that invents missing keys would invent names.

Two bounds are applied to values, where a rule on the shape of the text can be
walked around. Wherever an operand of ``*`` or ``+`` could be a tuple or list
that the expression itself built, the operation refuses such an operand, so a
sequence cannot be repeated or concatenated to a size the expression chooses.
Wherever the base of ``**`` is itself computed with ``**``, or its exponent is
anything but a numeric literal (whose size the gate limits), an integer result
longer than :data:`MAX_INTEGER_BITS` is refused before it is computed. Both
refusals are raised when the program runs. Operations that cannot meet these
cases use the plain operators and pay nothing.

Arithmetic on a plain name or a literal is the bulk of every equation, so the
binary translator reads such operands in place instead of calling a program
for each. The operand order and the operator are unchanged.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping
import operator
from typing import Any

from sc_neurocore.neurons.equation_refusals import EquationRefusal
from sc_neurocore.neurons.equation_safety import _ALLOWED_ATTRIBUTES, ExpressionSafetyValidator

EquationProgram = Callable[[dict[str, object]], Any]
"""A translated expression: called with the bindings, it returns the value."""

_BINARY: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
MAX_INTEGER_BITS = 4096
"""Longest integer, in bits, that a power of a power may produce."""


def _guarded_sequence(apply: Callable[[Any, Any], Any]) -> Callable[[Any, Any], Any]:
    """Return ``apply`` with a refusal of tuple and list operands."""

    def run(first: Any, second: Any) -> Any:
        if isinstance(first, (tuple, list)) or isinstance(second, (tuple, list)):
            raise EquationRefusal("A tuple or list cannot be repeated or concatenated")
        return apply(first, second)

    return run


def _guarded_power(base: Any, exponent: Any) -> Any:
    """Raise ``base`` to ``exponent``, refusing an integer result beyond the bit bound."""
    if (
        type(base) is int
        and type(exponent) is int
        and exponent > 0
        and base.bit_length() * exponent > MAX_INTEGER_BITS
    ):
        raise EquationRefusal("An integer power exceeds the size an equation may compute")
    return operator.pow(base, exponent)


def _holds(node: ast.AST, kinds: tuple[type[ast.AST], ...]) -> bool:
    """Tell whether a subtree holds a node of one of the given kinds."""
    return any(isinstance(inner, kinds) for inner in ast.walk(node))


def _holds_power(node: ast.AST) -> bool:
    """Tell whether a subtree computes a power."""
    return any(
        isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.Pow) for inner in ast.walk(node)
    )


def _plain_number(node: ast.AST) -> bool:
    """Tell whether a node is a numeric literal, with or without a sign.

    The gate limits the magnitude of such an exponent, so a power with one
    cannot grow past the bound unless its base is itself a power.
    """
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        node = node.operand
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    )


def _binary_operator(node: ast.BinOp) -> Callable[[Any, Any], Any]:
    """Choose the function of a binary operation, guarded only where its operands need it."""
    apply = _operator(_BINARY, node.op)
    if isinstance(node.op, (ast.Mult, ast.Add)) and (
        _holds(node.left, (ast.Tuple, ast.List)) or _holds(node.right, (ast.Tuple, ast.List))
    ):
        return _guarded_sequence(apply)
    if isinstance(node.op, ast.Pow) and (_holds_power(node.left) or not _plain_number(node.right)):
        return _guarded_power
    return apply


_UNARY: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Not: operator.not_,
}
_COMPARE: dict[type[ast.cmpop], Callable[[Any, Any], Any]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}


def compile_expression(
    expression: str, validator: ExpressionSafetyValidator | None = None
) -> EquationProgram:
    """Validate an expression and translate it into a program.

    Parameters
    ----------
    expression : str
        Authored equation, threshold, reset or rate expression.
    validator : ExpressionSafetyValidator, optional
        Gate that checks the expression; the default gate when omitted.

    Returns
    -------
    EquationProgram
        Callable taking the bindings and returning the expression value.

    Raises
    ------
    EquationRefusal
        The validator refuses the expression, or the checked tree holds a
        construct that no translator of this module covers.
    """
    gate = ExpressionSafetyValidator() if validator is None else validator
    return _translate(gate.checked_tree(expression).body)


def _operator(table: Mapping[type, Callable[..., Any]], node: ast.AST) -> Callable[..., Any]:
    """Return the function for an operator node, refusing operators outside the grammar."""
    try:
        return table[type(node)]
    except KeyError:
        raise EquationRefusal(
            f"Equation construct {type(node).__name__} cannot be evaluated"
        ) from None


def _translate(node: ast.expr) -> EquationProgram:
    """Translate one checked node; every admitted node type has one branch."""
    if isinstance(node, ast.Constant):
        return _constant(_literal(node))
    if isinstance(node, ast.Name):
        return _name(node.id)
    if isinstance(node, ast.BinOp):
        return _binary(_binary_operator(node), node.left, node.right)
    if isinstance(node, ast.UnaryOp):
        return _unary(_operator(_UNARY, node.op), _translate(node.operand))
    if isinstance(node, ast.Compare):
        return _comparison(
            _translate(node.left),
            tuple(_operator(_COMPARE, item) for item in node.ops),
            tuple(_translate(item) for item in node.comparators),
        )
    if isinstance(node, ast.BoolOp):
        return _choice(isinstance(node.op, ast.And), tuple(_translate(v) for v in node.values))
    if isinstance(node, ast.IfExp):
        return _conditional(_translate(node.test), _translate(node.body), _translate(node.orelse))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.keywords:
            raise EquationRefusal("Only a function under a plain name is called, by position")
        return _call(node.func.id, tuple(_translate(item) for item in node.args))
    if isinstance(node, ast.Attribute):
        if node.attr not in _ALLOWED_ATTRIBUTES:
            raise EquationRefusal(f"Attribute {node.attr!r} is not part of the equation grammar")
        return _attribute(_translate(node.value), node.attr)
    if isinstance(node, ast.Subscript):
        return _subscript(_translate(node.value), _translate(node.slice))
    if isinstance(node, ast.Slice):
        return _slice(
            *(
                None if part is None else _translate(part)
                for part in (node.lower, node.upper, node.step)
            )
        )
    if isinstance(node, (ast.Tuple, ast.List)):
        return _sequence(
            tuple if isinstance(node, ast.Tuple) else list, tuple(_translate(i) for i in node.elts)
        )
    raise EquationRefusal(f"Equation construct {type(node).__name__} cannot be evaluated")


def _literal(node: ast.expr) -> Any:
    """Return the value of a constant node, refusing text: equations are numeric."""
    value = node.value  # type: ignore[attr-defined]
    if isinstance(value, (str, bytes)):
        raise EquationRefusal("A string constant is not part of the equation grammar")
    return value


def _constant(value: object) -> EquationProgram:
    def run(bindings: dict[str, object]) -> Any:
        return value

    return run


def _undefined(name: object) -> NameError:
    """Build the error for a name that the bindings do not hold."""
    return NameError(f"name {name!r} is not defined")


def _name(name: str) -> EquationProgram:
    def run(bindings: dict[str, object]) -> Any:
        try:
            return bindings[name]
        except KeyError:
            raise _undefined(name) from None

    return run


def _binary(apply: Callable[[Any, Any], Any], left: ast.expr, right: ast.expr) -> EquationProgram:
    """Translate a binary operation, reading name and literal operands in place.

    Only the lookup of a name sits inside the handler that turns a missing
    binding into :class:`NameError`; a ``KeyError`` raised while an operand
    program runs is the expression's own and passes through.
    """
    left_name = left.id if isinstance(left, ast.Name) else None
    right_name = right.id if isinstance(right, ast.Name) else None
    left_literal = isinstance(left, ast.Constant)
    right_literal = isinstance(right, ast.Constant)

    if left_name is not None and right_name is not None:

        def run_names(bindings: dict[str, object]) -> Any:
            try:
                first = bindings[left_name]
            except KeyError:
                raise _undefined(left_name) from None
            try:
                second = bindings[right_name]
            except KeyError:
                raise _undefined(right_name) from None
            return apply(first, second)

        return run_names

    if left_name is not None and right_literal:
        literal = _literal(right)

        def run_name_literal(bindings: dict[str, object]) -> Any:
            try:
                first = bindings[left_name]
            except KeyError:
                raise _undefined(left_name) from None
            return apply(first, literal)

        return run_name_literal

    if left_literal and right_name is not None:
        literal = _literal(left)

        def run_literal_name(bindings: dict[str, object]) -> Any:
            try:
                second = bindings[right_name]
            except KeyError:
                raise _undefined(right_name) from None
            return apply(literal, second)

        return run_literal_name

    if left_name is not None:
        right_program = _translate(right)

        def run_name_program(bindings: dict[str, object]) -> Any:
            try:
                first = bindings[left_name]
            except KeyError:
                raise _undefined(left_name) from None
            return apply(first, right_program(bindings))

        return run_name_program

    left_program = _translate(left)
    if right_name is not None:

        def run_program_name(bindings: dict[str, object]) -> Any:
            first = left_program(bindings)
            try:
                second = bindings[right_name]
            except KeyError:
                raise _undefined(right_name) from None
            return apply(first, second)

        return run_program_name

    if right_literal:
        literal = _literal(right)

        def run_program_literal(bindings: dict[str, object]) -> Any:
            return apply(left_program(bindings), literal)

        return run_program_literal

    right_program = _translate(right)

    def run(bindings: dict[str, object]) -> Any:
        return apply(left_program(bindings), right_program(bindings))

    return run


def _unary(apply: Callable[[Any], Any], operand: EquationProgram) -> EquationProgram:
    def run(bindings: dict[str, object]) -> Any:
        return apply(operand(bindings))

    return run


def _comparison(
    first: EquationProgram,
    operators: tuple[Callable[[Any, Any], Any], ...],
    others: tuple[EquationProgram, ...],
) -> EquationProgram:
    if len(operators) == 1:
        compare, second = operators[0], others[0]

        def run_one(bindings: dict[str, object]) -> Any:
            return compare(first(bindings), second(bindings))

        return run_one

    def run(bindings: dict[str, object]) -> Any:
        # A chain compares neighbours left to right, evaluates each operand
        # once and stops at the first comparison that does not hold.
        left = first(bindings)
        outcome: Any = True
        for compare, other in zip(operators, others):
            right = other(bindings)
            outcome = compare(left, right)
            if not outcome:
                return outcome
            left = right
        return outcome

    return run


def _choice(conjunction: bool, values: tuple[EquationProgram, ...]) -> EquationProgram:
    def run(bindings: dict[str, object]) -> Any:
        # ``and`` returns the first operand that is false, ``or`` the first
        # that is true; either returns the last operand otherwise.
        outcome: Any = None
        for value in values:
            outcome = value(bindings)
            if bool(outcome) is not conjunction:
                return outcome
        return outcome

    return run


def _conditional(
    test: EquationProgram, chosen: EquationProgram, otherwise: EquationProgram
) -> EquationProgram:
    def run(bindings: dict[str, object]) -> Any:
        return chosen(bindings) if test(bindings) else otherwise(bindings)

    return run


def _call(name: str, arguments: tuple[EquationProgram, ...]) -> EquationProgram:
    """Translate a call of the function that the bindings hold under ``name``."""
    if len(arguments) == 1:
        only = arguments[0]

        def run_one(bindings: dict[str, object]) -> Any:
            try:
                target = bindings[name]
            except KeyError:
                raise _undefined(name) from None
            return target(only(bindings))  # type: ignore[operator]

        return run_one

    def run(bindings: dict[str, object]) -> Any:
        try:
            target = bindings[name]
        except KeyError:
            raise _undefined(name) from None
        return target(*[argument(bindings) for argument in arguments])  # type: ignore[operator]

    return run


def _attribute(value: EquationProgram, attribute: str) -> EquationProgram:
    read = operator.attrgetter(attribute)

    def run(bindings: dict[str, object]) -> Any:
        return read(value(bindings))

    return run


def _subscript(value: EquationProgram, index: EquationProgram) -> EquationProgram:
    def run(bindings: dict[str, object]) -> Any:
        return value(bindings)[index(bindings)]

    return run


def _slice(
    lower: EquationProgram | None, upper: EquationProgram | None, step: EquationProgram | None
) -> EquationProgram:
    def run(bindings: dict[str, object]) -> Any:
        return slice(*(None if part is None else part(bindings) for part in (lower, upper, step)))

    return run


def _sequence(
    build: Callable[[list[Any]], Any], items: tuple[EquationProgram, ...]
) -> EquationProgram:
    def run(bindings: dict[str, object]) -> Any:
        return build([item(bindings) for item in items])

    return run
