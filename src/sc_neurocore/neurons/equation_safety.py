# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Equation expression safety — the gate before evaluation

"""AST-level safety validation for equation strings — the gate before evaluation.

Equation-defined neurons accept authored expression strings. Each one is parsed
and checked here, and the checked tree is then translated into a program of
plain function calls by :mod:`sc_neurocore.neurons.equation_program`. No
expression reaches the interpreter's ``compile`` or ``eval``.

The gate admits a closed grammar: the listed arithmetic, comparison and choice
nodes; names; calls of a function under a plain name; the attributes in
:data:`_ALLOWED_ATTRIBUTES`; indexing; tuples and lists. Everything else is
refused. Calling the result of an attribute, an index or another call is
refused, because the objects an expression can hold (array scalars, unit
quantities, the supplied functions) carry methods that write files and expose
their classes. Reading any other attribute is refused for the same reason.

The earlier rules stay in force and report first: builtins and escape names
are refused by name, string constants are refused (a format string's field
names reach any attribute or index, dunders included, where no node check
sees them), and deep trees, giant constants and large or nested exponents are
refused before they can exhaust the evaluator. Three further rules bound what
an accepted expression can ask for: an array function of the namespace cannot
be handed an output array (a positional argument after its inputs must be the
literal ``None``), and a tuple or list literal cannot be repeated or
concatenated. A sequence reached through a slice, a choice or a call, and an
integer power of a power, are bounded where the values are known, by the
operations of the program (:mod:`sc_neurocore.neurons.equation_program`).
"""

from __future__ import annotations

import ast

from sc_neurocore.neurons.equation_refusals import EquationRefusal

_ALLOWED_AST_NODES = {
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Compare,
    ast.BoolOp,
    ast.IfExp,
    ast.Call,
    ast.Name,
    ast.Constant,
    ast.Attribute,
    ast.Subscript,
    ast.Index,
    ast.Slice,
    ast.Load,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.FloorDiv,
    ast.USub,
    ast.UAdd,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.And,
    ast.Or,
    ast.Not,
    ast.Tuple,
    ast.List,
}
"""AST node types an equation expression may contain."""

_ALLOWED_ATTRIBUTES = frozenset({"real", "imag"})
"""Attributes an expression may read: the two parts of a numeric value.

Any other attribute of a bound object is outside the grammar. Numeric scalars
and unit quantities carry attributes that return their class, their unit
registry and methods that write to disk; none of them is equation arithmetic.
"""

_INPUT_ARGUMENTS = {
    "exp": 1,
    "log": 1,
    "sin": 1,
    "cos": 1,
    "tanh": 1,
    "cosh": 1,
    "sinh": 1,
    "exp2": 1,
    "log2": 1,
    "log10": 1,
    "expm1": 1,
    "log1p": 1,
    "tan": 1,
    "arctan": 1,
    "atan2": 2,
    "softplus": 1,
    "arcsin": 1,
    "arccos": 1,
    "erf": 1,
    "floor": 1,
    "ceil": 1,
    "trunc": 1,
    "sign": 1,
    "clip": 3,
}
"""Number of input arguments of each array function of the equation namespace.

These functions accept one further positional argument naming an output array,
which would let an expression overwrite an array that its bindings hold. A
positional argument after the inputs must be the literal ``None``, the value
that means "no output array". How many inputs a call must supply is left to
the function itself and to the callers' own signature checks.
"""

_DEFAULT_MAX_AST_DEPTH = 20
"""Maximum expression AST nesting depth (stack-exhaustion / obfuscation guard)."""

_MAX_POW_EXPONENT = 64
"""Largest literal ``**`` exponent; neuron dynamics use small integer/rational powers."""

_MAX_CONSTANT_MAGNITUDE = 1e15
"""Largest permitted numeric literal, rejecting giant constants used for eval blow-up."""

_BLOCKED_NAMES = {
    # Python builtins that enable code execution or introspection
    "__import__",
    "eval",
    "exec",
    "compile",
    "globals",
    "locals",
    "getattr",
    "setattr",
    "delattr",
    "open",
    "input",
    "breakpoint",
    "type",
    "vars",
    "dir",
    "help",
    "print",
    "exit",
    "quit",
    # Dunder attributes used in sandbox escape chains
    "__builtins__",
    "__class__",
    "__subclasses__",
    "__mro__",
    "__bases__",
    "__globals__",
    "__code__",
    "__reduce__",
    "__reduce_ex__",
    "__dict__",
    "__init_subclass__",
    "__getattr__",
    "__setattr__",
    "__delattr__",
    # String formatting resolves attribute and index lookups inside the format
    # string, where the dunder rule on ast.Attribute cannot see them.
    "format",
    "format_map",
    # Module names that must never appear as identifiers
    "os",
    "sys",
    "subprocess",
    "importlib",
    "shutil",
    "pathlib",
    "socket",
    "ctypes",
    "pickle",
}
"""Identifier and attribute names rejected regardless of AST position."""


class ExpressionSafetyValidator:
    """Validate equation expression strings against the AST allowlist.

    Holds the allowed node set, the blocked-name set, and the maximum AST depth,
    and exposes :meth:`validate` and :meth:`checked_tree`, which raise
    :class:`ValueError` on any disallowed construct. The same validator instance
    is reused for a neuron's dynamics, threshold, reset rules, and (for
    exponential Euler) its symbolic Jacobian. A program is only ever built from
    the tree that :meth:`checked_tree` returns, so no expression is evaluated
    without having passed this gate.
    """

    def __init__(self, *, max_depth: int = _DEFAULT_MAX_AST_DEPTH) -> None:
        """Create a validator with the given maximum AST depth."""
        self._max_depth = max_depth

    def validate(self, expr: str) -> None:
        """Validate an expression against the AST whitelist."""
        self.checked_tree(expr)

    def checked_tree(self, expr: str) -> ast.Expression:
        """Parse an expression, apply every rule and return the checked tree.

        Parameters
        ----------
        expr : str
            Authored expression.

        Returns
        -------
        ast.Expression
            The parsed tree, returned only after every rule accepted it.

        Raises
        ------
        EquationRefusal
            The expression does not parse or breaks a rule of the gate.

        """
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as e:
            raise EquationRefusal(f"Invalid equation syntax: {expr!r}") from e
        except RecursionError as exc:
            raise EquationRefusal("Equation expression is too deep to parse") from exc

        # Reject excessively deep ASTs (stack exhaustion / obfuscation)
        max_depth = self._ast_depth(tree)
        if max_depth > self._max_depth:
            raise EquationRefusal(
                f"Equation AST depth {max_depth} exceeds limit {self._max_depth}: {expr!r}"
            )

        for node in ast.walk(tree):
            if type(node) not in _ALLOWED_AST_NODES:
                raise EquationRefusal(
                    f"Unsafe AST node {type(node).__name__} in equation: {expr!r}"
                )
            if isinstance(node, ast.Name) and node.id in _BLOCKED_NAMES:
                raise EquationRefusal(f"Blocked function {node.id!r} in equation: {expr!r}")
            if isinstance(node, ast.Attribute):
                # Block all double-underscore attribute access
                if node.attr.startswith("__") and node.attr.endswith("__"):
                    raise EquationRefusal(
                        f"Dunder attribute access {node.attr!r} blocked in equation: {expr!r}"
                    )
                if node.attr in _BLOCKED_NAMES:
                    raise EquationRefusal(f"Blocked attribute {node.attr!r} in equation: {expr!r}")
            if isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)):
                # Equations are numeric. A string constant carried a format
                # string whose field names ('{0.__class__.__mro__}') read any
                # attribute or index, dunders included, past every other rule
                # here (CEO security review of CodeQL #484, 2026-09-30).
                raise EquationRefusal(f"Blocked string constant in equation: {expr!r}")
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, (int, float))
                and not isinstance(node.value, bool)
                and abs(node.value) > _MAX_CONSTANT_MAGNITUDE
            ):
                raise EquationRefusal(
                    f"Numeric constant {node.value!r} exceeds magnitude limit "
                    f"{_MAX_CONSTANT_MAGNITUDE:g} in equation: {expr!r}"
                )
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
                exponent = node.right
                if (
                    isinstance(exponent, ast.Constant)
                    and isinstance(exponent.value, (int, float))
                    and not isinstance(exponent.value, bool)
                    and abs(exponent.value) > _MAX_POW_EXPONENT
                ):
                    raise EquationRefusal(
                        f"Exponent {exponent.value!r} exceeds limit {_MAX_POW_EXPONENT} "
                        f"in equation: {expr!r}"
                    )
                # Chained exponents (``a ** b ** c``) blow up under eval even with small
                # literals, so reject any ``**`` nested inside another's exponent.
                if any(
                    isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.Pow)
                    for inner in ast.walk(exponent)
                ):
                    raise EquationRefusal(
                        f"Nested exponentiation blocked (eval blow-up risk): {expr!r}"
                    )

        # The closed grammar. These rules run after the rules above so that a
        # blocked name or a dunder keeps its specific refusal.
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr not in _ALLOWED_ATTRIBUTES:
                raise EquationRefusal(
                    f"Attribute {node.attr!r} is not part of the equation grammar: {expr!r}"
                )
            if isinstance(node, ast.Call) and not isinstance(node.func, ast.Name):
                raise EquationRefusal(
                    f"Only a function under a plain name can be called in an equation: {expr!r}"
                )
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                inputs = _INPUT_ARGUMENTS.get(node.func.id)
                if inputs is not None and any(
                    not (isinstance(extra, ast.Constant) and extra.value is None)
                    for extra in node.args[inputs:]
                ):
                    raise EquationRefusal(
                        f"Function {node.func.id!r} cannot be given an output argument "
                        f"in an equation: {expr!r}"
                    )
            if (
                isinstance(node, ast.BinOp)
                and isinstance(node.op, (ast.Mult, ast.Add))
                and isinstance(node.left, (ast.Tuple, ast.List))
                | isinstance(node.right, (ast.Tuple, ast.List))
            ):
                raise EquationRefusal(
                    f"A tuple or list cannot be repeated or concatenated in an equation: {expr!r}"
                )
        return tree

    @staticmethod
    def _ast_depth(node: ast.AST) -> int:
        """Return the maximum nesting depth of an AST."""
        deepest = 1
        pending = [(node, 1)]
        while pending:
            current, depth = pending.pop()
            deepest = max(deepest, depth)
            pending.extend((child, depth + 1) for child in ast.iter_child_nodes(current))
        return deepest
