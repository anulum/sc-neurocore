# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Equation expression safety — the eval sandbox gate

"""AST-level safety validation for equation strings — the eval sandbox gate.

Equation-defined neurons compile user-supplied expression strings and evaluate
them with :func:`eval`. Before an expression is ever compiled it is validated
here against an AST allowlist: only the whitelisted maths/comparison node types
survive, dangerous builtins and sandbox-escape dunder chains are rejected by
name, and pathologically deep trees are refused. Together with the empty
``__builtins__`` in :data:`EVAL_GLOBALS` (so ``eval``/``exec``/``__import__``
are unreachable from inside a compiled expression) this is the security boundary
that makes the ``# nosec B307`` eval sites at the runtime call sites sound.

Extracting the validator into its own module makes the sandbox independently
testable and keeps the rationale for every ``nosec`` co-located with the gate it
depends on.
"""

from __future__ import annotations

import ast

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

EVAL_GLOBALS = {
    "__builtins__": {"__import__": __import__},
}
"""Globals for the compiled-expression ``eval`` sites: an empty builtins sandbox."""


class ExpressionSafetyValidator:
    """Validate equation expression strings against the AST allowlist.

    Holds the allowed node set, the blocked-name set, and the maximum AST depth,
    and exposes :meth:`validate`, which raises :class:`ValueError` on any
    disallowed construct. The same validator instance is reused for a neuron's
    dynamics, threshold, reset rules, and (for exponential Euler) its symbolic
    Jacobian, so every expression that reaches an ``eval`` site has passed the
    same gate.
    """

    def __init__(self, *, max_depth: int = _DEFAULT_MAX_AST_DEPTH) -> None:
        """Create a validator with the given maximum AST depth."""
        self._max_depth = max_depth

    def validate(self, expr: str) -> None:
        """Validate an expression against the AST whitelist."""
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as e:
            raise ValueError(f"Invalid equation syntax: {expr!r}") from e

        # Reject excessively deep ASTs (stack exhaustion / obfuscation)
        max_depth = self._ast_depth(tree)
        if max_depth > self._max_depth:
            raise ValueError(
                f"Equation AST depth {max_depth} exceeds limit {self._max_depth}: {expr!r}"
            )

        for node in ast.walk(tree):
            if type(node) not in _ALLOWED_AST_NODES:
                raise ValueError(f"Unsafe AST node {type(node).__name__} in equation: {expr!r}")
            if isinstance(node, ast.Name) and node.id in _BLOCKED_NAMES:
                raise ValueError(f"Blocked function {node.id!r} in equation: {expr!r}")
            if isinstance(node, ast.Attribute):
                # Block all double-underscore attribute access
                if node.attr.startswith("__") and node.attr.endswith("__"):
                    raise ValueError(
                        f"Dunder attribute access {node.attr!r} blocked in equation: {expr!r}"
                    )
                if node.attr in _BLOCKED_NAMES:
                    raise ValueError(f"Blocked attribute {node.attr!r} in equation: {expr!r}")
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, (int, float))
                and not isinstance(node.value, bool)
                and abs(node.value) > _MAX_CONSTANT_MAGNITUDE
            ):
                raise ValueError(
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
                    raise ValueError(
                        f"Exponent {exponent.value!r} exceeds limit {_MAX_POW_EXPONENT} "
                        f"in equation: {expr!r}"
                    )
                # Chained exponents (``a ** b ** c``) blow up under eval even with small
                # literals, so reject any ``**`` nested inside another's exponent.
                if any(
                    isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.Pow)
                    for inner in ast.walk(exponent)
                ):
                    raise ValueError(f"Nested exponentiation blocked (eval blow-up risk): {expr!r}")

    @staticmethod
    def _ast_depth(node: ast.AST) -> int:
        """Return the maximum nesting depth of an AST."""
        children = list(ast.iter_child_nodes(node))
        if not children:
            return 1
        return 1 + max(ExpressionSafetyValidator._ast_depth(c) for c in children)
