# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Symbolic partial derivative of a DSL equation expression

"""Symbolic partial derivative of a neuron-DSL equation, ``∂f/∂x``.

Exponential-integrator schemes (linearised exponential Euler / Rush–Larsen) need
the diagonal Jacobian term ``A = ∂f/∂x`` for each state variable ``x`` whose
dynamics are ``dx/dt = f(x, …)``. This module differentiates the equation's
right-hand-side *string* with respect to one variable and returns another
equation string in the **same restricted grammar** the golden model and the
Verilog emitter already evaluate — so one derivative expression drives every
backend and the numerics stay consistent by construction.

The derivative is computed symbolically (SymPy) over the smooth subset of the
grammar (arithmetic, powers, and ``exp/log/sqrt/sin/cos/tanh/sinh/cosh`` plus the
conductance helpers ``exprel/sigmoid``). A sub-expression the differentiator
cannot see through (``abs/min/max/clip``, comparisons, conditionals, ``%``,
``//``) is admissible **only** when it does not depend on the differentiation
variable — then its derivative is zero and it survives verbatim as a coefficient.
A genuine dependence on such a term raises
:class:`ExpressionDifferentiationError` rather than silently linearising through a
kink — an exponential step through a non-differentiable term would be a
fabrication, not a faithful discretisation.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from typing import cast

import sympy


class exprel(sympy.Function):  # noqa: N801 - DSL/print token
    """SymPy image of the DSL ``exprel(x) = (exp(x) - 1) / x`` (exprel(0) = 1)."""

    def fdiff(self, argindex: int = 1) -> sympy.Expr:
        """Return d/dx (exp(x) - 1)/x = (exp(x)·(x - 1) + 1) / x²."""
        x = _as_expr(self.args[0])
        numerator = _as_expr(sympy.exp(x) * (x - 1) + 1)
        return _as_expr(numerator / x**2)


class sigmoid(sympy.Function):  # noqa: N801 - DSL/print token
    """SymPy image of the DSL logistic ``sigmoid(x) = 1/(1 + exp(-x))``."""

    def fdiff(self, argindex: int = 1) -> sympy.Expr:
        """Return sigmoid(x)·(1 - sigmoid(x))."""
        x = _as_expr(self.args[0])
        sigma = _sigmoid_expr(x)
        return _as_expr(sigma * (1 - sigma))


# Binary operators that are smooth wherever their operands are defined.
_BINOPS: dict[type[ast.operator], Callable[[sympy.Expr, sympy.Expr], sympy.Expr]] = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a**b,
}

_PLACEHOLDER_PREFIX = "_opaque_"


class ExpressionDifferentiationError(ValueError):
    """Raised when an expression cannot be faithfully differentiated in-grammar."""


def differentiate(expr: str, wrt: str) -> str:
    """Return ``∂expr/∂wrt`` as an equation string in the DSL grammar.

    Parameters
    ----------
    expr:
        The right-hand-side of ``d(wrt)/dt`` — an equation string already valid
        under the neuron-DSL grammar.
    wrt:
        The state-variable name to differentiate with respect to.

    Returns
    -------
    str
        The partial derivative as a new equation string using only DSL-grammar
        tokens. ``"0"`` when the expression does not depend on ``wrt``.

    Raises
    ------
    ExpressionDifferentiationError
        If the expression depends on ``wrt`` through a construct whose derivative
        is not expressible in the smooth grammar (``abs``/``clip``/``min``/
        ``max``, a comparison, a conditional, ``%``, ``//``, or an unknown
        function).
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:  # pragma: no cover - equations are pre-validated
        raise ExpressionDifferentiationError(f"invalid expression syntax: {expr!r}") from exc

    converter = _Converter(wrt)
    sympy_expr = converter.convert(tree.body)
    derivative = _diff(sympy_expr, _symbol(wrt))
    return converter.render(derivative)


def _as_expr(value: object) -> sympy.Expr:
    """Return a SymPy expression value behind SymPy's partially typed API."""
    return cast(sympy.Expr, value)


def _diff(expr: sympy.Expr, symbol: sympy.Expr) -> sympy.Expr:
    """Return ``d(expr)/d(symbol)`` through SymPy's untyped differentiator."""
    return cast(sympy.Expr, sympy.diff(expr, symbol))  # type: ignore[no-untyped-call] # SymPy API is untyped.


def _float(value: float) -> sympy.Expr:
    """Return a SymPy float through SymPy's untyped constructor."""
    return cast(sympy.Expr, sympy.Float(value))  # type: ignore[no-untyped-call] # SymPy API is untyped.


def _integer(value: int) -> sympy.Expr:
    """Return a SymPy integer through SymPy's partially typed constructor."""
    return cast(sympy.Expr, sympy.Integer(value))


def _symbol(name: str) -> sympy.Expr:
    """Return a SymPy symbol through SymPy's untyped constructor."""
    return cast(sympy.Expr, sympy.Symbol(name))  # type: ignore[no-untyped-call] # SymPy API is untyped.


def _exprel_expr(*args: sympy.Expr) -> sympy.Expr:
    """Return an ``exprel`` SymPy call as a typed expression."""
    return cast(sympy.Expr, exprel(*args))


def _sigmoid_expr(*args: sympy.Expr) -> sympy.Expr:
    """Return a ``sigmoid`` SymPy call as a typed expression."""
    return cast(sympy.Expr, sigmoid(*args))


# DSL call names whose derivative is expressible in the same grammar.
_SMOOTH_FUNCS: dict[str, Callable[..., sympy.Expr]] = {
    "exp": sympy.exp,
    "log": sympy.log,
    "sqrt": sympy.sqrt,
    "sin": sympy.sin,
    "cos": sympy.cos,
    "tanh": sympy.tanh,
    "sinh": sympy.sinh,
    "cosh": sympy.cosh,
    "exprel": _exprel_expr,
    "sigmoid": _sigmoid_expr,
}


class _Converter:
    """Convert a validated DSL AST to SymPy and back, tracking opaque terms.

    A non-smooth sub-expression that does not depend on the differentiation
    variable is replaced by a fresh placeholder symbol whose original source text
    is remembered, so it round-trips verbatim when it survives as a coefficient.
    """

    def __init__(self, wrt: str) -> None:
        self._wrt = wrt
        self._sources: dict[str, str] = {}

    def convert(self, node: ast.expr) -> sympy.Expr:
        """Return the SymPy image of a DSL AST node."""
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ExpressionDifferentiationError(f"non-numeric constant {node.value!r}")
            if isinstance(node.value, int):
                return _integer(node.value)
            return _float(node.value)
        if isinstance(node, ast.Name):
            return _symbol(node.id)
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                return _as_expr(-self.convert(node.operand))
            if isinstance(node.op, ast.UAdd):
                return self.convert(node.operand)
            return self._opaque(node)
        if isinstance(node, ast.BinOp):
            builder = _BINOPS.get(type(node.op))
            if builder is None:
                return self._opaque(node)
            return builder(self.convert(node.left), self.convert(node.right))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            func = _SMOOTH_FUNCS.get(node.func.id)
            if func is not None:
                return func(*[self.convert(arg) for arg in node.args])
        return self._opaque(node)

    def _opaque(self, node: ast.expr) -> sympy.Expr:
        """Stand an opaque sub-expression in as a constant, if it is one."""
        if any(isinstance(sub, ast.Name) and sub.id == self._wrt for sub in ast.walk(node)):
            raise ExpressionDifferentiationError(
                f"cannot differentiate through {ast.unparse(node)!r} with respect to {self._wrt!r}"
            )
        name = f"{_PLACEHOLDER_PREFIX}{len(self._sources)}"
        self._sources[name] = ast.unparse(node)
        return _symbol(name)

    def render(self, expr: sympy.Expr) -> str:
        """Render a differentiated SymPy expression back to a DSL string."""
        return _GrammarPrinter(self._sources).render(expr)


class _GrammarPrinter(sympy.printing.str.StrPrinter):
    """Print SymPy expressions using DSL-grammar tokens and opaque source text."""

    def __init__(self, sources: dict[str, str]) -> None:
        super().__init__()  # type: ignore[no-untyped-call] # SymPy printer base is untyped.
        self._sources = sources

    def render(self, expr: sympy.Expr) -> str:
        """Return the DSL string representation of ``expr``."""
        return str(self.doprint(expr))  # type: ignore[no-untyped-call] # SymPy printer API is untyped.

    def _print_Symbol(self, expr: sympy.Symbol) -> str:
        source = self._sources.get(expr.name)
        return f"({source})" if source is not None else expr.name


__all__ = ["ExpressionDifferentiationError", "differentiate", "exprel", "sigmoid"]
