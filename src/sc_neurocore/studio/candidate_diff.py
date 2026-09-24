# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — What a candidate changes against the catalogue model it derives from

"""Diff a candidate model against its parent, mathematically and semantically.

Text comparison would call ``(v - a) / tau`` and ``v / tau - a / tau``
different. Each equation is therefore compared as mathematics as well: both
sides are converted to SymPy expressions and their difference is simplified.
A rewritten but equal equation is reported as ``equivalent``, a real change as
``changed`` with the simplified difference, and an equation that uses a
construct with no exact symbolic reading (a comparison, a conditional, a
modulo) as ``not_comparable`` rather than guessed at.

Expressions are converted from their Python AST after they have passed the
equation safety gate, node by node, so an uploaded equation is never handed
to ``sympify`` or ``eval``. The numerical helpers whose exact behaviour SymPy
does not share — ``exprel``, ``sigmoid``, ``clip``, ``max``, ``min`` — become
uninterpreted functions: two equations using them are equal only when they use
them identically.

The semantic part compares what is not an equation: threshold and detection,
reset rules, the integration method and timestep, and the initial state and
parameter values.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping
from typing import Any

from sc_neurocore.neurons.equation_safety import ExpressionSafetyValidator

DIFF_SCHEMA_VERSION = "sc-neurocore.studio.candidate-diff.v1"

_MAX_SIMPLIFY_OPS = 400
"""Largest difference, in SymPy operations, that is simplified in full."""


class _NotSymbolic(ValueError):
    """The expression has no exact symbolic reading."""


def _to_sympy(expression: str) -> Any:
    """Convert one safe DSL expression to SymPy without evaluating it."""
    import sympy

    ExpressionSafetyValidator().validate(expression)
    exact: dict[str, Callable[..., Any]] = {
        "exp": sympy.exp,
        "log": sympy.log,
        "sqrt": sympy.sqrt,
        "abs": sympy.Abs,
        "sin": sympy.sin,
        "cos": sympy.cos,
        "tanh": sympy.tanh,
        "cosh": sympy.cosh,
        "sinh": sympy.sinh,
    }
    operators: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
        ast.Add: lambda left, right: left + right,
        ast.Sub: lambda left, right: left - right,
        ast.Mult: lambda left, right: left * right,
        ast.Div: lambda left, right: left / right,
        ast.Pow: lambda left, right: left**right,
    }

    def convert(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return convert(node.body)
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
        ):
            # A decimal literal is taken exactly, so 0.5 and 1/2 are the same number.
            return sympy.Rational(repr(node.value))
        if isinstance(node, ast.Name):
            return sympy.pi if node.id == "pi" else sympy.Symbol(node.id, real=True)
        if isinstance(node, ast.BinOp) and type(node.op) in operators:
            return operators[type(node.op)](convert(node.left), convert(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            operand = convert(node.operand)
            return -operand if isinstance(node.op, ast.USub) else operand
        # Keyword arguments never reach here: the safety gate refuses them.
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            arguments = [convert(argument) for argument in node.args]
            if node.func.id in exact:
                return exact[node.func.id](*arguments)
            return sympy.Function(node.func.id)(*arguments)
        raise _NotSymbolic(type(node).__name__)

    return convert(ast.parse(expression, mode="eval"))


def compare_expressions(parent: str, candidate: str) -> dict[str, Any]:
    """Compare two equations as text and as mathematics.

    Returns
    -------
    dict
        ``status`` is ``unchanged`` (same text), ``equivalent`` (equal after
        simplification), ``changed`` (with ``difference``, candidate minus
        parent), ``undecided`` (too large to simplify here) or
        ``not_comparable`` (with ``reason``).
    """
    if parent.strip() == candidate.strip():
        return {"status": "unchanged"}
    import sympy

    try:
        difference = _to_sympy(candidate) - _to_sympy(parent)
    except _NotSymbolic as exc:
        return {"status": "not_comparable", "reason": f"uses {exc}, which has no symbolic reading"}
    expanded = sympy.expand(difference)
    if expanded == 0:
        return {"status": "equivalent"}
    if sympy.count_ops(expanded) > _MAX_SIMPLIFY_OPS:
        return {"status": "undecided", "reason": "the difference is too large to simplify here"}
    simplified = sympy.simplify(expanded)
    if simplified == 0:
        return {"status": "equivalent"}
    return {"status": "changed", "difference": str(simplified)}


def _value_rows(parent: Mapping[str, Any], candidate: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in sorted(set(parent) | set(candidate)):
        if name not in parent:
            rows.append({"name": name, "status": "added", "candidate": candidate[name]})
        elif name not in candidate:
            rows.append({"name": name, "status": "removed", "parent": parent[name]})
        else:
            same = float(parent[name]) == float(candidate[name])
            rows.append(
                {
                    "name": name,
                    "status": "unchanged" if same else "changed",
                    "parent": parent[name],
                    "candidate": candidate[name],
                }
            )
    return rows


def _equation_rows(parent: Mapping[str, Any], candidate: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in sorted(set(parent) | set(candidate)):
        if name not in parent:
            rows.append({"variable": name, "status": "added", "candidate": candidate[name]})
        elif name not in candidate:
            rows.append({"variable": name, "status": "removed", "parent": parent[name]})
        else:
            rows.append(
                {
                    "variable": name,
                    "parent": parent[name],
                    "candidate": candidate[name],
                    **compare_expressions(str(parent[name]), str(candidate[name])),
                }
            )
    return rows


def _section(model: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = model.get(key)
    return value if isinstance(value, Mapping) else {}


def diff_models(parent: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Return what ``candidate`` changes against ``parent``, section by section.

    Both are Universal DSL schema documents.
    """
    parent_threshold = _section(parent, "threshold")
    candidate_threshold = _section(candidate, "threshold")
    threshold: dict[str, Any] = {
        "detection": {
            "parent": parent_threshold.get("detection"),
            "candidate": candidate_threshold.get("detection"),
        },
    }
    if "condition" in parent_threshold and "condition" in candidate_threshold:
        threshold.update(
            compare_expressions(
                str(parent_threshold["condition"]), str(candidate_threshold["condition"])
            )
        )
    else:
        present = ("condition" in parent_threshold, "condition" in candidate_threshold)
        threshold["status"] = {
            (False, False): "unchanged",
            (True, False): "removed",
            (False, True): "added",
        }[present]
    parent_integration = _section(parent, "integration")
    candidate_integration = _section(candidate, "integration")
    integration_keys = sorted(set(parent_integration) | set(candidate_integration))
    changed_integration = [
        key
        for key in integration_keys
        if parent_integration.get(key) != candidate_integration.get(key)
    ]
    return {
        "state": _value_rows(_section(parent, "state"), _section(candidate, "state")),
        "parameters": _value_rows(
            _section(parent, "parameters"), _section(candidate, "parameters")
        ),
        "dynamics": _equation_rows(_section(parent, "dynamics"), _section(candidate, "dynamics")),
        "reset": _equation_rows(_section(parent, "reset"), _section(candidate, "reset")),
        "threshold": threshold,
        "integration": {
            "status": "changed" if changed_integration else "unchanged",
            "changed_fields": changed_integration,
            "parent": dict(parent_integration),
            "candidate": dict(candidate_integration),
        },
    }


def diff_candidate(document: Mapping[str, Any]) -> dict[str, Any]:
    """Diff a validated candidate against the catalogue model it names as parent.

    Returns
    -------
    dict
        The diff, or a statement of why there is none: the candidate names no
        parent, or the parent has no canonical schema to compare with.
    """
    parent = document.get("parent")
    base: dict[str, Any] = {"schema_version": DIFF_SCHEMA_VERSION, "parent": parent}
    if parent is None:
        return {**base, "status": "no_parent"}
    from sc_neurocore.neurons.model_identity import ModelIdentityError, schema_for_class
    from sc_neurocore.neurons.universal_dsl import load_schema

    try:
        schema_name = schema_for_class(str(parent))
        parent_model = load_schema(schema_name)
    except (ModelIdentityError, FileNotFoundError, ValueError):
        return {**base, "status": "parent_has_no_schema"}
    return {
        **base,
        "status": "compared",
        "parent_schema": schema_name,
        **diff_models(parent_model, document["model"]),
    }


__all__ = ["DIFF_SCHEMA_VERSION", "compare_expressions", "diff_candidate", "diff_models"]
