# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source-located function argument domains and numerical evidence
"""Keep separate argument domains for every authored or implicit function call.

An authored call can be replicated by integration stages. Its interval covers
every instance because state and input words are initially unconstrained. No
trajectory invariant, stage correlation, generated latency or area improvement
is inferred. This metadata does not change unit selection or generated values.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from ..neurons.equation_builder import EquationNeuron
from . import expr_lut_tables
from .equation_lowering import equation_expressions, prepare_neuron
from .fixed_expression_ranges import CodeRange, expression_code_range
from .function_call_accuracy import CallAccuracy, measure_call
from .lut_accuracy import GEOMETRY


@dataclass(frozen=True, slots=True)
class FunctionCallContract:
    """One source-located call, its conservative operand ranges and measured error.

    Parameters
    ----------
    location : str
        Equation, event, Jacobian or generated integration-expression location.
    line, column : int
        One-based source line and zero-based UTF-8 byte column within that expression.
    function : str
        Canonical unit name, independently of spelling aliases.
    expression : str
        Complete call or power expression.
    arguments : tuple of str
        Operand expressions, in call order; literal exponents are separate.
    ranges : tuple of CodeRange
        Inclusive conservative signed-code intervals at the function boundary.
    accuracy : CallAccuracy or None
        Measured numerical scope, or None when admission failed.
    exponent : int or float or None, optional
        Independent literal exponent; None also denotes a dynamic power.
    implicit : bool, optional
        Whether the integration method introduces this call.
    refusal : str, optional
        Why argument inference or accuracy measurement was refused.
    """

    location: str
    line: int
    column: int
    function: str
    expression: str
    arguments: tuple[str, ...]
    ranges: tuple[CodeRange, ...]
    accuracy: CallAccuracy | None
    exponent: int | float | None = None
    implicit: bool = False
    refusal: str = ""

    def to_public_dict(self, fraction: int) -> dict[str, object]:
        """Return source identity, actual argument hulls and explicit measured scope."""
        result: dict[str, object] = {
            "location": self.location,
            "line": self.line,
            "column": self.column,
            "function": self.function,
            "expression": self.expression,
            "arguments": list(self.arguments),
            "argument_ranges": [bounds.to_public_dict(fraction) for bounds in self.ranges],
            "accuracy": None if self.accuracy is None else self.accuracy.to_public_dict(fraction),
            "implicit": self.implicit,
            "refusal": self.refusal,
        }
        if self.function == "power":
            result["exponent"] = (
                {"kind": "dynamic"}
                if self.exponent is None
                else {"kind": "literal", "value": self.exponent}
            )
        return result


def function_call_contracts(
    neuron: EquationNeuron,
    data_width: int,
    fraction: int,
    *,
    rounding: str = "truncate",
    function_lowering: str = "v2",
) -> tuple[FunctionCallContract, ...]:
    """Infer and measure each approximation call without deduplicating its name.

    Parameters
    ----------
    neuron : EquationNeuron
        Authored model; compiler-owned noise and threshold intrinsics are lowered once.
    data_width, fraction : int
        Shared signed word format.
    rounding : str, optional
        Fixed-point multiply policy.
    function_lowering : str, optional
        ``v2`` reports all registered units and powers. ``lut-v1`` reports its
        approximation-table calls; exact arithmetic is not a table call.

    Returns
    -------
    tuple of FunctionCallContract
        One record per source occurrence, plus each implicit exponential-Euler
        exprel call. Invalid call arguments retain a refusal instead of an error claim.

    Raises
    ------
    ValueError
        For an invalid lowering, or an authored call shadowed by a numeric binding.
    """
    if function_lowering not in {"v2", "lut-v1"}:
        raise ValueError(f"unknown function lowering {function_lowering!r}")
    model = prepare_neuron(neuron)
    values = {**model.parameters, **model.constants}
    variables = frozenset({*model.equations, *model.initial_state})
    bound = set(values) | variables
    expressions = [
        (where, ast.parse(expression, mode="eval").body, False)
        for where, expression in equation_expressions(model)
    ]
    if model.method == "exp_euler":
        for variable, expression in model.jacobian_expressions.items():
            argument = ast.BinOp(
                ast.parse(expression, mode="eval").body, ast.Mult(), ast.Constant(model.dt)
            )
            call = ast.fix_missing_locations(
                ast.Call(ast.Name("exprel", ast.Load()), [argument], [])
            )
            expressions.append((f"exponential Euler {variable}", call, True))
    calls: list[FunctionCallContract] = []
    for where, root, implicit in expressions:
        nodes = [root] if implicit else ast.walk(root)
        for node in sorted(
            (child for child in nodes if isinstance(child, (ast.Call, ast.BinOp))),
            key=lambda child: (child.lineno, child.col_offset),
        ):
            function: str | None = None
            exponent: int | float | None = None
            arguments: list[ast.expr] = []
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                function = expr_lut_tables.V2_UNIT_FOR.get(node.func.id)
                arguments = node.args
            elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
                kind, exponent = expr_lut_tables.classify_power(
                    node.right, function_lowering, bound
                )
                if kind != "multiply":
                    function = kind
                    arguments = [node.left] + (
                        [node.right] if kind == "power" and exponent is None else []
                    )
            if function is None or function_lowering == "lut-v1" and function not in GEOMETRY:
                continue
            text = tuple(ast.unparse(argument) for argument in arguments)
            ranges: tuple[CodeRange, ...] = ()
            accuracy, refusal = None, ""
            try:
                if isinstance(node, ast.Call):
                    assert isinstance(node.func, ast.Name)
                    expr_lut_tables.check_call_arguments(node.func.id, node)
                ranges = tuple(
                    expression_code_range(
                        argument,
                        data_width,
                        fraction,
                        values=values,
                        variables=variables,
                        rounding=rounding,
                        function_lowering=function_lowering,
                    )
                    for argument in text
                )
            except ValueError as exc:
                refusal = str(exc)
            else:
                accuracy = measure_call(
                    function,
                    data_width,
                    fraction,
                    ranges,
                    lowering=function_lowering,
                    exponent=exponent if function == "power" else None,
                )
            calls.append(
                FunctionCallContract(
                    where,
                    node.lineno,
                    node.col_offset,
                    function,
                    ast.unparse(node),
                    text,
                    ranges,
                    accuracy,
                    exponent if function == "power" else None,
                    implicit,
                    refusal,
                )
            )
    return tuple(calls)
